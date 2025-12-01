import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from katabatic.models.base_model import Model
from .utils import EmbeddingGenerator, EmbeddingDiscriminator




class PateGan(Model):
    def __init__(
        self,
        delta: float = 1e-5,
        epsilon: float = 5.0,
        batch_size: int = 128,
        max_iterations: int = 100000,
        lr: float = 1e-3,
        latent_dim: int = 64,
        n_teachers: int = 15,
        n_teacher_iters: int = 50,
        n_student_iters: int = 10,
        noise_lambda: float = 10.0,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        device: str = None,
        use_class_weights: bool = True,
        **kwargs
    ):
        super().__init__()
        self.delta = delta
        self.target_epsilon = epsilon
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.lr = lr
        self.latent_dim = latent_dim
        self.n_teachers = n_teachers
        self.n_teacher_iters = n_teacher_iters
        self.n_student_iters = n_student_iters
        self.noise_lambda = noise_lambda
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_class_weights = use_class_weights

        # Models
        self.generator = None
        self.student = None
        self.teachers = []
        self.partitions = []

        # Data preprocessing
        self.cat_cols = []
        self.cat_dims = [] 
        self._target_col = None
        self._train_df = None
        self.class_weights = None
        # Privacy accounting
        self.alpha_accum = None
        self.queries_made = 0
        self.is_fitted = False

    def _preprocess_data(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Data is already label-encoded. Just convert to tensor.
        Returns: LongTensor of shape (n_samples, n_features) with category indices.
        """
        # Assume data is already integer-encoded
        X = df.values.astype(np.int64)  # (n_samples, n_features)
        
        # Determine number of categories per feature
        self.cat_dims = []
        for col in df.columns:
            n_cats = int(df[col].max()) + 1
            self.cat_dims.append(n_cats)
        
        # NEW: Calculate class weights for the target column
        if self.use_class_weights:
            target_idx = df.columns.get_loc(self._target_col)
            n_classes = self.cat_dims[target_idx]
            class_counts = df[self._target_col].value_counts().sort_index()
            total = len(df)
            
            # Inverse frequency weighting
            self.class_weights = torch.FloatTensor([
                total / (n_classes * max(class_counts.get(i, 1), 1)) 
                for i in range(n_classes)
            ]).to(self.device)
            
            print(f"Class distribution: {class_counts.to_dict()}")
            print(f"Class weights: {self.class_weights.cpu().numpy()}")
        
        return torch.LongTensor(X).to(self.device)

    def _postprocess_data(self, tensor: torch.Tensor) -> pd.DataFrame:
        """
        Convert tensor back to DataFrame with integer codes.
        tensor: LongTensor of shape (n_samples, n_features)
        """
        data = tensor.cpu().numpy()
        
        # Clamp to valid category ranges
        for i in range(data.shape[1]):
            data[:, i] = np.clip(data[:, i], 0, self.cat_dims[i] - 1)
        
        # Create DataFrame with original column names
        df_dict = {}
        for i, col in enumerate(self._train_df.columns):
            df_dict[col] = data[:, i].astype(int)
        
        return pd.DataFrame(df_dict)

    def partition_data(self, data: torch.Tensor):
        """Partition data into disjoint subsets for teachers"""
        n = len(data)
        idx = torch.randperm(n)
        data = data[idx]
        size = n // self.n_teachers
        return [data[i*size:(i+1)*size] for i in range(self.n_teachers)]

    def pate_mechanism(self, samples: torch.Tensor):
        batch_size = samples.size(0)
        
        with torch.no_grad():
            votes = []
            for teacher in self.teachers:
                logits = teacher(samples)
                vote = (torch.sigmoid(logits) > 0.5).float().squeeze(-1)
                votes.append(vote)
            
            votes = torch.stack(votes, dim=0)  # (n_teachers, batch)
            
            # Count votes for each class
            n1 = votes.sum(dim=0)  # votes for class 1, shape (batch,)
            n0 = self.n_teachers - n1  # votes for class 0
        
        # Step 2: Add Laplace(λ) noise
        laplace_dist = torch.distributions.Laplace(
            loc=0.0,
            scale=self.noise_lambda
        )
        noise = laplace_dist.sample((batch_size,)).to(self.device)
        
        # Add noise to counts
        noisy_n1 = n1 + noise
        noisy_n0 = n0 - noise
        
        # Step 3: Argmax to get noisy label
        labels = (noisy_n1 > noisy_n0).float().unsqueeze(1)  # (batch, 1)
        
        # Step 4: Update privacy accountant
        self._update_privacy_accountant(n0, n1)
        self.queries_made += 1
        
        return labels

    def _update_privacy_accountant(self, n0: torch.Tensor, n1: torch.Tensor):
        if self.alpha_accum is None:
            self.alpha_accum = torch.zeros(100, device=self.device)
        
        # Compute vote difference across batch
        diff = torch.abs(n0 - n1).float()
        
        mean_diff = diff.mean().item()
        
        if self.noise_lambda * mean_diff > 50.0:
            mean_diff = 50.0 / self.noise_lambda
        
        # Compute q
        numerator = 2.0 + self.noise_lambda * mean_diff
        denominator = 4.0 * np.exp(self.noise_lambda * mean_diff)
        q = numerator / denominator
        q = np.clip(q, 1e-12, 0.5)
        
        # Update alpha for each l
        for l in range(1, 101):
            # Term 1: 2λ²l(l+1)
            term1 = 2.0 * (self.noise_lambda ** 2) * l * (l + 1)
            
            # Term 2: log((1-q)^a + q·e^(2λl))
            exp_2lambda = np.exp(2.0 * self.noise_lambda)
            denom = 1.0 - exp_2lambda * q
            
            if denom <= 1e-10 or q >= 0.499:
                increment = term1
            else:
                base = (1.0 - q) / denom
                
                if base <= 0:
                    increment = term1
                else:
                    try:
                        log_term1 = np.log(1.0 - q) + l * np.log(base)
                        log_term2 = np.log(q) + 2.0 * self.noise_lambda * l
                        
                        max_log = max(log_term1, log_term2)
                        term2 = max_log + np.log(
                            np.exp(log_term1 - max_log) + np.exp(log_term2 - max_log)
                        )
                        
                        increment = min(term1, term2)
                    except (ValueError, FloatingPointError):
                        increment = term1
            
            self.alpha_accum[l-1] += increment

    def get_current_epsilon(self) -> float:
        """
        Compute current epsilon using basic composition.
        """
        if self.alpha_accum is None or self.queries_made == 0:
            return 0.0
        
        return self.alpha_accum[0].item()

    def train(self, output_dir: str, **kwargs) -> 'PateGan':
        df = pd.read_csv(os.path.join(output_dir, "train_full.csv"))
        self._train_df = df.copy()
        self._target_col = df.columns[-1]
        
        self.cat_cols = list(df.columns)
        
        X = self._preprocess_data(df)
        n_features = X.shape[1]

        self.generator = EmbeddingGenerator(
            self.latent_dim, 
            self.cat_dims,
            hidden_dim=self.hidden_dim,
            embed_dim=self.embed_dim
        ).to(self.device)
        
        self.student = EmbeddingDiscriminator(
            self.cat_dims,
            hidden_dim=self.hidden_dim,
            embed_dim=self.embed_dim
        ).to(self.device)
        
        self.teachers = [
            EmbeddingDiscriminator(
                self.cat_dims,
                hidden_dim=16,
                embed_dim=4
            ).to(self.device) 
            for _ in range(self.n_teachers)
        ]
        
        # Partition data
        self.partitions = self.partition_data(X)
        
        # Optimizers
        g_opt = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        s_opt = optim.Adam(self.student.parameters(), lr=self.lr, betas=(0.5, 0.999))
        t_opts = [optim.Adam(t.parameters(), lr=self.lr, betas=(0.5, 0.999)) for t in self.teachers]

        # Reset privacy accounting
        self.alpha_accum = None
        self.queries_made = 0

        print(f"[PATE-GAN] Training | ε_target={self.target_epsilon} | λ={self.noise_lambda} | Teachers={self.n_teachers} | Weighted={self.use_class_weights}")

        for it in range(self.max_iterations):
            # Train teachers
            for _ in range(self.n_teacher_iters):
                z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
                fake = self.generator(z).detach()
                
                for t, part, opt in zip(self.teachers, self.partitions, t_opts):
                    if len(part) < self.batch_size:
                        continue
                    
                    idx = torch.randint(0, len(part), (self.batch_size,))
                    real = part[idx]
                    
                    opt.zero_grad()
                    
                    real_pred = t(real)
                    fake_pred = t(fake)
                    
                    if self.use_class_weights and self.class_weights is not None:
                        target_idx = self._train_df.columns.get_loc(self._target_col)
                        real_classes = real[:, target_idx]
                        weights = self.class_weights[real_classes]
                        
                        loss_real = (nn.BCEWithLogitsLoss(reduction='none')(
                            real_pred, torch.ones_like(real_pred)
                        ).squeeze() * weights).mean()
                        
                        loss_fake = nn.BCEWithLogitsLoss()(
                            fake_pred, torch.zeros_like(fake_pred)
                        )
                        
                        loss = loss_real + loss_fake
                    else:
                        # Standard unweighted loss
                        loss = nn.BCEWithLogitsLoss()(real_pred, torch.ones_like(real_pred)) + \
                               nn.BCEWithLogitsLoss()(fake_pred, torch.zeros_like(fake_pred))
                    
                    loss.backward()
                    opt.step()

            z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
            fake = self.generator(z).detach()
            
            labels = self.pate_mechanism(fake)
            
            for _ in range(self.n_student_iters):
                s_opt.zero_grad()
                loss = nn.BCEWithLogitsLoss()(self.student(fake), labels)
                loss.backward()
                s_opt.step()

            # Train generator
            z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
            fake = self.generator(z)
            
            g_opt.zero_grad()
            loss_g = -self.student(fake).mean()
            loss_g.backward()
            g_opt.step()

            
            if it % 20 == 0:
                eps = self.get_current_epsilon()
                print(f"Iter {it:5d} | ε ≈ {eps:.4f}")

            if self.get_current_epsilon() >= self.target_epsilon:
                eps = self.get_current_epsilon()
                print(f"[PATE-GAN] Target ε={self.target_epsilon} reached at ε={eps:.4f}")
                break

        self.is_fitted = True
        
        # Generate synthetic data
        if kwargs.get('auto_generate_synthetic', True) and kwargs.get('synthetic_dir'):
            self.sample(num_samples=len(df), synthetic_dir=kwargs['synthetic_dir'])

        return self

    def sample(self, num_samples: int, synthetic_dir: str = None, oversample_minority: bool = True, min_samples_per_class: int = None, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic samples with optional minority class oversampling.
        
        Args:
            num_samples: Total number of samples to generate
            synthetic_dir: Directory to save synthetic data
            oversample_minority: If True, ensure minimum representation of minority classes
            min_samples_per_class: Minimum samples per class (default: num_samples // (n_classes * 10))
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained first")

        self.generator.eval()
        
        # Strategy 1: Generate and filter by class until we have balanced representation
        if oversample_minority and self.class_weights is not None:
            target_idx = self._train_df.columns.get_loc(self._target_col)
            n_classes = len(self.class_weights)
            
            # Set minimum samples per class
            if min_samples_per_class is None:
                min_samples_per_class = max(50, num_samples // (n_classes * 5))
            
            print(f"Generating with minority oversampling (min {min_samples_per_class} per class)...")
            
            # Track samples per class
            class_samples = {i: [] for i in range(n_classes)}
            
            # Generate samples until we have enough of each class
            generation_rounds = 0
            max_rounds = 100
            
            with torch.no_grad():
                while generation_rounds < max_rounds:
                    generation_rounds += 1
                    
                    z = torch.randn(self.batch_size * 2, self.latent_dim, device=self.device)
                    gen = self.generator(z)
                    
                    classes = gen[:, target_idx].cpu()
                    for class_idx in range(n_classes):
                        mask = (classes == class_idx)
                        if mask.sum() > 0:
                            class_samples[class_idx].append(gen[mask].cpu())
                    
                    min_class_count = min(
                        sum(len(batch) for batch in batches) 
                        for batches in class_samples.values()
                    )
                    
                    if min_class_count >= min_samples_per_class:
                        break
            
            final_samples = []
            for class_idx in range(n_classes):
                if len(class_samples[class_idx]) == 0:
                    continue
                
                class_data = torch.cat(class_samples[class_idx])
                
                original_prop = (self._train_df[self._target_col] == class_idx).sum() / len(self._train_df)
                target_n = max(min_samples_per_class, int(num_samples * original_prop))
                
                # Sample
                if len(class_data) >= target_n:
                    indices = torch.randperm(len(class_data))[:target_n]
                    final_samples.append(class_data[indices])
                else:
                    n_repeats = (target_n // len(class_data)) + 1
                    repeated = class_data.repeat(n_repeats, 1)
                    indices = torch.randperm(len(repeated))[:target_n]
                    final_samples.append(repeated[indices])
                
                print(f"Class {class_idx}: Generated {len(class_data)}, Using {target_n} (target: {original_prop:.2%})")
            
            tensor = torch.cat(final_samples)
            
            indices = torch.randperm(len(tensor))
            tensor = tensor[indices][:num_samples]
            
        else:
            samples = []
            with torch.no_grad():
                for i in range(0, num_samples, self.batch_size):
                    bs = min(self.batch_size, num_samples - i)
                    z = torch.randn(bs, self.latent_dim, device=self.device)
                    gen = self.generator(z)
                    samples.append(gen.cpu())
            
            tensor = torch.cat(samples)[:num_samples]
        
        df = self._postprocess_data(tensor)
        

        if self.class_weights is not None:
            target_idx = self._train_df.columns.get_loc(self._target_col)
            print("\nSynthetic vs Original Distribution:")
            synth_counts = df[self._target_col].value_counts().sort_index()
            orig_counts = self._train_df[self._target_col].value_counts().sort_index()
            for class_idx in sorted(set(synth_counts.index) | set(orig_counts.index)):
                synth_n = synth_counts.get(class_idx, 0)
                orig_n = orig_counts.get(class_idx, 0)
                synth_pct = synth_n / len(df) * 100
                orig_pct = orig_n / len(self._train_df) * 100
                print(f"Class {class_idx}: Synth={synth_n:4d} ({synth_pct:5.1f}%)  |  Orig={orig_n:4d} ({orig_pct:5.1f}%)")

        if synthetic_dir:
            os.makedirs(synthetic_dir, exist_ok=True)
            y = df[[self._target_col]]
            x = df.drop(columns=[self._target_col])
            x.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            y.to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)

        return df

    def evaluate(self):
        """Placeholder for evaluation"""
        return 0.0
    
    def get_required_dependencies(self) -> list[str]:
        return ["torch", "sklearn", "pandas", "numpy"]