import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from katabatic.models.base_model import Model
from .utils import Generator, Discriminator

class PATEGAN(Model):
    def __init__(
        self,
        target_epsilon: float = 50.0,
        delta: float = 1e-5,
        batch_size: int = 256, 
        max_iterations: int = 10000,
        lr: float = 1e-4, 
        latent_dim: int = None,
        n_teachers: int = 10,
        n_teacher_iters: int = 5,
        n_student_iters: int = 5,
        noise_lambda: float = 50.0,
        mechanism: str = 'laplace',
        hidden_dim: int = None,
        device: str = None,
        gumbel_temp: float = 0.5,
        entropy_coeff: float = 0.1,
        **kwargs
    ):
        """Initialize PATE-GAN for differentially private synthetic data generation.

        Args:
            target_epsilon: Privacy budget threshold.
            delta: Privacy loss parameter (usually 1e-5).
            batch_size: Training batch size.
            max_iterations: Max iterations.
            lr: Learning rate.
            noise_lambda: Laplace noise SCALE (b). Higher = More Noise = Better Privacy.
                          If None, defaults to 1/target_epsilon (heuristic).
        """
        super().__init__()
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.lr = lr
        self.latent_dim = latent_dim
        self.n_teachers = n_teachers
        self.n_teacher_iters = n_teacher_iters
        self.n_student_iters = n_student_iters
        self.noise_lambda = noise_lambda
        self.hidden_dim = hidden_dim
        self.gumbel_temp = gumbel_temp
        self.mechanism = mechanism
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.entropy_coeff = entropy_coeff

        self.generator = None
        self.student = None
        self.teachers = []
        self.partitions = []

        self.cat_dims = [] 
        self._target_col = None
        self._train_df = None
        self._target_classes = None

        self.alpha_accum = None
        self.rdp = None
        self.rdp_orders = np.arange(2, 101)
        self.queries_made = 0

    def _preprocess_data(self, df: pd.DataFrame) -> torch.Tensor:
        n = len(df)
        self.cat_dims = []
        self.cat_col_idx = []
        self.num_col_idx = []
        self.feature_types = []

        for i, col in enumerate(df.columns):
            series = df[col]
            if pd.api.types.is_float_dtype(series):
                self.feature_types.append('num')
                self.num_col_idx.append(i)
            else:
                nunique = series.nunique()
                if nunique <= 50 or nunique <= max(2, 0.05 * n):
                    self.feature_types.append('cat')
                    self.cat_col_idx.append(i)
                    n_cats = int(series.max()) + 1
                    self.cat_dims.append(n_cats)
                else:
                    self.feature_types.append('num')
                    self.num_col_idx.append(i)

        if len(self.cat_col_idx) > 0:
            X_cat = df.iloc[:, self.cat_col_idx].values.astype(np.int64)
            X_cat = torch.LongTensor(X_cat).to(self.device)
        else:
            X_cat = torch.empty((len(df), 0), dtype=torch.long, device=self.device)

        if len(self.num_col_idx) > 0:
            X_num = df.iloc[:, self.num_col_idx].values.astype(np.float32)
            X_num = torch.FloatTensor(X_num).to(self.device)
        else:
            X_num = torch.empty((len(df), 0), dtype=torch.float32, device=self.device)

        return X_cat, X_num

    def _indices_to_onehot(self, data_indices: torch.Tensor) -> list[torch.Tensor]:
        one_hots = []
        for idx, n_cat in enumerate(self.cat_dims):
            if data_indices.numel() == 0:
                continue
            oh = F.one_hot(data_indices[:, idx], num_classes=n_cat).float()
            one_hots.append(oh)
        return one_hots

    def _postprocess_data(self, samples_list: list[torch.Tensor]) -> pd.DataFrame:
        cat_count = len(self.cat_dims)
        n_samples = None

        vals = []
        for i, s in enumerate(samples_list):
            if i < cat_count:
                col = s.argmax(dim=1).cpu().numpy()
            else:
                col = s.squeeze(1).cpu().numpy()

            vals.append(col)
            if n_samples is None:
                n_samples = len(col)

        data = np.stack(vals, axis=1)

        df_dict = {}
        col_names = list(self._train_df.columns)
        cat_idx = 0
        num_idx = 0
        for i, t in enumerate(self.feature_types):
            if t == 'cat':
                df_dict[col_names[i]] = data[:, cat_idx].astype(int)
                cat_idx += 1
            else:
                df_dict[col_names[i]] = data[:, cat_count + num_idx]
                num_idx += 1

        return pd.DataFrame(df_dict)

    def partition_data(self, data: torch.Tensor):
        X_cat, X_num = data
        n = X_cat.size(0) if X_cat.numel() > 0 else X_num.size(0)
        
        # 1. Try to get labels for stratified split
        labels = None
        if self._train_df is not None and self._target_col is not None:
             labels = self._train_df[self._target_col].values

        if labels is None:
            # Fallback: Random split
            idx = torch.randperm(n)
            X_cat = X_cat[idx] if X_cat.numel() > 0 else X_cat
            X_num = X_num[idx] if X_num.numel() > 0 else X_num

            size = n // self.n_teachers
            partitions = []
            for i in range(self.n_teachers):
                start = i * size
                end = (i + 1) * size
                cat_part = X_cat[start:end] if X_cat.numel() > 0 else X_cat
                num_part = X_num[start:end] if X_num.numel() > 0 else X_num
                partitions.append((cat_part, num_part))
            return partitions

        # 2. Stratified Split (Balanced classes across teachers)
        labels = np.asarray(labels)
        teacher_indices = [[] for _ in range(self.n_teachers)]
        unique_classes = np.unique(labels)
        
        for cls in unique_classes:
            cls_idx = np.where(labels == cls)[0]
            np.random.shuffle(cls_idx)
            # Distribute this class indices evenly among teachers
            for i, ind in enumerate(cls_idx):
                teacher_indices[i % self.n_teachers].append(int(ind))

        partitions = []
        for inds in teacher_indices:
            inds_t = torch.LongTensor(inds).to(self.device)
            cat_part = X_cat[inds_t] if X_cat.numel() > 0 else X_cat
            num_part = X_num[inds_t] if X_num.numel() > 0 else X_num
            partitions.append((cat_part, num_part))

        return partitions

    def pate_mechanism(self, fake_samples_list: list[torch.Tensor]):
        """
        Aggregate teacher votes with Laplace noise.
        fake_samples_list: generated samples (detached)
        """
        batch_size = fake_samples_list[0].size(0)

        with torch.no_grad():
            votes = []
            for teacher in self.teachers:
                teacher.eval()
                logits = teacher(fake_samples_list)
                vote = (torch.sigmoid(logits) > 0.5).float().squeeze(-1)
                votes.append(vote)

            votes = torch.stack(votes, dim=0)
            n1 = votes.sum(dim=0)     
            n0 = self.n_teachers - n1 
        laplace = torch.distributions.Laplace(loc=0.0, scale=self.noise_lambda)
        
        noise_n1 = laplace.sample(n1.size()).to(self.device)
        noise_n0 = laplace.sample(n0.size()).to(self.device)
        
        noisy_n1 = n1 + noise_n1
        noisy_n0 = n0 + noise_n0
        labels = (noisy_n1 > noisy_n0).float().unsqueeze(1)

        # Update Accountant
        if self.mechanism == 'gaussian':
            pass
        else:
            self._update_privacy_accountant_laplace(n0, n1)
            
        self.queries_made += 1
        return labels

    def _update_privacy_accountant_laplace(self, n0: torch.Tensor, n1: torch.Tensor):
        """
        Update moments accountant.
        Ref: PATE-GAN Theorem 5 / PATE Paper.
        """
        if self.alpha_accum is None:
            self.alpha_accum = torch.zeros(100, device=self.device)

        diff = torch.abs(n0 - n1).float()
        mean_diff = diff.mean().item()
        lam = 1.0 / self.noise_lambda 

        # Numerical stability clip
        if lam * mean_diff > 50.0:
            mean_diff = 50.0 / lam

        numerator = 2.0 + lam * mean_diff
        denominator = 4.0 * np.exp(lam * mean_diff)
        q = numerator / denominator
        q = np.clip(q, 1e-12, 0.5)

        # Update moments
        for l in range(1, 101):
            term1 = 2.0 * (lam ** 2) * l * (l + 1)
            
            exp_2lam = np.exp(2.0 * lam)
            denom = 1.0 - exp_2lam * q

            increment = term1
            if denom > 1e-10 and q < 0.499:
                base = (1.0 - q) / denom
                if base > 0:
                    try:
                        log_term1 = np.log(1.0 - q) + l * np.log(base)
                        log_term2 = np.log(q) + 2.0 * lam * l

                        max_log = max(log_term1, log_term2)
                        term2 = max_log + np.log(
                            np.exp(log_term1 - max_log) + np.exp(log_term2 - max_log)
                        )

                        increment = min(term1, term2)
                    except (ValueError, FloatingPointError):
                        increment = term1

            self.alpha_accum[l-1] += increment

    def get_current_epsilon(self) -> float:
        if self.alpha_accum is None:
            return 0.0
        
        A = self.alpha_accum.cpu().numpy()
        orders = np.arange(1, len(A) + 1)

        eps_candidates = (A + math.log(1.0 / max(self.delta, 1e-10))) / (orders) 
        
        return float(np.min(eps_candidates))

    def _get_gumbel_temperature(self, progress: float) -> float:
        # Anneal from gumbel_temp down to 0.1
        temp_min = 0.1
        return self.gumbel_temp * (1 - 0.8 * progress) + temp_min * (0.8 * progress)

    def train(self, output_dir: str, **kwargs) -> 'PATEGAN':
        df = pd.read_csv(os.path.join(output_dir, "train_full.csv"))
        self._train_df = df.copy()
        self._target_col = df.columns[-1]
        self._target_classes = df[self._target_col].unique() # Needed for stratification
        self.cat_cols = list(df.columns)

        X_cat, X_num = self._preprocess_data(df)
        n_samples = len(X_cat)
        total_dim = sum(self.cat_dims)

        if self.latent_dim is None:
            self.latent_dim = 128 # Fixed default
        if self.hidden_dim is None:
            self.hidden_dim = total_dim

        # Heuristic for teachers
        if self.n_teachers is None:
            self.n_teachers = 5 # Standard starting point for PATE

        if self.noise_lambda is None:
            self.noise_lambda = 1.0 / self.target_epsilon
            self.noise_lambda = max(self.noise_lambda, 0.1)

        print(f"[PATE-GAN] Teachers: {self.n_teachers}, Laplace Scale: {self.noise_lambda:.4f}")
        self.generator = Generator(
            self.latent_dim,
            self.cat_dims,
            hidden_dim=self.hidden_dim
        ).to(self.device)

        self.student = Discriminator(
            self.cat_dims,
            hidden_dim=self.hidden_dim,
            depth=3 # Student is deeper
        ).to(self.device)

        # Teachers (Depth 1)
        self.teachers = [
            Discriminator(
                self.cat_dims,
                hidden_dim=self.hidden_dim,
                depth=1
            ).to(self.device)
            for _ in range(self.n_teachers)
        ]
        
        # Partition Data (Stratified)
        self.partitions = self.partition_data((X_cat, X_num))

        g_opt = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.9))
        s_opt = optim.Adam(self.student.parameters(), lr=self.lr, betas=(0.5, 0.9))
        t_opts = [optim.Adam(t.parameters(), lr=self.lr, betas=(0.5, 0.9)) for t in self.teachers]

        self.alpha_accum = None
        self.queries_made = 0
        current_epsilon = 0.0
        iteration = 0

        print(f"[PATE-GAN] Training Start | Target ε={self.target_epsilon}")
        pbar = tqdm(total=self.target_epsilon, desc="Privacy Budget", unit="ε")

        while current_epsilon < self.target_epsilon and iteration < self.max_iterations:
            z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
            with torch.no_grad():
                # Use current temp
                progress = min(current_epsilon / self.target_epsilon, 1.0)
                temp = self._get_gumbel_temperature(progress)
                fake_soft = self.generator(z, temperature=temp, hard=False)

            for _ in range(self.n_teacher_iters):
                for i, (teacher, part, opt) in enumerate(zip(self.teachers, self.partitions, t_opts)):
                    cat_part, num_part = part
                    if cat_part.size(0) < self.batch_size: 
                        continue 
                    idx = torch.randint(0, cat_part.size(0), (self.batch_size,), device=self.device)
                    real_cat_batch = cat_part[idx]
                    real_num_batch = num_part[idx]
                    
                    real_inputs = self._indices_to_onehot(real_cat_batch)
                    
                    opt.zero_grad()
                    real_logits = teacher(real_inputs)
                    fake_logits = teacher([f.detach() for f in fake_soft])
                    
                    loss_real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                    loss_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                    
                    loss_t = loss_real + loss_fake
                    loss_t.backward()
                    opt.step()

            # --- 2. Train Student ---
            for _ in range(self.n_student_iters):
                z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
                fake_soft = self.generator(z, temperature=temp, hard=False)
                fake_inputs = [f.detach() for f in fake_soft]
                labels = self.pate_mechanism(fake_inputs)

                s_opt.zero_grad()
                student_logits = self.student(fake_inputs)
                loss_s = F.binary_cross_entropy_with_logits(student_logits, labels)
                loss_s.backward()
                s_opt.step()
                current_epsilon = self.get_current_epsilon()
                if current_epsilon >= self.target_epsilon:
                    break
            
            if current_epsilon >= self.target_epsilon:
                break
            z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
            fake_soft = self.generator(z, temperature=temp, hard=False)

            g_opt.zero_grad()
            student_logits = self.student(fake_soft)
            loss_g = F.binary_cross_entropy_with_logits(student_logits, torch.ones_like(student_logits))
            cat_count = len(self.cat_dims)
            if self.entropy_coeff > 0:
                ent_loss = 0
                for i in range(cat_count):
                    probs = fake_soft[i] 
                    ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                    ent_loss += ent
                loss_g -= self.entropy_coeff * ent_loss

            loss_g.backward()
            g_opt.step()
            pbar.n = min(current_epsilon, self.target_epsilon)
            pbar.set_postfix({"ε": f"{current_epsilon:.4f}", "iter": iteration})
            pbar.refresh()
            iteration += 1

        pbar.close()
        print(f"[PATE-GAN] Finished. Final ε={current_epsilon:.4f}, Iterations={iteration}")
        self.is_fitted = True

        if kwargs.get('auto_generate_synthetic', True) and kwargs.get('synthetic_dir'):
            self.sample(num_samples=len(df), synthetic_dir=kwargs['synthetic_dir'])

        return self

    def sample(self, num_samples: int, synthetic_dir: str = None, **kwargs) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be trained first")

        self.generator.eval()

        samples_list = []
        with torch.no_grad():
            for i in range(0, num_samples, self.batch_size):
                bs = min(self.batch_size, num_samples - i)
                z = torch.randn(bs, self.latent_dim, device=self.device)
                gen_soft = self.generator(z, temperature=0.1, hard=True)

                if not samples_list:
                    samples_list = [[] for _ in range(len(gen_soft))]

                for feat_idx, feat_tensor in enumerate(gen_soft):
                    samples_list[feat_idx].append(feat_tensor.cpu())

        concat_samples = [torch.cat(s_list)[:num_samples] for s_list in samples_list]
        df = self._postprocess_data(concat_samples)

        if synthetic_dir:
            os.makedirs(synthetic_dir, exist_ok=True)
            y = df[[self._target_col]]
            x = df.drop(columns=[self._target_col])
            x.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            y.to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)

        return df
    
    def evaluate(self):
        return 0.0
    
    def get_required_dependencies(self) -> list[str]:
        return ["torch", "sklearn", "pandas", "numpy", "tqdm"]