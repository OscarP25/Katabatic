from katabatic.models.base_model import Model
from typing import List, Optional
from torch import nn   
import torch
from torch.nn import functional as F
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tqdm import tqdm

# Import custom modules
from .gaussian_diffusion import GaussianDiffusion
from .multinomial_diffusion import MultinomialDiffusion
from .mlp_model import DenoisingMLP

class TabDDPM(Model):
    def __init__(
        self,
        num_timesteps=1000,
        scheduler='cosine',
        hidden_dims=[256, 512, 512, 512],
        dropout=0.0,
        lr=0.0001,
        batch_size=1024,
        train_iterations=20000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.train_iterations = train_iterations
        self.device = device
        
        self.model = None
        self.gaussian_diffusion = None
        self.multinomial_diffusions = None
        self.preprocessor = None
        self.label_encoder = None
        self.num_features = None
        self.num_numerical = 0
        self.categorical_dims = None
        self.num_classes = None
        self.class_proportions = None

    def _preprocess_data(self, df: pd.DataFrame):
        """
        Preprocess features X (excluding target).
        """
        if self.preprocessor is None:
            self.preprocessor = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_processed = self.preprocessor.fit_transform(df)
            self.categorical_dims = [len(cats) for cats in self.preprocessor.categories_]
        else:
            X_processed = self.preprocessor.transform(df)
        
        return torch.tensor(X_processed, dtype=torch.float32)

    def _inverse_preprocess(self, X_tensor: torch.Tensor) -> pd.DataFrame:
        X_numpy = X_tensor.cpu().numpy()
        X_original = self.preprocessor.inverse_transform(X_numpy)
        
        # Reconstruct DataFrame with feature names
        # Note: We store feature names during training to ensure safety
        df = pd.DataFrame(X_original, columns=self.feature_names)
        
        # Convert to int as data is originally discrete/categorical
        for col in df.columns:
            try:
                df[col] = df[col].astype(float).round().astype(int)
            except:
                pass # Keep as is if conversion fails
        return df

    def train(self, output_dir: str, **kwargs) -> 'TabDDPM':
        """Train the TabDDPM model with Optimized GPU Utilization"""
        df = pd.read_csv(os.path.join(output_dir, "train_full.csv"))
        self._target_col = df.columns[-1]
        
        # 1. Separate Features (X) and Target (y)
        y_raw = df[self._target_col]
        X_raw = df.drop(columns=[self._target_col])
        self.feature_names = X_raw.columns.tolist()

        # 2. Process Features
        X_tensor = self._preprocess_data(X_raw)
        self.num_features = X_tensor.shape[1]

        # 3. Process Target (Conditioning)
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_raw)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)
        
        self.num_classes = len(self.label_encoder.classes_)
        
        # Save class proportions for sampling
        unique, counts = np.unique(y_encoded, return_counts=True)
        self.class_proportions = counts / len(y_encoded)
        
        # --- OPTIMIZATION START: Move ENTIRE dataset to GPU upfront ---
        if self.device != 'cpu':
            print(f"Moving entire dataset to {self.device}...")
            X_tensor = X_tensor.to(self.device)
            y_tensor = y_tensor.to(self.device)
        # --- OPTIMIZATION END ---

        # 4. Initialize Diffusions
        self.gaussian_diffusion = GaussianDiffusion(
            num_timesteps=self.num_timesteps, 
            scheduler=self.scheduler
        )
        
        self.multinomial_diffusions = []
        for cat_dim in self.categorical_dims:
            self.multinomial_diffusions.append(
                MultinomialDiffusion(
                    num_classes=cat_dim,
                    num_timesteps=self.num_timesteps,
                    scheduler=self.scheduler
                )
            )
        
        # 5. Initialize Model
        self.model = DenoisingMLP(
            input_dim=self.num_features,
            output_dim=self.num_features,
            hidden_dims=self.hidden_dims,
            num_classes=self.num_classes,
            dropout=self.dropout
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 6. Training Loop
        self.model.train()
        
        # Create dataset (Tensors are ALREADY on GPU)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        
        # IMPORTANT: When using CUDA tensors in DataLoader, num_workers must be 0
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=0 
        )
        
        pbar = tqdm(range(self.train_iterations), desc="Training TabDDPM")
        data_iter = iter(dataloader)
        
        for iteration in pbar:
            try:
                # No .to(device) needed here anymore! Speed is now limited only by compute.
                batch_X, batch_y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch_X, batch_y = next(data_iter)
            
            # ... [Rest of the training loop code remains the same] ...
            t = torch.randint(0, self.num_timesteps, (batch_X.shape[0],), device=self.device)
            
            # Split X into categorical parts
            X_cat_parts = []
            cat_start_idx = 0
            for cat_dim in self.categorical_dims:
                X_cat = batch_X[:, cat_start_idx:cat_start_idx + cat_dim]
                cat_start_idx += cat_dim
                X_cat_parts.append(X_cat)            
            X_t = torch.cat(X_cat_parts, dim=1)
            
            # Predict with Conditioning (pass batch_y)
            pred = self.model(X_t, t, batch_y)
            
            # Calculate Loss
            multinomial_losses = []
            pred_start_idx = 0
            for i, cat_dim in enumerate(self.categorical_dims):
                pred_cat = pred[:, pred_start_idx:pred_start_idx + cat_dim]
                pred_cat = F.softmax(pred_cat, dim=-1)
                pred_start_idx += cat_dim
                
                cat_loss = self.multinomial_diffusions[i].compute_loss(
                    pred_cat, X_cat_parts[i], X_cat_parts[i], t
                )
                multinomial_losses.append(cat_loss)
            
            total_loss = sum(multinomial_losses) / len(multinomial_losses)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': total_loss.item()})
        
        self.is_fitted = True
        
        if kwargs.get('auto_generate_synthetic', True) and kwargs.get('synthetic_dir'):
            self.sample(num_samples=len(df), synthetic_dir=kwargs['synthetic_dir'])

        return self
    
    def sample(
        self, 
        num_samples: int, 
        synthetic_dir: str = None, 
        **kwargs
    ):
        if not self.is_fitted:
            raise RuntimeError("Model must be trained first")
        
        self.model.eval()
        
        all_samples_x = []
        all_samples_y = []
        
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for _ in tqdm(range(num_batches), desc="Generating samples"):
                current_batch_size = min(self.batch_size, num_samples - len(all_samples_x) * self.batch_size)
                
                # 1. Sample Labels (y) based on training distribution
                y_indices = np.random.choice(
                    self.num_classes, 
                    size=current_batch_size, 
                    p=self.class_proportions
                )
                y_batch = torch.tensor(y_indices, device=self.device, dtype=torch.long)
                
                # 2. Start from Noise
                X_t = torch.randn(current_batch_size, self.num_features, device=self.device)
                
                # 3. Reverse Diffusion
                for t in reversed(range(self.num_timesteps)):
                    t_batch = torch.full((current_batch_size,), t, device=self.device, dtype=torch.long)
                    
                    # Predict with Conditioning
                    pred = self.model(X_t, t_batch, y_batch)
                    
                    X_cat_parts = []
                    pred_start_idx = 0
                    cat_start_idx = 0
                    
                    for i, cat_dim in enumerate(self.categorical_dims):
                        pred_cat = pred[:, pred_start_idx:pred_start_idx + cat_dim]
                        pred_cat = F.softmax(pred_cat, dim=-1)
                        pred_start_idx += cat_dim
                        
                        X_cat_t = X_t[:, cat_start_idx:cat_start_idx + cat_dim]
                        cat_start_idx += cat_dim
                        
                        X_cat_t_minus_1 = self.multinomial_diffusions[i].p_sample(
                            X_cat_t, pred_cat, t_batch
                        )
                        X_cat_parts.append(X_cat_t_minus_1)
                    
                    X_t = torch.cat(X_cat_parts, dim=1)
                
                all_samples_x.append(X_t.cpu())
                all_samples_y.append(y_batch.cpu())
        
        # Combine
        X_synthetic_tensor = torch.cat(all_samples_x, dim=0)[:num_samples]
        y_synthetic_tensor = torch.cat(all_samples_y, dim=0)[:num_samples]
        
        # Inverse transform X
        df_x = self._inverse_preprocess(X_synthetic_tensor)
        
        # Inverse transform y
        y_original = self.label_encoder.inverse_transform(y_synthetic_tensor.numpy())
        df_y = pd.DataFrame(y_original, columns=[self._target_col])
        
        # Combine into one DF (optional, but good for inspection)
        df_final = pd.concat([df_x, df_y], axis=1)
        
        if synthetic_dir:
            os.makedirs(synthetic_dir, exist_ok=True)
            
            # Ensure integer types for target to prevent downstream errors
            df_y[self._target_col] = df_y[self._target_col].round().astype(int)
            
            df_x.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            df_y.to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)

        return df_final

    def evaluate(self):
        return 0.0

    def get_required_dependencies(self) -> List[str]:
        return ["torch", "numpy", "scikit-learn", "pandas", "tqdm"]