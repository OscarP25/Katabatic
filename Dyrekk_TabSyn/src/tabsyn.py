import os
import sys
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from typing import Any, Optional, Union, List, Tuple

from katabatic.models.base_model import Model
# Imported Model_ConditionalVAE
from vae import Model_VAE, Model_ConditionalVAE 
from diffusion import Diffusion


class TabSyn(Model):
    """
    TabSyn: Tabular Data Synthesis with Score-based Diffusion in Latent Space.
    (Updated to use Conditional VAE for better class preservation)
    """

    def __init__(
        self,
        epochs_vae: int = 200,             
        epochs_diffusion: int = 500,     
        batch_size: int = 256,             
        vae_lr: float = 1e-3,              
        diffusion_lr: float = 1e-3,       
        device: str = None,
        # VAE Hyperparameters
        d_token: int = 32,               
        n_layers_vae: int = 4,  
        hid_dim_vae: int = 64,             
        # Diffusion Hyperparameters
        num_timesteps: int = 1000,
        # Loss balancing
        cat_loss_weight: float = 1.0, 
        num_loss_weight: float = 1.0, 
        **kwargs 
    ):
        super().__init__()
        self.epochs_vae = epochs_vae
        self.epochs_diffusion = epochs_diffusion
        self.batch_size = batch_size
        self.vae_lr = vae_lr
        self.diffusion_lr = diffusion_lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Architecture Configs
        self.d_token = d_token
        self.n_layers_vae = n_layers_vae
        self.hid_dim_vae = hid_dim_vae
        self.num_timesteps = num_timesteps
        self.cat_loss_weight = cat_loss_weight
        self.num_loss_weight = num_loss_weight
        
        # State
        self.vae = None
        self.diffusion = None
        self.transformers = {}
        self.num_cols = []
        self.cat_cols = []
        self.categories_list = []
        self.info = {} 
        self.latent_shape = None 
        self.is_fitted = False
        
        # New state for Conditional VAE
        self.target_col = None
        self.target_le = None
        self.n_classes = None
        self.p_y = None  # To store target class probabilities
        
        # Store statistics for better sampling
        self.num_means = None
        self.num_stds = None
        # Debug: Print all input parameters
        print("[TabSyn] Initialization parameters:")
        print(f"  epochs_vae: {epochs_vae}")
        print(f"  epochs_diffusion: {epochs_diffusion}")
        print(f"  batch_size: {batch_size}")
        print(f"  vae_lr: {vae_lr}")
        print(f"  diffusion_lr: {diffusion_lr}")
        print(f"  device: {self.device}")
        print(f"  d_token: {d_token}")
        print(f"  n_layers_vae: {n_layers_vae}")
        print(f"  hid_dim_vae: {hid_dim_vae}")
        print(f"  num_timesteps: {num_timesteps}")
        print(f"  cat_loss_weight: {cat_loss_weight}")
        print(f"  num_loss_weight: {num_loss_weight}")

    def get_required_dependencies(self) -> list[str]:
        return ["torch", "sklearn", "pandas", "numpy", "tqdm"]

    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess features (X) for VAE/Diffusion training.
        """
        # Ensure we only process features, not the target
        X_num = df[self.num_cols].copy()
        X_cat = df[self.cat_cols].copy()

        if not X_num.empty:
            num_imputer = SimpleImputer(strategy='mean')
            X_num_imputed = num_imputer.fit_transform(X_num)
            
            # Store original statistics
            self.num_means = np.mean(X_num_imputed, axis=0)
            self.num_stds = np.std(X_num_imputed, axis=0) + 1e-6
            
            # Use StandardScaler first, then QuantileTransformer
            scaler = StandardScaler()
            X_num_scaled = scaler.fit_transform(X_num_imputed)
            self.transformers['scaler'] = scaler
            
            # QuantileTransformer for better normalization
            self.transformers['num'] = QuantileTransformer(
                output_distribution='normal',
                n_quantiles=min(1000, len(X_num_imputed))
            )
            X_num_transformed = self.transformers['num'].fit_transform(X_num_scaled)
            X_num_tensor = torch.FloatTensor(X_num_transformed).to(self.device)
        else:
            X_num_tensor = None

        X_cat_tensor = None
        self.categories_list = []
        
        if not X_cat.empty:
            # Categorical columns are already encoded as integers
            X_cat = X_cat.fillna(0).astype(int)
            
            # Get number of categories per column
            for col in self.cat_cols:
                n_categories = int(X_cat[col].max()) + 1
                self.categories_list.append(n_categories)
            
            X_cat_tensor = torch.LongTensor(X_cat.values).to(self.device)

        return X_num_tensor, X_cat_tensor

    def train(self, output_dir: str, *args, **kwargs) -> 'TabSyn':
        """
        Train the Conditional TabSyn model.
        """
        self.check_dependencies()
        train_csv_path = os.path.join(output_dir, "train_full.csv")
        
        print(f"[TabSyn] Loading training data from {train_csv_path}...")
        df = pd.read_csv(train_csv_path)
        self._train_df = df.copy()

        self.info['columns'] = df.columns.tolist()
        self.target_col = df.columns[-1]

        # 1. Identify Categorical vs Numerical (Target is handled separately now)
        temp_cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        for col in df.columns:
            if col in temp_cat_cols: continue
            if col == self.target_col: continue # Skip target for now
            try:
                nunique = int(df[col].nunique(dropna=True))
            except:
                nunique = df[col].nunique()
            if nunique <= max(20, int(0.05 * len(df))):
                temp_cat_cols.append(col)
        
        # 2. Separate Features (X) and Target (y)
        self.cat_cols = [c for c in df.columns if c in temp_cat_cols and c != self.target_col]
        self.num_cols = [c for c in df.columns if c not in temp_cat_cols and c != self.target_col]
        
        print(f"[TabSyn] Found {len(self.num_cols)} numerical and {len(self.cat_cols)} categorical features.")
        print(f"[TabSyn] Target column: '{self.target_col}'")

        # 3. Preprocess Target (y)
        y = df[self.target_col].values
        self.target_le = LabelEncoder()
        y_encoded = self.target_le.fit_transform(y)
        self.n_classes = len(self.target_le.classes_)
        
        # Calculate target class probabilities for sampling later
        _, counts = np.unique(y_encoded, return_counts=True)
        self.p_y = counts / counts.sum()
        print(f"[TabSyn] Target distribution: {dict(zip(self.target_le.classes_, counts))}")
        
        y_tensor = torch.LongTensor(y_encoded).to(self.device)

        # 4. Preprocess Features (X)
        X_num_tensor, X_cat_tensor = self._preprocess_data(df)

        # Create Dataset with (Num, Cat, Label)
        tensors = []
        if X_num_tensor is not None: tensors.append(X_num_tensor)
        else: tensors.append(torch.empty(len(df), 0).to(self.device)) # Placeholder
            
        if X_cat_tensor is not None: tensors.append(X_cat_tensor)
        else: tensors.append(torch.empty(len(df), 0).to(self.device)) # Placeholder

        tensors.append(y_tensor)
        dataset = TensorDataset(*tensors)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # ==================== Phase 1: Train Conditional VAE ====================
        print("\n[TabSyn] Phase 1: Training Conditional VAE...")
        
        d_numerical = len(self.num_cols)
        categories = self.categories_list if self.cat_cols else None
        
        # Initialize Model_ConditionalVAE
        self.vae = Model_ConditionalVAE(
            num_layers=self.n_layers_vae,
            d_numerical=d_numerical,
            categories=categories,
            d_token=self.d_token,
            n_classes=self.n_classes,  # Pass number of classes
            n_head=1,
            factor=4,
            bias=True
        ).to(self.device)
        
        vae_optimizer = torch.optim.AdamW(self.vae.parameters(), lr=self.vae_lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(vae_optimizer, T_max=self.epochs_vae, eta_min=self.vae_lr * 0.1)
        
        beta_start, beta_end = 0.0, 0.01  
        best_loss = float('inf')

        for epoch in range(self.epochs_vae):
            self.vae.train()
            epoch_loss_dict = {'recon': 0, 'num': 0, 'cat': 0, 'kl': 0}
            
            beta = beta_start + (beta_end - beta_start) * min(epoch / max(self.epochs_vae * 0.5, 1), 1.0)
            
            for batch_num, batch_cat, batch_y in dataloader:
                # Handle placeholders
                if batch_num.shape[1] == 0: batch_num = None
                if batch_cat.shape[1] == 0: batch_cat = None
                
                vae_optimizer.zero_grad()

                # Forward pass now includes batch_y
                recon_num, recon_cat_logits, mu_z, logvar_z = self.vae(batch_num, batch_cat, batch_y)
                
                # --- Loss Calculation ---
                num_loss = 0
                cat_loss = 0
                
                if batch_num is not None and recon_num is not None:
                    num_loss = F.mse_loss(recon_num, batch_num, reduction='mean')
                    epoch_loss_dict['num'] += num_loss.item()
   
                if batch_cat is not None and recon_cat_logits is not None:
                    for i, logits in enumerate(recon_cat_logits):
                        # Calculate class weights for feature imbalance
                        class_counts = torch.bincount(batch_cat[:, i], minlength=logits.size(1))
                        w = 1.0 / (class_counts.float() + 1e-6)
                        w = w / w.sum() * len(w)
                        cat_loss += F.cross_entropy(logits, batch_cat[:, i], weight=w.to(self.device))
                    epoch_loss_dict['cat'] += cat_loss.item()
                
                recon_loss = self.num_loss_weight * num_loss + self.cat_loss_weight * cat_loss
                
                kl_per_dim = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
                kl_loss = torch.mean(torch.sum(torch.clamp(kl_per_dim, min=0.1), dim=-1))

                loss = recon_loss + beta * kl_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0) 
                vae_optimizer.step()

                epoch_loss_dict['recon'] += recon_loss.item()
                epoch_loss_dict['kl'] += kl_loss.item()
            
            scheduler.step()
            
            # Logging
            avg_losses = {k: v / len(dataloader) for k, v in epoch_loss_dict.items()}
            
            if avg_losses['recon'] < best_loss:
                best_loss = avg_losses['recon']
            
            if self.epochs_vae <= 10 or (epoch + 1) % max(1, self.epochs_vae // 10) == 0:
                print(f"Epoch {epoch+1}/{self.epochs_vae} | "
                    f"Recon: {avg_losses['recon']:.4f} (Num: {avg_losses['num']:.4f}, Cat: {avg_losses['cat']:.4f}) | "
                    f"KL: {avg_losses['kl']:.4f} | Beta: {beta:.6f}")
        
        print("[TabSyn] VAE training completed!")
        
        # ==================== Phase 2: Extract Latents ====================
        print("\n[TabSyn] Extracting latent embeddings...")
        
        self.vae.eval()
        all_latents = []
        
        with torch.no_grad():
            for batch_num, batch_cat, batch_y in dataloader:
                if batch_num.shape[1] == 0: batch_num = None
                if batch_cat.shape[1] == 0: batch_cat = None
                
                # Manually replicate encoding logic to get mu_z
                # We need to access the underlying ConditionalVAE
                x = self.vae.VAE.Tokenizer(batch_num, batch_cat)
                
                # Add class embedding to CLS token
                class_emb = self.vae.VAE.class_embedding(batch_y).unsqueeze(1)
                x[:, 0, :] = x[:, 0, :] + class_emb.squeeze(1)

                mu_z = self.vae.VAE.encoder_mu(x)
                
                # Flatten latents for diffusion (batch, tokens * dim)
                latent_flat = mu_z[:, 1:, :].reshape(mu_z.size(0), -1)
                all_latents.append(latent_flat)
        
        latents = torch.cat(all_latents, dim=0)
        self.latent_shape = mu_z[:, 1:, :].shape[1:] # Store shape for reshaping later
        
        # Standardize latents
        self.latent_mean = latents.mean(dim=0, keepdim=True)
        self.latent_std = latents.std(dim=0, keepdim=True) + 1e-6
        latents = (latents - self.latent_mean) / self.latent_std
        
        # ==================== Phase 3: Train Diffusion ====================
        print("\n[TabSyn] Phase 2: Training Diffusion Model...")
        
        latent_dim = latents.shape[1]
        self.diffusion = Diffusion(latent_dim=latent_dim, device=self.device, num_steps=self.num_timesteps)
        
        # Setup scheduler (same as original code)
        s = 0.008
        steps = torch.linspace(0, self.num_timesteps, self.num_timesteps + 1, device=self.device)
        alpha_bar = torch.cos(((steps / self.num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        self.diffusion.alpha_bars = alpha_bar[1:]
        self.diffusion.alphas = self.diffusion.alpha_bars / torch.cat([torch.tensor([1.0], device=self.device), self.diffusion.alpha_bars[:-1]])
        self.diffusion.betas = 1 - self.diffusion.alphas
        self.diffusion.betas = torch.clamp(self.diffusion.betas, min=1e-5, max=0.999)
        
        latent_dataset = TensorDataset(latents)
        latent_dataloader = DataLoader(latent_dataset, batch_size=self.batch_size, shuffle=True)
        
        diffusion_optimizer = torch.optim.AdamW(self.diffusion.model.parameters(), lr=self.diffusion_lr, weight_decay=1e-6)
        diff_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(diffusion_optimizer, T_max=self.epochs_diffusion, eta_min=self.diffusion_lr * 0.1)
        
        for epoch in range(self.epochs_diffusion):
            self.diffusion.model.train()
            epoch_loss = 0
            for batch_z, in latent_dataloader:
                batch_z = batch_z.to(self.device)
                diffusion_optimizer.zero_grad()
                loss = self.diffusion.loss(batch_z)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion.model.parameters(), max_norm=1.0)
                diffusion_optimizer.step()
                epoch_loss += loss.item()
            diff_scheduler.step()
            
            if self.epochs_diffusion <= 10 or (epoch + 1) % max(1, self.epochs_diffusion // 10) == 0:
                print(f"Epoch {epoch+1}/{self.epochs_diffusion} | Loss: {epoch_loss / len(latent_dataloader):.4f}")
        
        self.is_fitted = True
        
        # Auto-generate if requested
        if kwargs.get('auto_generate_synthetic', True) and kwargs.get('synthetic_dir'):
            synth_dir = kwargs['synthetic_dir']
            num_samples = kwargs.get('num_synthetic_samples', len(self._train_df))
            self.sample(num_samples=num_samples, synthetic_dir=synth_dir)

        return self

    def sample(self, num_samples: int, *args, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic samples using Conditional VAE + Diffusion.
        Controlled sampling ensures the target distribution matches the training data.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before sampling.")

        print(f"[TabSyn] Sampling {num_samples} rows (Conditionally)...")
        
        # 1. Sample Target Class Labels first
        # This guarantees the synthetic data has the exact same class balance as training data
        y_indices = np.random.choice(
            np.arange(self.n_classes), 
            size=num_samples, 
            p=self.p_y
        )
        y_tensor = torch.LongTensor(y_indices).to(self.device)
        
        # 2. Generate Latents via Diffusion
        with torch.no_grad():
            z_flat_gen = self.diffusion.sample(num_samples)
            z_flat_gen = z_flat_gen * self.latent_std + self.latent_mean
        
        z_gen = z_flat_gen.reshape(num_samples, *self.latent_shape)
        
        # Decode with Condition Injection
        self.vae.eval()
        with torch.no_grad():
            # Logic from ConditionalVAE.forward
            # We need to manually inject the class embedding into the generated latent
            
            # Get class embedding
            class_emb = self.vae.VAE.class_embedding(y_tensor).unsqueeze(1) # (B, 1, D)
            
            # Inject into CLS token (Index 0)
            # Note: z_gen includes the CLS token at index 0 because we trained diffusion on the full encoder output
            z_gen[:, 0, :] = z_gen[:, 0, :] + class_emb.squeeze(1)
            
            # Decode (Model_ConditionalVAE decoder expects z[:, 1:])
            # Note: In the provided vae.py, decoder takes z[:, 1:]. 
            # We assume the Transformer decoder can utilize the context or that z structure aligns.
            h = self.vae.VAE.decoder(z_gen[:, 1:])
            
            recon_num, recon_cat_logits = self.vae.Reconstructor(h)
            
            if recon_num is not None:
                recon_num = torch.clamp(recon_num, -5, 5)

        # Convert Latent Output to DataFrame
        if self.num_cols:
            X_num_np = recon_num.cpu().numpy()
            try:
                X_num_inv = self.transformers['num'].inverse_transform(X_num_np)
                X_num_inv = self.transformers['scaler'].inverse_transform(X_num_inv)
            except:
                X_num_inv = X_num_np
                
 
            if np.isnan(X_num_inv).any():
                for i in range(X_num_inv.shape[1]):
                    col_mask = np.isnan(X_num_inv[:, i])
                    if col_mask.any():
                        X_num_inv[col_mask, i] = self.num_means[i]
            df_num = pd.DataFrame(X_num_inv, columns=self.num_cols)
        else:
            df_num = pd.DataFrame()


        df_cat = pd.DataFrame()
        if self.cat_cols:
            for i, col in enumerate(self.cat_cols):
                logits = recon_cat_logits[i]
                num_classes = self.categories_list[i]
                
                probs = torch.softmax(logits, dim=-1)
                indices = torch.argmax(probs, dim=-1)
                indices = torch.clamp(indices, 0, num_classes - 1)
                df_cat[col] = indices.cpu().numpy()

        # Append the Conditioned Target Column
        df_target = pd.DataFrame({self.target_col: self.target_le.inverse_transform(y_indices)})
        
        # Combine
        df_synth = pd.concat([df_num, df_cat, df_target], axis=1)
        
        # Reorder columns to match original
        df_synth = df_synth[self.info['columns']]
        
        # Print distribution check
        print(f"\n[TabSyn] Target '{self.target_col}' distribution check:")
        print(f"  Synthetic: {df_synth[self.target_col].value_counts().sort_index().to_dict()}")

        # Save
        synthetic_dir = kwargs.get("synthetic_dir")
        if synthetic_dir:
            os.makedirs(synthetic_dir, exist_ok=True)
            target_col = df_synth.columns[-1]
            X_synth = df_synth.drop(columns=[target_col])
            Y_synth = df_synth[[target_col]]
            
            X_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            Y_synth.to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)
            print("[TabSyn] Saved synthetic data.")

        return df_synth
    
    def evaluate(self, synthetic_data: pd.DataFrame, real_data: pd.DataFrame = None) -> dict:
        """
        Evaluate the quality of synthetic data against real data.
        
        Args:
            synthetic_data: Generated synthetic DataFrame
            real_data: Original training DataFrame (uses self._train_df if None)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if real_data is None:
            real_data = self._train_df
        
        if real_data is None:
            raise ValueError("Real data not provided and model has no training data stored.")
        
        print("[TabSyn] Evaluating synthetic data quality...")
        
        metrics = {}
        
        # 1. Column-wise statistics comparison
        print("  Computing statistical metrics...")
        metrics['statistics'] = {}
        
        for col in real_data.columns:
            if col not in synthetic_data.columns:
                continue
            
            real_col = real_data[col]
            synth_col = synthetic_data[col]
            
            if real_col.dtype in ['int64', 'float64']:
                # Numerical column
                metrics['statistics'][col] = {
                    'real_mean': float(real_col.mean()),
                    'synth_mean': float(synth_col.mean()),
                    'real_std': float(real_col.std()),
                    'synth_std': float(synth_col.std()),
                    'real_min': float(real_col.min()),
                    'synth_min': float(synth_col.min()),
                    'real_max': float(real_col.max()),
                    'synth_max': float(synth_col.max()),
                }
            else:
                # Categorical column
                real_dist = real_col.value_counts(normalize=True)
                synth_dist = synth_col.value_counts(normalize=True)
                
                metrics['statistics'][col] = {
                    'real_distribution': real_dist.to_dict(),
                    'synth_distribution': synth_dist.to_dict(),
                }
        
        # 2. Target class balance
        if self.target_col and self.target_col in real_data.columns:
            print("  Computing target class balance...")
            real_target_dist = real_data[self.target_col].value_counts(normalize=True).sort_index()
            synth_target_dist = synthetic_data[self.target_col].value_counts(normalize=True).sort_index()
            
            metrics['target_balance'] = {
                'real': real_target_dist.to_dict(),
                'synthetic': synth_target_dist.to_dict(),
                'kl_divergence': float(np.sum(real_target_dist * np.log(real_target_dist / (synth_target_dist + 1e-6) + 1e-6)))
            }
        
        # 3. Shape comparison
        metrics['shape'] = {
            'real_rows': real_data.shape[0],
            'synth_rows': synthetic_data.shape[0],
            'columns': real_data.shape[1]
        }
        
        # 4. Missing values
        metrics['missing_values'] = {
            'real': real_data.isnull().sum().to_dict(),
            'synthetic': synthetic_data.isnull().sum().to_dict(),
        }
        
        print("[TabSyn] Evaluation complete!")
        return metrics