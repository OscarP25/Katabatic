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
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from typing import Any, Optional, Union, List, Tuple

from katabatic.models.base_model import Model
from vae import Model_VAE
from diffusion import Diffusion


class TabSyn(Model):
    """
    TabSyn: Tabular Data Synthesis with Score-based Diffusion in Latent Space.
    """

    def __init__(
        self,
        epochs_vae: int = 100,             
        epochs_diffusion: int = 100,     
        batch_size: int = 64,             
        vae_lr: float = 1e-3,              
        diffusion_lr: float = 1e-3,       
        device: str = None,
        # VAE Hyperparameters
        d_token: int = 32,               
        n_layers_vae: int = 3,  
        hid_dim_vae: int = 64,             
        # Diffusion Hyperparameters
        num_timesteps: int = 1000,
        # Loss balancing
        cat_loss_weight: float = 2.0,  # Weight for categorical loss
        num_loss_weight: float = 1.0,  # Weight for numerical loss
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
        
        # Store statistics for better sampling
        self.num_means = None
        self.num_stds = None

    def get_required_dependencies(self) -> list[str]:
        return ["torch", "sklearn", "pandas", "numpy", "tqdm"]

    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess data for VAE/Diffusion training.
        """
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
        Train the TabSyn model in two phases:
        1. Train VAE to learn latent representations
        2. Train Diffusion model in the latent space
        """
        self.check_dependencies()
        train_csv_path = os.path.join(output_dir, "train_full.csv")
        
        print(f"[TabSyn] Loading training data from {train_csv_path}...")
        df = pd.read_csv(train_csv_path)

        self._train_df = df.copy()

        self.info['columns'] = df.columns.tolist()
        target_col = df.columns[-1]
        self._target_col = target_col

        # Identify categorical columns (including target)
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        for col in df.columns:
            if col in cat_cols:
                continue
            try:
                nunique = int(df[col].nunique(dropna=True))
            except Exception:
                nunique = df[col].nunique()
            if nunique <= max(20, int(0.05 * len(df))):
                cat_cols.append(col)
        
        if target_col not in cat_cols:
            cat_cols.append(target_col)

        self.cat_cols = [c for c in df.columns if c in cat_cols]
        self.num_cols = [c for c in df.columns if c not in self.cat_cols]

        print(f"[TabSyn] Found {len(self.num_cols)} numerical and {len(self.cat_cols)} categorical columns (target='{target_col}')")

        X_num_tensor, X_cat_tensor = self._preprocess_data(df)

        if X_num_tensor is not None and X_cat_tensor is not None:
            dataset = TensorDataset(X_num_tensor, X_cat_tensor)
        elif X_num_tensor is not None:
            dataset = TensorDataset(X_num_tensor)
        else:
            dataset = TensorDataset(X_cat_tensor)
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # ==================== Phase 1: Train VAE ====================
        print("\n[TabSyn] Phase 1: Training VAE...")
        
        d_numerical = len(self.num_cols)
        categories = self.categories_list if self.cat_cols else None
        
        self.vae = Model_VAE(
            num_layers=self.n_layers_vae,
            d_numerical=d_numerical,
            categories=categories,
            d_token=self.d_token,
            n_head=1,
            factor=4,
            bias=True
        ).to(self.device)
        
        vae_optimizer = torch.optim.AdamW(
            self.vae.parameters(), 
            lr=self.vae_lr,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            vae_optimizer, 
            T_max=self.epochs_vae,
            eta_min=self.vae_lr * 0.1
        )
        
        beta_start = 0.0
        beta_end = 0.008  
        best_loss = float('inf')
        patience_counter = 0
        patience = 20

        for epoch in range(self.epochs_vae):
            self.vae.train()
            epoch_recon_loss = 0
            epoch_num_loss = 0
            epoch_cat_loss = 0
            epoch_kl_loss = 0
            
            beta = beta_start + (beta_end - beta_start) * min(epoch / max(self.epochs_vae * 0.5, 1), 1.0)
            
            for batch_data in dataloader:
                if X_num_tensor is not None and X_cat_tensor is not None:
                    batch_num, batch_cat = batch_data
                elif X_num_tensor is not None:
                    batch_num = batch_data[0]
                    batch_cat = None
                else:
                    batch_num = None
                    batch_cat = batch_data[0]
                
                vae_optimizer.zero_grad()

                recon_num, recon_cat_logits, mu_z, logvar_z = self.vae(batch_num, batch_cat)
                
                num_loss = 0
                cat_loss = 0
                
                if batch_num is not None and recon_num is not None:
                    num_loss = F.mse_loss(recon_num, batch_num, reduction='mean')
                    epoch_num_loss += num_loss.item()
   
                if batch_cat is not None and recon_cat_logits is not None and len(recon_cat_logits) > 0:
                    for i, logits in enumerate(recon_cat_logits):
                        class_counts = torch.bincount(batch_cat[:, i], minlength=logits.size(1))
                        class_weights = 1.0 / (class_counts.float() + 1e-6)
                        class_weights = class_weights / class_weights.sum() * len(class_weights)
                        class_weights = class_weights.to(self.device)
                        
                        cat_loss += F.cross_entropy(
                            logits, 
                            batch_cat[:, i], 
                            weight=class_weights,
                            reduction='mean'
                        )
                    epoch_cat_loss += cat_loss.item()
                
                # Total reconstruction loss with balancing
                recon_loss = self.num_loss_weight * num_loss + self.cat_loss_weight * cat_loss
                
                if recon_loss == 0:
                    raise ValueError("No reconstruction loss computed - check data preprocessing")

                kl_per_dim = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
                kl_loss = torch.mean(torch.sum(torch.clamp(kl_per_dim, min=0.1), dim=-1))

                loss = recon_loss + beta * kl_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0) 
                vae_optimizer.step()

                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
            
            scheduler.step()
            
            avg_recon_loss = epoch_recon_loss / len(dataloader)
            avg_num_loss = epoch_num_loss / len(dataloader) if epoch_num_loss > 0 else 0
            avg_cat_loss = epoch_cat_loss / len(dataloader) if epoch_cat_loss > 0 else 0
            avg_kl_loss = epoch_kl_loss / len(dataloader)
            
            if avg_recon_loss < best_loss:
                best_loss = avg_recon_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"[TabSyn] Early stopping at epoch {epoch+1}")
                break
            
            if self.epochs_vae <= 10 or (epoch + 1) % max(1, self.epochs_vae // 10) == 0:
                print(f"Epoch {epoch+1}/{self.epochs_vae} | "
                    f"Recon: {avg_recon_loss:.4f} (Num: {avg_num_loss:.4f}, Cat: {avg_cat_loss:.4f}) | "
                    f"KL: {avg_kl_loss:.4f} | Beta: {beta:.6f}")
        
        print("[TabSyn] VAE training completed!")
        
        # ==================== Phase 2: Extract Latent Embeddings ====================
        print("\n[TabSyn] Extracting latent embeddings...")
        
        self.vae.eval()
        all_latents = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                if X_num_tensor is not None and X_cat_tensor is not None:
                    batch_num, batch_cat = batch_data
                elif X_num_tensor is not None:
                    batch_num = batch_data[0]
                    batch_cat = None
                else:
                    batch_num = None
                    batch_cat = batch_data[0]
                
                x = self.vae.VAE.Tokenizer(batch_num, batch_cat)
                mu_z = self.vae.VAE.encoder_mu(x)
                
                latent_flat = mu_z[:, 1:, :].reshape(mu_z.size(0), -1)
                all_latents.append(latent_flat)
        
        latents = torch.cat(all_latents, dim=0)
        self.latent_shape = mu_z[:, 1:, :].shape[1:]
        
        # Standardize latents for better diffusion training
        latent_mean = latents.mean(dim=0, keepdim=True)
        latent_std = latents.std(dim=0, keepdim=True) + 1e-6
        latents = (latents - latent_mean) / latent_std
        
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        
        print(f"[TabSyn] Latent embeddings shape: {latents.shape}")
        print(f"[TabSyn] Latent stats - mean: {latents.mean():.4f}, std: {latents.std():.4f}")
        
        # ==================== Phase 3: Train Diffusion Model ====================
        print("\n[TabSyn] Phase 2: Training Diffusion Model...")
        
        latent_dim = latents.shape[1]
        self.diffusion = Diffusion(
            latent_dim=latent_dim,
            device=self.device,
            num_steps=self.num_timesteps
        )
        
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
        
        diffusion_optimizer = torch.optim.AdamW(
            self.diffusion.model.parameters(), 
            lr=self.diffusion_lr,
            weight_decay=1e-6
        )
        
        # Cosine scheduler for diffusion
        diff_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            diffusion_optimizer,
            T_max=self.epochs_diffusion,
            eta_min=self.diffusion_lr * 0.1
        )
        
        best_diff_loss = float('inf')
        
        for epoch in range(self.epochs_diffusion):
            self.diffusion.model.train()
            epoch_loss = 0
            
            for batch_latents, in latent_dataloader:
                batch_latents = batch_latents.to(self.device)
                diffusion_optimizer.zero_grad()
                loss = self.diffusion.loss(batch_latents)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion.model.parameters(), max_norm=1.0)
                diffusion_optimizer.step()
                epoch_loss += loss.item()
            
            diff_scheduler.step()
            
            avg_loss = epoch_loss / len(latent_dataloader)
            
            if avg_loss < best_diff_loss:
                best_diff_loss = avg_loss

            if self.epochs_diffusion <= 10 or (epoch + 1) % max(1, self.epochs_diffusion // 10) == 0:
                print(f"Epoch {epoch+1}/{self.epochs_diffusion} | Loss: {avg_loss:.4f}")
        
        print("[TabSyn] Diffusion training completed!")
        
        self.is_fitted = True
        if kwargs.get('auto_generate_synthetic', True) and kwargs.get('synthetic_dir'):
            synth_dir = kwargs['synthetic_dir']
            num_samples = kwargs.get('num_synthetic_samples', None)
            if num_samples is None:
                num_samples = len(pd.read_csv(os.path.join(output_dir, "train_full.csv")))
            
            print(f"[TabSyn] Auto-generating {num_samples} synthetic samples in {synth_dir} ...")
            self.sample(num_samples=num_samples, synthetic_dir=synth_dir)

        return self

    def sample(self, num_samples: int, *args, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic samples using the trained TabSyn model.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before sampling.")

        print(f"[TabSyn] Sampling {num_samples} rows...")
        
        # Generate latent codes via diffusion
        with torch.no_grad():
            z_flat_gen = self.diffusion.sample(num_samples)
            
            # Denormalize latents
            z_flat_gen = z_flat_gen * self.latent_std + self.latent_mean
        
        z_gen = z_flat_gen.reshape(num_samples, *self.latent_shape)
        
        # Decode latents to data space
        self.vae.eval()
        with torch.no_grad():
            h = self.vae.VAE.decoder(z_gen)
            recon_num, recon_cat_logits = self.vae.Reconstructor(h)
            
            # Handle numerical outputs
            if recon_num is not None:
                # Clip extreme values
                recon_num = torch.clamp(recon_num, -5, 5)
        
        # Process numerical features
        if self.num_cols:
            X_num_np = recon_num.cpu().numpy()
            
            # Inverse transform with both transformers
            try:
                X_num_inv = self.transformers['num'].inverse_transform(X_num_np)
                X_num_inv = self.transformers['scaler'].inverse_transform(X_num_inv)
            except Exception as e:
                print(f"[WARNING] Inverse transform failed: {e}")
                X_num_inv = X_num_np
            
            # Handle any NaNs
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
                

                temperature = 0.8
                probs = torch.softmax(logits / temperature, dim=-1)
                
    
                probs = probs + 1e-10
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                try:
                    indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
                except RuntimeError:
                    indices = torch.argmax(probs, dim=-1)
                
                # Ensure indices are valid
                indices = torch.clamp(indices, 0, num_classes - 1)
                df_cat[col] = indices.cpu().numpy()

        # Combine numerical and categorical
        df_synth = pd.concat([df_num, df_cat], axis=1)
        df_synth = df_synth[self.info['columns']]
        
        # Check target distribution
        target_col = df_synth.columns[-1]
        if target_col in df_cat.columns:
            print(f"\n[TabSyn] Target column '{target_col}' distribution:")
            synth_dist = df_synth[target_col].value_counts().sort_index().to_dict()
            orig_dist = self._train_df[target_col].value_counts().sort_index().to_dict()
            print(f"  Synthetic: {synth_dist}")
            print(f"  Original:  {orig_dist}")
        
        # Handle any remaining NaNs
        for col in df_synth.columns:
            if df_synth[col].isna().any():
                if col in self.num_cols:
                    df_synth[col].fillna(df_synth[col].mean(), inplace=True)
                else:
                    df_synth[col].fillna(df_synth[col].mode()[0] if not df_synth[col].mode().empty else 0, inplace=True)
        
        # Save to disk
        synthetic_dir = kwargs.get("synthetic_dir")
        if synthetic_dir:
            os.makedirs(synthetic_dir, exist_ok=True)

            target_col = df_synth.columns[-1]
            X_synth = df_synth.drop(columns=[target_col])
            Y_synth = df_synth[[target_col]]

            # Fallback for collapsed target distribution
            if Y_synth[target_col].nunique() <= 1 and hasattr(self, '_train_df') and self._train_df is not None:
                print("[WARNING] Target collapsed to single class, resampling from original")
                orig_y = self._train_df[target_col]
                sampled = np.random.choice(orig_y.values, size=len(Y_synth), replace=True)
                Y_synth = pd.DataFrame(sampled, columns=[target_col])

            X_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            Y_synth.to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)

            print("\n[TabSyn] Synthetic data saved:")
            print("  X ->", os.path.join(synthetic_dir, "x_synth.csv"))
            print("  y ->", os.path.join(synthetic_dir, "y_synth.csv"))

        return df_synth

    def evaluate(self):
        return 0.0