"""Utility classes for the MTabGen model.

This module provides the Transformer-based Denoiser and Diffusion wrapper 
for the MTabGen model (Conditional Diffusion with enhanced conditioning).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TransformerDenoiser(nn.Module):
    """Transformer-based denoiser for Tabular Diffusion (MTabGen).

    Processes numerical and categorical inputs as tokens in a sequence, 
    fusing them with time embeddings.
    """
    def __init__(
        self,
        d_in: int,         # Total input dimension (num + cat_onehot)
        n_num: int,        # Number of numerical features
        n_cat: int,        # Number of categorical features
        cat_cards: List[int], # Cardinality of each categorical feature
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.0,
        d_feedforward: int = 512,
    ):
        super().__init__()
        self.d_in = d_in
        self.n_num = n_num
        self.n_cat = n_cat
        self.cat_cards = cat_cards
        self.d_model = d_model

        # 1. Input Projection
        # We treat the entire row as a sequence of (n_num + n_cat) tokens if we did per-feature embedding
        # BUT for compatibility with the flat diffusions, we will project the flat vector 
        # to a sequence of embeddings, or just use a simple projection + Transformer on top.
        #
        # Better approach for Tabular Transformer: 
        # Project each Numerical feature to d_model
        # Embed each Categorical feature to d_model (requires re-indexing inputs, but diffusion inputs are flat/one-hot)
        #
        # SIMPLIFIED MTabGen: 
        # Treat the WHOLE row as one embedding? No, that's just MLP.
        # We need token-per-feature to use attention.
        
        # We assume input x is [Batch, N_num + Sum(Div_Cards)] (Flat)
        # We need to map this back to features. 
        # This implementation requires x to be reconstructable into features.
        
        # Mapping layers for features
        self.num_proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_num)])
        self.cat_embs = nn.ModuleList([nn.Linear(c, d_model) for c in cat_cards if c > 0]) # Input is one-hot/logits from diffusion

        self.num_features = n_num + len([c for c in cat_cards if c > 0])
        
        # Time Embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_feedforward, 
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output Heads (Predicting noise/x_0)
        # We need to reconstruct the flat vector.
        # We apply a head to each token and concat.
        self.num_tails = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(n_num)])
        self.cat_tails = nn.ModuleList([nn.Linear(d_model, c) for c in cat_cards if c > 0])


    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x is [Batch, d_in]
        # t is [Batch]
        
        batch = x.shape[0]
        
        # 1. Time Embed
        t_emb = self.time_embed(t.float()) # [Batch, d_model]
        t_emb = t_emb.unsqueeze(1) # [Batch, 1, d_model]
        
        # 2. Reshape/Project Input to Tokens
        tokens = []
        
        # Numerical
        # x[:, :n_num] are numericals
        curr = 0
        for i in range(self.n_num):
            val = x[:, curr:curr+1]
            tokens.append(self.num_proj[i](val))
            curr += 1
            
        # Categorical
        # One-hot/Logit chunks
        for i, card in enumerate(self.cat_cards):
            if card > 0:
                val = x[:, curr:curr+card]
                tokens.append(self.cat_embs[i](val))
                curr += card
                
        # Stack tokens
        if not tokens: # Should not happen if check_dependencies done right
             # Fallback
             return torch.zeros_like(x)
             
        h = torch.stack(tokens, dim=1) # [Batch, num_feats, d_model]
        
        # Add time embedding to all tokens (or append as CLS token)
        # Adding is standard for diffusion transformers
        h = h + t_emb
        
        # 3. Transformer
        h = self.transformer(h) # [Batch, num_feats, d_model]
        
        # 4. Output Projection
        outs = []
        
        # Numerical Heads
        for i in range(self.n_num):
            outs.append(self.num_tails[i](h[:, i, :]))
            
        # Categorical Heads
        # Note: h indices must match input order
        cat_token_start = self.n_num
        for i, card in enumerate(self.cat_cards):
            if card > 0:
                outs.append(self.cat_tails[i](h[:, cat_token_start + i, :]))
        
        return torch.cat(outs, dim=1)


class GaussianMultinomialDiffusion(nn.Module):
    """Minimal diffusion wrapper (Adapted for MTabGen)."""

    def __init__(
        self,
        num_classes: Sequence[int] | np.ndarray,
        num_numerical_features: int,
        denoise_fn: nn.Module,
        gaussian_loss_type: str = "mse",
        num_timesteps: int = 1000,
        scheduler: str = "cosine",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._denoise_fn = denoise_fn
        self.register_buffer("_dummy", torch.zeros(1))
        
        self._K = np.array(num_classes, dtype=int) if len(num_classes) > 0 else np.array([0], dtype=int)
        self._n_num = int(num_numerical_features)
        self._n_cat = int(np.sum(self._K > 0))
        self._steps = int(num_timesteps)
        
        # Beta Schedule
        if scheduler == "cosine":
            betas = self._cosine_beta_schedule(self._steps)
        else: # linear
            betas = torch.linspace(1e-4, 0.02, self._steps)
            
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def forward(self, x, t, y=None):
        return self._denoise_fn(x, t, y)

    def mixed_loss(self, x_start: torch.Tensor, out: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x_start.device
        batch_size = x_start.shape[0]
        
        # Sample time
        t = torch.randint(0, self._steps, (batch_size,), device=device).long()
        
        # Noise
        noise = torch.randn_like(x_start)
        
        # q_sample: x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * eps
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        x_t = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        
        # Predict noise (epsilon matching) or x_0
        # Common for tabular is predicting x_0 directly or epsilon.
        # TabDDPM often predicts x_0 for categorical and eps for numerical?
        # Here we simplify: Predict x_0 directly for valid reconstruction loss
        
        pred_x0 = self._denoise_fn(x_t, t)
        
        # Split loss
        n_num = min(self._n_num, x_start.shape[1])
        
        loss_gauss = torch.tensor(0.0, device=device)
        loss_multi = torch.tensor(0.0, device=device)

        if n_num > 0:
            loss_gauss = F.mse_loss(pred_x0[:, :n_num], x_start[:, :n_num])

        if self._n_cat > 0:
            # We treat categorical parts as logits for CE loss?
            # Or just MSE on one-hots? simple MSE on one-hots is stable for diffusion
            # But CE is better. 
            # TabDDPM does specialized loss.
            # Simplified MTabGen: MSE on everything for robustness + stability
            loss_multi = F.mse_loss(pred_x0[:, n_num:], x_start[:, n_num:])

        return loss_multi, loss_gauss

    @torch.no_grad()
    def sample_all(
        self,
        num_samples: int,
        batch_size: int,
        y_dist: torch.Tensor = None, # Metadata only here
        impute_mask: Optional[torch.Tensor] = None, # For inpainting [1=Keep, 0=Missing]
        impute_values: Optional[torch.Tensor] = None, # Partial observed data
    ) -> torch.Tensor:
        
        device = self._dummy.device
        n = num_samples
        d_total = self._n_num + int(self._K.sum())
        
        # Start from noise
        x = torch.randn(n, d_total, device=device)
        
        # Sampling loop (DDPM)
        for i in reversed(range(0, self._steps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            
            # Predict x_0
            pred_x0 = self._denoise_fn(x, t)
            
            # Clamp/Threshold x_0
            if self._n_num > 0:
                pred_x0[:, :self._n_num] = pred_x0[:, :self._n_num].clamp(-3.0, 3.0)
            
            # Re-noising to x_{t-1}
            # mean = (beta * x_0 + (1-beta) * sqrt(1-beta) * x_t) ... 
            # Using simple posterior mean formula from Ho et al. 2020
            
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            alpha_cumprod_prev = self.alphas_cumprod[i-1] if i > 0 else torch.tensor(1.0).to(device)
            beta = self.betas[i]
            
            # Posterior mean
            post_mean_coef1 = beta * torch.sqrt(alpha_cumprod_prev) / (1. - alpha_cumprod)
            post_mean_coef2 = (1. - alpha_cumprod_prev) * torch.sqrt(alpha) / (1. - alpha_cumprod)
            
            post_mean = post_mean_coef1 * pred_x0 + post_mean_coef2 * x
            
            if i > 0:
                noise = torch.randn_like(x)
                # Posterior variance
                post_variance = beta * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)
                log_var = torch.log(torch.clamp(post_variance, min=1e-20))
                x = post_mean + torch.exp(0.5 * log_var) * noise
            else:
                x = post_mean
                
            # IMPUTATION / INPAINTING STEP - The "Conditioning" part of MTabGen
            if impute_mask is not None and impute_values is not None:
                # We enforce the known values at step t-1
                # To do this correctly: add noise to known values to match noise level at t-1
                if i > 0:
                    noise_known = torch.randn_like(impute_values)
                    known_t = self.sqrt_alphas_cumprod[i-1] * impute_values + \
                              torch.sqrt(1 - self.alphas_cumprod[i-1]) * noise_known
                    x = x * (1 - impute_mask) + known_t * impute_mask
                else:
                    x = x * (1 - impute_mask) + impute_values * impute_mask

        return x, torch.zeros(n)
