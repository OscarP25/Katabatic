import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from .utils import (
    get_named_beta_schedule, log_add_exp, log_1_min_a, extract, 
    index_to_log_onehot, ohe_to_categories, sum_except_batch, 
    mean_flat, normal_kl, log_categorical, timestep_embedding
)

# --- Components ---

class ResBlock(nn.Module):
    def __init__(self, n_in_channels, d_t_emb, n_out_channels=None, n_groups=32):
        super().__init__()
        if n_out_channels is None: n_out_channels = n_in_channels
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(n_groups, n_in_channels), nn.SiLU(),
            nn.Conv2d(n_in_channels, n_out_channels, 3, 1, 1)
        )
        self.t_emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(d_t_emb, n_out_channels))
        self.out_layers = nn.Sequential(
            nn.GroupNorm(n_groups, n_out_channels), nn.SiLU(), nn.Dropout(0.),
            nn.Conv2d(n_out_channels, n_out_channels, 3, 1, 1)
        )
        self.skip = nn.Identity() if n_out_channels == n_in_channels else nn.Conv2d(n_in_channels, n_out_channels, 1, 1)

    def forward(self, x, t_emb):
        h = self.in_layers(x)
        t = self.t_emb_layers(t_emb).type(h.dtype)
        h = h + t[:, :, None, None]
        h = self.out_layers(h)
        return self.skip(x) + h

class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, t_emb, cond_emb=None):
        for layer in self:
            if isinstance(layer, ResBlock): x = layer(x, t_emb)
            else: x = layer(x)
        return x

# --- Robust U-Net Architecture ---

class Unet(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_base_channels, n_channels_factors, n_res_blocks, d_t_emb, d_cond_emb, n_groups=8):
        super().__init__()
        self.n_in_channels = n_in_channels
        
        # Initial Convolution
        self.init_conv = TimestepEmbedSequential(nn.Conv2d(n_in_channels, n_base_channels, 3, 1, 1))
        
        curr_channels = n_base_channels
        
        # Downsampling Path
        self.down_levels = nn.ModuleList()
        for factor in n_channels_factors:
            out_channels = n_base_channels * factor
            level_blocks = nn.ModuleList()
            
            # 1. Residual Blocks (Features to be skipped)
            for _ in range(n_res_blocks):
                level_blocks.append(TimestepEmbedSequential(
                    ResBlock(curr_channels, d_t_emb, out_channels, n_groups)
                ))
                curr_channels = out_channels
            
            # 2. Downsample (Features passed to next level, NOT skipped)
            level_blocks.append(TimestepEmbedSequential(nn.Conv2d(curr_channels, curr_channels, 3, 2, 1)))
            self.down_levels.append(level_blocks)

        # Middle Block (Bottleneck)
        self.middle_block = TimestepEmbedSequential(
            ResBlock(curr_channels, d_t_emb, curr_channels, n_groups),
            ResBlock(curr_channels, d_t_emb, curr_channels, n_groups)
        )
        
        # Upsampling Path
        self.up_levels = nn.ModuleList()
        for factor in reversed(n_channels_factors):
            # Target output channels for this level
            # Note: We work backwards. The output of this level matches the input of the corresponding down level.
            # But for simplicity in construction, we look at the factors.
            # We want to reduce channels back.
            
            # Logic: Input is `curr_channels` (from deep). 
            # We concat with skip (also `curr_channels` roughly).
            # Output should ideally go back to `n_base_channels * factor`.
            
            out_channels = n_base_channels * factor
            level_blocks = nn.ModuleList()
            
            # 1. Upsample first
            level_blocks.append(TimestepEmbedSequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(curr_channels, curr_channels, 3, 1, 1)
            ))
            
            # 2. Residual Blocks with Concatenation
            for _ in range(n_res_blocks):
                # Input channels = curr + skip (skip size is same as out_channels of this level in down path)
                # Actually, skip size is what we produced in the down path: `out_channels`.
                # Current size is also `curr_channels`.
                # We map to `out_channels`.
                level_blocks.append(TimestepEmbedSequential(
                    ResBlock(curr_channels + out_channels, d_t_emb, out_channels, n_groups)
                ))
                curr_channels = out_channels
                
            self.up_levels.append(level_blocks)
        
        # Final resolution blocks (after last upsample level)
        # Skip connection from Init Conv
        self.final_res = TimestepEmbedSequential(
            ResBlock(curr_channels + n_base_channels, d_t_emb, n_base_channels, n_groups)
        )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(n_groups, n_base_channels), nn.SiLU(),
            nn.Conv2d(n_base_channels, n_out_channels, 3, 1, 1)
        )

    def forward(self, x, t_emb, cond_emb):
        # 1. Init
        x = self.init_conv(x, t_emb)
        skips = [x]
        
        # 2. Down
        for level_blocks in self.down_levels:
            # Run ResBlocks (and save to skips)
            for block in level_blocks[:-1]: 
                x = block(x, t_emb)
                skips.append(x)
            # Run Downsample (do NOT save to skips)
            x = level_blocks[-1](x, t_emb)
            
        # 3. Middle
        x = self.middle_block(x, t_emb)
        
        # 4. Up
        for level_blocks in self.up_levels:
            # Run Upsample
            x = level_blocks[0](x, t_emb)
            
            # Run ResBlocks (consume skips)
            for block in level_blocks[1:]:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x, t_emb)
        
        # 5. Final
        skip = skips.pop()
        x = torch.cat([x, skip], dim=1)
        x = self.final_res(x, t_emb)
        
        return self.out_layers(x)

# --- Wrapper & Diffusion Logic ---

class DenoiseFn(nn.Module):
    def __init__(self, input_dim, d_t_emb, d_cond_emb, n_channels=32):
        super().__init__()
        self.d_t_emb = d_t_emb
        self.input_dim = input_dim
        self.side = 4 # Projects data to 4x4 image
        self.n_channels = n_channels
        
        self.projection_in = nn.Linear(input_dim, n_channels * self.side * self.side)
        self.projection_out = nn.Linear(n_channels * self.side * self.side, input_dim)
        
        self.unet = Unet(
            n_in_channels=n_channels,
            n_out_channels=n_channels,
            n_base_channels=n_channels,
            n_channels_factors=[2, 4], 
            n_res_blocks=2,
            d_t_emb=d_t_emb,
            d_cond_emb=d_cond_emb
        )
        self.cond_proj = nn.Linear(1, d_cond_emb)

    def forward(self, x, t, cond):
        t_emb = timestep_embedding(t, self.d_t_emb)
        cond = cond.float()
        if cond.ndim == 1: cond = cond.unsqueeze(1)
        cond_emb = self.cond_proj(cond).unsqueeze(1) 
        
        x_img = self.projection_in(x)
        x_img = x_img.view(-1, self.n_channels, self.side, self.side)
        
        out_img = self.unet(x_img, t_emb, cond_emb)
        
        out_flat = out_img.view(x.shape[0], -1)
        out = self.projection_out(out_flat)
        return out

class GaussianMultinomialDiffusion(nn.Module):
    def __init__(self, num_classes, num_numerical_features, denoise_fn, num_timesteps=1000, device='cpu', is_fair=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_numerical_features = num_numerical_features
        self._denoise_fn = denoise_fn
        self.num_timesteps = num_timesteps
        self.is_fair = is_fair
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))])
        ).to(device)
        self.slices_for_classes = [np.arange(self.num_classes[0])]
        offsets = np.cumsum(self.num_classes)
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i-1], offsets[i]))

        betas = get_named_beta_schedule('cosine', num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        
        self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod).float().to(device))
        self.register_buffer('sqrt_alphas_cumprod', torch.tensor(np.sqrt(alphas_cumprod)).float().to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.tensor(np.sqrt(1. - alphas_cumprod)).float().to(device))
        self.register_buffer('log_alpha', torch.tensor(np.log(alphas)).float().to(device))
        self.register_buffer('log_1_min_alpha', log_1_min_a(torch.tensor(np.log(alphas)).float()).to(device))
        self.register_buffer('log_cumprod_alpha', torch.tensor(np.cumsum(np.log(alphas))).float().to(device))
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_a(torch.tensor(np.cumsum(np.log(alphas))).float()).to(device))

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)
        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.log(self.num_classes_expanded)
        )
        return log_probs

    def q_sample(self, log_x_start, t):
        log_ev_qxt_x0 = self.q_pred(log_x_start, t)
        return self.log_sample_categorical(log_ev_qxt_x0)

    def log_sample_categorical(self, logits):
        full_sample = []
        for i in range(len(self.num_classes)):
            one_class_logits = logits[:, self.slices_for_classes[i]]
            uniform = torch.rand_like(one_class_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample = (gumbel_noise + one_class_logits).argmax(dim=1)
            full_sample.append(F.one_hot(sample, self.num_classes[i]))
        x_onehot = torch.cat(full_sample, dim=1)
        return torch.log(x_onehot.float().clamp(min=1e-30))

    def mixed_loss(self, x, cond):
        b = x.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x.device).long()
        log_x_cat = index_to_log_onehot(x.long(), self.num_classes)
        log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t)
        model_out = self._denoise_fn(log_x_cat_t, t, cond)
        loss = 0
        for i in range(len(self.num_classes)):
            slc = self.slices_for_classes[i]
            logits = model_out[:, slc]
            target = x[:, i].long()
            loss += F.cross_entropy(logits, target)
        return loss / len(self.num_classes)

    @torch.no_grad()
    def p_sample(self, model_out, log_x, t):
        log_x_recon = torch.zeros_like(model_out)
        for ix in self.slices_for_classes:
            log_x_recon[:, ix] = F.log_softmax(model_out[:, ix], dim=1)
        return self.log_sample_categorical(log_x_recon)

    @torch.no_grad()
    def sample(self, n_samples, cond):
        b = n_samples
        device = self.log_alpha.device
        uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
        log_z = self.log_sample_categorical(uniform_logits)
        
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(log_z, t, cond)
            log_z = self.p_sample(model_out, log_z, t)
            
        z_ohe = torch.exp(log_z).round()
        z_cat = ohe_to_categories(z_ohe, self.num_classes)
        return z_cat