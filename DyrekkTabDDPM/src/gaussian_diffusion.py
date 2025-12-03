"""
Gaussian diffusion process for numerical features
"""
import torch
import torch.nn as nn
import numpy as np


class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, scheduler='cosine'):
        """
        Args:
            num_timesteps: Number of diffusion timesteps (T)
            scheduler: Noise schedule type ('linear' or 'cosine')
        """
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler
        
        # Compute beta schedule
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
    def _get_beta_schedule(self):
        """Get beta schedule for noise"""
        if self.scheduler == 'linear':
            beta_start = 0.0001
            beta_end = 0.02
            return torch.linspace(beta_start, beta_end, self.num_timesteps)
        elif self.scheduler == 'cosine':
            return self._cosine_beta_schedule()
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")
    
    def _cosine_beta_schedule(self):
        """Cosine schedule as proposed in Improved DDPM"""
        steps = self.num_timesteps + 1
        s = 0.008
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_0: Initial data [batch_size, num_features]
            t: Timestep [batch_size]
            noise: Optional noise tensor
            
        Returns:
            Noised data x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x_t, t, y=None):
        """
        Reverse diffusion process: p(x_{t-1} | x_t)
        
        Args:
            model: Denoising model
            x_t: Noised data at timestep t
            t: Current timestep
            y: Optional class labels
            
        Returns:
            Denoised data x_{t-1}
        """
        # Predict noise
        eps_pred = model(x_t, t, y)
        
        # Compute mean
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        
        mean = sqrt_recip_alphas_t * (
            x_t - betas_t * eps_pred / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return mean
        else:
            noise = torch.randn_like(x_t)
            variance = betas_t
            return mean + torch.sqrt(variance) * noise
    
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def compute_loss(self, model, x_0, t, y=None):
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        
        predicted_noise = model(x_t, t, y)
        
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        return loss