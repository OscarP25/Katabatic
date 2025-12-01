"""
Multinomial diffusion process for categorical features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultinomialDiffusion:
    def __init__(self, num_classes, num_timesteps=1000, scheduler='cosine'):
        """
        Args:
            num_classes: Number of categories (K)
            num_timesteps: Number of diffusion timesteps (T)
            scheduler: Noise schedule type
        """
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler
        
        # Compute beta schedule (same as Gaussian)
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
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
        """Cosine schedule"""
        steps = self.num_timesteps + 1
        s = 0.008
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_0, t):
        """
        Forward diffusion: q(x_t | x_0)
        
        Args:
            x_0: One-hot encoded categorical [batch_size, num_classes]
            t: Timestep [batch_size]
            
        Returns:
            x_t: Noised categorical distribution
        """
        batch_size = x_0.shape[0]
        
        # Get alpha_bar_t
        alpha_bar_t = self.alphas_cumprod.to(x_0.device)[t].view(batch_size, 1)
        
        # q(x_t | x_0) = Cat(x_t; alpha_bar_t * x_0 + (1 - alpha_bar_t) / K)
        probs = alpha_bar_t * x_0 + (1 - alpha_bar_t) / self.num_classes
        
        # Sample from categorical
        x_t = F.gumbel_softmax(torch.log(probs + 1e-10), tau=1.0, hard=True)
        
        return x_t
    
    def q_posterior(self, x_t, x_0, t):
        """
        Posterior q(x_{t-1} | x_t, x_0)
        
        Args:
            x_t: Noised data at timestep t
            x_0: Predicted x_0
            t: Timestep
            
        Returns:
            Posterior probabilities
        """
        batch_size = x_0.shape[0]
        
        alpha_t = self.alphas.to(x_0.device)[t].view(batch_size, 1)
        alpha_bar_t = self.alphas_cumprod.to(x_0.device)[t].view(batch_size, 1)
        alpha_bar_t_prev = self.alphas_cumprod.to(x_0.device)[t - 1].view(batch_size, 1)
        alpha_bar_t_prev[t == 0] = 1.0
        
        # Compute posterior: pi = [alpha_t * x_t + (1 - alpha_t) / K] * [alpha_bar_{t-1} * x_0 + (1 - alpha_bar_{t-1}) / K]
        term1 = alpha_t * x_t + (1 - alpha_t) / self.num_classes
        term2 = alpha_bar_t_prev * x_0 + (1 - alpha_bar_t_prev) / self.num_classes
        
        pi = term1 * term2
        
        # Normalize
        pi = pi / (pi.sum(dim=-1, keepdim=True) + 1e-10)
        
        return pi
    
    def p_sample(self, x_t, x_0_pred, t):
        """
        Reverse sampling: p(x_{t-1} | x_t)
        
        Args:
            x_t: Current noised categorical
            x_0_pred: Predicted x_0 from model
            t: Current timestep
            
        Returns:
            x_{t-1}: Sample from reverse process
        """
        # Get posterior
        posterior_probs = self.q_posterior(x_t, x_0_pred, t)
        
        # Sample from categorical
        if t[0] == 0:
            return x_0_pred
        else:
            x_t_minus_1 = F.gumbel_softmax(torch.log(posterior_probs + 1e-10), tau=1.0, hard=True)
            return x_t_minus_1
    
    def compute_loss(self, x_0_pred, x_0_true, x_t, t):
        """
        Compute KL divergence loss
        
        Args:
            x_0_pred: Predicted x_0 from model
            x_0_true: True x_0
            x_t: Noised x_t
            t: Timestep
            
        Returns:
            KL divergence loss
        """
        # True posterior
        true_posterior = self.q_posterior(x_t, x_0_true, t)
        
        # Predicted posterior
        pred_posterior = self.q_posterior(x_t, x_0_pred, t)
        
        # KL divergence
        kl = (true_posterior * (torch.log(true_posterior + 1e-10) - torch.log(pred_posterior + 1e-10))).sum(dim=-1)
        
        return kl.mean()