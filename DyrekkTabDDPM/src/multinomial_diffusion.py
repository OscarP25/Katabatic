"""
Multinomial diffusion process for categorical features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultinomialDiffusion:
    def __init__(self, num_classes, num_timesteps=1000, scheduler='cosine'):
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def _get_beta_schedule(self):
        if self.scheduler == 'linear':
            beta_start = 0.0001
            beta_end = 0.02
            return torch.linspace(beta_start, beta_end, self.num_timesteps)
        elif self.scheduler == 'cosine':
            return self._cosine_beta_schedule()
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")
    
    def _cosine_beta_schedule(self):
        steps = self.num_timesteps + 1
        s = 0.008
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_0, t):
        batch_size = x_0.shape[0]
        alpha_bar_t = self.alphas_cumprod.to(x_0.device)[t].view(batch_size, 1)
        probs = alpha_bar_t * x_0 + (1 - alpha_bar_t) / self.num_classes
        x_t = F.gumbel_softmax(torch.log(probs + 1e-10), tau=1.0, hard=True)
        return x_t
    
    def q_posterior(self, x_t, x_0, t):
        batch_size = x_0.shape[0]
        alpha_t = self.alphas.to(x_0.device)[t].view(batch_size, 1)
        alpha_bar_t = self.alphas_cumprod.to(x_0.device)[t].view(batch_size, 1)
        alpha_bar_t_prev = self.alphas_cumprod.to(x_0.device)[t - 1].view(batch_size, 1)
        alpha_bar_t_prev[t == 0] = 1.0
        term1 = alpha_t * x_t + (1 - alpha_t) / self.num_classes
        term2 = alpha_bar_t_prev * x_0 + (1 - alpha_bar_t_prev) / self.num_classes
        pi = term1 * term2
        pi = pi / (pi.sum(dim=-1, keepdim=True) + 1e-10)
        return pi
    
    def p_sample(self, x_t, x_0_pred, t):
        posterior_probs = self.q_posterior(x_t, x_0_pred, t)
        if t[0] == 0:
            return x_0_pred
        else:
            x_t_minus_1 = F.gumbel_softmax(torch.log(posterior_probs + 1e-10), tau=1.0, hard=True)
            return x_t_minus_1
    
    def compute_loss(self, x_0_pred, x_0_true, x_t, t):
        #KL
        # true_posterior = self.q_posterior(x_t, x_0_true, t)
        # pred_posterior = self.q_posterior(x_t, x_0_pred, t)
        # kl = (
        #     true_posterior * (torch.log(true_posterior + 1e-10) - 
        #                       torch.log(pred_posterior + 1e-10))
        #                       ).sum(dim=-1)
        # return kl.mean()

        #cce
        log_x_0_pred = torch.log(x_0_pred + 1e-10)
        cce_loss = -(x_0_true * log_x_0_pred).sum(dim=-1)
        return cce_loss.mean()