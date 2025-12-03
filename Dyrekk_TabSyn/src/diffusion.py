import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Sinusoidal timestep embeddings."""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32,
                                 device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class DenoisingMLP(nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 time_emb_dim: int = 128,  
                 hidden_dim: int = 512    
                 ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.latent_dim = latent_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        self.net = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()),
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()),
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()),
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()),
        ])
        
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)
        
        h = self.input_proj(x) 
        
        h = h + t_emb
        
        for layer in self.net:
            h = h + layer(h) 
        
        return self.output_proj(h)


class Diffusion(nn.Module):
    def __init__(self, latent_dim: int, device: str = None, num_steps: int = 1000):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.num_steps = num_steps

        self.model = DenoisingMLP(latent_dim, time_emb_dim=128, hidden_dim=512).to(self.device)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alpha_bars[t]  
        return torch.sqrt(1.0 - alpha_bar_t)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alpha_bars[t].view(-1, 1)  
        return torch.sqrt(alpha_bar_t) * z0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    def loss(self, z_0: torch.Tensor) -> torch.Tensor:
        self.train()
        batch_size = z_0.size(0)
        device = z_0.device

        t = torch.randint(0, self.num_steps, (batch_size,), device=device, dtype=torch.long)
        
        eps = torch.randn_like(z_0)

        z_t = self.q_sample(z_0, t, eps)
        
        eps_pred = self.model(z_t, t)

        loss = F.mse_loss(eps_pred, eps, reduction='mean')
        return loss

    def sample(self, num_samples: int, steps: int = None) -> torch.Tensor:
        self.eval()
        steps = steps or self.num_steps
        steps = min(steps, self.num_steps)

        device = self.device
        
        x = torch.randn(num_samples, self.latent_dim, device=device)

        for i in reversed(range(steps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            beta_t = self.betas[i]
            alpha_t = self.alphas[i]
            alpha_bar_t = self.alpha_bars[i]

            with torch.no_grad():
                eps_theta = self.model(x, t)
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
            )

            if i > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

        return x