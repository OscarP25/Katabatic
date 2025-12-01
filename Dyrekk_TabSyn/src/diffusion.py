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
    """
    MLP-based denoising network for diffusion.
    """
    def __init__(self, 
                 latent_dim: int, 
                 time_emb_dim: int = 128,  
                 hidden_dim: int = 512    
                 ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.latent_dim = latent_dim
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Main network: deeper architecture for better capacity
        self.net = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()),
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()),
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()),
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()),
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize output layer with small weights (helps training stability)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Embed timestep
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)  # (B, hidden_dim)
        
        # Project input
        h = self.input_proj(x)  # (B, hidden_dim)
        
        # Add time embedding
        h = h + t_emb
        
        # Pass through network with residual connections
        for layer in self.net:
            h = h + layer(h)  # Residual connection
        
        # Project to output
        return self.output_proj(h)


class Diffusion(nn.Module):
    """
    Diffusion model for latent space generation.
    """
    def __init__(self, latent_dim: int, device: str = None, num_steps: int = 1000):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.num_steps = num_steps

        self.model = DenoisingMLP(latent_dim, time_emb_dim=128, hidden_dim=512).to(self.device)
        
        # Schedules will be set during training

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Standard deviation at timestep t."""
        alpha_bar_t = self.alpha_bars[t]  
        return torch.sqrt(1.0 - alpha_bar_t)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion: q(z_t | z_0) = N(sqrt(α_bar_t) * z_0, (1 - α_bar_t) * I)
        """
        alpha_bar_t = self.alpha_bars[t].view(-1, 1)  
        return torch.sqrt(alpha_bar_t) * z0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    def loss(self, z_0: torch.Tensor) -> torch.Tensor:
        """
        Training loss: predict noise ε from z_t.
        """
        self.train()
        batch_size = z_0.size(0)
        device = z_0.device

        # Sample random timesteps
        t = torch.randint(0, self.num_steps, (batch_size,), device=device, dtype=torch.long)
        
        # Sample noise
        eps = torch.randn_like(z_0)

        # Get noisy samples
        z_t = self.q_sample(z_0, t, eps)
        
        # Predict noise
        eps_pred = self.model(z_t, t)

        # MSE loss
        loss = F.mse_loss(eps_pred, eps, reduction='mean')
        return loss

    def sample(self, num_samples: int, steps: int = None) -> torch.Tensor:
        """
        Reverse diffusion sampling using DDPM.
        """
        self.eval()
        steps = steps or self.num_steps
        steps = min(steps, self.num_steps)

        device = self.device
        
        # Start from pure noise
        x = torch.randn(num_samples, self.latent_dim, device=device)

        # Reverse iteration from T-1 to 0
        for i in reversed(range(steps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            
            # Get schedule parameters
            beta_t = self.betas[i]
            alpha_t = self.alphas[i]
            alpha_bar_t = self.alpha_bars[i]

            with torch.no_grad():
                # Predict noise
                eps_theta = self.model(x, t)

            # Compute mean of p(x_{t-1} | x_t)
            # μ_θ(x_t, t) = (1 / sqrt(α_t)) * (x_t - (β_t / sqrt(1 - α_bar_t)) * ε_θ(x_t, t))
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
            )

            # Add noise (except for last step)
            if i > 0:
                noise = torch.randn_like(x)
                # Variance: σ_t = sqrt(β_t)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

        return x