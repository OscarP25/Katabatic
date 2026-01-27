"""
TAEGAN (Tabular Auto-Encoder GAN)

This file acts as a thin wrapper around the original TAEGAN implementation.
No training logic, hyperparameters, or architecture details are modified.

Source:
Official TAEGAN GitHub implementation by Zhaozilong et al.
"""

from typing import Tuple
import torch

# Import the original implementation
from .taegan_core import TAEGAN


class TAEGANModel:
    """
    Minimal wrapper for TAEGAN.

    This class exists only for structural consistency with the project layout.
    All training, generation, and hyperparameters remain unchanged.
    """

    def __init__(
        self,
        cache_dir: str,
        embed_dim: int = 256,
        cat_noise_dim: int = 64,
        cont_noise_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 6,
    ):
        self.model = TAEGAN(
            cache_dir=cache_dir,
            embed_dim=embed_dim,
            cat_noise_dim=cat_noise_dim,
            cont_noise_dim=cont_noise_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

    def train(
        self,
        batch_size: int = 500,
        epochs: int = 300,
        warmup_epochs: int = 90,
        lr: float = 2e-4,
        l2scale: float = 1e-5,
        discrimination_steps: int = 2,
        generation_steps: int = 1,
        reconstruction_steps: int = 2,
        min_recon_weight: float = 0.1,
        max_recon_weight: float = 1.0,
        gp_lambda: float = 10,
    ):
        """
        Train TAEGAN using the original GitHub hyperparameters.
        """
        self.model.train(
            batch_size=batch_size,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            lr=lr,
            l2scale=l2scale,
            discrimination_steps=discrimination_steps,
            generation_steps=generation_steps,
            reconstruction_steps=reconstruction_steps,
            min_recon_weight=min_recon_weight,
            max_recon_weight=max_recon_weight,
            gp_lambda=gp_lambda,
        )

    @torch.no_grad()
    def generate(
        self,
        n: int,
        batch_size: int = 100,
        temperature: Tuple[float, float] = (0.5, 1.2),
    ) -> torch.FloatTensor:
        """
        Generate synthetic samples using the trained TAEGAN generator.
        """
        return self.model.generate(
            n=n,
            batch_size=batch_size,
            temperature=temperature,
        )
