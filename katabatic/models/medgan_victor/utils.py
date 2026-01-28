# katabatic/models/medgan/utils.py

import torch
import torch.nn as nn


def sample_noise(n: int, dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(n, dim, device=device)


class Autoencoder(nn.Module):
    """
    Simple AE with encode/decode helpers (MedGAN-style).
    Uses Sigmoid output for BCE training on [0,1] normalized data.
    """
    def __init__(self, input_dim: int, encoder_dim: int, latent_dim: int, bn_decay: float = 0.99):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class Generator(nn.Module):
    """
    Generator in latent space.
    """
    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int = 2, bn_decay: float = 0.99):
        super().__init__()

        layers = []
        in_dim = latent_dim
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, noise):
        return self.net(noise)


class Discriminator(nn.Module):
    """
    Discriminator in latent space.
    """
    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        layers = []
        in_dim = latent_dim
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)
