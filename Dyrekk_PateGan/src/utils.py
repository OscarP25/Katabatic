# utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ============================================================================
# EMBEDDING-BASED ARCHITECTURES (No one-hot encoding!)
# ============================================================================

class EmbeddingGenerator(nn.Module):
    """
    Generator that outputs categorical indices directly (not one-hot vectors).
    Uses Gumbel-Softmax for differentiable sampling during training.
    
    Args:
        latent_dim: Dimension of input noise vector
        cat_dims: List of number of categories per feature [n_cat1, n_cat2, ...]
        hidden_dim: Hidden layer dimension
        embed_dim: Embedding dimension (not used in generator, kept for consistency)
    """
    def __init__(
        self,
        latent_dim: int,
        cat_dims: List[int],
        hidden_dim: int = 256,
        embed_dim: int = 16  # Not used but kept for API consistency
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cat_dims = cat_dims
        self.n_features = len(cat_dims)
        self.hidden_dim = hidden_dim
        
        # Shared encoder with batch normalization
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        
        # Per-feature logit heads (one per categorical feature)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim // 2, n_cat)
            )
            for n_cat in cat_dims
        ])
    
    def forward(self, z: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        """
        Generate categorical samples using Gumbel-Softmax.
        
        Args:
            z: Noise tensor of shape (batch, latent_dim)
            temperature: Gumbel-Softmax temperature (lower = harder, higher = softer)
        
        Returns:
            LongTensor of shape (batch, n_features) with category indices
        """
        h = self.encoder(z)
        
        # Generate categorical samples for each feature
        all_indices = []
        for i, head in enumerate(self.heads):
            logits = head(h)  # (batch, n_categories_i)
            
            # Gumbel-Softmax trick for differentiable categorical sampling
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            gumbel_logits = (logits + gumbel_noise) / temperature
            
            # Straight-through estimator:
            # Forward pass: hard argmax (discrete)
            # Backward pass: soft gradients
            soft = F.softmax(gumbel_logits, dim=-1)
            hard = torch.zeros_like(soft)
            hard_indices = soft.argmax(dim=1, keepdim=True)
            hard.scatter_(1, hard_indices, 1.0)
            
            # Straight-through: gradient flows through soft, output is hard
            output = hard - soft.detach() + soft
            indices = output.argmax(dim=1)  # (batch,)
            all_indices.append(indices)
        
        return torch.stack(all_indices, dim=1)  # (batch, n_features)


class EmbeddingDiscriminator(nn.Module):
    """
    Discriminator that uses learnable embeddings for categorical features.
    No one-hot encoding needed!
    
    Args:
        cat_dims: List of number of categories per feature
        hidden_dim: Hidden layer dimension
        embed_dim: Embedding dimension per feature
    """
    def __init__(
        self,
        cat_dims: List[int],
        hidden_dim: int = 256,
        embed_dim: int = 16
    ):
        super().__init__()
        self.cat_dims = cat_dims
        self.embed_dim = embed_dim
        self.n_features = len(cat_dims)
        
        # Embedding layer for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cat, embed_dim) for n_cat in cat_dims
        ])
        
        # Discriminator network
        total_dim = self.n_features * embed_dim
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LongTensor of shape (batch, n_features) with category indices
        
        Returns:
            Logits of shape (batch, 1)
        """
        # Clamp indices to valid range (safety check)
        x = torch.clamp(x, min=0)
        for i in range(self.n_features):
            x[:, i] = torch.clamp(x[:, i], max=self.cat_dims[i] - 1)
        
        # Embed each feature
        embedded = []
        for i in range(self.n_features):
            emb = self.embeddings[i](x[:, i])  # (batch, embed_dim)
            embedded.append(emb)
        
        # Concatenate all embeddings
        x_embedded = torch.cat(embedded, dim=1)  # (batch, n_features * embed_dim)
        
        return self.net(x_embedded)


# ============================================================================
# LEGACY COMPONENTS (kept for backward compatibility if needed)
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.fc1(x)), 0.2)
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return residual + out


# Keep old class names as aliases for backward compatibility
TeacherDiscriminator = EmbeddingDiscriminator
StudentDiscriminator = EmbeddingDiscriminator
Generator = EmbeddingGenerator

class AuxiliaryClassifierDiscriminator(nn.Module):
    """Discriminator with auxiliary classifier for class prediction"""
    def __init__(self, cat_dims, n_classes, hidden_dim=64, embed_dim=8):
        super().__init__()
        self.cat_dims = cat_dims
        self.n_classes = n_classes
        self.n_features = len(cat_dims)
        
        # Embedding for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, embed_dim) for cat_dim in cat_dims
        ])
        
        total_embed_dim = len(cat_dims) * embed_dim
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(total_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Real/Fake head
        self.real_fake_head = nn.Linear(hidden_dim, 1)
        
        # Class prediction head
        self.class_head = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, x, return_class_logits=False):
        """
        x: LongTensor of shape (batch, n_features)
        return_class_logits: If True, also return class predictions
        """
        # Embed each feature
        embeds = []
        for i, emb_layer in enumerate(self.embeddings):
            embeds.append(emb_layer(x[:, i]))
        
        x = torch.cat(embeds, dim=1)
        features = self.feature_net(x)
        
        real_fake_logits = self.real_fake_head(features)
        
        if return_class_logits:
            class_logits = self.class_head(features)
            return real_fake_logits, class_logits
        
        return real_fake_logits
