"""
Optimized MLP architecture for TabDDPM with Residual Connections and LayerNorm
"""
import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal time embedding (Standard)"""
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """
    Optimized Block: Linear -> Activation -> Dropout -> Residual Connection
    Includes LayerNorm for stability.
    """
    def __init__(self, in_features, out_features, dropout=0.0, use_layer_norm=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm
        
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_features)
        
        # Project input if dimensions don't match for residual connection
        self.project_residual = None
        if in_features != out_features:
            self.project_residual = nn.Linear(in_features, out_features)

    def forward(self, x):
        residual = x
        
        # Main path
        out = self.linear(x)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.project_residual is not None:
            residual = self.project_residual(residual)
            
        out = out + residual
        
        # Layer Norm (Applied after residual in Pre-LN or Post-LN config; here Post-LN)
        if self.use_layer_norm:
            out = self.layer_norm(out)
            
        return out

class DenoisingMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[256, 512, 512, 512], # Deeper defaults
        time_embed_dim=128,
        num_classes=None,
        dropout=0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        # Time Embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Class Embedding
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, time_embed_dim)
        else:
            self.class_embed = None
        
        # Input Projection [cite: 112]
        self.input_proj = nn.Linear(input_dim, time_embed_dim)
        
        # Residual MLP Blocks
        self.blocks = nn.ModuleList()
        prev_dim = time_embed_dim
        
        for hidden_dim in hidden_dims:
            self.blocks.append(
                ResidualBlock(
                    prev_dim, 
                    hidden_dim, 
                    dropout=dropout,
                    use_layer_norm=True
                )
            )
            prev_dim = hidden_dim
        
        # Final Output Projection
        self.output_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x, t, y=None):
        # 1. Embeddings
        t_emb = self.time_embed(t)
        x_emb = self.input_proj(x)
        
        # 2. Combine (Equation 5 in TabDDPM paper) [cite: 112]
        x = x_emb + t_emb
        
        if self.class_embed is not None and y is not None:
            y_emb = self.class_embed(y)
            x = x + y_emb
            
        # 3. Pass through Residual Blocks
        for block in self.blocks:
            x = block(x)
        
        # 4. Output
        x = self.output_layer(x)
        
        return x