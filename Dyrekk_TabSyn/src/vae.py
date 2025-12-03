import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch import Tensor

import typing as ty
import math

class Tokenizer(nn.Module):
    def __init__(
            self,
            d_numerical,
            categories,
            d_token,
            bias
            ):
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.weight = nn.Parameter(torch.empty(d_numerical + 1, d_token))
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(d_bias, d_token))
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        else:
            self.bias = None

    @property
    def n_tokens(self):
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num, x_cat):
        if x_num is None and x_cat is None:
            raise ValueError("Both x_num and x_cat cannot be None")

        if x_num is not None:
            batch_size, device, dtype = x_num.shape[0], x_num.device, x_num.dtype
        else:
            batch_size, device, dtype = x_cat.shape[0], x_cat.device, torch.float32
        
        cls_token = torch.ones(batch_size, 1, device=device, dtype=dtype)

        if x_num is not None:
            x_num_with_cls = torch.cat([cls_token, x_num], dim=1)
            x = self.weight[None] * x_num_with_cls.unsqueeze(-1)
        else:
            cls_emb = self.weight[0][None, None, :].to(device=device, dtype=dtype)
            x = cls_emb.expand(batch_size, 1, cls_emb.shape[-1]).contiguous()

        if self.bias is not None:
            full_bias = torch.cat([
                torch.zeros(1, self.bias.shape[1], device=device, dtype=dtype),
                self.bias
            ], dim=0)
            x = x + full_bias[None]

        if x_cat is not None:
            assert self.category_embeddings is not None, "Categories not initialized"
            cat_emb = self.category_embeddings(x_cat + self.category_offsets[None])
            x = torch.cat([x, cat_emb], dim=1)

        return x.contiguous()


class MultiheadAttention(nn.Module):
    def __init__(
            self,
            d,
            n_heads,
            dropout,
            initialization='kaiming'
            ):
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
            self,
            x_q,
            x_kv,
            key_compression=None,
            value_compression=None
            ):
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)

        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)

        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)

        return x


class Transformer(nn.Module):
    def __init__(
            self,
            n_layers: int,
            d_token: int,
            n_heads: int,
            d_out: int,
            d_ffn_factor: int,
            attention_dropout=0.1,
            ffn_dropout=0.1,
            residual_dropout=0.1,
            activation='relu',
            prenormalization=True,
            initialization='kaiming'
            ):
        super().__init__()

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(d_token, d_hidden),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()

            self.layers.append(layer)

        self.activation = nn.GELU() 
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(
            self,
            x,
            x_residual,
            layer,
            norm_idx
            ):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x):
        for layer_idx, layer in enumerate(self.layers):
            # Self-attention
            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](x_residual, x_residual)
            x = self._end_residual(x, x_residual, layer, 0)

            # FFN
            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)
        
        return x


class VAE(nn.Module):
    def __init__(
            self,
            d_numerical,
            categories,
            num_layers,
            hid_dim,
            n_head=1,
            factor=4,
            bias=True
            ):
        super(VAE, self).__init__()

        self.d_numerical = d_numerical
        self.categories = categories
        self.hid_dim = hid_dim
        d_token = hid_dim
        self.n_head = n_head

        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias=bias)

        self.encoder_mu = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)
        self.encoder_logvar = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

        self.decoder = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick with clamping for stability."""
        logvar = torch.clamp(logvar, min=-10, max=2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_num, x_cat):
        x = self.Tokenizer(x_num, x_cat)
        x = x.contiguous()
        mu_z = self.encoder_mu(x)
        logvar_z = self.encoder_logvar(x)
        z = self.reparameterize(mu_z, logvar_z)
        h = self.decoder(z[:, 1:])
        return h, mu_z, logvar_z


class Reconstructor(nn.Module):
    def __init__(
            self,
            d_numerical,
            categories,
            d_token
            ):
        super().__init__()
        self.d_numerical = d_numerical
        self.d_token = d_token
        self.categories = categories

        self.num_recons = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_token, d_token // 2),
                nn.GELU(),
                nn.Linear(d_token // 2, 1)
            ) for _ in range(d_numerical)
        ]) if d_numerical > 0 else nn.ModuleList()

        self.cat_recons = nn.ModuleList()
        if categories is not None:
            for d in categories:
                self.cat_recons.append(
                    nn.Sequential(
                        nn.Linear(d_token, d_token // 2),
                        nn.GELU(),
                        nn.Linear(d_token // 2, d)
                    )
                )

    def forward(self, h):
        recon_x_num = None
        recon_x_cat = []
        if len(self.num_recons) > 0:
            h_num = h[:, :self.d_numerical, :]
            num_outputs = []
            for i, recon in enumerate(self.num_recons):
                num_outputs.append(recon(h_num[:, i, :]))
            recon_x_num = torch.cat(num_outputs, dim=-1)
        if self.categories is not None and len(self.categories) > 0:
            h_cat = h[:, self.d_numerical:, :]
            for i, recon in enumerate(self.cat_recons):
                recon_x_cat.append(recon(h_cat[:, i, :]))

        return recon_x_num, recon_x_cat


class Model_VAE(nn.Module):
    def __init__(
            self, 
            num_layers, 
            d_numerical, 
            categories, 
            d_token, 
            n_head=1, 
            factor=4, 
            bias=True
            ):
        super(Model_VAE, self).__init__()

        self.VAE = VAE(
            d_numerical, 
            categories, 
            num_layers, 
            d_token, 
            n_head=n_head, 
            factor=factor, 
            bias=bias
            )
        self.Reconstructor = Reconstructor(d_numerical, categories, d_token)

    def forward(self, x_num, x_cat):
        h, mu_z, logvar_z = self.VAE(x_num, x_cat)
        recon_x_num, recon_x_cat = self.Reconstructor(h)
        return recon_x_num, recon_x_cat, mu_z, logvar_z


class Encoder_model(nn.Module):
    def __init__(
            self,
            num_layers,
            d_numerical,
            categories,
            d_token,
            n_head,
            factor,
            bias=True
            ):
        super(Encoder_model, self).__init__()
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias)
        self.VAE_Encoder = Transformer(num_layers, d_token, n_head, d_token, factor)

    def load_weights(self, Pretrained_VAE):
        self.Tokenizer.load_state_dict(Pretrained_VAE.VAE.Tokenizer.state_dict())
        self.VAE_Encoder.load_state_dict(Pretrained_VAE.VAE.encoder_mu.state_dict())

    def forward(self, x_num, x_cat):
        x = self.Tokenizer(x_num, x_cat)
        z = self.VAE_Encoder(x)
        return z


class Decoder_model(nn.Module):
    def __init__(
            self,
            num_layers,
            d_numerical,
            categories,
            d_token,
            n_head,
            factor,
            bias=True
            ):
        super(Decoder_model, self).__init__()
        self.VAE_Decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.Detokenizer = Reconstructor(d_numerical, categories, d_token)

    def load_weights(self, Pretrained_VAE):
        self.VAE_Decoder.load_state_dict(Pretrained_VAE.VAE.decoder.state_dict())
        self.Detokenizer.load_state_dict(Pretrained_VAE.Reconstructor.state_dict())

    def forward(self, z):
        h = self.VAE_Decoder(z)
        x_hat_num, x_hat_cat = self.Detokenizer(h)
        return x_hat_num, x_hat_cat
    

class ConditionalVAE(nn.Module):
    """
    Conditional VAE that takes class labels as conditioning input.
    This naturally learns class-specific distributions.
    """
    def __init__(
            self,
            d_numerical,
            categories,
            num_layers,
            hid_dim,
            n_classes,
            n_head=1,
            factor=4,
            bias=True
            ):
        super(ConditionalVAE, self).__init__()

        self.d_numerical = d_numerical
        self.categories = categories
        self.hid_dim = hid_dim
        self.n_classes = n_classes
        d_token = hid_dim
        self.n_head = n_head

        # Class embedding
        self.class_embedding = nn.Embedding(n_classes, hid_dim)
        
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias=bias)

        self.encoder_mu = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)
        self.encoder_logvar = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

        self.decoder = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick with clamping for stability."""
        logvar = torch.clamp(logvar, min=-10, max=2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
            self,
            x_num,
            x_cat,
            y_class
            ):
        """
        x_num: numerical features
        x_cat: categorical features  
        y_class: target class labels (B,)
        """
        # Tokenize input
        x = self.Tokenizer(x_num, x_cat)
        
        # Add class embedding to CLS token
        class_emb = self.class_embedding(y_class).unsqueeze(1)  # (B, 1, hid_dim)
        x[:, 0, :] = x[:, 0, :] + class_emb.squeeze(1)  # Add to CLS token

        # Encode
        mu_z = self.encoder_mu(x)
        logvar_z = self.encoder_logvar(x)

        # Reparameterize
        z = self.reparameterize(mu_z, logvar_z)

        # Decode - inject class info again
        z[:, 0, :] = z[:, 0, :] + class_emb.squeeze(1)
        h = self.decoder(z[:, 1:])

        return h, mu_z, logvar_z


class Model_ConditionalVAE(nn.Module):
    """Wrapper for conditional VAE with reconstructor"""
    def __init__(
            self,
            num_layers,
            d_numerical,
            categories,
            d_token,
            n_classes,
            n_head=1,
            factor=4,
            bias=True
            ):
        super(Model_ConditionalVAE, self).__init__()

        self.VAE = ConditionalVAE(d_numerical, categories, num_layers, d_token, n_classes, n_head=n_head, factor=factor, bias=bias)
        self.Reconstructor = Reconstructor(d_numerical, categories, d_token)

    def forward(
            self,
            x_num,
            x_cat,
            y_class
            ):
        h, mu_z, logvar_z = self.VAE(x_num, x_cat, y_class)
        recon_x_num, recon_x_cat = self.Reconstructor(h)
        return recon_x_num, recon_x_cat, mu_z, logvar_z