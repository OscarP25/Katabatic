

import torch
from torch import nn, Tensor
from typing import Optional, Tuple, Union
import dgl
from dgl.nn import GraphConv, SAGEConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, masked_select_nnz, matmul

class Encoder(nn.Module):
    def __init__(self, input_dim, encoder_dim, encoder_l, device):
        super().__init__()

        layers = [nn.Linear(input_dim, encoder_dim), nn.ReLU()]
        for _ in range(encoder_l - 2):
            encoder_dim_ = encoder_dim // 2
            layers += [nn.Linear(encoder_dim, encoder_dim_), nn.ReLU()]
            encoder_dim = encoder_dim_

        self.encoder = nn.Sequential(*layers)
        self.mu = nn.Linear(encoder_dim, input_dim)
        self.logvar = nn.Linear(encoder_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, (mu, logvar)


class LearnedGraph(nn.Module):
    def __init__(self, input_dim, graph_prior, prior_mask, threshold, device):
        super().__init__()

        self.graph = nn.Parameter(torch.zeros(input_dim, input_dim, device=device))
        self.threshold = nn.Threshold(threshold, 0)
        self.act = nn.Sigmoid()
        self.device = device

        if graph_prior is not None and prior_mask is not None:
            self.graph_prior = graph_prior.to(device)
            self.prior_mask = prior_mask.to(device)
            self.use_prior = True
        else:
            self.use_prior = False

    def forward(self, iteration):
        g = self.graph
        if self.use_prior:
            g = self.prior_mask * self.graph_prior + (1 - self.prior_mask) * g

        g = self.act(g)
        g = g * (1 - torch.eye(g.size(0), device=self.device)) + torch.eye(
            g.size(0), device=self.device
        )

        if iteration > 50:
            g = self.threshold(g)

        return g

class GraphInputProcessorHomo(nn.Module):
    def __init__(self, input_dim, decoder_dim, het_encoding, device):
        super().__init__()
        self.device = device
        feat_dim = input_dim + 1 if het_encoding else 1

        self.embeddings = nn.ModuleList(
            [nn.Sequential(nn.Linear(feat_dim, decoder_dim), nn.Tanh()) for _ in range(input_dim)]
        )

        self.het_encoding = het_encoding

    def forward(self, z, adj):
        b, n = z.shape
        z = z.unsqueeze(-1)

        if self.het_encoding:
            eye = torch.eye(n, device=self.device)
            enc = torch.stack([eye] * b)
            z = torch.cat([z, enc], dim=-1)

        z = torch.stack([f(z[:, i]) for i, f in enumerate(self.embeddings)], dim=1)
        z = z.flatten(0, 1)

        edge_index = adj.nonzero().t()
        edge_weight = adj[edge_index[0], edge_index[1]]

        g = dgl.graph((edge_index[0], edge_index[1]))
        g = dgl.batch([g] * b)

        return z, g, edge_weight.repeat(b)

class GraphInputProcessorHet(nn.Module):
    def __init__(self, input_dim, decoder_dim, n_edge_types, het_encoding, device):
        super().__init__()
        self.device = device
        feat_dim = input_dim + 1 if het_encoding else 1

        self.embeddings = nn.ModuleList(
            [nn.Sequential(nn.Linear(feat_dim, decoder_dim), nn.Tanh()) for _ in range(input_dim)]
        )

        self.n_edge_types = n_edge_types
        self.het_encoding = het_encoding

    def forward(self, z, adj):
        b, n = z.shape
        z = z.unsqueeze(-1)

        if self.het_encoding:
            eye = torch.eye(n, device=self.device)
            enc = torch.stack([eye] * b)
            z = torch.cat([z, enc], dim=-1)

        z = torch.stack([f(z[:, i]) for i, f in enumerate(self.embeddings)], dim=1)
        z = z.reshape(b * n, -1)

        edge_index, edge_weight = dense_to_sparse(adj.repeat(b, 1, 1))
        edge_types = torch.arange(1, self.n_edge_types + 1, device=self.device)
        edge_types = edge_types[(edge_index[0] % n) * n + (edge_index[1] % n)]

        return z, edge_index, edge_weight, edge_types

class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, root_weight=True):
        super().__init__(aggr="mean")
        self.weight = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        self.root = nn.Parameter(torch.Tensor(in_channels, out_channels)) if root_weight else None
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        out = torch.zeros(x.size(0), self.weight.size(-1), device=x.device)
        for r in range(self.weight.size(0)):
            mask = edge_type == r
            if mask.any():
                out += self.propagate(edge_index[:, mask], x=x) @ self.weight[r]
        if self.root is not None:
            out += x @ self.root
        return out + self.bias


class GraphDecoderHomo(nn.Module):
    def __init__(self, dim, layers, arch):
        super().__init__()
        mods = []
        for i in range(layers):
            out = 1 if i == layers - 1 else dim // 2
            mods.append(GraphConv(dim, out))
            dim = out
        self.net = nn.Sequential(*mods)

    def forward(self, g_in, b):
        z, g, w = g_in
        for layer in self.net:
            z = layer(g, z, edge_weight=w)
        return z.view(b, -1)

class GraphDecoderHet(nn.Module):
    def __init__(self, dim, layers, n_edge_types):
        super().__init__()
        mods = []
        for i in range(layers):
            out = 1 if i == layers - 1 else dim // 2
            mods.append(RGCNConv(dim, out, n_edge_types + 1))
            dim = out
        self.net = nn.Sequential(*mods)

    def forward(self, g_in, b):
        z, ei, ew, et = g_in
        for layer in self.net:
            z = layer(z, ei, et, ew)
        return z.view(b, -1)

class Goggle(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_dim,
        encoder_l,
        het_encoding,
        decoder_dim,
        decoder_l,
        threshold,
        decoder_arch,
        graph_prior,
        prior_mask,
        device,
    ):
        super().__init__()

        self.encoder = Encoder(input_dim, encoder_dim, encoder_l, device)

        # 🔧 FIX: explicit learned_graph name
        self.learned_graph = LearnedGraph(
            input_dim, graph_prior, prior_mask, threshold, device
        )

        if decoder_arch == "het":
            n_edge_types = input_dim * input_dim
            self.processor = GraphInputProcessorHet(
                input_dim, decoder_dim, n_edge_types, het_encoding, device
            )
            self.decoder = GraphDecoderHet(decoder_dim, decoder_l, n_edge_types)
        else:
            self.processor = GraphInputProcessorHomo(
                input_dim, decoder_dim, het_encoding, device
            )
            self.decoder = GraphDecoderHomo(decoder_dim, decoder_l, decoder_arch)

    def forward(self, x, iteration):
        z, (mu, logvar) = self.encoder(x)
        adj = self.learned_graph(iteration)
        g_in = self.processor(z, adj)
        x_hat = self.decoder(g_in, x.size(0))
        return x_hat, adj, mu, logvar

    def sample(self, n):
        z = torch.randn(n, self.encoder.mu.out_features, device=self.learned_graph.device)
        adj = self.learned_graph(100)
        g_in = self.processor(z, adj)
        return self.decoder(g_in, n)


class GoggleLoss(nn.Module):
    def __init__(self, alpha, beta, graph_prior, device):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss(reduction="sum")
        self.graph_prior = graph_prior.to(device) if graph_prior is not None else None

    def forward(self, x_hat, x, mu, logvar, graph):
        rec = self.mse(x_hat, x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        graph_loss = graph.norm(1) / graph.numel()
        return rec + self.alpha * kld + self.beta * graph_loss, rec, kld, graph_loss
