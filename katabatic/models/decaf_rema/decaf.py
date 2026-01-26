from collections import OrderedDict
from typing import Any, List, Optional, Union

import networkx as nx
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from . import logger as log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_nonlin(name: str) -> nn.Module:
    if name == "none":
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "softmax":
        return nn.Softmax(dim=-1)
    else:
        raise ValueError(f"Unknown nonlinearity {name}")


class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, data: torch.Tensor) -> torch.Tensor:
        E = torch.linalg.matrix_exp(data)
        f = torch.trace(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=data.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (E,) = ctx.saved_tensors
        return grad_output * E.t()


trace_expm = TraceExpm.apply


class Generator_causal(nn.Module):
    def __init__(
        self,
        z_dim: int,
        x_dim: int,
        h_dim: int,
        f_scale: float = 0.1,
        dag_seed: list = [],
        nonlin_out: Optional[List] = None,
    ) -> None:
        super().__init__()

        self.x_dim = x_dim
        self.nonlin_out = nonlin_out

        def block(in_feat: int, out_feat: int):
            return [
                nn.Linear(in_feat, out_feat),
                nn.ReLU(inplace=True),
            ]

        self.shared = nn.Sequential(
            *block(h_dim, h_dim),
            *block(h_dim, h_dim),
        ).to(DEVICE)

        if len(dag_seed) > 0:
            M_init = torch.zeros(x_dim, x_dim)
            for i, j in dag_seed:
                M_init[i, j] = 1.0
            self.M = nn.Parameter(M_init.to(DEVICE), requires_grad=False)
        else:
            M_init = torch.rand(x_dim, x_dim) * 0.2
            M_init.fill_diagonal_(0.0)
            self.M = nn.Parameter(M_init.to(DEVICE))

        self.fc_i = nn.ModuleList(
            [nn.Linear(x_dim + 1, h_dim) for _ in range(x_dim)]
        )
        self.fc_f = nn.ModuleList(
            [nn.Linear(h_dim, 1) for _ in range(x_dim)]
        )

        for layer in self.parameters():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                layer.weight.data *= f_scale

    def sequential(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        gen_order: Optional[list] = None,
        biased_edges: dict = {},
    ) -> torch.Tensor:

        out = x.clone()
        gen_order = gen_order or list(range(self.x_dim))

        for i in gen_order:
            x_masked = out * self.M[:, i]
            x_masked[:, i] = 0.0

            if i in biased_edges:
                for j in biased_edges[i]:
                    perm = torch.randperm(len(x_masked), device=x_masked.device)
                    x_masked[:, j] = x_masked[perm, j]

            h = self.fc_i[i](torch.cat([x_masked, z[:, i:i+1]], dim=1))
            h = self.shared(h)
            new_val = self.fc_f[i](h).squeeze()

            out = torch.cat(
                [
                    out[:, :i],
                    new_val.unsqueeze(1),
                    out[:, i + 1 :],
                ],
                dim=1,
            )

        return out


class Discriminator(nn.Module):
    def __init__(self, x_dim: int, h_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DECAF(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        dag_seed: list = [],
        h_dim: int = 200,
        lr: float = 1e-3,
        lambda_gp: float = 10.0,
        lambda_privacy: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = Generator_causal(
            z_dim=input_dim,
            x_dim=input_dim,
            h_dim=h_dim,
            dag_seed=dag_seed,
        )
        self.discriminator = Discriminator(input_dim, h_dim)

    def sample_z(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.hparams.input_dim, device=self.device)

    def training_step(self, batch, batch_idx):
        opt_d, opt_g = self.optimizers()

        z = self.sample_z(batch.size(0))
        fake = self.generator.sequential(batch, z)

        opt_d.zero_grad()
        d_loss = self.discriminator(fake.detach()).mean() - self.discriminator(batch).mean()
        self.manual_backward(d_loss)
        opt_d.step()

        opt_g.zero_grad()
        fake = self.generator.sequential(batch, self.sample_z(batch.size(0)))
        g_loss = -self.discriminator(fake).mean()
        self.manual_backward(g_loss)
        opt_g.step()

        return g_loss

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=self.hparams.lr)
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=self.hparams.lr)
        return [opt_d, opt_g]
