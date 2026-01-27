import torch
import torch.nn as nn
from typing import Any, List, Optional, Union

def get_nonlin(name: str) -> nn.Module:
    match name:
        case "none":
            return nn.Identity()
        case "elu":
            return nn.ELU()
        case "relu":
            return nn.ReLU()
        case "leaky_relu":
            return nn.LeakyReLU()
        case "selu":
            return nn.SELU()
        case "tanh":
            return nn.Tanh()
        case "sigmoid":
            return nn.Sigmoid()
        case "softmax":
            return nn.Softmax(dim=-1)
        case _:
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
        grad_input = grad_output * E.t()
        return grad_input

trace_expm = TraceExpm.apply

class Generator_causal(nn.Module):
    def __init__(self, z_dim: int, x_dim: int, h_dim: int, device: str, f_scale: float = 0.1, 
                 dag_seed: list = [], nonlin_out: Optional[List] = None):
        super().__init__()
        self.device = device
        self.x_dim = x_dim
        self.nonlin_out = nonlin_out

        def block(in_feat: int, out_feat: int, normalize: bool = False) -> list:
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.shared = nn.Sequential(*block(h_dim, h_dim), *block(h_dim, h_dim)).to(device)

        # Initialise adjacency matrix
        if len(dag_seed) > 0:
            M_init = torch.zeros(x_dim, x_dim)
            for pair in dag_seed:
                M_init[pair[0], pair[1]] = 1
            M_init = M_init.to(device)
            # Freeze if DAG is provided
            self.M = torch.nn.parameter.Parameter(M_init, requires_grad=False).to(device)
        else:
            # Learn DAG if not provided
            M_init = torch.rand(x_dim, x_dim) * 0.2
            M_init[torch.eye(x_dim, dtype=bool)] = 0
            M_init = M_init.to(device)
            self.M = torch.nn.parameter.Parameter(M_init).to(device)

        self.fc_i = nn.ModuleList([nn.Linear(x_dim + 1, h_dim) for i in range(self.x_dim)]).to(device)
        self.fc_f = nn.ModuleList([nn.Linear(h_dim, 1) for i in range(self.x_dim)]).to(device)

        # Weight initialisation
        for layer in self.shared.parameters():
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer.weight)
                layer.weight.data *= f_scale

        for i, layer in enumerate(self.fc_i):
            torch.nn.init.xavier_normal_(layer.weight)
            layer.weight.data *= f_scale
            layer.weight.data[:, i] = 1e-16

        for i, layer in enumerate(self.fc_f):
            torch.nn.init.xavier_normal_(layer.weight)
            layer.weight.data *= f_scale

    def sequential(self, x: torch.Tensor, z: torch.Tensor, gen_order: Union[list, dict, None] = None, biased_edges: dict = {}) -> torch.Tensor:
        out = x.clone().detach()
        if gen_order is None:
            gen_order = list(range(self.x_dim))

        for i in gen_order:
            x_masked = out.clone() * self.M[:, i]
            x_masked[:, i] = 0.0
            
            # Handling biased edges for counterfactuals/interventions
            if i in biased_edges:
                for j in biased_edges[i]:
                    x_j = x_masked[:, j]
                    perm = torch.randperm(len(x_j))
                    x_masked[:, j] = x_j[perm]
            
            inp = torch.cat([x_masked, z[:, i].unsqueeze(1)], axis=1)
            out_i = self.fc_i[i](inp)
            out_i = nn.ReLU()(out_i)
            out_i = self.shared(out_i)
            out_i = self.fc_f[i](out_i).squeeze()
            out[:, i] = out_i

        return out

class Discriminator(nn.Module):
    def __init__(self, x_dim: int, h_dim: int, device: str) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        ).to(device)

        for layer in self.model.parameters():
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer)

    def forward(self, x_hat: torch.Tensor) -> torch.Tensor:
        return self.model(x_hat)