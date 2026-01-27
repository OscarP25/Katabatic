import torch
import torch.nn.functional as f
from torch import nn


class Generator(nn.Module):
    def __init__(self, input_dim, continuous_columns, discrete_columns):
        super(Generator, self).__init__()
        self._input_dim = input_dim
        self._discrete_columns = discrete_columns
        self._num_continuous_columns = len(continuous_columns)

        self.lin1 = nn.Linear(self._input_dim, self._input_dim)
        self.lin_numerical = nn.Linear(
            self._input_dim, self._num_continuous_columns)

        self.lin_cat = nn.ModuleDict()
        for key, value in self._discrete_columns.items():
            self.lin_cat[key] = nn.Linear(self._input_dim, value)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        x_numerical = f.relu(self.lin_numerical(x))
        x_cat = []
        for key in self.lin_cat:
            x_cat.append(f.gumbel_softmax(self.lin_cat[key](x), tau=0.2))
        x_final = torch.cat((x_numerical, *x_cat), 1)
        return x_final


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self._input_dim = input_dim
        self.dense1 = nn.Linear(self._input_dim, self._input_dim)
        self.dense2 = nn.Linear(self._input_dim, self._input_dim)

    def forward(self, x):
        x = f.leaky_relu(self.dense1(x))
        x = f.leaky_relu(self.dense2(x))
        return x


class FairLossFunc(nn.Module):
    def __init__(
            self,
            S_start_index,
            Y_start_index,
            underpriv_index,
            priv_index,
            undesire_index,
            desire_index):
        super(FairLossFunc, self).__init__()
        self._S_start_index = S_start_index
        self._Y_start_index = Y_start_index
        self._underpriv_index = underpriv_index
        self._priv_index = priv_index
        self._undesire_index = undesire_index
        self._desire_index = desire_index

    def forward(self, x, crit_fake_pred, lamda):
        G = x[:, self._S_start_index:self._S_start_index + 2]
        I = x[:, self._Y_start_index:self._Y_start_index + 2]
        disp = -1.0 * lamda * (torch.mean(G[:, self._underpriv_index] * I[:, self._desire_index]) / (x[:, self._S_start_index + self._underpriv_index].sum()) - torch.mean(G[:, self._priv_index] * I[:, self._desire_index]) / (x[:, self._S_start_index + self._priv_index].sum())) - 1.0 * torch.mean(crit_fake_pred)
        return disp


def get_gradient(crit, real, fake, epsilon):
    mixed_data = real * epsilon + fake * (1 - epsilon)

    mixed_scores = crit(mixed_data)

    gradient = torch.autograd.grad(
        inputs=mixed_data,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)

    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)

    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - \
        torch.mean(crit_real_pred) + c_lambda * gp

    return crit_loss
