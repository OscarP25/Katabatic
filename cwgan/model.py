import torch, torch.nn as nn, torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=256, depth=3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LeakyReLU(0.2, inplace=True)]
            d = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self, z_dim, y_dim, num_dim, cat_sizes, hidden=256, depth=3):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.num_dim = num_dim
        self.cat_sizes = cat_sizes
        self.backbone = MLP(z_dim + y_dim, hidden, depth)
        if num_dim > 0:
            self.num_head = nn.Linear(hidden, num_dim)
        else:
            self.num_head = None
        self.cat_heads = nn.ModuleList([nn.Linear(hidden, k) for k in cat_sizes])

    def forward(self, z, y_onehot, tau=0.5):
        h = self.backbone(torch.cat([z, y_onehot], dim=1))
        if self.num_head is not None:
            x_num = torch.sigmoid(self.num_head(h))  # keep in [0,1]
        else:
            x_num = None
        x_cats = []
        for head in self.cat_heads:
            logits = head(h)
            # straight-through Gumbel-Softmax
            x_cat = F.gumbel_softmax(logits, tau=tau, hard=True)
            x_cats.append(x_cat)
        if len(x_cats) > 0:
            x_cat_concat = torch.cat(x_cats, dim=1)
        else:
            x_cat_concat = None
        return x_num, x_cat_concat

class Discriminator(nn.Module):
    def __init__(self, y_dim, num_dim, cat_sizes, hidden=256, depth=3):
        super().__init__()
        in_dim = y_dim + num_dim + sum(cat_sizes)
        self.backbone = MLP(in_dim, hidden, depth)
        self.d_out = nn.Linear(hidden, 1)      # WGAN score
        self.ac_out = nn.Linear(hidden, y_dim) # Aux classifier logits

    def forward(self, x_num, x_cat, y_onehot):
        parts = [y_onehot]
        if x_num is not None:
            parts.append(x_num)
        if x_cat is not None:
            parts.append(x_cat)
        x = torch.cat(parts, dim=1)
        h = self.backbone(x)
        return self.d_out(h), self.ac_out(h)
