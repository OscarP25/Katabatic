# D:\PROJECT MEG\katabatic\models\meg.py
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---- determinism (reasonable defaults)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(False)

class MaskedGenerator(nn.Module):
    """
    Shared trunk + per-feature heads with correct output sizes (cardinalities).
    Trains with proxy loss: sum_j CE(head_j(x), x_j) on discretized inputs.
    """
    def __init__(self, n_features: int, cardinalities, hidden: int = 128):
        super().__init__()
        self.n_features = int(n_features)
        self.cardinalities = list(map(int, cardinalities))
        self.shared = nn.Sequential(
            nn.Linear(self.n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden, self.cardinalities[j]) for j in range(self.n_features)
        ])

    def forward(self, x: torch.Tensor, target_idx: int) -> torch.Tensor:
        h = self.shared(x)
        return self.heads[target_idx](h)  # logits [B, K_j]

class Discriminator(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MEG:
    """
    Research-aligned MEG skeleton:
      - Train on discretized integer features
      - Per-feature heads with true cardinalities
      - Proxy loss = sum_j CE(head_j(x), x_j)
      - Adversarial loss (beta)
      - Gibbs-like sampling at generation
    """
    def __init__(self, epochs=50, batch_size=256, lr=1e-3, beta=1.0, cardinalities=None):
        self.epochs = int(epochs)
        self.bs = int(batch_size)
        self.lr = float(lr)
        self.beta = float(beta)
        self.cardinalities = cardinalities  # list[int] or None
        self.n_features = None

        self.generator: MaskedGenerator | None = None
        self.discriminator: Discriminator | None = None

    def fit(self, X_disc: np.ndarray):
        """
        X_disc: np.int array (N, d), each col in [0..K_j-1]
        If cardinalities not given, inferred as max+1 per feature.
        """
        X_disc = np.asarray(X_disc)
        if not np.issubdtype(X_disc.dtype, np.integer):
            raise ValueError("MEG.fit expects discretized integer inputs per feature")

        self.n_features = X_disc.shape[1]
        if self.cardinalities is None:
            self.cardinalities = [(int(X_disc[:, j].max()) + 1) for j in range(self.n_features)]

        # models & opt
        self.generator = MaskedGenerator(self.n_features, self.cardinalities)
        self.discriminator = Discriminator(self.n_features)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        ce = nn.CrossEntropyLoss()
        bce = nn.BCELoss()

        # data
        X_tensor = torch.tensor(X_disc, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.bs, shuffle=True, drop_last=False)

        for epoch in range(self.epochs):
            g_loss_sum, d_loss_sum = 0.0, 0.0
            for (xb,) in loader:
                # ---------------- Discriminator ----------------
                self.discriminator.train(); self.generator.eval()
                opt_d.zero_grad()

                # real
                real_pred = self.discriminator(xb)
                d_real = bce(real_pred, torch.ones_like(real_pred))

                # fake from gibbs-like sampler
                fake = self._gibbs_sample(n=xb.size(0), sweeps=2)
                fake_pred = self.discriminator(fake.detach())
                d_fake = bce(fake_pred, torch.zeros_like(fake_pred))

                d_loss = d_real + d_fake
                d_loss.backward()
                opt_d.step()

                # ---------------- Generator ----------------
                self.generator.train(); self.discriminator.eval()
                opt_g.zero_grad()

                proxy_loss = 0.0
                for j in range(self.n_features):
                    logits = self.generator(xb, j)   # [B, K_j]
                    y = xb[:, j].long()             # [B]
                    proxy_loss = proxy_loss + ce(logits, y)

                adv_pred = self.discriminator(fake)
                adv_loss = bce(adv_pred, torch.ones_like(adv_pred))

                g_loss = proxy_loss + self.beta * adv_loss
                g_loss.backward()
                opt_g.step()

                g_loss_sum += g_loss.item()
                d_loss_sum += d_loss.item()

            print(f"Epoch {epoch+1}/{self.epochs} | G_loss={g_loss_sum:.3f} | D_loss={d_loss_sum:.3f}")

    def _gibbs_sample(self, n: int, sweeps: int = 3) -> torch.Tensor:
        """
        Iteratively resample each feature conditioned on current others.
        """
        cols = []
        for j in range(self.n_features):
            K = int(self.cardinalities[j])
            cols.append(torch.randint(low=0, high=K, size=(n, 1)))
        x = torch.cat(cols, dim=1).float()

        with torch.no_grad():
            for _ in range(sweeps):
                for j in range(self.n_features):
                    logits = self.generator(x, j)           # [n, K_j]
                    probs = torch.softmax(logits, dim=1)
                    x[:, j] = torch.multinomial(probs, 1).squeeze(1).float()
        return x

    def sample(self, n: int) -> np.ndarray:
        with torch.no_grad():
            synth_disc = self._gibbs_sample(n, sweeps=3)
        return synth_disc.numpy().astype(np.int64)
