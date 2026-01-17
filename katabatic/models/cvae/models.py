from __future__ import annotations

from typing import Any, Optional, Dict, List, Tuple
import os
import json
import importlib

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model as BaseModel
from .utils import (
    infer_schema,
    fit_transformers,
    encode_df,
    decode_batch,
    build_conditioning,
    sample_conditions,
    ColumnMeta,
)


def _try_import(module: str):
    try:
        return importlib.import_module(module)
    except Exception:
        return None


class CVAE(BaseModel):
    """
    Conditional Variational Autoencoder (CVAE) for tabular data generation.
    - Encodes inputs and conditions into a latent space.
    - Decodes latent vectors and conditions back to data space.
    """

    def __init__(
        self,
        *,
        epochs: int = 300,
        batch_size: int = 512,
        latent_dim: int = 256,
        encoder_hidden: Tuple[int, ...] = (256, 256),
        decoder_hidden: Tuple[int, ...] = (256, 256),
        lr: float = 1e-3,
        kl_weight: float = 0.01,
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.cfg = {
            "epochs": epochs,
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "encoder_hidden": list(encoder_hidden),
            "decoder_hidden": list(decoder_hidden),
            "lr": lr,
            "kl_weight": kl_weight,
            "seed": seed,
            "device": device,
        }
        self.schema: List[ColumnMeta] | None = None
        self.output_order: List[str] = []
        self.cat_blocks: Dict[str, Tuple[int, int]] = {}
        self.cond_blocks: Dict[str, Tuple[int, int]] = {}
        self.empirical_probs: Dict[str, np.ndarray] = {}
        self._train_df: Optional[pd.DataFrame] = None

        # Torch-related members
        self.encoder = None
        self.decoder = None
        self._enc_dim: int = 0
        self._cond_dim: int = 0
        self._device = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["torch", "sklearn"]

    def _build_torch_networks(self, torch, nn) -> None:
        in_dim = self._enc_dim + self._cond_dim
        latent_dim = self.cfg["latent_dim"]
        
        # Encoder
        class Encoder(nn.Module):
            def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
                super().__init__()
                layers = []
                last = input_dim
                for h in hidden_dims:
                    layers.append(nn.Linear(last, h))
                    layers.append(nn.ReLU())
                    last = h
                self.net = nn.Sequential(*layers)
                self.fc_mu = nn.Linear(last, latent_dim)
                self.fc_logvar = nn.Linear(last, latent_dim)

            def forward(self, x, c):
                inputs = torch.cat([x, c], dim=1)
                hidden = self.net(inputs)
                return self.fc_mu(hidden), self.fc_logvar(hidden)

        # Decoder
        class Decoder(nn.Module):
            def __init__(self, latent_dim: int, cond_dim: int, hidden_dims: List[int], output_dim: int):
                super().__init__()
                layers = []
                last = latent_dim + cond_dim
                for h in hidden_dims:
                    layers.append(nn.Linear(last, h))
                    layers.append(nn.ReLU())
                    last = h
                layers.append(nn.Linear(last, output_dim))
                self.net = nn.Sequential(*layers)

            def forward(self, z, c):
                inputs = torch.cat([z, c], dim=1)
                return self.net(inputs)

        self.encoder = Encoder(in_dim, self.cfg["encoder_hidden"], latent_dim).to(self._device)
        self.decoder = Decoder(latent_dim, self._cond_dim, self.cfg["decoder_hidden"], self._enc_dim).to(self._device)

    def reparameterize(self, mu, logvar, torch):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "CVAE":
        # Load data
        train_full = os.path.join(data_dir, "train_full.csv")
        x_path = os.path.join(data_dir, "x_train.csv")
        y_path = os.path.join(data_dir, "y_train.csv")
        
        if os.path.exists(train_full):
            df = pd.read_csv(train_full)
        else:
            if not (os.path.exists(x_path) and os.path.exists(y_path)):
                raise FileNotFoundError(
                    f"Could not find training data in {data_dir}. Expected train_full.csv or x_train.csv/y_train.csv.")
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path)
            if y.shape[1] != 1:
                raise ValueError("y_train.csv must have exactly one column (the target).")
            y_col = y.columns[0]
            df = pd.concat([X, y[y_col]], axis=1)

        self._train_df = df.copy()

        # Schema & encoders
        self.schema = infer_schema(df)
        fit_transformers(df, self.schema)

        torch = _try_import("torch")
        nn = _try_import("torch.nn")
        data_utils = _try_import("torch.utils.data")
        F = _try_import("torch.nn.functional")
        
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for CVAE.")

        torch.manual_seed(self.cfg["seed"])
        max_dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(self.cfg["device"] or max_dev)

        enc, self.cat_blocks, self.output_order = encode_df(df, self.schema)
        cond_full, self.cond_blocks = build_conditioning(df, self.schema)

        # Empirical probabilities
        self.empirical_probs = {}
        for col in self.schema:
            if col.kind == 'categorical':
                start, end = self.cond_blocks[col.name]
                counts = cond_full[:, start:end].sum(axis=0) + 1e-8
                p = counts / counts.sum()
                self.empirical_probs[col.name] = p.astype(np.float32)

        self._enc_dim = int(enc.shape[1])
        self._cond_dim = int(cond_full.shape[1])
        
        self._build_torch_networks(torch, nn)

        # DataLoader
        TensorDataset = data_utils.TensorDataset
        DataLoader = data_utils.DataLoader
        enc_t = torch.tensor(enc, dtype=torch.float32)
        cond_t = torch.tensor(cond_full, dtype=torch.float32)
        loader = DataLoader(TensorDataset(enc_t, cond_t), batch_size=self.cfg["batch_size"], shuffle=True, drop_last=False)

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), 
            lr=self.cfg["lr"]
        )

        self.encoder.train()
        self.decoder.train()

        for epoch in range(self.cfg["epochs"]):
            total_loss = 0
            for batch_x, batch_c in loader:
                batch_x = batch_x.to(self._device)
                batch_c = batch_c.to(self._device)

                mu, logvar = self.encoder(batch_x, batch_c)
                z = self.reparameterize(mu, logvar, torch)
                recon_x = self.decoder(z, batch_c)

                # Loss: reconstruction + KL
                # Simple MSE for reconstruction for now (improve for categorical/continuous mix)
                recon_loss = F.mse_loss(recon_x, batch_x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                loss = recon_loss + self.cfg["kl_weight"] * kl_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.cfg['epochs']}, Loss: {total_loss/len(loader.dataset):.4f}")

        self.is_fitted = True

        # Save outputs
        synth_dir = synthetic_dir
        if not synth_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "cvae")
        os.makedirs(synth_dir, exist_ok=True)

        df_s = self.sample(n=len(df))
        label = df.columns[-1]
        x_synth = df_s[df.columns[:-1]].copy()
        y_synth = df_s[[label]].copy()

        x_path_out = os.path.join(synth_dir, "x_synth.csv")
        y_path_out = os.path.join(synth_dir, "y_synth.csv")
        x_synth.to_csv(x_path_out, index=False)
        y_synth.to_csv(y_path_out, index=False, header=True)

        meta = {
            "schema": {
                "columns": df.columns.tolist(),
                "label": label,
                "dtypes": {c: str(df[c].dtype) for c in df.columns},
                "categorical_columns": [c.name for c in self.schema or [] if c.kind == 'categorical']
            },
            "training": self.cfg,
        }
        with open(os.path.join(synth_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[CVAE] Synthetic data saved:\n  X -> {x_path_out}\n  y -> {y_path_out}")
        return self

    def evaluate(self, *args, **kwargs) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return 0.0

    def sample(
        self,
        n: Optional[int] = None,
        conditional: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        if not self.is_fitted or self.schema is None:
            raise RuntimeError("Call train() before sample().")

        torch = _try_import("torch")
        self.decoder.eval()
        
        batch = int(n) if n is not None else 1000
        all_rows: List[pd.DataFrame] = []
        
        with torch.no_grad():
            steps = (batch + 1023) // 1024
            remain = batch
            for _ in range(steps):
                bsz = min(1024, remain)
                remain -= bsz
                
                z = torch.randn(bsz, self.cfg["latent_dim"], device=self._device)
                
                if conditional and len(conditional) == 1 and self.cond_blocks:
                    # Handle specific conditional sampling
                     name, val = next(iter(conditional.items()))
                     cond = np.zeros((bsz, self._cond_dim), dtype=np.float32)
                     if name in self.cond_blocks:
                         start, end = self.cond_blocks[name]
                         cats = next((c.categories for c in self.schema if c.name == name), [])
                         if cats:
                             try:
                                 idx = list(cats).index(str(val))
                                 cond[:, start + idx] = 1.0
                             except ValueError:
                                 pass
                     cond_t = torch.tensor(cond, dtype=torch.float32, device=self._device)
                else:
                    cond_np = sample_conditions(bsz, self.schema, self.cond_blocks, self.empirical_probs)
                    cond_t = torch.tensor(cond_np, dtype=torch.float32, device=self._device)

                recon_x = self.decoder(z, cond_t).cpu().numpy()
                df_batch = decode_batch(recon_x, self.schema)
                all_rows.append(df_batch)

        df_out = pd.concat(all_rows, axis=0).reset_index(drop=True)
        ordered_cols = [c.name for c in self.schema]
        return df_out[ordered_cols]
