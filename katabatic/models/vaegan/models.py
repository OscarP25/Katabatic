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


class VAEGANModel(BaseModel):
    """
    VAE-GAN (Variational Autoencoder - Generative Adversarial Network) for tabular data.
    
    Combines the strengths of VAEs and GANs:
    - VAE component: Structured latent space with encoder (mu, logvar) and decoder
    - GAN component: Adversarial training with discriminator for better sample quality
    - Hybrid training: Reconstruction loss + KL divergence + Adversarial loss
    
    Architecture:
    - Encoder: Maps input x -> latent distribution (mu, logvar)
    - Decoder/Generator: Maps latent z -> reconstructed/generated x (shared for VAE/GAN)
    - Discriminator: Distinguishes real from generated samples
    """

    def __init__(
        self,
        *,
        epochs: int = 300,
        batch_size: int = 512,
        latent_dim: int = 128,
        encoder_hidden: Tuple[int, ...] = (256, 256),
        decoder_hidden: Tuple[int, ...] = (256, 256),
        discriminator_hidden: Tuple[int, ...] = (256, 256),
        lr: float = 2e-4,
        betas: Tuple[float, float] = (0.5, 0.9),
        reconstruction_weight: float = 1.0,
        kl_weight: float = 0.01,
        adversarial_weight: float = 0.5,
        gumbel_tau: float = 0.2,
        seed: int = 42,
        device: Optional[str] = None,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        self.cfg = {
            "epochs": epochs,
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "encoder_hidden": list(encoder_hidden),
            "decoder_hidden": list(decoder_hidden),
            "discriminator_hidden": list(discriminator_hidden),
            "lr": lr,
            "betas": list(betas),
            "reconstruction_weight": reconstruction_weight,
            "kl_weight": kl_weight,
            "adversarial_weight": adversarial_weight,
            "gumbel_tau": gumbel_tau,
            "seed": seed,
            "device": device,
            "backend": backend,
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
        self.discriminator = None
        self._enc_dim: int = 0
        self._cond_dim: int = 0
        self._device = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["torch", "sklearn"]

    def _build_torch_networks(self, torch, nn) -> None:
        """Build encoder, decoder/generator, and discriminator networks."""
        in_enc = self._enc_dim + self._cond_dim
        latent_dim = self.cfg["latent_dim"]
        in_dec = latent_dim + self._cond_dim
        out_dec = self._enc_dim
        in_disc = self._enc_dim + self._cond_dim

        # Encoder: input -> (mu, logvar)
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

        # Decoder/Generator: latent z + condition -> output
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

        # Discriminator: input + condition -> real/fake score
        class Discriminator(nn.Module):
            def __init__(self, input_dim: int, hidden_dims: List[int]):
                super().__init__()
                layers = []
                last = input_dim
                for h in hidden_dims:
                    layers.append(nn.Linear(last, h))
                    layers.append(nn.LeakyReLU(0.2))
                    last = h
                layers.append(nn.Linear(last, 1))
                self.net = nn.Sequential(*layers)

            def forward(self, x, c):
                inputs = torch.cat([x, c], dim=1)
                return self.net(inputs)

        self.encoder = Encoder(in_enc, self.cfg["encoder_hidden"], latent_dim).to(self._device)
        self.decoder = Decoder(latent_dim, self._cond_dim, self.cfg["decoder_hidden"], out_dec).to(self._device)
        self.discriminator = Discriminator(in_disc, self.cfg["discriminator_hidden"]).to(self._device)

    def _reparameterize(self, mu, logvar, torch):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _gumbelize_cats(self, logits, torch, F) -> Any:
        """Apply Gumbel-Softmax to categorical columns and tanh to continuous."""
        out = []
        ptr = 0
        for name in self.output_order:
            if name in self.cat_blocks:
                start, end = self.cat_blocks[name]
                size = end - start
                block = logits[:, ptr:ptr+size]
                out.append(F.gumbel_softmax(
                    block, tau=self.cfg["gumbel_tau"], hard=False, dim=1))
                ptr += size
            else:
                val = torch.tanh(logits[:, ptr:ptr+1])
                out.append(val)
                ptr += 1
        return torch.cat(out, dim=1)

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "VAEGANModel":
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
                raise ValueError(
                    "y_train.csv must have exactly one column (the target).")
            y_col = y.columns[0]
            df = pd.concat([X, y[y_col]], axis=1)

        self._train_df = df.copy()

        # Schema & encoders
        self.schema = infer_schema(df)
        fit_transformers(df, self.schema)

        backend = (self.cfg.get("backend") or "torch").lower()
        if backend == "torch":
            torch = _try_import("torch")
            nn = _try_import("torch.nn")
            data_utils = _try_import("torch.utils.data")
            F = _try_import("torch.nn.functional")
            if torch is None or nn is None or data_utils is None or F is None:
                self.is_fitted = True
            else:
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
                loader = DataLoader(TensorDataset(enc_t, cond_t), 
                                  batch_size=self.cfg["batch_size"], shuffle=True, drop_last=True)

                # Optimizers
                enc_dec_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
                enc_dec_opt = torch.optim.Adam(enc_dec_params, lr=self.cfg["lr"], betas=tuple(self.cfg["betas"]))
                disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg["lr"], betas=tuple(self.cfg["betas"]))

                print(f"[VAE-GAN] Training with latent_dim={self.cfg['latent_dim']}")
                
                for epoch in range(self.cfg["epochs"]):
                    epoch_recon_loss = 0
                    epoch_kl_loss = 0
                    epoch_adv_loss = 0
                    epoch_d_loss = 0
                    n_batches = 0
                    
                    for real_batch, real_cond in loader:
                        real_batch = real_batch.to(self._device)
                        real_cond = real_cond.to(self._device)
                        batch_size = real_batch.size(0)

                        # ===== Train Encoder + Decoder =====
                        # Encode
                        mu, logvar = self.encoder(real_batch, real_cond)
                        z = self._reparameterize(mu, logvar, torch)
                        
                        # Decode
                        recon_logits = self.decoder(z, real_cond)
                        recon_x = self._gumbelize_cats(recon_logits, torch, F)
                        
                        # VAE losses
                        recon_loss = F.mse_loss(recon_x, real_batch, reduction='sum') / batch_size
                        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
                        
                        # GAN loss for encoder-decoder (fool discriminator)
                        d_fake_recon = self.discriminator(recon_x, real_cond).mean()
                        adv_loss_enc_dec = -d_fake_recon  # Want discriminator to think it's real
                        
                        # Combined loss for encoder-decoder
                        enc_dec_loss = (self.cfg["reconstruction_weight"] * recon_loss + 
                                       self.cfg["kl_weight"] * kl_loss + 
                                       self.cfg["adversarial_weight"] * adv_loss_enc_dec)
                        
                        enc_dec_opt.zero_grad(set_to_none=True)
                        enc_dec_loss.backward()
                        enc_dec_opt.step()

                        # ===== Train Discriminator =====
                        # Sample from prior and generate
                        z_prior = torch.randn(batch_size, self.cfg["latent_dim"], device=self._device)
                        cond_fake_np = sample_conditions(batch_size, self.schema, self.cond_blocks, self.empirical_probs)
                        cond_fake = torch.tensor(cond_fake_np, dtype=torch.float32, device=self._device)
                        
                        with torch.no_grad():
                            fake_logits = self.decoder(z_prior, cond_fake)
                            fake_batch = self._gumbelize_cats(fake_logits, torch, F)
                        
                        d_real = self.discriminator(real_batch, real_cond).mean()
                        d_fake = self.discriminator(fake_batch, cond_fake).mean()
                        
                        # Discriminator loss (maximize d_real, minimize d_fake)
                        d_loss = d_fake - d_real
                        
                        disc_opt.zero_grad(set_to_none=True)
                        d_loss.backward()
                        disc_opt.step()
                        
                        # Track losses
                        epoch_recon_loss += recon_loss.item()
                        epoch_kl_loss += kl_loss.item()
                        epoch_adv_loss += adv_loss_enc_dec.item()
                        epoch_d_loss += d_loss.item()
                        n_batches += 1

                    if (epoch + 1) % 10 == 0:
                        avg_recon = epoch_recon_loss / max(n_batches, 1)
                        avg_kl = epoch_kl_loss / max(n_batches, 1)
                        avg_adv = epoch_adv_loss / max(n_batches, 1)
                        avg_d = epoch_d_loss / max(n_batches, 1)
                        print(f"Epoch {epoch+1}/{self.cfg['epochs']}, "
                              f"Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, "
                              f"Adv: {avg_adv:.4f}, D: {avg_d:.4f}")

                self.is_fitted = True
        else:
            self.is_fitted = True

        # Save outputs
        synth_dir = synthetic_dir
        if not synth_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "vaegan")
        os.makedirs(synth_dir, exist_ok=True)

        df_s = self.sample(n=len(df))
        label = df.columns[-1]
        x_synth = df_s[df.columns[:-1]].copy()
        y_synth = df_s[[label]].copy()

        # Align names with real X
        real_x_train_path = os.path.join(data_dir, "x_train.csv")
        try:
            real_cols = pd.read_csv(real_x_train_path, nrows=0).columns.tolist()
            if len(real_cols) == x_synth.shape[1]:
                x_synth.columns = real_cols
                x_synth = x_synth.reindex(columns=real_cols)
        except Exception:
            pass

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

        print(f"[VAE-GAN] Synthetic data saved:\n  X -> {x_path_out}\n  y -> {y_path_out}")
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

        backend = (self.cfg.get("backend") or "torch").lower()
        if backend == "torch" and self.decoder is not None:
            torch = _try_import("torch")
            F = _try_import("torch.nn.functional")
            if torch is None:
                backend = "numpy"
            else:
                batch = int(n) if n is not None else 1000
                self.decoder.eval()
                all_rows: List[pd.DataFrame] = []
                with torch.no_grad():
                    steps = (batch + 1023) // 1024
                    remain = batch
                    for _ in range(steps):
                        bsz = min(1024, remain)
                        remain -= bsz
                        z = torch.randn(bsz, self.cfg["latent_dim"], device=self._device)

                        if conditional and len(conditional) == 1 and self.cond_blocks:
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

                        recon_logits = self.decoder(z, cond_t)
                        recon_x = self._gumbelize_cats(recon_logits, torch, F).cpu().numpy()
                        df_batch = decode_batch(recon_x, self.schema)
                        all_rows.append(df_batch)

                df_out = pd.concat(all_rows, axis=0).reset_index(drop=True)
                ordered_cols = [c.name for c in self.schema]
                return df_out[ordered_cols]

        # NumPy fallback
        n_rows = int(n) if n is not None else (len(self._train_df) if self._train_df is not None else 1000)
        rows: Dict[str, list] = {c.name: [] for c in self.schema}
        rng = np.random.default_rng(self.cfg.get("seed", 42))
        train_df = self._train_df if self._train_df is not None else None

        for _ in range(n_rows):
            for col in self.schema:
                if col.kind == 'categorical':
                    cats = col.categories or []
                    if train_df is not None and len(cats) > 0:
                        counts = train_df[col.name].astype(str).value_counts().reindex(
                            cats, fill_value=0).values + 1e-8
                        p = counts / counts.sum()
                        choice = rng.choice(cats, p=p)
                        rows[col.name].append(choice)
                    else:
                        choice = rng.choice(cats) if len(cats) else None
                        rows[col.name].append(choice)
                else:
                    qt = col.qt
                    if train_df is not None and qt is not None:
                        vals = train_df[[col.name]].astype(float)
                        norm = qt.transform(vals)
                        mu = float(norm.mean())
                        sigma = float(norm.std() + 1e-6)
                        z = rng.normal(mu, sigma)
                        orig = float(qt.inverse_transform(np.array([[z]])).ravel()[0])
                        rows[col.name].append(orig)
                    else:
                        rows[col.name].append(0.0)

        df_out = pd.DataFrame(rows)
        ordered_cols = [c.name for c in self.schema]
        return df_out[ordered_cols]
