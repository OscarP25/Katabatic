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


class CTGANModel(BaseModel):
    """
    CTGAN-style generator with conditional vectors.

    Fixes included:
    - Condition ONLY on label (better TSTR + avoids invalid label classes).
    - Force output label one-hot block == conditioning label one-hot (prevents contradictions).
    - Force label categories from training data (prevents phantom classes).
    - Clamp y_synth to valid labels before saving (extra safety).
    - Epoch logging every 10 epochs.
    """

    def __init__(
        self,
        *,
        epochs: int = 150,
        batch_size: int = 512,
        noise_dim: int = 128,
        generator_hidden: Tuple[int, ...] = (512, 512),
        discriminator_hidden: Tuple[int, ...] = (512, 512),
        lr: float = 2e-4,
        betas: Tuple[float, float] = (0.5, 0.9),
        lambda_gp: float = 10.0,
        use_gradient_penalty: bool = True,
        clip_value: float = 0.01,
        n_critic: int = 5,
        gumbel_tau: float = 0.5,
        seed: int = 42,
        device: Optional[str] = None,
        backend: str = "torch",
        balanced_label_sampling: bool = False,
        log_every: int = 10,
    ) -> None:
        super().__init__()
        self.cfg = {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "noise_dim": int(noise_dim),
            "generator_hidden": list(generator_hidden),
            "discriminator_hidden": list(discriminator_hidden),
            "lr": float(lr),
            "betas": list(betas),
            "lambda_gp": float(lambda_gp),
            "use_gradient_penalty": bool(use_gradient_penalty),
            "clip_value": float(clip_value),
            "n_critic": int(n_critic),
            "gumbel_tau": float(gumbel_tau),
            "seed": int(seed),
            "device": device,
            "backend": backend,
            "balanced_label_sampling": bool(balanced_label_sampling),
            "log_every": int(log_every),
        }

        # fitted state
        self.schema: List[ColumnMeta] | None = None
        self.output_order: List[str] = []
        self.cat_blocks: Dict[str, Tuple[int, int]] = {}

        # conditioning (label-only)
        self.cond_blocks: Dict[str, Tuple[int, int]] = {}
        self.empirical_probs: Dict[str, np.ndarray] = {}
        self._label_schema: List[ColumnMeta] = []

        self._train_df: Optional[pd.DataFrame] = None
        self._label: Optional[str] = None

        # torch members
        self.generator = None
        self.discriminator = None
        self._enc_dim: int = 0
        self._cond_dim: int = 0
        self._device = None

        # label alignment blocks (encoded output vs conditioning)
        self._label_out_block: Optional[Tuple[int, int]] = None
        self._label_cond_block: Optional[Tuple[int, int]] = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["sklearn", "torch"]

    # ---------------------------------------------------------------------
    # Torch nets
    # ---------------------------------------------------------------------
    def _build_torch_networks(self, torch, nn) -> None:
        in_g = self.cfg["noise_dim"] + self._cond_dim
        out_g = self._enc_dim
        in_d = self._enc_dim + self._cond_dim

        class LocalMLP(nn.Module):
            def __init__(self, in_dim: int, hidden: List[int], out_dim: int):
                super().__init__()
                layers: List[nn.Module] = []
                last = in_dim
                for h in hidden:
                    layers += [nn.Linear(last, h), nn.ReLU()]
                    last = h
                layers.append(nn.Linear(last, out_dim))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        self.generator = LocalMLP(in_g, self.cfg["generator_hidden"], out_g).to(self._device)
        self.discriminator = LocalMLP(in_d, self.cfg["discriminator_hidden"], 1).to(self._device)

    def _apply_output_activations(self, logits, *, hard: bool) -> Any:
        """
        - categorical blocks: gumbel-softmax
        - continuous dims: raw logits (no tanh squash)
        """
        torch = _try_import("torch")
        F = _try_import("torch.nn.functional")
        if torch is None or F is None:
            return logits

        out = []
        ptr = 0
        for name in self.output_order:
            if name in self.cat_blocks:
                start, end = self.cat_blocks[name]
                size = end - start
                block = logits[:, ptr : ptr + size]
                out.append(F.gumbel_softmax(block, tau=self.cfg["gumbel_tau"], hard=hard, dim=1))
                ptr += size
            else:
                out.append(logits[:, ptr : ptr + 1])
                ptr += 1

        return torch.cat(out, dim=1)

    def _force_label_match(self, torch, x_enc, cond):
        """Force output label block to equal the conditioning label block."""
        if self._label_out_block is None or self._label_cond_block is None:
            return x_enc

        out_start, out_end = self._label_out_block
        cond_start, cond_end = self._label_cond_block

        if (out_end - out_start) != (cond_end - cond_start):
            return x_enc

        x_enc[:, out_start:out_end] = cond[:, cond_start:cond_end]
        return x_enc

    def _forward_generator(self, torch, z, cond, *, hard: bool):
        logits = self.generator(torch.cat([z, cond], dim=1))
        x_enc = self._apply_output_activations(logits, hard=hard)
        x_enc = self._force_label_match(torch, x_enc, cond)
        return x_enc

    def _forward_discriminator(self, torch, x_enc, cond):
        return self.discriminator(torch.cat([x_enc, cond], dim=1))

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "CTGANModel":
        train_full = os.path.join(data_dir, "train_full.csv")
        x_path = os.path.join(data_dir, "x_train.csv")
        y_path = os.path.join(data_dir, "y_train.csv")

        if os.path.exists(train_full):
            df = pd.read_csv(train_full)
            label = df.columns[-1]
        else:
            if not (os.path.exists(x_path) and os.path.exists(y_path)):
                raise FileNotFoundError(
                    f"Could not find training data in {data_dir}. Expected train_full.csv or x_train.csv/y_train.csv."
                )
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path)
            if y.shape[1] != 1:
                raise ValueError("y_train.csv must have exactly one column (the target).")
            label = y.columns[0]
            df = pd.concat([X, y[label]], axis=1)

        self._train_df = df.copy()
        self._label = label

        # Schema (label-aware) + fit transformers
        self.schema = infer_schema(df, label_name=label)
        fit_transformers(df, self.schema)

        # Force label categories strictly from training data (prevents phantom labels)
        train_label_cats = sorted(df[label].astype(str).unique().tolist())
        for c in self.schema:
            if c.name == label and c.kind == "categorical":
                c.categories = train_label_cats

        backend = (self.cfg.get("backend") or "torch").lower()

        # -----------------------------------------------------------------
        # Torch backend training
        # -----------------------------------------------------------------
        if backend == "torch":
            torch = _try_import("torch")
            nn = _try_import("torch.nn")
            data_utils = _try_import("torch.utils.data")
            if torch is None or nn is None or data_utils is None:
                backend = "numpy"
            else:
                torch.manual_seed(self.cfg["seed"])
                np.random.seed(self.cfg["seed"])

                enc, self.cat_blocks, self.output_order = encode_df(df, self.schema)

                # Condition ONLY on label
                self._label_schema = [c for c in self.schema if c.name == label and c.kind == "categorical"]
                cond_full, self.cond_blocks = build_conditioning(df, self._label_schema)

                # Save blocks so we can force output label == condition label
                if label in self.cat_blocks and label in self.cond_blocks:
                    self._label_out_block = self.cat_blocks[label]
                    self._label_cond_block = self.cond_blocks[label]
                else:
                    self._label_out_block = None
                    self._label_cond_block = None

                # Label sampling probabilities
                self.empirical_probs = {}
                if self._label_schema and label in self.cond_blocks:
                    start, end = self.cond_blocks[label]
                    counts = cond_full[:, start:end].sum(axis=0) + 1e-8
                    p = (counts / counts.sum()).astype(np.float32)

                    if self.cfg["balanced_label_sampling"]:
                        p = (np.ones_like(p) / len(p)).astype(np.float32)

                    self.empirical_probs[label] = p

                self._enc_dim = int(enc.shape[1])
                self._cond_dim = int(cond_full.shape[1])

                dev = self.cfg.get("device")
                if dev is None:
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                self._device = torch.device(dev)

                self._build_torch_networks(torch, nn)

                TensorDataset = data_utils.TensorDataset
                DataLoader = data_utils.DataLoader

                enc_t = torch.tensor(enc, dtype=torch.float32, device=self._device)
                cond_t = torch.tensor(cond_full, dtype=torch.float32, device=self._device)

                loader = DataLoader(
                    TensorDataset(enc_t, cond_t),
                    batch_size=self.cfg["batch_size"],
                    shuffle=True,
                    drop_last=True,
                )

                G = self.generator
                D = self.discriminator

                g_opt = torch.optim.Adam(G.parameters(), lr=self.cfg["lr"], betas=tuple(self.cfg["betas"]))
                d_opt = torch.optim.Adam(D.parameters(), lr=self.cfg["lr"], betas=tuple(self.cfg["betas"]))

                def _gradient_penalty(D_net, real, fake, cond, device, lambda_gp: float):
                    bs = real.size(0)
                    alpha = torch.rand(bs, 1, device=device).expand_as(real)
                    interp = alpha * real + (1 - alpha) * fake
                    interp_in = torch.cat([interp, cond], dim=1)
                    interp_in.requires_grad_(True)
                    out = D_net(interp_in)
                    grads = torch.autograd.grad(
                        outputs=out,
                        inputs=interp_in,
                        grad_outputs=torch.ones_like(out),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                    grads = grads[:, : real.size(1)]
                    return ((grads.view(bs, -1).norm(2, dim=1) - 1) ** 2).mean() * lambda_gp

                G.train()
                D.train()

                # ----------------------------
                # TRAIN LOOP (WITH LOGGING)
                # ----------------------------
                for _epoch in range(self.cfg["epochs"]):
                    d_loss_epoch = 0.0
                    g_loss_epoch = 0.0
                    n_batches = 0

                    for real_batch, real_cond in loader:
                        n_batches += 1

                        # ===== Train D =====
                        for _ in range(self.cfg["n_critic"]):
                            z = torch.randn(real_batch.size(0), self.cfg["noise_dim"], device=self._device)

                            cond_fake_np = sample_conditions(
                                n=real_batch.size(0),
                                schema=self._label_schema,
                                cond_blocks=self.cond_blocks,
                                empirical_probs=self.empirical_probs,
                                seed=None,
                            )
                            cond_fake = torch.tensor(cond_fake_np, dtype=torch.float32, device=self._device)

                            fake_batch = self._forward_generator(torch, z, cond_fake, hard=False).detach()

                            d_real = self._forward_discriminator(torch, real_batch, real_cond).mean()
                            d_fake = self._forward_discriminator(torch, fake_batch, cond_fake).mean()

                            if self.cfg["use_gradient_penalty"]:
                                gp = _gradient_penalty(D, real_batch, fake_batch, cond_fake, self._device, self.cfg["lambda_gp"])
                                d_loss = d_fake - d_real + gp
                            else:
                                d_loss = d_fake - d_real

                            d_opt.zero_grad(set_to_none=True)
                            d_loss.backward()
                            d_opt.step()

                            if not self.cfg["use_gradient_penalty"]:
                                for p in D.parameters():
                                    p.data.clamp_(-self.cfg["clip_value"], self.cfg["clip_value"])

                        # ===== Train G =====
                        z = torch.randn(real_batch.size(0), self.cfg["noise_dim"], device=self._device)

                        cond_fake_np = sample_conditions(
                            n=real_batch.size(0),
                            schema=self._label_schema,
                            cond_blocks=self.cond_blocks,
                            empirical_probs=self.empirical_probs,
                            seed=None,
                        )
                        cond_fake = torch.tensor(cond_fake_np, dtype=torch.float32, device=self._device)

                        fake_batch = self._forward_generator(torch, z, cond_fake, hard=False)
                        g_loss = -self._forward_discriminator(torch, fake_batch, cond_fake).mean()

                        g_opt.zero_grad(set_to_none=True)
                        g_loss.backward()
                        g_opt.step()

                        d_loss_epoch += float(d_loss.item())
                        g_loss_epoch += float(g_loss.item())

                    # ===== Epoch Logging =====
                    log_every = int(self.cfg.get("log_every", 10))
                    if (_epoch + 1) % log_every == 0 or _epoch == 0:
                        denom = max(n_batches, 1)
                        print(
                            f"[CTGAN] Epoch {_epoch + 1}/{self.cfg['epochs']} | "
                            f"D_loss={d_loss_epoch / denom:.4f} | "
                            f"G_loss={g_loss_epoch / denom:.4f}"
                        )

                self.is_fitted = True

        # -----------------------------------------------------------------
        # NumPy fallback
        # -----------------------------------------------------------------
        if backend == "numpy":
            self.is_fitted = True

        # -----------------------------------------------------------------
        # Save outputs
        # -----------------------------------------------------------------
        synth_dir = synthetic_dir
        if not synth_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "ctgan")
        os.makedirs(synth_dir, exist_ok=True)

        df_s = self.sample(n=len(df))
        df_s = df_s[df.columns.tolist()]

        x_synth = df_s[df.columns[:-1]].copy()
        y_synth = df_s[[label]].copy()

        # Extra safety: clamp synthetic labels to valid labels from y_train.csv
        try:
            y_real = pd.read_csv(y_path)
            valid = sorted(y_real[label].astype(str).unique().tolist())
            rng = np.random.default_rng(self.cfg.get("seed", 42))

            ys = y_synth[label].astype(str)
            bad = ~ys.isin(valid)
            if bad.any():
                ys.loc[bad] = rng.choice(valid, size=int(bad.sum()), replace=True)
            y_synth[label] = ys
        except Exception:
            pass

        # Align X columns exactly to x_train.csv to avoid sklearn mismatch
        try:
            real_cols = pd.read_csv(x_path, nrows=0).columns.tolist()
            if len(real_cols) == x_synth.shape[1]:
                x_synth.columns = real_cols
                x_synth = x_synth.reindex(columns=real_cols)
        except Exception:
            pass

        x_out = os.path.join(synth_dir, "x_synth.csv")
        y_out = os.path.join(synth_dir, "y_synth.csv")
        x_synth.to_csv(x_out, index=False)
        y_synth.to_csv(y_out, index=False, header=True)

        meta = {
            "schema": {
                "columns": df.columns.tolist(),
                "label": label,
                "dtypes": {c: str(df[c].dtype) for c in df.columns},
                "categorical_columns": [c.name for c in (self.schema or []) if c.kind == "categorical"],
            },
            "training": self.cfg,
            "conditioning": {"condition_on": "label_only", "label": label},
        }
        with open(os.path.join(synth_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[CTGAN] Synthetic data saved:\n  X -> {x_out}\n  y -> {y_out}")
        return self

    def evaluate(self, *args, **kwargs) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return 0.0

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------
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

        # Torch sampling
        if backend == "torch" and self.generator is not None:
            torch = _try_import("torch")
            if torch is None:
                backend = "numpy"
            else:
                total = int(n) if n is not None else 1000
                self.generator.eval()

                all_rows: List[pd.DataFrame] = []
                with torch.no_grad():
                    steps = (total + 1023) // 1024
                    remain = total

                    for _ in range(steps):
                        bsz = min(1024, remain)
                        remain -= bsz

                        z = torch.randn(bsz, self.cfg["noise_dim"], device=self._device)

                        # If user gives a label condition, enforce it (e.g., {"class": 1})
                        if conditional and len(conditional) == 1 and self.cond_blocks:
                            name, val = next(iter(conditional.items()))
                            cond_np = np.zeros((bsz, self._cond_dim), dtype=np.float32)

                            if name in self.cond_blocks:
                                start, end = self.cond_blocks[name]
                                cats = next((c.categories for c in self._label_schema if c.name == name), []) or []
                                try:
                                    idx = list(cats).index(str(val))
                                    cond_np[:, start + idx] = 1.0
                                except ValueError:
                                    pass

                            cond_t = torch.tensor(cond_np, dtype=torch.float32, device=self._device)
                        else:
                            cond_np = sample_conditions(
                                n=bsz,
                                schema=self._label_schema,
                                cond_blocks=self.cond_blocks,
                                empirical_probs=self.empirical_probs,
                                seed=None,
                            )
                            cond_t = torch.tensor(cond_np, dtype=torch.float32, device=self._device)

                        # hard=True => clean one-hot categories
                        enc_fake = self._forward_generator(torch, z, cond_t, hard=True).cpu().numpy()
                        df_batch = decode_batch(enc_fake, self.schema)
                        all_rows.append(df_batch)

                df_out = pd.concat(all_rows, axis=0).reset_index(drop=True)
                ordered_cols = [c.name for c in self.schema]
                return df_out[ordered_cols]

        # NumPy fallback
        n_rows = int(n) if n is not None else (len(self._train_df) if self._train_df is not None else 1000)
        rng = np.random.default_rng(self.cfg.get("seed", 42))
        train_df = self._train_df

        rows: Dict[str, list] = {c.name: [] for c in self.schema}

        for _ in range(n_rows):
            for col in self.schema:
                if col.kind == "categorical":
                    cats = list(col.categories or [])
                    if train_df is not None and len(cats) > 0:
                        counts = (
                            train_df[col.name].astype(str)
                            .value_counts()
                            .reindex(cats, fill_value=0)
                            .values
                            + 1e-8
                        )
                        p = counts / counts.sum()
                        rows[col.name].append(rng.choice(cats, p=p))
                    else:
                        rows[col.name].append(rng.choice(cats) if len(cats) else None)
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
