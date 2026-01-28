from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model
from .models import TVAECore


def _resolve_label_name(df: pd.DataFrame, label_col):
    if label_col is None:
        raise ValueError("label_col is required (e.g. '6' or target column name).")

    if isinstance(label_col, int):
        return df.columns[label_col]

    if isinstance(label_col, str):
        s = label_col.strip()
        if s.isdigit():
            idx = int(s)
            if idx < 0 or idx >= len(df.columns):
                raise ValueError(f"label_col index {idx} out of range for columns={list(df.columns)}")
            return df.columns[idx]
        if s in df.columns:
            return s

    raise ValueError(f"Could not resolve label_col={label_col} in columns={list(df.columns)}")


class TVAEAdapter(Model):
    """
    Best fix: class-conditional TVAE (one TVAE per class).
    Guarantees multi-class y_synth, preventing TSTR special cases.
    """

    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 256,
        seed: int = 42,
        enable_gpu: bool = False,
        max_synth: int | None = 2000,
        conditional: bool = True,
        min_rows_per_class: int = 50,
    ):
        super().__init__()  # ✅ IMPORTANT for your base Model
        self.check_dependencies()

        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.enable_gpu = bool(enable_gpu)

        self.max_synth = max_synth
        self.conditional = bool(conditional)
        self.min_rows_per_class = int(min_rows_per_class)

        self._x_cols: list[str] | None = None
        self._y_name: str | None = None

        self._label_values: np.ndarray | None = None
        self._label_probs: np.ndarray | None = None

        self._models_by_label: dict[Any, TVAECore] = {}
        self._fallback_rows_by_label: dict[Any, pd.DataFrame] = {}

        self._rng = np.random.default_rng(self.seed)

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["sdv"]

    def _new_core(self, label_column: str | None) -> TVAECore:
        return TVAECore(
            epochs=self.epochs,
            batch_size=self.batch_size,
            seed=self.seed,
            enable_gpu=self.enable_gpu,
            enforce_rounding=True,
            force_categorical_metadata=True,
            table_name="data",
            label_column=label_column,
        )

    def train(self, *args, **kwargs) -> "TVAEAdapter":
        output_dir = (
            kwargs.get("output_dir")
            or kwargs.get("dataset_dir")
            or (args[0] if args else None)
        )
        if output_dir is None:
            raise ValueError("train() requires output_dir=... or dataset_dir=...")

        label_col = kwargs.get("label_col")
        synthetic_dir = kwargs.get("synthetic_dir")

        x_train_path = os.path.join(output_dir, "x_train.csv")
        y_train_path = os.path.join(output_dir, "y_train.csv")

        if not os.path.exists(x_train_path):
            raise FileNotFoundError(f"Missing: {x_train_path}")
        if not os.path.exists(y_train_path):
            raise FileNotFoundError(f"Missing: {y_train_path}")

        X_train = pd.read_csv(x_train_path)
        y_train_df = pd.read_csv(y_train_path)

        if y_train_df.shape[1] != 1:
            y_train_df = y_train_df.iloc[:, [0]]

        # Resolve y column name robustly
        if label_col is None:
            y_name = str(y_train_df.columns[0])
        else:
            tmp = pd.concat([X_train, y_train_df], axis=1)
            y_name = str(_resolve_label_name(tmp, label_col))

        y_train_df.columns = [y_name]

        self._x_cols = [str(c) for c in X_train.columns.tolist()]
        self._y_name = y_name

        # Keep labels as strings (stable categories)
        y_series = y_train_df[y_name].astype(str)

        label_counts = y_series.value_counts(dropna=False)
        label_values = label_counts.index.to_numpy(dtype=object)
        probs = (label_counts / label_counts.sum()).to_numpy(dtype=float)

        self._label_values = label_values
        self._label_probs = probs

        df_train = X_train.copy()
        df_train.columns = self._x_cols
        df_train[y_name] = y_series.values

        # reset
        self._models_by_label = {}
        self._fallback_rows_by_label = {}

        if (not self.conditional) or len(label_values) <= 1:
            core = self._new_core(label_column=y_name)
            core.fit(df_train)
            self._models_by_label["__all__"] = core
        else:
            for lab in label_values:
                lab = str(lab)
                sub = df_train[df_train[y_name] == lab].copy()

                if len(sub) < self.min_rows_per_class:
                    self._fallback_rows_by_label[lab] = sub
                    continue

                core = self._new_core(label_column=y_name)
                core.fit(sub)
                self._models_by_label[lab] = core

        self.is_fitted = True

        if synthetic_dir is not None:
            n = len(df_train)
            if self.max_synth is not None:
                n = min(n, int(self.max_synth))

            X_s, y_s = self.sample(int(n))

            os.makedirs(synthetic_dir, exist_ok=True)
            pd.DataFrame(X_s, columns=self._x_cols).to_csv(
                os.path.join(synthetic_dir, "x_synth.csv"), index=False
            )
            pd.DataFrame({self._y_name: y_s}).to_csv(
                os.path.join(synthetic_dir, "y_synth.csv"), index=False
            )

        return self

    def _allocate_counts(self, n: int) -> dict[str, int]:
        assert self._label_values is not None and self._label_probs is not None
        labels = [str(x) for x in self._label_values.tolist()]
        probs = self._label_probs

        k = len(labels)
        n = int(n)
        if k == 0:
            return {}
        if k == 1:
            return {labels[0]: n}

        counts = self._rng.multinomial(n, probs).astype(int)
        out = {labels[i]: int(counts[i]) for i in range(k)}

        # guarantee 1 per class if possible
        if n >= k:
            for lab in labels:
                if out[lab] == 0:
                    donor = max(out, key=lambda x: out[x])
                    if out[donor] > 1:
                        out[donor] -= 1
                        out[lab] += 1

        diff = n - sum(out.values())
        if diff != 0:
            donor = max(out, key=lambda x: out[x])
            out[donor] += diff

        return out

    def sample(self, n: int, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("TVAEAdapter not trained. Call train(...) first.")
        if self._x_cols is None or self._y_name is None:
            raise RuntimeError("Internal columns not initialized. Train again.")

        n = int(n)

        if "__all__" in self._models_by_label:
            df_synth = self._models_by_label["__all__"].sample(n)
            X_synth = df_synth[self._x_cols].to_numpy()
            y_synth = df_synth[self._y_name].astype(str).to_numpy()
            return X_synth, y_synth

        if self._label_values is None or self._label_probs is None:
            raise RuntimeError("Label distribution missing. Train again.")

        counts = self._allocate_counts(n)
        frames: list[pd.DataFrame] = []

        for lab, k in counts.items():
            if k <= 0:
                continue

            core = self._models_by_label.get(lab)
            if core is not None:
                df_k = core.sample(k)
                df_k[self._y_name] = lab
                frames.append(df_k)
                continue

            fb = self._fallback_rows_by_label.get(lab)
            if fb is None or len(fb) == 0:
                df_k = pd.DataFrame({c: [0] * k for c in self._x_cols})
                df_k[self._y_name] = lab
                frames.append(df_k)
            else:
                idx = self._rng.integers(0, len(fb), size=k)
                df_k = fb.iloc[idx].copy()
                df_k[self._y_name] = lab
                frames.append(df_k)

        df_synth = pd.concat(frames, ignore_index=True)
        df_synth = df_synth.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

        X_synth = df_synth[self._x_cols].to_numpy()
        y_synth = df_synth[self._y_name].astype(str).to_numpy()
        return X_synth, y_synth

    def evaluate(self, *args, **kwargs) -> float:
        return float("nan")
