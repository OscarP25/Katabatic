# katabatic/models/forestdiffusion/adapter.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model
from katabatic.models.forestdiffusion.models import ForestDiffusionCore


def _read_csv_keep_cols(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c) for c in df.columns]
    return df


def _resolve_label_name(y_df: pd.DataFrame, label_col: Optional[str | int]) -> str:
    cols = list(map(str, y_df.columns))

    if label_col is None:
        for cand in ("class", "label", "target", "y"):
            if cand in cols:
                return cand
        if len(cols) == 1:
            return cols[0]
        raise ValueError(f"label_col=None but y has multiple columns: {cols}")

    if isinstance(label_col, int):
        if label_col < 0 or label_col >= len(cols):
            raise ValueError(f"label_col index {label_col} out of range for columns={cols}")
        return cols[label_col]

    label_col = str(label_col)
    if label_col in cols:
        return label_col

    if len(cols) == 1:
        return cols[0]

    for cand in ("class", "label", "target", "y"):
        if cand in cols:
            return cand

    raise ValueError(f"Could not resolve label_col={label_col} in columns={cols}")


def _infer_feature_schema(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return categorical_cols, numeric_cols


def _allocate_per_class(
    n: int,
    label_values: np.ndarray,
    label_probs: np.ndarray,
    seed: int = 0,
) -> dict[Any, int]:
    """
    Allocate n samples across classes.

    - Multinomial by label_probs
    - If n >= K, guarantee at least 1 per class
    """
    n = int(n)
    lv = list(label_values)
    K = len(lv)
    rng = np.random.default_rng(seed)

    if K == 0:
        return {}

    if n <= 0:
        return {c: 0 for c in lv}

    counts = rng.multinomial(n, label_probs)

    if n >= K:
        zeros = np.where(counts == 0)[0].tolist()
        if zeros:
            donors = np.argsort(-counts).tolist()
            for z in zeros:
                for d in donors:
                    if counts[d] > 1:
                        counts[d] -= 1
                        counts[z] += 1
                        break

    return {lv[i]: int(counts[i]) for i in range(K)}


def _map_decoded_labels_to_original(y_decoded: np.ndarray, label_values: np.ndarray) -> np.ndarray:
    """
    Map decoded labels (typically strings from categorical decode) back to original dtype.
    Works for shuttle (int labels) and adult (string labels, sometimes with spaces).
    """
    lv = np.asarray(label_values)
    mapping = {str(v): v for v in lv}
    fallback = lv[0]
    y_out = np.array([mapping.get(str(v), fallback) for v in y_decoded], dtype=lv.dtype)
    return y_out


class ForestDiffusionAdapter(Model):
    """
    Katabatic adapter for ForestDiffusion (BEST FIX).

    Fixes:
    - Fits one diffusion model per class (prevents collapse)
    - Forces label to be categorical so it never turns into floats
    - IMPORTANT: train() also generates x_synth/y_synth into synthetic_dir
      (required for your pipeline + TSTR)
    """

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

        # per-class models
        self.cores_: Dict[Any, ForestDiffusionCore] = {}

        # metadata
        self.label_name_: Optional[str] = None
        self.label_values_: Optional[np.ndarray] = None
        self.label_probs_: Optional[np.ndarray] = None
        self.x_columns_: Optional[list[str]] = None

        # schema
        self.feature_categorical_cols_: Optional[list[str]] = None
        self.feature_numeric_cols_: Optional[list[str]] = None

        # seed for allocation/shuffle
        self.seed_: int = int(self.kwargs.get("seed", 0))

    def evaluate(self, *args, **kwargs):
        return {}

    # ------------------------------------------------------------------
    # IMPORTANT: Katabatic pipeline calls train(output_dir, **kwargs)
    # and expects model to write synthetic files to synthetic_dir.
    # ------------------------------------------------------------------
    def train(self, output_dir: str, label_col: Optional[str | int] = "class", **kwargs) -> "ForestDiffusionAdapter":
        out = Path(output_dir)

        x_train_path = out / "x_train.csv"
        y_train_path = out / "y_train.csv"

        if not x_train_path.exists() or not y_train_path.exists():
            raise FileNotFoundError(f"Missing x_train/y_train in {out}")

        X_train = _read_csv_keep_cols(x_train_path)
        y_train_df = _read_csv_keep_cols(y_train_path)

        y_name = _resolve_label_name(y_train_df, label_col)
        y_series = y_train_df[y_name]

        self.x_columns_ = list(X_train.columns)
        self.label_name_ = y_name

        labels = y_series.values
        self.label_values_, counts = np.unique(labels, return_counts=True)
        self.label_probs_ = counts / max(counts.sum(), 1)

        # schema on X
        cat_cols, num_cols = _infer_feature_schema(X_train)
        self.feature_categorical_cols_ = cat_cols
        self.feature_numeric_cols_ = num_cols

        # fit per-class cores
        self.cores_.clear()
        for lv in self.label_values_:
            mask = (labels == lv)
            X_c = X_train.loc[mask].copy()

            df_c = X_c.copy()
            # force label categorical
            df_c["__label__"] = str(lv)

            core = ForestDiffusionCore(
                **self.kwargs,
                categorical_cols=list(self.feature_categorical_cols_) + ["__label__"],
                numeric_cols=list(self.feature_numeric_cols_),
            )
            core.fit(df_c)
            self.cores_[lv] = core

        # after fitting, generate + save synthetic files (pipeline expects this)
        synthetic_dir = kwargs.get("synthetic_dir", str(out))
        self.generate(
            output_dir=str(synthetic_dir),
            n_samples=len(X_train),
            label_col=(str(label_col) if isinstance(label_col, str) else y_name),
        )

        return self

    def sample(self, n_samples: int) -> Tuple[pd.DataFrame, np.ndarray]:
        if (
            self.label_values_ is None
            or self.label_probs_ is None
            or self.x_columns_ is None
            or not self.cores_
        ):
            raise RuntimeError("Model not trained. Call train() before sample().")

        n = int(n_samples)

        per_class = _allocate_per_class(
            n=n,
            label_values=self.label_values_,
            label_probs=self.label_probs_,
            seed=self.seed_,
        )

        X_parts: list[pd.DataFrame] = []
        y_parts: list[np.ndarray] = []

        for lv, k in per_class.items():
            if k <= 0:
                continue

            core = self.cores_.get(lv)
            if core is None:
                continue

            df = core.sample(k)

            if "__label__" in df.columns:
                y_decoded = df["__label__"].values
                X_synth = df.drop(columns=["__label__"]).copy()
            else:
                y_decoded = df.iloc[:, -1].values
                X_synth = df.iloc[:, :-1].copy()

            X_synth = X_synth[self.x_columns_].copy()
            y_labels = _map_decoded_labels_to_original(y_decoded, self.label_values_)

            X_parts.append(X_synth)
            y_parts.append(y_labels)

        if not X_parts:
            # fallback: sample from first class
            lv0 = self.label_values_[0]
            df = self.cores_[lv0].sample(max(n, 1))
            y_decoded = df["__label__"].values if "__label__" in df.columns else df.iloc[:, -1].values
            X_synth = df.drop(columns=["__label__"]).copy() if "__label__" in df.columns else df.iloc[:, :-1].copy()
            X_synth = X_synth[self.x_columns_].copy()
            y_labels = _map_decoded_labels_to_original(y_decoded, self.label_values_)
            X_parts = [X_synth]
            y_parts = [y_labels]

        X_out = pd.concat(X_parts, ignore_index=True)
        y_out = np.concatenate(y_parts, axis=0)

        # shuffle
        rng = np.random.default_rng(self.seed_ + 1)
        idx = rng.permutation(len(X_out))
        X_out = X_out.iloc[idx].reset_index(drop=True)
        y_out = y_out[idx]

        return X_out, y_out

    def generate(
        self,
        output_dir: str,
        n_samples: Optional[int] = None,
        label_col: str = "class",
        **kwargs,
    ) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if n_samples is None:
            raise ValueError("n_samples must be provided in this adapter's generate().")

        X_synth, y_synth = self.sample(int(n_samples))

        X_synth.to_csv(out / "x_synth.csv", index=False)
        pd.DataFrame({label_col: y_synth}).to_csv(out / "y_synth.csv", index=False)
