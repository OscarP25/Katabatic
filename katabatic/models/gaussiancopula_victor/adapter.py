# katabatic/models/gaussianCopula/adapter.py

from __future__ import annotations

import os
import pandas as pd

from katabatic.models.base_model import Model
from .models import GaussianCopulaCore


def _resolve_label_name(df: pd.DataFrame, label_col):
    """
    Supports:
      - label_col as column name (str)
      - label_col as index (int)
      - label_col as numeric string like "6"
    """
    if label_col is None:
        raise ValueError("label_col is required (e.g. '6' or target column name).")

    # int index
    if isinstance(label_col, int):
        return df.columns[label_col]

    # str
    if isinstance(label_col, str):
        s = label_col.strip()
        if s.isdigit():
            idx = int(s)
            return df.columns[idx]
        if s in df.columns:
            return s

    raise ValueError(f"Could not resolve label_col={label_col} in columns={list(df.columns)}")


class GaussianCopulaAdapter(Model):
    """
    Katabatic adapter:
    - train(output_dir=..., label_col=...) reads x_train.csv + y_train.csv
    - sample(n) returns (x_synth, y_synth)
    - optional pipeline mode train(dataset_dir, synthetic_dir) writes x_synth/y_synth
    """

    def __init__(
        self,
        target_col: str = "target",
        seed: int = 42,
        max_synth: int | None = 2000,
        enforce_rounding: bool = True,
        force_categorical_metadata: bool = True,
    ):
        super().__init__()
        self.check_dependencies()

        self.target_col = target_col
        self.max_synth = max_synth

        self.core = GaussianCopulaCore(
            seed=seed,
            enforce_rounding=enforce_rounding,
            force_categorical_metadata=force_categorical_metadata,
        )

        self._x_cols: list[str] | None = None
        self._y_name: str | None = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["sdv"]

    def train(self, *args, **kwargs) -> "GaussianCopulaAdapter":
        """
        Supports:
        A) train(output_dir="sample_data/car", label_col="6")
        B) train(dataset_dir="sample_data/car", synthetic_dir="synthetic/..", label_col="6")
        """
        output_dir = kwargs.get("output_dir") or kwargs.get("dataset_dir") or (args[0] if args else None)
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

        # Decide output y name
        if label_col is None:
            y_name = y_train_df.columns[0]
        else:
            tmp = pd.concat([X_train, y_train_df], axis=1)
            y_name = _resolve_label_name(tmp, label_col)

        y_train_df.columns = [y_name]

        self._x_cols = X_train.columns.tolist()
        self._y_name = y_name

        df_train = pd.concat([X_train, y_train_df], axis=1)

        self.core.fit(df_train)
        self.is_fitted = True

        # If pipeline provided synthetic_dir, generate + write immediately
        if synthetic_dir is not None:
            n = len(df_train)
            if self.max_synth is not None:
                n = min(n, int(self.max_synth))
            x_s, y_s = self.sample(n)

            os.makedirs(synthetic_dir, exist_ok=True)
            pd.DataFrame(x_s, columns=self._x_cols).to_csv(
                os.path.join(synthetic_dir, "x_synth.csv"), index=False
            )
            pd.DataFrame({self._y_name: y_s}).to_csv(
                os.path.join(synthetic_dir, "y_synth.csv"), index=False
            )

        return self

    def sample(self, n: int, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("GaussianCopulaAdapter not trained. Call train(...) first.")
        if self._x_cols is None or self._y_name is None:
            raise RuntimeError("Internal columns not initialized. Train again.")

        df_synth = self.core.sample(int(n))

        # NOTE: core may have renamed duplicated columns internally (with __1, __2).
        # For X/y split we rely on the original x_train/y_train headers.
        # So we re-align: take first matching columns by position.
        if len(df_synth.columns) == (len(self._x_cols) + 1):
            # positional split safe
            X_synth = df_synth.iloc[:, : len(self._x_cols)].to_numpy()
            y_synth = df_synth.iloc[:, len(self._x_cols)].to_numpy()
        else:
            # fallback by name (if names still match)
            X_synth = df_synth[self._x_cols].to_numpy()
            y_synth = df_synth[self._y_name].to_numpy()

        return X_synth, y_synth

    def evaluate(self, *args, **kwargs) -> float:
        return float("nan")
