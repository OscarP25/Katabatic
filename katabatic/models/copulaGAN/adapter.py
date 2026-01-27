from __future__ import annotations

import os
import numpy as np
import pandas as pd

from katabatic.models.base_model import Model
from .models import CopulaGANCore


def _resolve_label_name(df: pd.DataFrame, label_col):
    """
    Supports:
      - label_col as column name (str)
      - label_col as index (int)
      - label_col as numeric string like "6" (matches your usage)
    """
    if label_col is None:
        raise ValueError("label_col is required (e.g. '6' or target column name).")

    # int index
    if isinstance(label_col, int):
        return df.columns[label_col]

    # str
    if isinstance(label_col, str):
        s = label_col.strip()
        # numeric string index
        if s.isdigit():
            idx = int(s)
            return df.columns[idx]
        # else column name
        if s in df.columns:
            return s

    raise ValueError(f"Could not resolve label_col={label_col} in columns={list(df.columns)}")


class CopulaGANAdapter(Model):
    """
    Katabatic adapter:
    - train(output_dir=..., label_col=...) reads x_train.csv + y_train.csv
    - sample(n) returns (x_synth, y_synth)
    - optional pipeline mode train(dataset_dir, synthetic_dir) writes x_synth/y_synth
    """

    def __init__(
        self,
        target_col: str = "target",
        epochs: int = 50,
        batch_size: int = 256,
        seed: int = 42,
        enable_gpu: bool = False,     # ✅ renamed
        pac: int = 1,                 # ✅ fixes CTGAN pac assertion
        max_synth: int | None = 2000,
        verbose: bool = False,
    ):
        super().__init__()
        self.check_dependencies()

        self.target_col = target_col
        self.max_synth = max_synth

        # ✅ IMPORTANT: pac=1 avoids "batch % pac == 0" assert inside CTGAN
        self.core = CopulaGANCore(
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            enable_gpu=enable_gpu,     # ✅ sdv new arg
            pac=pac,                   # ✅ fix
            enforce_rounding=True,
            force_categorical_metadata=True,
            verbose=verbose,
        )

        self._x_cols: list[str] | None = None
        self._y_name: str | None = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["sdv"]

    def train(self, *args, **kwargs) -> "CopulaGANAdapter":
        """
        Supports:
        A) train(output_dir="sample_data/car", label_col="6")
        B) train(dataset_dir="sample_data/car", synthetic_dir="synthetic/..", label_col="6")
        """
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

        # y_train.csv might have header "6" or "target"; we allow label_col to choose the output name
        if y_train_df.shape[1] != 1:
            y_train_df = y_train_df.iloc[:, [0]]

        # Decide label name for saving
        if label_col is None:
            y_name = y_train_df.columns[0]
        else:
            tmp = pd.concat([X_train, y_train_df], axis=1)
            y_name = _resolve_label_name(tmp, label_col)

        # Rename y to y_name (important for downstream expectations)
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
            raise RuntimeError("CopulaGANAdapter not trained. Call train(...) first.")
        if self._x_cols is None or self._y_name is None:
            raise RuntimeError("Internal columns not initialized. Train again.")

        df_synth = self.core.sample(int(n))

        X_synth = df_synth[self._x_cols].to_numpy()
        y_synth = df_synth[self._y_name].to_numpy()

        return X_synth, y_synth

    def evaluate(self, *args, **kwargs) -> float:
        # Katabatic uses TSTR evaluation in pipeline; keep minimal.
        return float("nan")
