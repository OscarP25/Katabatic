from __future__ import annotations

import inspect
from typing import Optional

import numpy as np
import pandas as pd


class TVAECore:
    """
    Core SDV TVAE implementation (no Katabatic folder knowledge).
    """

    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 256,
        seed: int = 42,
        enable_gpu: bool = False,
        cuda: bool | None = None,
        enforce_rounding: bool = True,
        force_categorical_metadata: bool = True,
        table_name: str = "data",
        label_column: Optional[str] = None,  # ✅ BEST FIX: force label categorical in metadata
    ):
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        if cuda is not None:
            self.enable_gpu = bool(cuda)
        else:
            self.enable_gpu = bool(enable_gpu)

        self.enforce_rounding = bool(enforce_rounding)
        self.force_categorical_metadata = bool(force_categorical_metadata)
        self.table_name = str(table_name)
        self.label_column = str(label_column) if label_column is not None else None

        self._synth = None
        self._columns: list[str] | None = None
        self._observed: dict[str, np.ndarray] = {}

        np.random.seed(self.seed)

    @staticmethod
    def required_dependency() -> str:
        return "sdv"

    def _ensure_series(self, df: pd.DataFrame, col: str) -> pd.Series:
        obj = df[col]
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:, 0]
        return obj

    def fit(self, df_train: pd.DataFrame) -> "TVAECore":
        from sdv.single_table import TVAESynthesizer
        from .utils.metadata import build_single_table_metadata

        df_train = df_train.copy()
        df_train.columns = [str(c) for c in df_train.columns]

        self._columns = df_train.columns.tolist()

        # observed values (safe if duplicate columns)
        obs = {}
        for c in df_train.columns:
            s = self._ensure_series(df_train, c)
            obs[c] = s.dropna().unique()
        self._observed = obs

        # ✅ Metadata with forced label categorical
        metadata = build_single_table_metadata(
            df_train,
            table_name=self.table_name,
            force_categorical=self.force_categorical_metadata,
            label_column=self.label_column,
        )

        # SDV param compatibility
        kwargs = {
            "metadata": metadata,
            "enforce_rounding": self.enforce_rounding,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }

        sig = inspect.signature(TVAESynthesizer.__init__)
        if "cuda" in sig.parameters:
            kwargs["cuda"] = self.enable_gpu
        elif "enable_gpu" in sig.parameters:
            kwargs["enable_gpu"] = self.enable_gpu

        self._synth = TVAESynthesizer(**kwargs)
        self._synth.fit(df_train)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._synth is None or self._columns is None:
            raise RuntimeError("TVAECore is not fitted. Call fit(df_train) first.")

        df = self._synth.sample(num_rows=int(n))

        # Ensure column presence + order
        for c in self._columns:
            if c not in df.columns:
                obs = self._observed.get(c)
                df[c] = np.random.choice(obs, size=len(df)) if obs is not None and len(obs) else 0

        df = df[self._columns].copy()

        # Clamp to observed values for ALL columns (safe, stabilizes label + categoricals)
        for c in self._columns:
            obs = self._observed.get(c)
            if obs is None or len(obs) == 0:
                continue
            bad = ~df[c].isin(obs)
            if bad.any():
                df.loc[bad, c] = np.random.choice(obs, size=int(bad.sum()))

        return df
