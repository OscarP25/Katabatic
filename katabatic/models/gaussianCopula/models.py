# katabatic/models/gaussianCopula/models.py

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_unique_columns(cols: list[str]) -> list[str]:
    """
    SDV + pandas can break if columns are duplicated.
    Ensure uniqueness by appending __1, __2, ...
    """
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out


class GaussianCopulaCore:
    """
    Core SDV GaussianCopula implementation.
    This class does not know about Katabatic folders.
    """

    def __init__(
        self,
        seed: int = 42,
        enforce_rounding: bool = True,
        force_categorical_metadata: bool = True,
    ):
        self.seed = int(seed)
        self.enforce_rounding = bool(enforce_rounding)
        self.force_categorical_metadata = bool(force_categorical_metadata)

        self._synth = None
        self._columns: list[str] | None = None
        self._observed: dict[str, np.ndarray] = {}

        np.random.seed(self.seed)

    @staticmethod
    def required_dependency() -> str:
        return "sdv"

    def fit(self, df_train: pd.DataFrame) -> "GaussianCopulaCore":
        from sdv.single_table import GaussianCopulaSynthesizer
        from .utils.metadata import build_single_table_metadata

        df_train = df_train.copy()

        # Ensure unique columns (fixes adult crash where df_train[c] returned DataFrame)
        df_train.columns = _make_unique_columns(list(df_train.columns))

        self._columns = df_train.columns.tolist()

        # Observed values (for clamping categories after sampling)
        self._observed = {}
        for c in df_train.columns:
            # Always Series now because columns are unique
            self._observed[c] = df_train[c].dropna().unique()

        metadata = build_single_table_metadata(
            df_train, force_categorical=self.force_categorical_metadata
        )

        self._synth = GaussianCopulaSynthesizer(
            metadata=metadata,
            enforce_rounding=self.enforce_rounding,
        )

        self._synth.fit(df_train)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._synth is None or self._columns is None:
            raise RuntimeError("GaussianCopulaCore is not fitted. Call fit(df_train) first.")

        df = self._synth.sample(num_rows=int(n))

        # Ensure column presence + order
        for c in self._columns:
            if c not in df.columns:
                obs = self._observed.get(c)
                df[c] = np.random.choice(obs, size=len(df)) if obs is not None and len(obs) else 0

        df = df[self._columns].copy()

        # Clamp to observed categories if needed (important for categorical/discretized)
        for c in self._columns:
            obs = self._observed.get(c)
            if obs is None or len(obs) == 0:
                continue
            bad = ~df[c].isin(obs)
            if bad.any():
                df.loc[bad, c] = np.random.choice(obs, size=int(bad.sum()))

        return df
