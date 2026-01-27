# katabatic/models/copulaGAN/models.py
from __future__ import annotations

import numpy as np
import pandas as pd


class CopulaGANCore:
    """
    Core SDV CopulaGAN implementation.

    Practical fixes:
    - CTGAN pac assertion: default pac=1
    - SDV API compatibility: enable_gpu vs cuda
    - High-cardinality categorical columns: use update_transformers(LabelEncoder)
    - Optional cap on training rows to avoid CPU RAM blowups (Adult dataset)
    """

    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 128,
        seed: int = 42,
        cuda: bool = False,
        enable_gpu: bool | None = None,
        enforce_rounding: bool = True,
        force_categorical_metadata: bool = True,
        pac: int = 1,
        # ✅ memory safety knobs
        max_train_rows: int | None = 15000,     # cap Adult; set None to disable
        high_card_threshold: int = 50,          # treat categorical with >50 uniques as "high card"
        **kwargs,
    ):
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        self.cuda = bool(enable_gpu) if enable_gpu is not None else bool(cuda)
        self.enforce_rounding = bool(enforce_rounding)
        self.force_categorical_metadata = bool(force_categorical_metadata)
        self.pac = int(pac)

        self.max_train_rows = max_train_rows
        self.high_card_threshold = int(high_card_threshold)

        self._synth = None
        self._columns: list[str] | None = None
        self._observed: dict[str, np.ndarray] = {}

        np.random.seed(self.seed)

    @staticmethod
    def required_dependency() -> str:
        return "sdv"

    @staticmethod
    def _ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
        cols = list(df.columns)
        seen: dict[str, int] = {}
        new_cols: list[str] = []
        changed = False

        for c in cols:
            if c not in seen:
                seen[c] = 0
                new_cols.append(c)
            else:
                seen[c] += 1
                new_cols.append(f"{c}__{seen[c]}")
                changed = True

        if changed:
            df = df.copy()
            df.columns = new_cols
        return df

    @staticmethod
    def _unique_nonnull_values(obj: pd.Series | pd.DataFrame) -> np.ndarray:
        if isinstance(obj, pd.DataFrame):
            values = obj.to_numpy().ravel()
            values = values[~pd.isna(values)]
            return pd.unique(values)

        s = obj.dropna()
        return pd.unique(s.to_numpy())

    @staticmethod
    def _is_categorical_series(s: pd.Series) -> bool:
        # Discretized datasets are often object dtype or category dtype
        dt = str(s.dtype).lower()
        return ("object" in dt) or ("category" in dt) or ("string" in dt)

    def fit(self, df_train: pd.DataFrame) -> "CopulaGANCore":
        from sdv.single_table import CopulaGANSynthesizer
        from .utils.metadata import build_single_table_metadata

        df_train = self._ensure_unique_columns(df_train)

        # ✅ optional row cap for memory safety
        if self.max_train_rows is not None and len(df_train) > int(self.max_train_rows):
            df_train = df_train.sample(int(self.max_train_rows), random_state=self.seed).reset_index(drop=True)

        self._columns = df_train.columns.tolist()
        self._observed = {c: self._unique_nonnull_values(df_train[c]) for c in self._columns}

        metadata = build_single_table_metadata(
            df_train, force_categorical=self.force_categorical_metadata
        )

        # ✅ batch size safety
        bs = min(int(self.batch_size), max(1, len(df_train)))
        # keep bs reasonably small on CPU to avoid huge allocations
        bs = min(bs, 256)

        # ✅ pac safety
        pac = max(1, int(self.pac))

        # Build synthesizer with SDV API compatibility
        def _make_synth(**extra):
            try:
                return CopulaGANSynthesizer(
                    metadata=metadata,
                    enforce_rounding=self.enforce_rounding,
                    epochs=self.epochs,
                    batch_size=bs,
                    enable_gpu=self.cuda,
                    **extra,
                )
            except TypeError:
                return CopulaGANSynthesizer(
                    metadata=metadata,
                    enforce_rounding=self.enforce_rounding,
                    epochs=self.epochs,
                    batch_size=bs,
                    cuda=self.cuda,
                    **extra,
                )

        # Some stacks accept pac; some don't
        try:
            self._synth = _make_synth(pac=pac)
        except TypeError:
            self._synth = _make_synth()

        # ✅ HIGH-CARD categorical fix: update_transformers(LabelEncoder)
        # This avoids massive one-hot-like expansions that cause RAM OOM.
        try:
            from rdt.transformers import LabelEncoder
            updates = {}
            for col in df_train.columns:
                s = df_train[col]
                if self._is_categorical_series(s):
                    nunique = int(pd.Series(s).nunique(dropna=True))
                    if nunique > self.high_card_threshold:
                        updates[col] = LabelEncoder()

            if updates:
                self._synth.update_transformers(updates)
        except Exception:
            # If rdt isn't available or API differs, we continue without crashing.
            pass

        # Fit (with a fallback for pac assertion)
        try:
            self._synth.fit(df_train)
        except AssertionError:
            # fallback: internal pac=10 often -> ensure bs multiple of 10
            pac_internal = 10
            bs2 = max(pac_internal, (bs // pac_internal) * pac_internal)
            if bs2 != bs:
                bs = bs2
                try:
                    self._synth = _make_synth(pac=1)
                except TypeError:
                    self._synth = _make_synth()
                self._synth.fit(df_train)
            else:
                raise

        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._synth is None or self._columns is None:
            raise RuntimeError("CopulaGANCore is not fitted. Call fit(df_train) first.")

        df = self._synth.sample(num_rows=int(n))

        # Ensure column presence + order
        for c in self._columns:
            if c not in df.columns:
                obs = self._observed.get(c)
                df[c] = np.random.choice(obs, size=len(df)) if obs is not None and len(obs) else 0
        df = df[self._columns].copy()

        # Clamp to observed values
        for c in self._columns:
            obs = self._observed.get(c)
            if obs is None or len(obs) == 0:
                continue
            bad = ~df[c].isin(obs)
            if bad.any():
                df.loc[bad, c] = np.random.choice(obs, size=int(bad.sum()))

        return df
