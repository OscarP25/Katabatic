from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.random import Generator, PCG64
from sklearn.mixture import GaussianMixture


@dataclass
class BaseSynthesizer:
    """
    Abstract base synthesizer.
    Subclasses should implement fit() and sample(n).

    Notes
    -----
    - Expect preprocessed, purely numeric OR purely categorical inputs
      (see utils.encode_mixed_frame to get there).
    - Keep RNG local and reproducible.
    """
    random_state: Optional[int] = 42

    def _rng(self) -> Generator:
        return np.random.default_rng(PCG64(self.random_state if self.random_state is not None else 42))

    def fit(self, X: pd.DataFrame) -> "BaseSynthesizer":
        raise NotImplementedError

    def sample(self, n: int) -> pd.DataFrame:
        raise NotImplementedError


@dataclass
class GaussianSynthesizer(BaseSynthesizer):
    """
    Simple numeric-only synthesizer using a diagonal-covariance Gaussian Mixture.

    Parameters
    ----------
    n_components : int
        Number of mixture components.
    max_iter : int
        EM iterations for GMM.
    """
    n_components: int = 8
    max_iter: int = 200
    _gmm: Optional[GaussianMixture] = None
    _columns: Optional[pd.Index] = None

    def fit(self, X: pd.DataFrame) -> "GaussianSynthesizer":
        if X.select_dtypes(exclude=[np.number]).shape[1] > 0:
            raise ValueError("GaussianSynthesizer expects numeric-only data.")
        self._columns = X.columns
        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="diag",
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self._gmm.fit(X.values)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._gmm is None or self._columns is None:
            raise RuntimeError("Call fit() before sample().")
        X, _ = self._gmm.sample(n)
        return pd.DataFrame(X, columns=self._columns)


@dataclass
class FrequencySynthesizer(BaseSynthesizer):
    """
    Categorical-only synthesizer that samples per-column
    from empirical frequency distributions, then concatenates.

    Notes
    -----
    - This assumes columns are independent (fast baseline).
    - Use utils.encode_mixed_frame to label-encode categoricals if needed,
      or pass already-categorical columns.
    """
    _probs: Optional[dict] = None
    _values: Optional[dict] = None
    _columns: Optional[pd.Index] = None
    _is_categorical: Optional[dict] = None

    def fit(self, X: pd.DataFrame) -> "FrequencySynthesizer":
        non_numeric_ok = True  # accepts any dtype, but treats each column as categories
        self._columns = X.columns
        self._probs = {}
        self._values = {}
        self._is_categorical = {}

        for c in X.columns:
            counts = X[c].astype("category").value_counts(dropna=False)
            vals = counts.index.to_numpy()
            probs = (counts / counts.sum()).to_numpy()
            self._values[c] = vals
            self._probs[c] = probs
            self._is_categorical[c] = True if non_numeric_ok else False
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._columns is None or self._probs is None or self._values is None:
            raise RuntimeError("Call fit() before sample().")

        rng = self._rng()
        out = {}
        for c in self._columns:
            vals = self._values[c]
            probs = self._probs[c]
            out[c] = rng.choice(vals, size=n, replace=True, p=probs)
        return pd.DataFrame(out, columns=self._columns)
