

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler


ArrayLike = Union[np.ndarray, pd.DataFrame]



# Basic helpers


def ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x


def compute_min_distances_cpu(A: ArrayLike, B: ArrayLike, k: int = 1) -> np.ndarray:
    """
    Returns distances from each point in A to its k nearest neighbor(s) in B.
    Output shape:
      - if k == 1: (len(A),)
      - else: (len(A), k)
    """
    A_np = np.asarray(A)
    B_np = np.asarray(B)
    A_np = ensure_2d(A_np)
    B_np = ensure_2d(B_np)

    tree = KDTree(B_np)
    dists, _ = tree.query(A_np, k=k)
    return dists



# Preprocess (simple, safe)


def preprocess_data(
    df: pd.DataFrame,
    normalize: bool = True
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    - Encodes object columns as category codes (starting from 1)
    - Optionally StandardScaler normalisation
    Returns:
      df_processed, mean_, var_
    """
    df_copy = df.copy()

    # ordinal-ish encoding for objects
    for col in df_copy.select_dtypes(include=["object"]).columns:
        df_copy[col] = df_copy[col].astype("category").cat.codes + 1

    df_copy = df_copy.astype(float)

    if not normalize:
        return df_copy, None, None

    scaler = StandardScaler()
    df_copy.loc[:, :] = scaler.fit_transform(df_copy.values)
    return df_copy, scaler.mean_, scaler.var_



# Empirical / Copula encoder 

@dataclass
class EmpiricalTransformer:
    """
    Minimal empirical CDF rank-transform per column.
    - fit(): stores sorted cols + rank matrix
    - convert(u_vectors): inverse empirical for each column
    """
    df: pd.DataFrame
    df_sorted: Optional[pd.DataFrame] = None
    df_ranks: Optional[pd.DataFrame] = None

    def fit(self, method: str = "average") -> pd.DataFrame:
        self.df_sorted = self.df.apply(np.sort, axis=0)
        self.df_ranks = self.df.rank(method=method).astype(int) / self.df.shape[0]
        return self.df_ranks

    @staticmethod
    def inverse_empirical(u: float, sorted_col: np.ndarray) -> float:
        n = len(sorted_col)
        ecdf = np.arange(1, n + 1) / n
        return float(np.interp(u, ecdf, sorted_col))

    def convert(self, u_vectors: np.ndarray) -> pd.DataFrame:
        if self.df_sorted is None:
            raise RuntimeError("EmpiricalTransformer not fitted.")
        u_vectors = ensure_2d(np.asarray(u_vectors))

        transformed = []
        for u_vec in u_vectors:
            row = [
                self.inverse_empirical(float(u), self.df_sorted.iloc[:, i].values)
                for i, u in enumerate(u_vec)
            ]
            transformed.append(row)

        return pd.DataFrame(np.asarray(transformed), columns=self.df.columns)



# Sampling utility for TabKDE style


def sample_points_via_dcp_distribution(
    X: np.ndarray,
    n_samples: int,
    gmm_model,
    noise_std: float = 1.0,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Core geometry sampler:
      pick anchor points from X
      sample distance r from fitted GMM over DCP distances
      sample random unit direction
      add optional gaussian noise
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    X = ensure_2d(X)
    m, d = X.shape

    # anchors
    idx = rng.integers(0, m, size=n_samples)
    anchors = X[idx]

    # directions
    directions = rng.normal(size=(n_samples, d))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    directions = directions / norms

    # distances from GMM
    distances, _ = gmm_model.sample(n_samples)
    distances = np.asarray(distances).reshape(n_samples, 1)

    # noise
    noise = rng.normal(scale=noise_std, size=(n_samples, d))

    return anchors + distances * directions + noise
