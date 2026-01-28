from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import os

from katabatic.models.base_model import Model  # <- so it works with the pipeline


class PATEGAN(Model):
    """
    Simplified, assignment-friendly PATE-GAN-style model.

    - Matches the public API from the Katabatic spec (epsilon, delta, etc.)
    - Provides both:
        * fit(X, y) / sample(n) for standalone use
        * train(dataset_dir, synthetic_dir=...) for pipeline use
    - Internally uses a differentially private label distribution
      + class-conditional feature distributions (no heavy TF / WGAN).
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        num_teachers: int = 10,
        niter: int = 10000,
        batch_size: int = 128,
        random_state: int = 42,
        synthetic_multiplier: float = 1.0,
        label_column: str = "label",
        *args,
        **kwargs,
    ):
        # Spec parameters (some are stored but not all are used in this simplified version)
        self.epsilon = epsilon
        self.delta = delta
        self.num_teachers = num_teachers
        self.niter = niter
        self.batch_size = batch_size
        self.random_state = random_state

        # Our extra knobs
        self.synthetic_multiplier = synthetic_multiplier
        self.label_column = label_column

        # Learned state
        self.feature_columns: Optional[List[str]] = None
        self.label_values_: Optional[np.ndarray] = None
        self.label_probs_: Optional[np.ndarray] = None
        # label -> feature_index -> (values, probs)
        self.class_feature_probs_: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}

        # For reproducibility
        if random_state is not None:
            np.random.seed(random_state)

    # ---------- Standalone API (spec) ----------

    def fit(self, X: pd.DataFrame, y: pd.Series, *args, **kwargs):
        """Fit on in-memory data (standalone usage)."""
        self.feature_columns = list(X.columns)
        X_np = X.values
        y_np = y.values
        self._fit_label_distribution(y_np)
        self._fit_feature_distributions(X_np, y_np)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        """Generate n synthetic rows (standalone usage)."""
        X_synth, y_synth = self._sample_synthetic(n)
        df = pd.DataFrame(X_synth, columns=self.feature_columns)
        df[self.label_column] = y_synth
        return df

    # ---------- Pipeline API (Katabatic.Model) ----------

    def train(self, dataset_dir: str, synthetic_dir: Optional[str] = None, *args, **kwargs):
        """
        Train the model from files in dataset_dir and
        write x_synth.csv / y_synth.csv into synthetic_dir.
        """
        dataset_dir = Path(dataset_dir)
        synthetic_dir = Path(synthetic_dir) if synthetic_dir is not None else dataset_dir

        # Prefer explicit X/y split if present
        x_path = dataset_dir / "x_train.csv"
        y_path = dataset_dir / "y_train.csv"

        if x_path.exists() and y_path.exists():
            X_train = pd.read_csv(x_path)
            y_df = pd.read_csv(y_path)
            label_col = y_df.columns[0]
            y_train = y_df[label_col]
        else:
            # Fallback to train_full.csv
            train_full_path = dataset_dir / "train_full.csv"
            train_full = pd.read_csv(train_full_path)
            if "label" in train_full.columns:
                label_col = "label"
            else:
                label_col = train_full.columns[-1]
            X_train = train_full.drop(columns=[label_col])
            y_train = train_full[label_col]

        # Fit in-memory
        self.fit(X_train, y_train)

        # How many synthetic samples to generate
        n_real = X_train.shape[0]
        n_synth = int(n_real * self.synthetic_multiplier)
        X_synth, y_synth = self._sample_synthetic(n_synth)

        # Save to synthetic_dir
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        x_out = synthetic_dir / "x_synth.csv"
        y_out = synthetic_dir / "y_synth.csv"

        pd.DataFrame(X_synth, columns=self.feature_columns).to_csv(x_out, index=False)
        pd.DataFrame({self.label_column: y_synth}).to_csv(y_out, index=False)

        print(f"Synthetic X saved to: {x_out}")
        print(f"Synthetic y saved to: {y_out}")

    # ---------- Internal helpers (same as you already had, just fixed indenting) ----------

    def _fit_label_distribution(self, y: np.ndarray):
        unique, counts = np.unique(y, return_counts=True)
        counts = counts.astype(float)

        # Laplace noise (PATE-style idea)
        sensitivity = 1.0
        scale = sensitivity / max(self.epsilon, 1e-6)
        noisy_counts = counts + np.random.laplace(
            loc=0.0,
            scale=scale,
            size=counts.shape,
        )
        noisy_counts = np.clip(noisy_counts, a_min=1e-6, a_max=None)

        probs = noisy_counts / noisy_counts.sum()
        self.label_values_ = unique
        self.label_probs_ = probs

    def _fit_feature_distributions(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.class_feature_probs_.clear()

        for label in np.unique(y):
            mask = (y == label)
            X_label = X[mask]
            label_int = int(label)
            self.class_feature_probs_[label_int] = {}

            for j in range(n_features):
                col = X_label[:, j]
                values, counts = np.unique(col, return_counts=True)
                probs = counts.astype(float) / counts.sum()
                self.class_feature_probs_[label_int][j] = (values, probs)

    def _sample_synthetic(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.label_values_ is not None
        assert self.label_probs_ is not None
        assert self.feature_columns is not None

        n_features = len(self.feature_columns)
        X_synth = np.zeros((n_samples, n_features), dtype=int)
        y_synth = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            # sample label
            label = np.random.choice(self.label_values_, p=self.label_probs_)
            label_int = int(label)
            y_synth[i] = label_int

            # sample features given label
            feature_dist = self.class_feature_probs_[label_int]
            for j in range(n_features):
                values, probs = feature_dist[j]
                X_synth[i, j] = np.random.choice(values, p=probs)

        return X_synth, y_synth

    def evaluate(self, *args, **kwargs):
        """
        Dummy evaluate method to satisfy the abstract base class interface.

        The real TSTR evaluation is done by katabatic.evaluate.tstr.TSTREvaluation,
        so this method is not used in our current pipeline.
        """
        print("PATEGAN.evaluate() was called, but evaluation is handled by TSTREvaluation.")
        return None
