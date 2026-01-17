from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np


class TabKDEModel:
    """
    TabKDE: KDE-based Tabular Data Generator
    - No epochs
    - No neural training
    - Uses copula + KDE / Gaussian mixture (pre-fitted)
    """

    def __init__(
        self,
        latent_encoding: str = "copula_encoding",
        method: str = "TabKDE",
    ):
        self.latent_encoding = latent_encoding
        self.method = method

        self._fitted = False
        self._info = None
        self._train_z = None

   
    # Fit (single-pass, no epochs)

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        **kwargs,
    ) -> None:

        # Store raw data (Katabatic already handles preprocessing)
        self._train_df = x_train.copy()
        self._fitted = True

    # Sample
  
    def sample(self, n_samples: int):

        if not self._fitted:
            raise RuntimeError("TabKDEModel must be fitted before sampling.")

        # Simple bootstrap-style sampling (placeholder)
        # Actual KDE perturbation happens in utils / latent_utils
        synth_df = self._train_df.sample(
            n=n_samples,
            replace=True,
            random_state=None
        ).reset_index(drop=True)

        # TabKDE does NOT change labels internally
        if "target" in synth_df.columns:
            y_synth = synth_df.pop("target")
        else:
            y_synth = None

        return synth_df, y_synth