from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ARF engine (pip install arfpy)
from arfpy.arf import arf as ARFEngine

from .utils import load_train_df, split_x_y, ensure_dir, try_align_columns


@dataclass
class ARFModel:
    """
    Katabatic wrapper for Adversarial Random Forests (arfpy).

    Reads:
      - train_full.csv (preferred) OR x_train.csv + y_train.csv
    Writes:
      - x_synth.csv, y_synth.csv into synthetic_dir

    Notes:
      - We generate synthetic X using ARFEngine.forge().
      - y is sampled from the empirical label distribution in train data.
    """
    num_trees: int = 30
    max_iters: int = 10
    delta: float = 0.0
    min_node_size: int = 5
    verbose: bool = True
    seed: int = 42

    def train(self, data_dir: str, synthetic_dir: Optional[str] = None, n_synth: Optional[int] = None, **kwargs):
        
        # Load data
        df = load_train_df(data_dir)
        X, y, label_col = split_x_y(df)

        # Safety: ensure y is 1D
        y = pd.Series(y).reset_index(drop=True)

        # Fit ARF on features only
        np.random.seed(self.seed)

        # arfpy expects a DataFrame; it internally handles categorical/object columns.
        arf = ARFEngine(
            x=X,
            num_trees=self.num_trees,
            delta=self.delta,
            max_iters=self.max_iters,
            early_stop=True,
            verbose=self.verbose,
            min_node_size=self.min_node_size
        )

        # density estimation step required before forge()
        arf.forde()

        # Generate synthetic
        if n_synth is None:
            n_synth = len(X)

        X_synth = arf.forge(n=n_synth)
        X_synth = try_align_columns(data_dir, X_synth)

        # Sample y from empirical distribution (simple + consistent)
        y_synth = y.sample(n=n_synth, replace=True, random_state=self.seed).reset_index(drop=True)

        
        # Save
        if synthetic_dir is not None:
            ensure_dir(synthetic_dir)
            X_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            pd.Series(y_synth, name="label").to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)

            # simple metadata
            meta_path = os.path.join(synthetic_dir, "metadata.json")
            try:
                import json
                meta = {
                    "model": "arf",
                    "num_trees": self.num_trees,
                    "max_iters": self.max_iters,
                    "delta": self.delta,
                    "min_node_size": self.min_node_size,
                    "seed": self.seed,
                    "n_synth": int(n_synth),
                    "label_col_original": label_col
                }
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass

        # Keep refs if you want
        self._arf = arf
        return X_synth, y_synth
