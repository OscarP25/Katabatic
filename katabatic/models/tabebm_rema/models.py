from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .utils import infer_schema, fit_schema_stats, encode_df, decode_df


# ---------------------------------------------------------------------
# Always import TabEBM from our LOCAL patched file (tabebm.py)
# ---------------------------------------------------------------------
def _load_tabebm_class():
    """
    We force using the local patched TabEBM implementation:
      katabatic/models/tabebm/tabebm.py

    Reason:
    - pip tabebm often breaks due to TabPFN API version mismatches
      (tabpfn.config / meta_dataset_collator changes).
    """
    from .tabebm import TabEBM  # local patched file
    return TabEBM


TabEBMUpstream = _load_tabebm_class()


@dataclass
class TabEBMConfig:
    """
    Paper-aligned defaults (do not tune unless explicitly allowed).
    TabEBM does not train epochs; generation uses SGLD steps.
    """
    max_data_size: int = 10000

    # SGLD params (paper defaults)
    starting_point_noise_std: float = 0.01
    sgld_step_size: float = 0.1
    sgld_noise_std: float = 0.01
    sgld_steps: int = 200

    # surrogate negative distance
    distance_negative_class: float = 5.0

    seed: int = 42
    debug: bool = False


class TabEBMModel:
    """
    Katabatic-friendly wrapper around TabEBM:
    - fits schema + stats on X_train
    - encodes X into numeric standardized space
    - calls TabEBM.generate()
    - decodes synthetic rows back into original column space
    """

    def __init__(self, target_col: str = "target", config: Optional[TabEBMConfig] = None):
        self.target_col = target_col
        self.config = config or TabEBMConfig()

        self._tabebm = TabEBMUpstream(max_data_size=self.config.max_data_size)

        self.schema_ = None
        self.col_order_ = None

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit only preprocessing metadata (schema + stats),
        because TabEBM internally fits TabPFN per class during sampling.
        """
        x_train = x_train.copy()
        y_train = pd.Series(y_train).copy()

        schema = infer_schema(x_train)
        fit_schema_stats(x_train, schema)

        self.schema_ = schema
        self.col_order_ = list(x_train.columns)

    def sample(self, n: int, x_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate n synthetic rows total across classes, split approximately proportional
        to class distribution in y_train.

        Returns:
          (x_synth_df, y_synth_series)
        """
        if self.schema_ is None or self.col_order_ is None:
            raise RuntimeError("TabEBMModel not fitted. Call fit() first.")

        x_train = x_train.copy()
        y_train = pd.Series(y_train).copy()

        # Encode X to numeric standardized space (TabEBM expects numeric array)
        X_enc, _, _ = encode_df(x_train, self.schema_)

        # Factorize labels -> ensures TabEBM sees classes as 0..K-1
        y_raw = y_train.to_numpy()
        y_fact, uniques = pd.factorize(y_raw)  # y_fact: 0..K-1, uniques: original labels

        # Class distribution in factorized space
        classes, counts = np.unique(y_fact, return_counts=True)
        probs = counts / counts.sum()
        per_class = np.floor(probs * n).astype(int)

        # Fix rounding to sum exactly n
        remainder = n - per_class.sum()
        if remainder > 0:
            for i in np.argsort(-probs)[:remainder]:
                per_class[i] += 1

        max_per_class = int(per_class.max()) if len(per_class) else 0
        if max_per_class == 0:
            return pd.DataFrame(columns=self.col_order_), pd.Series([], name=self.target_col)

        # Generate max_per_class per class, then slice needed count
        generated = self._tabebm.generate(
            X=X_enc,
            y=y_fact,
            num_samples=max_per_class,
            starting_point_noise_std=self.config.starting_point_noise_std,
            sgld_step_size=self.config.sgld_step_size,
            sgld_noise_std=self.config.sgld_noise_std,
            sgld_steps=self.config.sgld_steps,
            distance_negative_class=self.config.distance_negative_class,
            seed=self.config.seed,
            debug=self.config.debug,
        )

        xs = []
        ys = []

        # TabEBM returns class_0..class_{K-1} matching factorized order
        for idx in classes:
            need = int(per_class[idx])
            if need <= 0:
                continue

            key = f"class_{int(idx)}"
            Xc = generated[key][:need]

            dfc = decode_df(Xc, self.schema_)
            dfc = dfc[self.col_order_]

            xs.append(dfc)

            # Map back to original labels using uniques
            original_label = uniques[idx]
            ys.append(np.full(need, original_label))

        x_synth = pd.concat(xs, ignore_index=True) if xs else pd.DataFrame(columns=self.col_order_)
        y_synth = pd.Series(np.concatenate(ys) if ys else np.array([]), name=self.target_col)

        return x_synth, y_synth
