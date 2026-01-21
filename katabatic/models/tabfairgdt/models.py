"""Implementation of TabFairGDT model."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from katabatic.models.base_model import Model

ArrayLike = Union[pd.Series, pd.DataFrame, np.ndarray, Sequence]


class TabFairGDT(Model):
    """
    TabFairGDT: A fast, fair tabular data generator using Autoregressive Decision Trees.
    """

    # Default settings
    _defaults = dict(
        lambda_val=0.5,           # 0 = Max Utility, 1 = Max Fairness
        min_samples_leaf=10,      # Regularization for trees
        protected_attribute=None,  # The sensitive column (e.g., 'sex')
        target_column=None,       # The outcome column (e.g., 'income')
        seed=42,
    )

    def __init__(self) -> None:
        super().__init__()

        # Internal storage
        self._models: Dict[str, Any] = {}
        self._encoders: Dict[str, LabelEncoder] = {}
        self._columns_ordered: List[str] = []

        # Fairness specific variables
        self._target_fair_prob: float = 0.5
        self._detected_protected: Optional[str] = None
        self._detected_target: Optional[str] = None

        self._cfg: Dict[str, Any] = dict(self._defaults)
        self.is_fitted = False

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["numpy", "pandas", "sklearn"]

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _as_numpy(x: ArrayLike) -> np.ndarray:
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            return x.to_numpy()
        return np.asarray(x)

    def train(self, *args, **kwargs) -> Model:
        """
        Trains the model. Supports both Pipeline mode (folder path) and Array mode (X, y).
        """

        # --- PART A: Pipeline Mode (Handling file paths from main.py) ---
        if len(args) >= 1 and isinstance(args[0], str):
            dataset_dir = args[0]
            synthetic_dir = kwargs.get("synthetic_dir")
            config = kwargs.get("config")

            # Update defaults with user config (e.g. protected attribute)
            if config:
                self._cfg.update(
                    {k: v for k, v in config.items() if k in self._defaults})

            # Load data from standard locations
            x_train_path = os.path.join(dataset_dir, "x_train.csv")
            y_train_path = os.path.join(dataset_dir, "y_train.csv")

            if not os.path.exists(x_train_path) or not os.path.exists(y_train_path):
                raise FileNotFoundError(
                    f"Expected x_train.csv and y_train.csv in {dataset_dir}")

            X_train = pd.read_csv(x_train_path)
            y_train = pd.read_csv(y_train_path)

            # Handle y being a DataFrame or Series
            if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
                y_train_series = y_train.iloc[:, 0]
            else:
                y_train_series = y_train

            # Recursively call train() with actual data (PART B)
            self.train(X_train, y_train_series, config=config)

            # --- After Training: Generate & Save Immediately ---
            n_rows = len(X_train)
            df_synth = self.sample(n_rows, as_dataframe=True)

            # Split back into X and y for saving
            y_col_name = self._detected_target
            if y_col_name in df_synth.columns:
                y_synth = df_synth[y_col_name]
                x_synth = df_synth.drop(columns=[y_col_name])
            else:
                x_synth = df_synth
                y_synth = pd.Series(np.zeros(len(df_synth)), name="label")

            # Save to the specific synthetic directory
            if synthetic_dir is None:
                synthetic_dir = os.path.join(
                    "synthetic",
                    os.path.basename(os.path.normpath(dataset_dir)),
                    "tabfairgdt",
                )
            os.makedirs(synthetic_dir, exist_ok=True)

            x_synth.to_csv(os.path.join(
                synthetic_dir, "x_synth.csv"), index=False)
            pd.DataFrame(y_synth).to_csv(os.path.join(
                synthetic_dir, "y_synth.csv"), index=False)

            return self

        # --- PART B: Array Mode ---
        if len(args) < 2:
            raise TypeError(
                "train() missing required positional arguments: X, y")
        X, y = args[0], args[1]
        config: Optional[Dict[str, Any]] = kwargs.get("config")

        if config:
            self._cfg.update(
                {k: v for k, v in config.items() if k in self._defaults})

        self._set_seed(int(self._cfg["seed"]))

        # 1. Prepare Data
        if isinstance(X, pd.DataFrame):
            data = X.copy()
        else:
            data = pd.DataFrame(
                X, columns=[f"col_{i}" for i in range(X.shape[1])])

        if isinstance(y, pd.Series):
            target_name = y.name if y.name else "target"
        elif isinstance(y, pd.DataFrame):
            target_name = y.columns[0]
            y = y.iloc[:, 0]
        else:
            target_name = "target"
            y = pd.Series(y, name=target_name)

        data[target_name] = y.values
        self._detected_target = target_name
        self._columns_ordered = list(data.columns)

        # 2. Identify Protected Attribute
        user_prot = self._cfg.get("protected_attribute")
        if user_prot and user_prot in data.columns:
            self._detected_protected = user_prot
        else:
            # Fallback logic if user didn't specify
            candidates = ['sex', 'gender', 'race', 'ethnicity', 'protected']
            matches = [c for c in candidates if c in data.columns]
            self._detected_protected = matches[0] if matches else None

            if not self._detected_protected:
                candidates_fallback = [
                    c for c in self._columns_ordered if c != target_name]
                if candidates_fallback:
                    self._detected_protected = candidates_fallback[0]

        print(
            f"[TabFairGDT] Target: {self._detected_target} | Protected: {self._detected_protected}")

        # 3. Encode Data (Trees need numbers)
        for col in self._columns_ordered:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            self._encoders[col] = le

        # 4. Calculate Global Fair Probability (P(Target=1))
        y_encoded = data[self._detected_target]
        self._target_fair_prob = (y_encoded == y_encoded.max()).mean()

        # 5. Train Autoregressive Trees
        min_samples = int(self._cfg["min_samples_leaf"])

        for i, col in enumerate(self._columns_ordered):
            if i == 0:
                self._models[col] = data[col].value_counts(
                    normalize=True).sort_index()
            else:
                X_curr = data[self._columns_ordered[:i]]
                y_curr = data[col]

                clf = DecisionTreeClassifier(min_samples_leaf=min_samples)
                clf.fit(X_curr, y_curr)
                self._models[col] = clf

        self.is_fitted = True
        return self

    def evaluate(self, X=None, y=None, *, batches=None) -> float:
        return 0.0

    def sample(
        self,
        n: int,
        *,
        batch_size: Optional[int] = None,
        as_dataframe: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame]:

        if not self.is_fitted:
            raise RuntimeError("Call train() before sample().")

        generated = pd.DataFrame(index=range(n), columns=self._columns_ordered)
        lambda_val = float(self._cfg["lambda_val"])

        for i, col in enumerate(self._columns_ordered):
            if i == 0:
                probs = self._models[col]
                choices = probs.index.values
                weights = probs.values
                weights = weights / weights.sum()
                generated[col] = np.random.choice(choices, size=n, p=weights)
            else:
                X_prev = generated[self._columns_ordered[:i]].astype(float)
                tree = self._models[col]

                leaf_indices = tree.apply(X_prev)
                node_counts = tree.tree_.value[leaf_indices, 0, :]
                node_probs = node_counts / \
                    (node_counts.sum(axis=1, keepdims=True) + 1e-10)

                # --- FAIRNESS INTERVENTION ---
                if col == self._detected_target and self._detected_protected is not None:
                    pos_class_idx = -1
                    p_tree = node_probs[:, pos_class_idx]
                    p_fair = self._target_fair_prob

                    p_final_pos = (1 - lambda_val) * \
                        p_tree + lambda_val * p_fair

                    if node_probs.shape[1] == 2:
                        p_final_neg = 1.0 - p_final_pos
                        node_probs = np.stack(
                            [p_final_neg, p_final_pos], axis=1)

                cumsum = node_probs.cumsum(axis=1)
                rand_vals = np.random.rand(n, 1)
                choice_indices = (cumsum < rand_vals).sum(axis=1)
                choice_indices = np.clip(
                    choice_indices, 0, len(tree.classes_) - 1)

                classes = tree.classes_
                generated[col] = classes[choice_indices]

        # Decode
        for col in self._columns_ordered:
            le = self._encoders[col]
            generated[col] = le.inverse_transform(generated[col].astype(int))

        if as_dataframe:
            return generated
        return generated.to_numpy()
