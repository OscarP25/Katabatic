from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import Generator, PCG64
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


# ----------------------------
# Reproducibility & Splits
# ----------------------------

def set_seed(seed: int = 42) -> Generator:
    """Return a local RNG seeded via PCG64."""
    return np.random.default_rng(PCG64(seed))


def safe_train_val_test_split(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split into train/val/test with optional stratification on y.
    val_size is relative to the *train* portion.
    """
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    strat_val = y_train if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=strat_val
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ----------------------------
# Encoding for mixed data
# ----------------------------

@dataclass
class MixedEncoder:
    """
    Column-wise encoding:
      - Numeric: StandardScaler
      - Categorical: OneHotEncoder(handle_unknown="ignore")
    Stores ColumnTransformer pipeline.
    """
    numeric_cols: List[str]
    categorical_cols: List[str]
    pipeline: Optional[ColumnTransformer] = None

    def fit(self, X: pd.DataFrame) -> "MixedEncoder":
        transformers = []
        if self.numeric_cols:
            transformers.append(("num", StandardScaler(), self.numeric_cols))
        if self.categorical_cols:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            transformers.append(("cat", ohe, self.categorical_cols))
        self.pipeline = ColumnTransformer(transformers=transformers, remainder="drop")
        self.pipeline.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Call fit() before transform().")
        return self.pipeline.transform(X)

    def inverse_transform(self, arr: np.ndarray) -> pd.DataFrame:
        """
        Attempts to invert transformations for interpretability.
        Note: OHE inverse is exact; StandardScaler inverse is exact.
        Column order is preserved as [numeric..., categorical...expanded].
        """
        if self.pipeline is None:
            raise RuntimeError("Call fit() before inverse_transform().")

        # ColumnTransformer does not provide a direct inverse for the entire matrix,
        # so we reconstruct by applying the inverse of each step range-wise.
        # We rely on fitted transformers and their column indices.
        col_out = []
        start = 0
        for name, transformer, cols in self.pipeline.transformers_:
            if name == "remainder":
                continue
            n_out = transformer.transform(pd.DataFrame({c: [0] for c in cols}).iloc[:0]).shape[1]
            sub = arr[:, start:start + n_out]
            start += n_out

            if name == "num":
                inv = transformer.inverse_transform(sub)
                col_out.append(pd.DataFrame(inv, columns=cols))
            elif name == "cat":
                inv = transformer.inverse_transform(sub)
                col_out.append(pd.DataFrame(inv, columns=cols))
            else:
                # unknown block; pass through
                col_out.append(pd.DataFrame(sub, columns=[f"{name}_{i}" for i in range(n_out)]))

        return pd.concat(col_out, axis=1)[self.numeric_cols + self.categorical_cols]


def encode_mixed_frame(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, MixedEncoder]:
    """
    Fit a MixedEncoder and return encoded array and the encoder.
    If categorical_cols is None, infer as non-numeric.
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_cols = [c for c in df.columns if c not in categorical_cols]
    enc = MixedEncoder(numeric_cols=numeric_cols, categorical_cols=categorical_cols).fit(df)
    return enc.transform(df), enc


def decode_mixed_frame(arr: np.ndarray, encoder: MixedEncoder) -> pd.DataFrame:
    """Inverse transform an encoded array back to a mixed DataFrame."""
    return encoder.inverse_transform(arr)


# ----------------------------
# TSTR Evaluation
# ----------------------------

def _default_clfs(random_state: int = 42) -> Dict[str, object]:
    return {
        "logreg": LogisticRegression(max_iter=200, n_jobs=None, random_state=random_state),
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=random_state
        ),
        # You can add XGBoost here if available:
        # "xgb": xgboost.XGBClassifier(
        #     n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
        #     colsample_bytree=0.9, tree_method="hist", n_jobs=-1, random_state=random_state
        # ),
    }


def evaluate_tstr(
    real_train_X: pd.DataFrame,
    real_train_y: Union[pd.Series, np.ndarray],
    real_test_X: pd.DataFrame,
    real_test_y: Union[pd.Series, np.ndarray],
    synthetic_train_X: pd.DataFrame,
    synthetic_train_y: Union[pd.Series, np.ndarray],
    classifiers: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """
    Train classifiers on synthetic (X_synth, y_synth), then test on real (X_test_real, y_test_real).

    Returns
    -------
    pd.DataFrame with columns: ['clf', 'accuracy', 'f1_macro']
    """
    if classifiers is None:
        classifiers = _default_clfs()

    rows = []
    for name, clf in classifiers.items():
        pipe = Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=False) if _needs_scaler(real_test_X) else "passthrough"),
            ("clf", clf),
        ])
        pipe.fit(synthetic_train_X, synthetic_train_y)
        preds = pipe.predict(real_test_X)
        rows.append({
            "clf": name,
            "accuracy": float(accuracy_score(real_test_y, preds)),
            "f1_macro": float(f1_score(real_test_y, preds, average="macro")),
        })
    return pd.DataFrame(rows)


def _needs_scaler(X: pd.DataFrame) -> bool:
    """Heuristic: scale if any column isn't a small-integer categorical."""
    if not isinstance(X, pd.DataFrame):
        return True
    if X.select_dtypes(exclude=[np.number]).shape[1] > 0:
        return True
    # numeric: if many unique values -> likely continuous
    unique_ratio = X.nunique() / (len(X) + 1e-9)
    return bool((unique_ratio > 0.2).any())
