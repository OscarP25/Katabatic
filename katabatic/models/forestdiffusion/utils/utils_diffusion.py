from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


@dataclass
class TabularCodec:
    """
    Encode mixed-type dataframe to numeric matrix for diffusion,
    and decode back to original schema.

    Improvements applied:
    - Numeric columns are median-imputed and STANDARDIZED (mean/std) for stable diffusion.
    - Categorical columns are ordinal-encoded then normalized to [-1, 1] so they don't dominate.
    - Inverse transform restores original numeric scale and categorical values.

    Notes:
    - Ordinal encoding is still an approximation for categoricals. This is a baseline codec.
    """
    categorical_cols: Optional[List[str]] = None
    numeric_cols: Optional[List[str]] = None
    unknown_value: int = -1

    _enc: Optional[OrdinalEncoder] = None
    _cat_categories: Dict[str, np.ndarray] = None
    _col_order: List[str] = None

    # numeric stats (fit-time)
    _num_median: Optional[pd.Series] = None
    _num_mean: Optional[pd.Series] = None
    _num_std: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "TabularCodec":
        df = df.copy()
        self._col_order = [str(c) for c in df.columns]
        df.columns = self._col_order

        # detect columns if not provided
        if self.numeric_cols is None:
            self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.categorical_cols is None:
            self.categorical_cols = [c for c in df.columns if c not in self.numeric_cols]

        # numeric stats
        if self.numeric_cols:
            num = df[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
            self._num_median = num.median(numeric_only=True)
            num = num.fillna(self._num_median)
            self._num_mean = num.mean()
            self._num_std = num.std(ddof=0).replace(0, 1.0)  # avoid div0

        # categorical encoder
        self._cat_categories = {}
        if self.categorical_cols:
            cat_df = df[self.categorical_cols].astype(str).fillna("NA")
            self._enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=self.unknown_value,
            )
            self._enc.fit(cat_df)
            for i, c in enumerate(self.categorical_cols):
                self._cat_categories[c] = np.array(self._enc.categories_[i], dtype=object)

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df.columns = [str(c) for c in df.columns]

        # ensure all columns exist
        for c in self._col_order:
            if c not in df.columns:
                df[c] = np.nan
        df = df[self._col_order]

        parts = []

        # numeric: median-impute + standardize
        if self.numeric_cols:
            num = df[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
            num = num.fillna(self._num_median)
            num = ((num - self._num_mean) / self._num_std).astype(np.float32)
            parts.append(num.to_numpy(dtype=np.float32))

        # categorical -> ordinal codes -> normalize to [-1, 1]
        if self.categorical_cols:
            if self._enc is None:
                raise RuntimeError("TabularCodec categorical encoder is not fitted.")
            cat_df = df[self.categorical_cols].astype(str).fillna("NA")
            codes = self._enc.transform(cat_df).astype(np.float32)  # (N, C)

            # normalize each categorical column by its cardinality to [-1, 1]
            for j, c in enumerate(self.categorical_cols):
                cats = self._cat_categories[c]
                m = len(cats)
                if m <= 1:
                    codes[:, j] = 0.0
                else:
                    # map [0, m-1] -> [-1, 1]
                    codes[:, j] = (codes[:, j] / (m - 1)) * 2.0 - 1.0

            parts.append(codes.astype(np.float32))

        if not parts:
            raise ValueError("No columns to encode.")

        X = np.concatenate(parts, axis=1).astype(np.float32)
        return X

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        X = np.asarray(X, dtype=np.float32)
        out: Dict[str, np.ndarray] = {}
        idx = 0

        # numeric: unstandardize
        if self.numeric_cols:
            k = len(self.numeric_cols)
            num = X[:, idx: idx + k]
            idx += k

            mean = self._num_mean.to_numpy(dtype=np.float32)
            std = self._num_std.to_numpy(dtype=np.float32)
            num = (num * std) + mean

            for j, c in enumerate(self.numeric_cols):
                out[c] = num[:, j].astype(np.float32)

        # categorical: denormalize [-1,1] -> [0,m-1] -> category
        if self.categorical_cols:
            k = len(self.categorical_cols)
            cat = X[:, idx: idx + k]
            idx += k

            for j, c in enumerate(self.categorical_cols):
                cats = self._cat_categories[c]
                m = len(cats)
                if m <= 1:
                    out[c] = np.array([cats[0]] * X.shape[0], dtype=object)
                    continue

                # [-1,1] -> [0, m-1]
                v = (cat[:, j] + 1.0) * 0.5 * (m - 1)
                cat_idx = np.rint(v).astype(int)
                cat_idx = np.clip(cat_idx, 0, m - 1)
                out[c] = cats[cat_idx]

        df = pd.DataFrame(out)
        df = df[self._col_order]
        return df
