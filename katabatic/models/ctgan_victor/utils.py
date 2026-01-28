from __future__ import annotations

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer


def infer_categorical_columns(
    df: pd.DataFrame,
    *,
    categorical_threshold: int = 20,
    max_ratio: float = 0.05,
    label_name: Optional[str] = None,
) -> List[str]:
    cat_cols: List[str] = []
    n_rows = max(len(df), 1)

    for col in df.columns:
        s = df[col]
        dt = str(s.dtype)

        if label_name is not None and col == label_name:
            nunique = s.nunique(dropna=True)
            if nunique <= categorical_threshold:
                cat_cols.append(col)
                continue

        if dt == "object" or dt.startswith("category"):
            cat_cols.append(col)
            continue

        nunique = s.nunique(dropna=True)

        if dt.startswith("int"):
            if nunique <= categorical_threshold and (nunique / n_rows) < max_ratio:
                cat_cols.append(col)
                continue

        if dt.startswith("float"):
            if nunique <= categorical_threshold and (nunique / n_rows) < max_ratio:
                cat_cols.append(col)
                continue

    return cat_cols


@dataclass
class ColumnMeta:
    name: str
    kind: str  # 'continuous' | 'categorical'
    categories: List[str] | None = None
    qt: QuantileTransformer | None = None


def infer_schema(
    df: pd.DataFrame,
    *,
    label_name: Optional[str] = None,
    categorical_threshold: int = 20,
) -> List[ColumnMeta]:
    schema: List[ColumnMeta] = []
    cat_cols = set(
        infer_categorical_columns(
            df,
            categorical_threshold=categorical_threshold,
            label_name=label_name,
        )
    )
    for c in df.columns:
        if c in cat_cols:
            cats = sorted([str(v) for v in df[c].dropna().unique().tolist()])
            schema.append(ColumnMeta(name=c, kind="categorical", categories=cats))
        else:
            schema.append(ColumnMeta(name=c, kind="continuous"))
    return schema


def fit_transformers(df: pd.DataFrame, schema: List[ColumnMeta]) -> None:
    for col in schema:
        if col.kind == "continuous":
            qt = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=min(1000, max(len(df), 10)),
                random_state=0,
            )
            vals = df[[col.name]].astype(float)
            qt.fit(vals)
            col.qt = qt


def encode_df(
    df: pd.DataFrame,
    schema: List[ColumnMeta],
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]], List[str]]:
    arrays: List[np.ndarray] = []
    cat_blocks: Dict[str, Tuple[int, int]] = {}
    output_order: List[str] = []

    def _cur_dim() -> int:
        return int(sum(a.shape[1] for a in arrays)) if arrays else 0

    for col in schema:
        output_order.append(col.name)

        if col.kind == "continuous":
            vals = df[[col.name]].astype(float).values
            if col.qt is not None:
                vals = col.qt.transform(vals)
            arrays.append(vals.astype(np.float32))
        else:
            cats = col.categories or []
            cat_to_idx = {v: i for i, v in enumerate(cats)}

            idxs = (
                df[col.name]
                .astype(str)
                .map(lambda v: cat_to_idx.get(v, -1))
                .values
            )

            one_hot = np.zeros((len(df), len(cats)), dtype=np.float32)
            valid = idxs >= 0
            one_hot[np.where(valid)[0], idxs[valid]] = 1.0

            start = _cur_dim()
            arrays.append(one_hot)
            end = _cur_dim()
            cat_blocks[col.name] = (start, end)

    enc = np.concatenate(arrays, axis=1) if arrays else np.zeros((len(df), 0), dtype=np.float32)
    return enc, cat_blocks, output_order


def decode_batch(enc: np.ndarray, schema: List[ColumnMeta]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for i in range(enc.shape[0]):
        row: Dict[str, object] = {}
        ptr = 0
        for col in schema:
            if col.kind == "continuous":
                val = float(enc[i, ptr])
                if col.qt is not None:
                    val = float(col.qt.inverse_transform(np.array([[val]]))[0, 0])
                row[col.name] = val
                ptr += 1
            else:
                size = len(col.categories or [])
                block = enc[i, ptr : ptr + size]
                idx = int(np.argmax(block)) if size > 0 else -1
                cats = col.categories or []
                row[col.name] = cats[idx] if 0 <= idx < len(cats) else None
                ptr += size
        rows.append(row)
    return pd.DataFrame(rows)


# Safe: skip zero-width categoricals so cond_blocks matches actual dim
def build_conditioning(
    df: pd.DataFrame,
    schema: List[ColumnMeta],
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    arrays: List[np.ndarray] = []
    cond_blocks: Dict[str, Tuple[int, int]] = {}

    def _cur_dim() -> int:
        return int(sum(a.shape[1] for a in arrays)) if arrays else 0

    for col in schema:
        if col.kind != "categorical":
            continue

        cats = col.categories or []
        if len(cats) == 0:
            continue

        cat_to_idx = {v: i for i, v in enumerate(cats)}
        idxs = (
            df[col.name]
            .astype(str)
            .map(lambda v: cat_to_idx.get(v, -1))
            .values
        )

        one_hot = np.zeros((len(df), len(cats)), dtype=np.float32)
        valid = idxs >= 0
        one_hot[np.where(valid)[0], idxs[valid]] = 1.0

        start = _cur_dim()
        arrays.append(one_hot)
        end = _cur_dim()
        cond_blocks[col.name] = (start, end)

    cond = np.concatenate(arrays, axis=1) if arrays else np.zeros((len(df), 0), dtype=np.float32)
    return cond, cond_blocks


def sample_conditions(
    n: int,
    schema: List[ColumnMeta],
    cond_blocks: Dict[str, Tuple[int, int]],
    empirical_probs: Optional[Dict[str, np.ndarray]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    spans: List[Tuple[str, int]] = []
    total_dim = 0

    for col in schema:
        if col.kind != "categorical":
            continue
        dim = len(col.categories or [])
        if dim <= 0:
            continue
        if col.name not in cond_blocks:
            continue
        spans.append((col.name, dim))
        total_dim += dim

    if total_dim == 0:
        return np.zeros((n, 0), dtype=np.float32)

    rng = np.random.default_rng(seed)
    conds = np.zeros((n, total_dim), dtype=np.float32)

    for i in range(n):
        name, dim = spans[int(rng.integers(0, len(spans)))]
        start, end = cond_blocks[name]

        if empirical_probs is not None and name in empirical_probs and len(empirical_probs[name]) == dim:
            p = empirical_probs[name]
            k = int(rng.choice(dim, p=p))
        else:
            k = int(rng.integers(0, dim))

        conds[i, start + k] = 1.0

    return conds.astype(np.float32)
