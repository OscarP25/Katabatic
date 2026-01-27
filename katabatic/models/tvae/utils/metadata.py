from __future__ import annotations

from typing import Optional, Iterable
import pandas as pd


def _is_probably_categorical(s: pd.Series, max_unique: int = 50) -> bool:
    dt = str(s.dtype)
    if dt in ("object", "category", "bool"):
        return True
    if dt.startswith("int"):
        nunique = s.nunique(dropna=True)
        return nunique <= max_unique
    return False


def _ensure_series(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


def _add_single_table_metadata_new(df: pd.DataFrame, table_name: str):
    from sdv.metadata import Metadata

    if hasattr(Metadata, "detect_from_dataframe"):
        return Metadata.detect_from_dataframe(data=df, table_name=table_name)

    md = Metadata()
    if hasattr(md, "add_table"):
        md.add_table(name=table_name)

    if hasattr(md, "detect_table_from_dataframe"):
        md.detect_table_from_dataframe(table_name=table_name, data=df)
    elif hasattr(md, "detect_from_dataframe"):
        md.detect_from_dataframe(data=df, table_name=table_name)
    else:
        for c in df.columns:
            md.update_column(table_name=table_name, column_name=c, sdtype="categorical")

    return md


def _update_column_safe(md, table_name: str, col: str, sdtype: str):
    try:
        md.update_column(table_name=table_name, column_name=col, sdtype=sdtype)
        return
    except Exception:
        pass

    try:
        md.update_column(column_name=col, sdtype=sdtype, table_name=table_name)
        return
    except Exception:
        pass

    try:
        md.tables[table_name].update_column(column_name=col, sdtype=sdtype)
        return
    except Exception:
        pass


def build_single_table_metadata(
    df: pd.DataFrame,
    table_name: str = "data",
    force_categorical: bool = True,
    categorical_columns: Optional[Iterable[str]] = None,
    label_column: Optional[str] = None,
):
    df = df.copy()
    df.columns = [str(c) for c in df.columns]

    cat_set = set(map(str, categorical_columns)) if categorical_columns is not None else set()
    if label_column is not None:
        cat_set.add(str(label_column))

    # New SDV
    try:
        md = _add_single_table_metadata_new(df, table_name=table_name)

        if force_categorical:
            for c in df.columns:
                s = _ensure_series(df, c)
                if c in cat_set or _is_probably_categorical(s):
                    _update_column_safe(md, table_name, c, "categorical")

        return md

    except Exception:
        # Old SDV fallback
        try:
            from sdv.single_table import SingleTableMetadata

            stm = SingleTableMetadata()
            stm.detect_from_dataframe(df)

            if force_categorical:
                for c in df.columns:
                    s = _ensure_series(df, c)
                    if c in cat_set or _is_probably_categorical(s):
                        stm.update_column(column_name=c, sdtype="categorical")

            return stm

        except Exception as e:
            raise ImportError(
                "Could not build SDV metadata for TVAE. "
                "Your SDV version appears incompatible with this environment."
            ) from e
