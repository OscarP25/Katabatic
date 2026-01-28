# katabatic/models/gaussianCopula/utils/metadata.py

from __future__ import annotations

import pandas as pd


def build_single_table_metadata(df: pd.DataFrame, force_categorical: bool = True):
    """
    Build SDV metadata for a single table.

    - If force_categorical=True: treat non-numeric columns as categorical.
    - Works across SDV versions (may raise deprecation warnings, but runs fine).
    """
    # SDV new metadata API is "Metadata", older is "SingleTableMetadata".
    # We'll use the widely supported SingleTableMetadata for compatibility.
    from sdv.metadata import SingleTableMetadata

    md = SingleTableMetadata()
    md.detect_from_dataframe(data=df)

    if force_categorical:
        for col in df.columns:
            # If not numeric => categorical
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    md.update_column(column_name=col, sdtype="categorical")
                except Exception:
                    # If SDV version differs, just ignore and proceed.
                    pass

    return md
