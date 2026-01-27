from __future__ import annotations

import pandas as pd


def build_single_table_metadata(df: pd.DataFrame, force_categorical: bool = True):
    """
    Build SDV SingleTableMetadata from a dataframe.

    For Katabatic discretized datasets, forcing categorical is usually best
    (keeps values within observed categories and avoids float drift).
    """
    from sdv.metadata import SingleTableMetadata

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    if force_categorical:
        for c in df.columns:
            metadata.update_column(column_name=c, sdtype="categorical")

    return metadata
