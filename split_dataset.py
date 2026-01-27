# split_dataset.py

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    input_csv: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
    label_column: str = "label",
    *args,
    **kwargs,
):
    """
    Split a discretized dataset into train/test and save:
      - train_full.csv, test_full.csv
      - X_train.csv, X_test.csv, y_train.csv, y_test.csv
    """
    input_path = Path(input_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    print(f"Loaded data with shape: {df.shape}")

    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in {input_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_column],
    )

    train_full_path = out_dir / "train_full.csv"
    test_full_path = out_dir / "test_full.csv"
    train_df.to_csv(train_full_path, index=False)
    test_df.to_csv(test_full_path, index=False)
    print("Saved train/test full data")
    print(f"Train size: {train_df.shape}, Test size: {test_df.shape}")

    def label_distribution(d):
        label_counts = d[label_column].value_counts(normalize=True).sort_index()
        return label_counts.rename("proportion")

    print("Train label distribution:\n", label_distribution(train_df))
    print("Test label distribution:\n", label_distribution(test_df))

    X_train = train_df.drop(columns=[label_column])
    y_train = train_df[label_column]
    X_test = test_df.drop(columns=[label_column])
    y_test = test_df[label_column]

    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    print("Saved X/y split")
    print(f"Training shape: {X_train.shape} {y_train.shape}")
    print(f"Test shape: {X_test.shape} {y_test.shape}")
