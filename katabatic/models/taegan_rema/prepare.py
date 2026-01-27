"""
TAEGAN Prepare Script
--------------------
Handles Steps 2–4 only:
1. Load x_train from sample_data
2. Force numeric float32 encoding
3. Create span-info.json
4. Create mask_train.csv

This file is intentionally minimal and model-agnostic.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def infer_span_info(x_df: pd.DataFrame):
    """
    Infer span-info from already-encoded x_train.

    Assumptions:
    - One-hot encoded categorical columns contain '_'
    - Continuous columns do not
    """
    span_info = []
    for col in x_df.columns:
        if "_" in col:
            span_info.append((1, "onehot"))
        else:
            span_info.append((1, "normal"))
    return span_info


def create_mask(x_df: pd.DataFrame, p: float = 0.5):
    """
    Create random known/unknown mask.
    """
    mask = np.random.binomial(
        n=1,
        p=p,
        size=x_df.shape
    ).astype(bool)
    return pd.DataFrame(mask, columns=x_df.columns)


# ------------------------------------------------------------
# Main entry
# ------------------------------------------------------------
def prepare_taegan(
    *,
    root_dir: str,
    dataset: str,
    mask_prob: float = 0.5,
):
    """
    Prepare TAEGAN cache directory.

    Parameters
    ----------
    root_dir : str
        Katabatic project root
    dataset : str
        Dataset name (e.g. 'adult')
    mask_prob : float
        Probability that a value is known (default = 0.5)
    """

    root = Path(root_dir)
    sample_dir = root / "sample_data" / dataset
    synth_dir = root / "synthetic" / dataset / "taegan"
    synth_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing TAEGAN data")
    print("Sample dir:", sample_dir)
    print("TAEGAN dir:", synth_dir)

    # --------------------------------------------------------
    # Load x_train
    # --------------------------------------------------------
    x_train = pd.read_csv(sample_dir / "x_train.csv")

    # Force numeric float32 (CRITICAL for PyTorch)
    x_train = x_train.apply(pd.to_numeric, errors="raise").astype(np.float32)

    x_train.to_csv(synth_dir / "x_train.csv", index=False)
    print(" x_train.csv saved (float32)")

    # --------------------------------------------------------
    # span-info.json
    # --------------------------------------------------------
    span_info = infer_span_info(x_train)
    with open(synth_dir / "span-info.json", "w") as f:
        json.dump(span_info, f)

    print("span-info.json created")
    print("Total spans:", len(span_info))

    # --------------------------------------------------------
    # mask_train.csv
    # --------------------------------------------------------
    mask_df = create_mask(x_train, p=mask_prob)
    mask_df.to_csv(synth_dir / "mask_train.csv", index=False)

    print(" mask_train.csv created")
    print("Mask shape:", mask_df.shape)

    print(" TAEGAN preparation complete")


# ------------------------------------------------------------
# Optional CLI usage
# ------------------------------------------------------------
if __name__ == "__main__":
    prepare_taegan(
        root_dir="/content/drive/MyDrive/katabatic1",
        dataset="adult",
    )
