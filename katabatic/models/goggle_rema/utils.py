# =========================
# utils.py
# =========================

# Standard
import random
import numpy as np
import torch

# PyTorch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Reproducibility (paper-faithful)
# --------------------------------------------------
def seed_worker(worker_id):
    """
    Ensures reproducibility across DataLoader workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------
# DataLoader helper (used by GoggleModel)
# --------------------------------------------------
def get_dataloader(X, batch_size, seed):
    """
    Splits data into train/validation and returns DataLoaders.

    Parameters
    ----------
    X : pandas.DataFrame
        Input real dataset (features only)
    batch_size : int
        Batch size (paper default = 32)
    seed : int
        Random seed

    Returns
    -------
    dict
        {"train": DataLoader, "val": DataLoader}
    """

    # ---- Train / validation split (paper default: 80/20) ----
    X_train, X_val = train_test_split(
        X,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )

    # ---- Convert to tensors (CPU only here) ----
    train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32))

    # ---- Generator for reproducibility ----
    g = torch.Generator()
    g.manual_seed(seed)

    # ---- DataLoaders ----
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g,
        ),
    }

    return loaders
