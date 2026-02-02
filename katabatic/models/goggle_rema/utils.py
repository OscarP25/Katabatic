

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split



def seed_worker(worker_id):
    """
    Ensures reproducibility across DataLoader workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def get_dataloader(X, batch_size, seed):
    """
    Prepare dataloaders for GOGGLE.
    Handles categorical data via ordinal encoding (paper-faithful).
    """


    if isinstance(X, pd.DataFrame):
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        X_enc = enc.fit_transform(X)
    else:
        raise ValueError("Input must be a pandas DataFrame")

   
    X_train, X_val = train_test_split(
        X_enc,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )


    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train)
    val_dataset = TensorDataset(X_val)


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
    }
