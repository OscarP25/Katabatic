
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


class TrainingDataLoader(DataLoader):
    def __init__(self, cache_dir: str, batch_size: int):
        cache_dir = Path(cache_dir)

        x = pd.read_csv(cache_dir / "x_train.csv").values
        mask = pd.read_csv(cache_dir / "mask_train.csv").values

        x = torch.tensor(x, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.bool)

        dataset = TensorDataset(x, mask)
        super().__init__(dataset, batch_size=batch_size, shuffle=True)

    def __iter__(self):
        for x, m in super().__iter__():
            yield x, m


class InferenceDataLoader(DataLoader):
    def __init__(self, n: int, cache_dir: str, batch_size: int):
        cache_dir = Path(cache_dir)

        x = pd.read_csv(cache_dir / "x_train.csv").values
        mask = pd.read_csv(cache_dir / "mask_train.csv").values

        x = torch.tensor(x[:n], dtype=torch.float32)
        mask = torch.tensor(mask[:n], dtype=torch.bool)

        dataset = TensorDataset(x, mask)
        super().__init__(dataset, batch_size=batch_size, shuffle=False)

    def __iter__(self):
        for x, m in super().__iter__():
            yield x, m
