from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from . import logger as log


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray | list) -> None:
        data = np.asarray(data, dtype="float32")
        self.x = torch.from_numpy(data).to(DEVICE)
        self.n_samples = self.x.shape[0]

        log.info("***** DATA *****")
        log.info(f"n_samples = {self.n_samples}")

    def __getitem__(self, index: int) -> Any:
        return self.x[index]

    def __len__(self) -> int:
        return self.n_samples


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: np.ndarray | list,
        batch_size: int = 64,
        num_workers: int = 0,
        data_dir: Path = Path.cwd(),
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = Dataset(data)
        self.dims = self.dataset.x.shape[1:]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
