# katabatic/models/great/great_trainer.py

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer


def _seed_worker(_):
    """Set deterministic worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)


class GReaTTrainer(Trainer):
    """
    Custom Trainer for GReaT.

    IMPORTANT:
    - Keeps unused columns (do NOT call _remove_unused_columns)
    - Forces safe DataLoader settings for Windows stability
    """

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer requires a train_dataset.")

        data_collator = self.data_collator
        train_dataset = self.train_dataset
        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,

            # 🔒 SAFETY SETTINGS
            num_workers=0,          # Windows-safe
            pin_memory=False,       # CPU-friendly
            worker_init_fn=_seed_worker,
        )
