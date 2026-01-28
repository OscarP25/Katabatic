from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.model_selection import train_test_split


def seed_everything(seed: int) -> None:
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_numpy(X: Union[np.ndarray, torch.Tensor, pd.DataFrame, None]) -> Optional[np.ndarray]:
    if isinstance(X, np.ndarray):
        return X
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    if X is None:
        return None
    raise ValueError("Unsupported input type")


class TabEBM:
    """
    TabEBM (proxy implementation).

    Energy-based sampling using distance-based surrogate energy
    and Stochastic Gradient Langevin Dynamics (SGLD).

    This version intentionally removes TabPFN to ensure
    stability and reproducibility inside Katabatic.
    """

    def __init__(self, max_data_size: int = 10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_data_size = max_data_size

    def generate(
        self,
        X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        y: Union[np.ndarray, torch.Tensor, pd.Series],
        num_samples: int,
        starting_point_noise_std: float = 0.01,
        sgld_step_size: float = 0.01,
        sgld_noise_std: float = 0.01,
        sgld_steps: int = 200,
        distance_negative_class: float = 5.0,
        seed: int = 42,
    ) -> Dict[str, np.ndarray]:

        seed_everything(seed)

        X = to_numpy(X)
        y = to_numpy(y).reshape(-1)

        if X.shape[0] > self.max_data_size:
            X, _, y, _ = train_test_split(
                X,
                y,
                train_size=self.max_data_size,
                stratify=y,
                random_state=seed,
            )

        results: Dict[str, np.ndarray] = {}
        unique_classes = np.unique(y)

        for class_idx, cls in enumerate(unique_classes):
            X_cls = X[y == cls]

            X_ebm, y_ebm = self.add_surrogate_negative_samples(
                X_cls, distance_negative_class
            )

            X_ebm_t = torch.tensor(X_ebm, device=self.device, dtype=torch.float32)
            y_ebm_t = torch.tensor(y_ebm, device=self.device, dtype=torch.long)

            start_idx = torch.randint(
                0, X_cls.shape[0], (num_samples,), device=self.device
            )

            X_sgld = X_ebm_t[y_ebm_t == 0][start_idx]
            X_sgld = X_sgld + torch.randn_like(X_sgld) * starting_point_noise_std
            X_sgld.requires_grad_(True)

            noise = torch.randn(
                (sgld_steps, num_samples, X.shape[1]),
                device=self.device,
            )

            for t in range(sgld_steps):
                if X_sgld.grad is not None:
                    X_sgld.grad.zero_()

                energy = self.compute_energy_distance(X_sgld, X_ebm_t, y_ebm_t)
                loss = energy.mean()
                loss.backward()

                with torch.no_grad():
                    X_sgld -= sgld_step_size * X_sgld.grad
                    X_sgld += sgld_noise_std * noise[t]
                    X_sgld.requires_grad_(True)

            results[f"class_{class_idx}"] = X_sgld.detach().cpu().numpy()

        return results

    @staticmethod
    def compute_energy_distance(
        X_synth: torch.Tensor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> torch.Tensor:

        distances = torch.cdist(X_synth, X_train)
        min_dist, _ = distances.min(dim=1)

        same_class = y_train == 0
        class_dist = distances[:, same_class].mean(dim=1)

        return min_dist + class_dist

    @staticmethod
    def add_surrogate_negative_samples(
        X: Union[np.ndarray, torch.Tensor],
        distance_negative_class: float,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:

        num_features = X.shape[1]

        if num_features == 2:
            surrogates = [
                [-distance_negative_class, -distance_negative_class],
                [distance_negative_class, distance_negative_class],
                [-distance_negative_class, distance_negative_class],
                [distance_negative_class, -distance_negative_class],
            ]
        else:
            surrogates = set()
            while len(surrogates) < 4:
                pt = np.random.choice(
                    [-distance_negative_class, distance_negative_class], num_features
                )
                surrogates.add(tuple(pt))
                surrogates.add(tuple(-np.array(pt)))
            surrogates = list(surrogates)

        if isinstance(X, np.ndarray):
            X_sur = np.array(surrogates, dtype=X.dtype)
            X_ebm = np.concatenate([X, X_sur], axis=0)
            y_ebm = np.concatenate(
                [np.zeros(X.shape[0]), np.ones(len(X_sur))]
            ).astype(int)
            return X_ebm, y_ebm

        X_sur = torch.tensor(surrogates, dtype=X.dtype, device=X.device)
        X_ebm = torch.cat([X, X_sur], dim=0)
        y_ebm = torch.cat(
            [
                torch.zeros(X.shape[0], device=X.device, dtype=torch.long),
                torch.ones(X_sur.shape[0], device=X.device, dtype=torch.long),
            ],
            dim=0,
        )
        return X_ebm, y_ebm
