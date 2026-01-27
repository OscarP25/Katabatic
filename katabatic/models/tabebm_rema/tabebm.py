# katabatic/models/tabebm/tabebm.py
from __future__ import annotations

import os
import random
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tabpfn import TabPFNClassifier

# ---------------------------------------------------------------------
# ✅ PATCH 1: TabPFN API changes — some versions don't have tabpfn.config
# ---------------------------------------------------------------------
try:
    from tabpfn.config import ModelInterfaceConfig, PreprocessorConfig  # type: ignore
except Exception:
    ModelInterfaceConfig = None
    PreprocessorConfig = None

# ---------------------------------------------------------------------
# ✅ PATCH 2: TabPFN API changes — some versions don't have meta_dataset_collator
# ---------------------------------------------------------------------
try:
    from tabpfn.utils import meta_dataset_collator  # type: ignore
except Exception:
    # TabEBM uses batch_size=1, so returning the single item is enough
    def meta_dataset_collator(batch):
        return batch[0]


def to_numpy(X: Union[np.ndarray, torch.Tensor, pd.DataFrame, None]) -> Optional[np.ndarray]:
    if isinstance(X, np.ndarray):
        return X
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    if X is None:
        return None
    raise ValueError("X must be np.ndarray, torch.Tensor, pd.DataFrame, or None")


def seed_everything(seed: int) -> None:
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TabEBM:
    """
    TabEBM (patched): Energy-based sampling using TabPFN gradients.

    Notes:
    - No 'epochs' like GANs.
    - Main hyperparameter is SGLD steps (default 200).
    """

    def __init__(self, max_data_size: int = 10000):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_data_size = max_data_size
        self._fitted_models_cache: Dict[int, bool] = {}

        # Build inference_config only if supported by this TabPFN version
        inference_config = None
        if ModelInterfaceConfig is not None and PreprocessorConfig is not None:
            inference_config = ModelInterfaceConfig(
                FINGERPRINT_FEATURE=False,
                FEATURE_SHIFT_METHOD=None,
                CLASS_SHIFT_METHOD=None,
                PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="none")],
            )

        # Important: keep n_estimators=1 for stable gradients
        self.model = TabPFNClassifier(
            device=self.device,
        )

        

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def generate(
        self,
        X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        y: Union[np.ndarray, torch.Tensor, pd.Series],
        num_samples: int,
        starting_point_noise_std: float = 0.01,
        sgld_step_size: float = 0.1,
        sgld_noise_std: float = 0.01,
        sgld_steps: int = 200,
        distance_negative_class: float = 5.0,
        seed: int = 42,
        debug: bool = False,
    ) -> Dict[str, np.ndarray]:

        data = self._preprocess(X, y)
        X_np = data["X"]
        y_np = data["y"]

        res = self._sampling_internal(
            X=X_np,
            y=y_np,
            num_samples=num_samples,
            starting_point_noise_std=starting_point_noise_std,
            sgld_step_size=sgld_step_size,
            sgld_noise_std=sgld_noise_std,
            sgld_steps=sgld_steps,
            distance_negative_class=distance_negative_class,
            seed=seed,
            debug=debug,
        )

        out: Dict[str, np.ndarray] = {}
        unique_classes = np.unique(y_np)
        for idx, _ in enumerate(unique_classes):
            key = f"class_{int(idx)}"
            out[key] = res[key]["sampling_paths"]
        return out

    # ------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------
    def _preprocess(
        self,
        X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        y: Union[np.ndarray, torch.Tensor, pd.Series],
    ) -> Dict[str, np.ndarray]:
        if not isinstance(X, np.ndarray):
            X = to_numpy(X)
        if not isinstance(y, np.ndarray):
            y = to_numpy(y).reshape(-1)

        assert X is not None and y is not None

        # Stratified downsample for stability + speed
        if X.shape[0] > self.max_data_size:
            X_s, _, y_s, _ = train_test_split(
                X,
                y,
                train_size=self.max_data_size,
                random_state=42,
                stratify=y,
            )
            X, y = X_s, y_s

        return {"X": X, "y": y}

    # ------------------------------------------------------------
    # Sampling internal
    # ------------------------------------------------------------
    def _sampling_internal(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_samples: int,
        starting_point_noise_std: float,
        sgld_step_size: float,
        sgld_noise_std: float,
        sgld_steps: int,
        distance_negative_class: float,
        seed: int,
        debug: bool,
    ) -> Dict[str, Dict[str, np.ndarray]]:

        if debug:
            print("=== TabEBM Sampling ===")
            print("Device:", self.device)
            print(f"SGLD steps={sgld_steps}, step_size={sgld_step_size}, noise_std={sgld_noise_std}")
            print("Neg distance:", distance_negative_class)

        seed_everything(seed)

        unique_classes = np.unique(y)
        results: Dict[str, Dict[str, np.ndarray]] = {}

        for class_index, target_class in enumerate(unique_classes):
            if debug:
                print(f"\n--- Class {target_class} (key class_{class_index}) ---")

            X_one = X[y == target_class]
            X_ebm, y_ebm = self.add_surrogate_negative_samples(
                X_one, distance_negative_class=distance_negative_class
            )

            X_ebm_t = torch.from_numpy(X_ebm).float().to(self.device)
            y_ebm_t = torch.from_numpy(y_ebm).long().to(self.device)

            self._fit_predictor_cached(X_ebm_t, y_ebm_t, cache_key=class_index)

            start = self._initialize_sgld_starting_points(
                X_ebm_t, y_ebm_t, num_samples, starting_point_noise_std, seed
            )
            X_start = start["X_start"]
            y_start = start["y_start"]

            batch = self._prepare_tabpfn_batch_data(X_start, y_start)
            X_sgld = batch["X_train"][0].to(self.device).requires_grad_(True)

            noise_tensor = torch.randn(
                (sgld_steps, num_samples, X.shape[1]),
                device=self.device,
                dtype=X_sgld.dtype,
            )

            X_final = self._perform_sgld_sampling(
                X_sgld=X_sgld,
                noise_tensor=noise_tensor,
                sgld_step_size=sgld_step_size,
                sgld_noise_std=sgld_noise_std,
                sgld_steps=sgld_steps,
                debug=debug,
            )

            key = f"class_{int(class_index)}"
            results[key] = {"sampling_paths": X_final.detach().cpu().squeeze(0).numpy()}

        return results

    # ------------------------------------------------------------
    # Fit caching
    # ------------------------------------------------------------
    def _fit_predictor_cached(self, X_ebm: torch.Tensor, y_ebm: torch.Tensor, cache_key: int) -> None:
        if cache_key in self._fitted_models_cache:
            return

        batch = self._prepare_tabpfn_batch_data(X_ebm, y_ebm)
        X_list = [xb.to(self.device) for xb in batch["X_train"]]
        y_list = [yb.to(self.device) for yb in batch["y_train"]]

        self.model.fit_from_preprocessed(
            X_list,
            y_list,
            cat_ix=batch["cat_ixs"],
            configs=batch["confs"],
        )

        self._fitted_models_cache[cache_key] = True

    # ------------------------------------------------------------
    # Init starting points
    # ------------------------------------------------------------
    def _initialize_sgld_starting_points(
        self,
        X_ebm: torch.Tensor,
        y_ebm: torch.Tensor,
        num_samples: int,
        starting_point_noise_std: float,
        seed: int,
    ) -> Dict[str, torch.Tensor]:
        seed_everything(seed)

        real_mask = (y_ebm == 0)
        real_samples = X_ebm[real_mask]
        n_real = real_samples.shape[0]

        idx = torch.randint(0, n_real, (num_samples,), device=self.device)
        X_start = real_samples[idx]
        y_start = torch.zeros(num_samples, dtype=torch.long, device=self.device)

        if starting_point_noise_std > 0:
            X_start = X_start + torch.randn_like(X_start) * starting_point_noise_std

        return {"X_start": X_start, "y_start": y_start}

    # ------------------------------------------------------------
    # Prepare TabPFN batch
    # ------------------------------------------------------------
    def _prepare_tabpfn_batch_data(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        splitter = partial(self.train_test_split_allow_full_train, test_size=0, random_state=42, shuffle=False)

        X_cpu = X.detach().cpu()
        y_cpu = y.detach().cpu()

        batched_datasets = self.model.get_preprocessed_datasets(
            X_cpu, y_cpu, splitter, max_data_size=self.max_data_size
        )

        dl = DataLoader(
            batched_datasets,
            batch_size=1,
            collate_fn=meta_dataset_collator,
            pin_memory=(self.device == "cuda"),
        )

        batch = next(iter(dl))
        X_train, X_val, y_train, y_val, cat_ixs, confs = batch

        return {
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
            "cat_ixs": cat_ixs,
            "confs": confs,
        }

    # ------------------------------------------------------------
    # SGLD loop
    # ------------------------------------------------------------
    def _perform_sgld_sampling(
        self,
        X_sgld: torch.Tensor,
        noise_tensor: torch.Tensor,
        sgld_step_size: float,
        sgld_noise_std: float,
        sgld_steps: int,
        debug: bool,
    ) -> torch.Tensor:
        X_list = [X_sgld]

        for t in range(sgld_steps):
            if X_list[0].grad is not None:
                X_list[0].grad.zero_()

            logits = self.model.forward(X_list, return_logits=True)
            energy = self.compute_energy(logits.reshape(logits.shape[-1], -1))
            total_energy = energy.sum() / X_list[0].shape[1]

            total_energy.backward()

            if debug and t % 10 == 0:
                print(f"Step {t}: energy={total_energy.item():.4f}")

            with torch.no_grad():
                X_new = X_list[0] - sgld_step_size * X_list[0].grad + sgld_noise_std * noise_tensor[t]
                X_list[0] = X_new.requires_grad_(True)

        return X_list[0]

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    @staticmethod
    def compute_energy(
        logits: Union[torch.Tensor, np.ndarray],
        return_unnormalized_prob: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(logits, torch.Tensor):
            energy = -torch.logsumexp(logits, dim=1)
            return torch.exp(-energy) if return_unnormalized_prob else energy
        if isinstance(logits, np.ndarray):
            energy = -scipy.special.logsumexp(logits, axis=1)
            return np.exp(-energy) if return_unnormalized_prob else energy
        raise ValueError("logits must be torch.Tensor or np.ndarray")

    @staticmethod
    def add_surrogate_negative_samples(
        X: Union[np.ndarray, torch.Tensor],
        distance_negative_class: float = 5.0,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        num_features = X.shape[1]

        if num_features == 2:
            surrogate_negatives = [
                [-distance_negative_class, -distance_negative_class],
                [distance_negative_class, distance_negative_class],
                [-distance_negative_class, distance_negative_class],
                [distance_negative_class, -distance_negative_class],
            ]
        else:
            surrogate_set = set()
            while len(surrogate_set) < 4:
                pt = np.random.choice([-distance_negative_class, distance_negative_class], num_features)
                surrogate_set.add(tuple(pt))
                surrogate_set.add(tuple(-np.array(pt)))
            surrogate_negatives = list(surrogate_set)

        num_surrogates = len(surrogate_negatives)

        if isinstance(X, np.ndarray):
            X_sur = np.array(surrogate_negatives, dtype=X.dtype)
            X_ebm = np.concatenate([X, X_sur], axis=0)
            y_ebm = np.concatenate(
                [np.zeros(X.shape[0], dtype=np.int64), np.ones(num_surrogates, dtype=np.int64)]
            )
            return X_ebm, y_ebm

        if isinstance(X, torch.Tensor):
            X_sur = torch.tensor(surrogate_negatives, dtype=X.dtype, device=X.device)
            X_ebm = torch.cat([X, X_sur], dim=0)
            y_ebm = torch.cat(
                [
                    torch.zeros(X.shape[0], dtype=torch.long, device=X.device),
                    torch.ones(num_surrogates, dtype=torch.long, device=X.device),
                ],
                dim=0,
            )
            return X_ebm, y_ebm

        raise ValueError("X must be np.ndarray or torch.Tensor")

    @staticmethod
    def train_test_split_allow_full_train(
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        test_size: Optional[float] = None,
        train_size: Optional[float] = None,
        random_state: Optional[int] = None,
        shuffle: bool = True,
        stratify: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        full_train_mode = (test_size == 0)
        if full_train_mode:
            test_size = None

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

        if full_train_mode:
            X_train = X
            y_train = y

        return X_train, X_val, y_train, y_val
