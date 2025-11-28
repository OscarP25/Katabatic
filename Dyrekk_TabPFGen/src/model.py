import os
import sys
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from typing import Any, Optional, Union, List, Tuple
from katabatic.models.base_model import Model

class TabPFGen(Model):
    def __init__(
        self,
        n_steps: int = 1000,
        step_size: float = 0.01,
        noise_scale: float = 0.01,
        device: str = "auto"
    ):
        """
        ===========================================================
        Initialize TabPFGen with initialize SGLD parameters
        ===========================================================
        Args:
        ========
        - n_steps : int 
            number of SGLD steps (Default: 1000)
        - step_size : float
            step size for SGLD update (Default: 0.01)
        - noise_scale : float
            noise scale for SGLD update (Default 0.01)
        - device : str, torch.device
            computation device (Default: "auto"), 
            If `"auto"`, the device is `"cuda"` if available, otherwise `"cpu"`.
        """
        super().__init__()

    def _infer_device(self, device: str | torch.device | None) -> torch.device:
        """
        ===========================================================
        Infer the device and data type from the given device string.
        ===========================================================
        Args:
        ========
        - device : str
            The device to infer the type from.

        Returns:
        =========
        - device : torch.device
            The inferred device
        """
        if (device is None) or (isinstance(device, str) and device == "auto"):
            device_type_ = "cuda" if torch.cuda.is_available() else "cpu"
            return torch.device(device_type_)
        if isinstance(device, str):
            return torch.device(device)
        if isinstance(device, torch.device):
            return device
        raise ValueError(f"Invalid device: {device}")

    def _compute_energy(
        self,
        x_synth: torch.Tensor,
        y_synth: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        """
        ===========================================================
        Compute differentiable energy score using surrogate network
        ===========================================================
        """
        # Use difference from training samples as a proxy for energy
        distances = torch.cdist(x_synth, x_train)
        min_distances, _ = distances.min(dim=1)

        # Add class-conditional term
        class_mask = y_synth.unsqueeze(1) == y_train.unsqueeze(0)
        class_distances = distances * class_mask.float()
        class_distances = class_distances.sum(dim=1) / (
            class_mask.float().sum(dim=1) + 1e-6
        )

        # Combine terms
        energy = min_distances + class_distances
        return energy

    def _sgld_step(
        self,
        x_synth: torch.Tensor,
        y_synth: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one SGLD step
        """
        x_synth = x_synth.clone().detach().requires_grad_(True)

        # Compute energy and its gradient
        energy = self._compute_energy(x_synth, y_synth, x_train, y_train)
        energy_sum = energy.sum()

        # Compute gradients with allow_unused=True
        grad = torch.autograd.grad(
            energy_sum,
            x_synth,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )[0]

        if grad is None:
            grad = torch.zeros_like(x_synth)

        # Update using gradients and noise
        noise = torch.randn_like(x_synth) * np.sqrt(2 * self.sgld_step_size)
        x_synth_new = (
            x_synth - self.sgld_step_size * grad + self.sgld_noise_scale * noise
        )

        return x_synth_new
    def balance_dataset(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        target_per_class: Optional[int] = None,
        min_class_size: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Balance dataset by generating synthetic samples for underrepresented classes.

        Note:
            The final class distribution may be approximately balanced rather than
            perfectly balanced due to TabPFN's label refinement process, which
            prioritizes data quality over exact class counts. Classes smaller than
            min_class_size are excluded from synthetic generation but remain in
            the combined dataset.

        Args:
        =========
        - X_train: np.ndarray
            Input features for training, shape (n_samples, n_features)
        - y_train: np.ndarray
            Target labels for training, shape (n_samples,)
        - target_per_class: Optional[int]
            Target number of samples per class. If None, uses majority class size.
        - min_class_size: int
            Minimum class size to include in balancing (Default: 5)

        Returns:
        =========
        - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            X_synthetic, y_synthetic, X_combined, y_combined
        """
        pass

    def generate_classification(
        self,
        X_train: np.ndarray, 
        y_train: np.ndarray,
        n_samples:int,
        balance_classes: bool =True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        =======================================
        Generate synthetic classification data.
        =======================================
        
        Algorithm:
        =========
        Scaling -> Initialize Synthetic Data -> GLD Refinement
        -> TabPFN Labeling -> Unscaling
        
        Args:
        =========
        - X_train: np.ndarray
            Input features for training, shape (n_samples, n_features)
        - y_train: np.ndarray
            Target labels for training, shape (n_samples,)
        - n_samples: int
            Number of synthetic samples to generate
        - balance_classes: bool
            Whether to balance classes in synthetic data (Default: True)


        Returns: 
        =========
        - Tuple[np.ndarray, np.ndarray]: Tuple of synthetic features and labels
        """

        # ==============================================================
        # 1. Scaling
        # ==============================================================
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X_train)

        # Convert to Tensor
        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)
        y_train = torch.tensor(y_train, device=self.device)



        # ==============================================================
        # 2. Initialize synthetic seeds
        # ==============================================================

        if balance_classes:
            classes = np.unique(y_train.cpu().numpy())
            n_per_class = n_samples // len(classes)
            x_synth_list = []
            y_synth_list = []

            for cls in classes:
                idx = np.where(y_train.cpu().numpy() == cls)[0]
                sample_idx = np.random.choice(idx, size=n_per_class)
                x_init = (
                    x_train[sample_idx]
                    + torch.randn(n_per_class, X_train.shape[1], device=self.device)
                    * 0.01
                )
                y_init = torch.full((n_per_class,), cls, device=self.device)

                x_synth_list.append(x_init)
                y_synth_list.append(y_init)
            
            x_synth = torch.cat(x_synth_list, dim=0)
            y_synth = torch.cat(y_synth_list, dim=0)

        else:
            x_synth = (
                torch.randn(n_samples, X_train.shape[1], device=self.device) * 0.01
            )
            y_synth = torch.randint(
                0, len(np.unique(y_train)), (n_samples,), device=self.device
            )

        # ==============================================================
        # 3. SGLD REFINEMENT
        # ==============================================================
        
        for step in range(self.n_sgld_steps):
            x_synth = self._sgld_step(x_synth, y_synth, x_train, y_train)

            if step % 100 == 0:
                print(f"Step {step}/{self.n_sgld_steps}")

        # ==============================================================
        # 4. TabPFN LABELING
        # ==============================================================
        
        # Generate final samples using TabPFN
        x_synth_np = x_synth.detach().cpu().numpy()
        x_train_np = x_train.cpu().numpy()
        y_train_np = y_train.cpu().numpy()

        # Fit TabPFN classifier
        clf = TabPFNClassifier(device=self.device)
        clf.fit(x_train_np, y_train_np)
        probs = clf.predict_proba(x_synth_np)

        # Refine labels based on TabPFN predictions
        y_synth = torch.tensor(probs.argmax(axis=1), device=self.device)

        # ==============================================================
        # 5. Inverse scale
        # ==============================================================
        
        # Convert back to numpy and inverse transform
        X_synth = self.scaler.inverse_transform(x_synth.detach().cpu().numpy())
        y_synth = y_synth.cpu().numpy()

        return X_synth, y_synth

    def generate_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_samples: int,
        use_quantiles: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ==========================================
        Generate synthetic samples for regression.
        ==========================================
        Args:
        =========
        - X_train: np.ndarray
            Input features for training, shape (n_samples, n_features)
        - y_train: np.ndarray
            Target values for training, shape (n_samples,)
        - n_samples: int
            Number of synthetic samples to generate
        - use_quantiles: bool
                Whether to use quantile regression for synthetic data (Default: True)

        Returns:
        =========
        - Tuple[np.ndarray, np.ndarray]: Tuple of synthetic features and target values
        """

        # ==============================================================
        # 1. Scaling
        # ==============================================================

        X_scaled = self.scaler.fit_transform(X_train)
        y_mean, y_std = y_train.mean(), y_train.std()
        y_scaled = (y_train - y_mean) / y_std

        # Convert to tensor
        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)

        n_strata = 10
        y_strata = np.quantile(y_train, np.linspace(0, 1, n_strata + 1))
        x_synth_list = []
        samples_per_stratum = n_samples // n_strata

        for i in range(n_strata):
            # Get indices for this stratum
            mask = (y_train >= y_strata[i]) & (y_train <= y_strata[i + 1])
            stratum_indices = np.where(mask)[0]
            
            if len(stratum_indices) > 0:
                # Sample indices with replacement if needed
                sampled_indices = np.random.choice(
                    stratum_indices, size=samples_per_stratum
                )
                x_stratum = X_scaled[sampled_indices]

                # Add noise scaled by local variance
                stratum_std = np.std(x_stratum, axis=0)
                noise = np.random.normal(
                    0, stratum_std * 0.1, (samples_per_stratum, X_train.shape[1])
                )
                x_synth_list.append(x_stratum + noise)

        x_synth_init = np.vstack(x_synth_list)
        x_synth = torch.tensor(x_synth_init, device=self.device, dtype=torch.float32)

        pass




    def train():
        pass

    def sample():
        pass

    def evaluate(self):
        return 0.0