from __future__ import annotations

import os
import json
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model as BaseModel


class MAFModel(BaseModel):
    """
    Class-Conditional Masked Autoregressive Flow (MAF)

    - Trains one MAF per class: p(X | y)
    - Samples y based on observed label proportions
    - Generates X conditionally using the correct flow
    - Saves x_synth.csv and y_synth.csv into synthetic_dir
    """

    def __init__(
        self,
        epochs: int = 60,
        batch_size: int = 512,
        lr: float = 1e-3,
        hidden_features: int = 128,
        num_transforms: int = 5,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_features = hidden_features
        self.num_transforms = num_transforms
        self.device = device
        self.seed = seed

        # Learned state
        self.is_fitted = False
        self.label_col: Optional[str] = None
        self.feature_columns: List[str] = []
        self.label_probs: Dict[int, float] = {}   # keys are ENCODED classes (0..K-1)

        self.flows: Dict[int, Any] = {}           # one flow per ENCODED class
        self.mean: Dict[int, np.ndarray] = {}     # per ENCODED class
        self.std: Dict[int, np.ndarray] = {}      # per ENCODED class

        # Needed to save correct ORIGINAL labels in y_synth.csv
        self._label_encoder = None

 
    # TRAIN
   
    def train(
        self,
        data_dir: str,
        label_col: str = "target",
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> str:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from torch import optim

        from sklearn.preprocessing import LabelEncoder
        from nflows.flows.base import Flow
        from nflows.distributions.normal import StandardNormal
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.permutations import ReversePermutation
        from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform

        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        # Load training data
     
        train_full = os.path.join(data_dir, "train_full.csv")
        df = pd.read_csv(train_full)

        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in train_full.csv")

        self.label_col = label_col

        X = df.drop(columns=[label_col]).copy()
        y = df[label_col].copy()

        # Ensure numeric (MAF requires numeric)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        self.feature_columns = list(X.columns)

    
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))

        self._label_encoder = le  # ✅ store for inverse_transform

        classes = np.unique(y_enc)

        # Save label probabilities in encoded space
        counts = np.bincount(y_enc)
        probs = counts / counts.sum()
        self.label_probs = {int(cls): float(probs[int(cls)]) for cls in classes}

      
        # Train one flow per encoded class
        
        for cls in classes:
            idx = np.where(y_enc == cls)[0]
            Xc = X.iloc[idx].to_numpy(dtype=np.float32)

            # Per-class normalisation
            mu = Xc.mean(axis=0)
            sd = Xc.std(axis=0) + 1e-6

            self.mean[int(cls)] = mu
            self.std[int(cls)] = sd

            Xn = (Xc - mu) / sd

            data = torch.tensor(Xn, dtype=torch.float32)
            loader = DataLoader(
                TensorDataset(data),
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
            )

            # Build MAF (stacked transforms)
            transforms = []
            for _ in range(self.num_transforms):
                transforms.append(ReversePermutation(features=data.shape[1]))
                transforms.append(
                    MaskedAffineAutoregressiveTransform(
                        features=data.shape[1],
                        hidden_features=self.hidden_features,
                    )
                )

            transform = CompositeTransform(transforms)
            base_dist = StandardNormal([data.shape[1]])
            flow = Flow(transform, base_dist).to(device)

            optimizer = optim.Adam(flow.parameters(), lr=self.lr)

            flow.train()
            for _ in range(self.epochs):
                for (batch,) in loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    loss = -flow.log_prob(batch).mean()
                    loss.backward()
                    optimizer.step()

            self.flows[int(cls)] = flow

        
        # Synthetic output directory
        
        if synthetic_dir is None:
            dataset_name = os.path.basename(os.path.normpath(data_dir))
            synthetic_dir = os.path.join("synthetic", dataset_name, "maf")

        os.makedirs(synthetic_dir, exist_ok=True)

       
        # Save synthetic data
       
        n = len(df)

        # (A) Sample labels in ENCODED space (0..K-1) using original distribution
        enc_classes = np.array(list(self.label_probs.keys()), dtype=int)
        enc_probs   = np.array(list(self.label_probs.values()), dtype=float)

        y_synth_enc = rng.choice(enc_classes, size=n, p=enc_probs)

        # (B) Convert labels back to ORIGINAL labels for y_synth.csv
        #     This fixes your "Accuracy ~ 0" problem.
        y_synth_orig = self._label_encoder.inverse_transform(y_synth_enc)

        # (C) FAST generation: sample X in batches per class (encoded)
        Xs = np.empty((n, len(self.feature_columns)), dtype=np.float32)

        with torch.no_grad():
            for cls in np.unique(y_synth_enc):
                idx = np.where(y_synth_enc == cls)[0]
                m = len(idx)
                if m == 0:
                    continue

                flow = self.flows[int(cls)]
                flow.eval()

                z = flow.sample(m)  # (m, D) on device
                z = z.detach().cpu().numpy().astype(np.float32)

                x = (z * self.std[int(cls)]) + self.mean[int(cls)]
                Xs[idx] = x.astype(np.float32)

        x_synth = pd.DataFrame(Xs, columns=self.feature_columns)
        y_synth = pd.DataFrame({self.label_col: y_synth_orig})

        x_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
        y_synth.to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)

        with open(os.path.join(synthetic_dir, "metadata.json"), "w") as f:
            json.dump(
                {
                    "model": "Class-Conditional MAF",
                    "label_col": self.label_col,
                    "feature_columns": self.feature_columns,
                    "label_probs_encoded_space": self.label_probs,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                    "hidden_features": self.hidden_features,
                    "num_transforms": self.num_transforms,
                    "device": str(device),
                    "seed": self.seed,
                },
                f,
                indent=2,
            )

        self.is_fitted = True
        return f"[MAF] Conditional synthetic data saved to {synthetic_dir}"

    
    # SAMPLE
  
    def sample(
        self,
        n: Optional[int] = None,
        conditional: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Returns only X samples (Katabatic may use this internally).
        """
        if not self.is_fitted:
            raise RuntimeError("Call train() before sample().")

        import torch

        n_samples = int(n) if n is not None else 1000
        rng = np.random.default_rng(self.seed)

        enc_classes = np.array(list(self.label_probs.keys()), dtype=int)
        enc_probs   = np.array(list(self.label_probs.values()), dtype=float)

        y_samples_enc = rng.choice(enc_classes, size=n_samples, p=enc_probs)

        Xs = np.empty((n_samples, len(self.feature_columns)), dtype=np.float32)

        with torch.no_grad():
            for cls in np.unique(y_samples_enc):
                idx = np.where(y_samples_enc == cls)[0]
                m = len(idx)
                if m == 0:
                    continue

                flow = self.flows[int(cls)]
                flow.eval()

                z = flow.sample(m)
                z = z.detach().cpu().numpy().astype(np.float32)

                x = (z * self.std[int(cls)]) + self.mean[int(cls)]
                Xs[idx] = x.astype(np.float32)

        return pd.DataFrame(Xs, columns=self.feature_columns)

    
    # EVALUATE 
   
    def evaluate(self, *args, **kwargs):
        """
        Required by BaseModel.
        Your evaluation is done in the notebook (TSTR),
        so this is just to satisfy the abstract interface.
        """
        return {
            "status": "MAF evaluate() placeholder (TSTR done externally)."
        }