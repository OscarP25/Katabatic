import pandas as pd
import numpy as np
import torch
import os
from typing import Union, Optional
from torch.utils.data import DataLoader, TensorDataset
from katabatic.models.base_model import Model as BaseModel
from .models import FairTabDiffusion

class KatabaticFairTabDiffusion(BaseModel):
    def __init__(self, epochs=100, batch_size=256, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        
        # internal storage
        self.columns = []
        self.columns_x = []
        self.target_col = None
        self.y_dist = None
        self.y_values = None

    def train(self, X: Union[pd.DataFrame, str], y: Optional[Union[pd.Series, pd.DataFrame]] = None, **kwargs):
        """
        Orchestrates loading, training, and splitting artifacts.
        """
        # 1. Load Data
        if isinstance(X, str):
            print(f"Loading FairTabDiffusion data from: {X}")
            try:
                X_df = pd.read_csv(os.path.join(X, 'x_train.csv'), skipinitialspace=True)
                y_df = pd.read_csv(os.path.join(X, 'y_train.csv'), skipinitialspace=True)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Pipeline artifacts missing in {X}. {e}")
            
            if y_df.shape[1] == 1: 
                y_df = y_df.iloc[:, 0]
                
            self.fit(X_df, y_df, **kwargs)
        else:
            self.fit(X, y, **kwargs)

        # 2. Generate Artifacts for Evaluation
        synthetic_dir = kwargs.get('synthetic_dir')
        if synthetic_dir:
            print(f"Generating synthetic data to: {synthetic_dir}")
            os.makedirs(synthetic_dir, exist_ok=True)
            
            n_samples = kwargs.get('n_samples', 1000)
            synth_df = self.sample(n_samples)
            
            # Identify Target Column for Splitting
            target_name = self.target_col
            if not target_name:
                fairness_cfg = kwargs.get('fairness_config', {})
                target_name = fairness_cfg.get('Y') or kwargs.get('target_col', 'target')

            # Split and Save
            if target_name and target_name in synth_df.columns:
                y_synth = synth_df[target_name]
                x_synth = synth_df.drop(columns=[target_name])
                
                x_synth.to_csv(os.path.join(synthetic_dir, 'x_synth.csv'), index=False)
                y_synth.to_csv(os.path.join(synthetic_dir, 'y_synth.csv'), index=False)
                print(f"Saved artifacts: x_synth.csv, y_synth.csv ({len(synth_df)} rows)")
            else:
                print(f"Warning: Target '{target_name}' not found. Saving single file.")
                synth_df.to_csv(os.path.join(synthetic_dir, 'synthetic.csv'), index=False)

    def evaluate(self, X, y=None, **kwargs):
        """Satisfies BaseModel contract. Actual evaluation is done by Pipeline."""
        return {}

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame], **kwargs):
        # 1. Clean Data Headers
        X = X.copy()
        X.columns = X.columns.str.strip()
        
        # 2. Setup Target
        y_name = 'target'
        if isinstance(y, pd.Series): 
            y_name = y.name or 'target'
        elif isinstance(y, pd.DataFrame): 
            y_name = y.columns[0]
        
        self.target_col = y_name.strip()
        self.columns_x = X.columns.tolist()
        self.columns = self.columns_x + [self.target_col]
        
        # 3. Calculate Cardinalities (for Multinomial Diffusion)
        col_cardinalities = []
        for col in self.columns_x:
            max_val = int(X[col].max())
            col_cardinalities.append(max_val + 1)
            
        # 4. Prepare Tensors
        x_tensor = torch.from_numpy(X.values.astype(np.int64))
        y_tensor = torch.from_numpy(y.values.astype(np.int64))
        if y_tensor.ndim == 2:
            y_tensor = y_tensor.squeeze()

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 5. Initialise Model
        epochs = kwargs.get('epochs', self.epochs)
        print(f"Initialising FairTabDiffusion (Features:{len(self.columns_x)}, Epochs:{epochs})...")
        
        self.model = FairTabDiffusion(
            num_classes=col_cardinalities,
            input_dim=len(self.columns_x),
            device=self.device
        )
        
        # 6. Train
        self.model.train(loader, epochs=epochs)
        
        # 7. Store Label Distribution for Sampling
        if isinstance(y, (pd.Series, pd.DataFrame)):
            vals = y.values.flatten() if isinstance(y, pd.DataFrame) else y.values
            unique, counts = np.unique(vals, return_counts=True)
            self.y_values = unique
            self.y_dist = counts / counts.sum()
        else:
            unique, counts = np.unique(y, return_counts=True)
            self.y_values = unique
            self.y_dist = counts / counts.sum()

    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        if not self.model:
            raise RuntimeError("Model not fitted")
            
        # 1. Sample Conditions (Y) from empirical distribution
        y_samples = np.random.choice(self.y_values, size=n_samples, p=self.y_dist)
        
        # 2. Generate Features (X) given Y
        x_gen = self.model.sample(n_samples, y_samples)
        
        # 3. Construct DataFrame
        df_gen = pd.DataFrame(x_gen, columns=self.columns_x)
        df_gen[self.target_col] = y_samples
        
        return df_gen.astype(int)

    def save(self, path: str):
        if self.model:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)

    def load(self, path: str):
        pass