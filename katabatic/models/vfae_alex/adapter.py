import pandas as pd
import numpy as np
import torch
import os
from typing import Union, Optional, Dict
from katabatic.models.base_model import Model as BaseModel
from .models import VFAE

class KatabaticVFAE(BaseModel):
    def __init__(self, epochs=50, batch_size=64, z_dim=50, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = None
        self.fitted = False
        
        # Internal storage for generation reconstruction
        self.x_cols = []
        self.s_col = None
        self.y_col = None
        self.s_data_np = None
        self.y_data_np = None
        
        # Store original feature order to align x_synth with x_test
        self.original_feature_order = None

    def train(self, X: Union[pd.DataFrame, str], y: Optional[Union[pd.Series, pd.DataFrame]] = None, **kwargs):
        """
        Loads data, trains VFAE, and generates split x_synth/y_synth files for TSTR.
        """
        # 1. Update Config
        self.epochs = kwargs.get('epochs', self.epochs)
        self.z_dim = kwargs.get('z_dim', self.z_dim)
        fairness_config = kwargs.get('fairness_config', {})

        # 2. Robust Data Loading
        if isinstance(X, str):
            print(f"Loading VFAE training data from: {X}")
            try:
                # skipinitialspace fixes " sex" -> "sex"
                X_df = pd.read_csv(os.path.join(X, 'x_train.csv'), skipinitialspace=True)
                y_df = pd.read_csv(os.path.join(X, 'y_train.csv'), skipinitialspace=True)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Pipeline artifacts missing in {X}. {e}")
            
            if y_df.shape[1] == 1:
                y_df = y_df.iloc[:, 0]
                
            self.fit(X_df, y_df, fairness_config=fairness_config)
        else:
            self.fit(X, y, fairness_config=fairness_config)

        # 3. Auto-Generate Split Artifacts for TSTR Evaluation
        synthetic_dir = kwargs.get('synthetic_dir')
        if synthetic_dir:
            print(f"Generating synthetic data to: {synthetic_dir}")
            os.makedirs(synthetic_dir, exist_ok=True)
            
            # Generate default amount (same as training size)
            n_samples = kwargs.get('n_samples', len(self.s_data_np))
            synth_df = self.sample(n_samples)
            
            # --- CRITICAL FIX: Split X and Y for Evaluator ---
            # self.y_col is guaranteed to be set by fit()
            if self.y_col and self.y_col in synth_df.columns:
                y_synth = synth_df[self.y_col]
                x_synth = synth_df.drop(columns=[self.y_col])
                
                # Save split files
                x_synth.to_csv(os.path.join(synthetic_dir, 'x_synth.csv'), index=False)
                y_synth.to_csv(os.path.join(synthetic_dir, 'y_synth.csv'), index=False)
                print(f"Saved artifacts: x_synth.csv ({x_synth.shape}), y_synth.csv ({y_synth.shape})")
            else:
                # Fallback (Should not happen if fit worked)
                print("Warning: Target column not found in synthetic data. Saving single file.")
                synth_df.to_csv(os.path.join(synthetic_dir, 'synthetic.csv'), index=False)

    def evaluate(self, X, y=None, **kwargs):
        """
        Satisfies BaseModel contract. Actual evaluation is done by Pipeline > TSTREvaluation.
        """
        return {}

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame], fairness_config: Dict):
        # 1. Clean Headers
        X = X.copy()
        X.columns = X.columns.str.strip()
        
        # Capture exact column order (features only) for reconstruction alignment
        self.original_feature_order = X.columns.tolist()
        
        # 2. Config Extraction
        s_col = fairness_config.get('S')
        y_col = fairness_config.get('Y', 'target') 

        if not s_col or s_col not in X.columns:
            raise ValueError(f"VFAE requires sensitive attribute '{s_col}' in feature columns.")

        # 3. Data Splitting
        s_data = X[[s_col]].values.astype(np.float32)
        
        if isinstance(y, pd.Series):
            y_data = y.values.reshape(-1, 1).astype(np.float32)
        else:
            y_data = y.values.astype(np.float32)

        x_cols = [c for c in X.columns if c != s_col]
        x_data = X[x_cols].values.astype(np.float32)

        # 4. Store Metadata
        self.x_cols = x_cols
        self.s_col = s_col
        self.y_col = y_col
        self.s_data_np = s_data
        self.y_data_np = y_data

        # 5. Initialize & Train
        print(f"Initializing VFAE (X:{x_data.shape[1]}, S:{s_data.shape[1]}, Y:{y_data.shape[1]})...")
        self.model = VFAE(
            x_dim=x_data.shape[1],
            s_dim=s_data.shape[1],
            y_dim=y_data.shape[1],
            z_dim=self.z_dim,
            device=self.device
        )

        self.model.train(
            x_data, s_data, y_data,
            epochs=self.epochs,
            batch_size=self.batch_size
        )
        self.fitted = True

    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        if not self.fitted or self.model is None:
            raise RuntimeError("VFAE must be fitted before sampling.")
            
        x_gen, s_gen, y_gen = self.model.generate(
            n_samples, 
            self.s_data_np, 
            self.y_data_np
        )
        
        # --- FIX: Sanitize NaNs/Infs before casting ---
        # Prevents "IntCastingNaNError" if model becomes unstable
        if np.isnan(x_gen).any() or np.isinf(x_gen).any():
            x_gen = np.nan_to_num(x_gen, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isnan(s_gen).any() or np.isinf(s_gen).any():
            s_gen = np.nan_to_num(s_gen, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isnan(y_gen).any() or np.isinf(y_gen).any():
            y_gen = np.nan_to_num(y_gen, nan=0.0, posinf=0.0, neginf=0.0)

        # Reconstruct DataFrame
        # 1. Create base frame with X (features sans S)
        df_gen = pd.DataFrame(np.round(x_gen), columns=self.x_cols)
        # 2. Add S and Y
        df_gen[self.s_col] = np.round(s_gen)
        df_gen[self.y_col] = np.round(y_gen)
        
        # --- FIX: Restore Original Column Order ---
        # Ensures x_synth matches x_test structure exactly (prevents "Feature names must match" error)
        if self.original_feature_order:
            ordered_cols = self.original_feature_order + [self.y_col]
            if all(c in df_gen.columns for c in ordered_cols):
                df_gen = df_gen[ordered_cols]
        
        return df_gen.astype(int)

    def save(self, path: str):
        if self.model:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.model.model.state_dict(), path)

    def load(self, path: str):
        pass