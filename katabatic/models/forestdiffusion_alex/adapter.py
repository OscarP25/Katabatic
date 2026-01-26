import pandas as pd
import numpy as np
import os
import pickle
from typing import Union, Optional
from ForestDiffusion import ForestDiffusionModel
from katabatic.models.base_model import Model as BaseModel

class ForestDiffusion(BaseModel):
    def __init__(self, n_t=50, diffusion_type='vp', **kwargs):
        """
        n_t: Number of diffusion steps (default 50 is usually sufficient for ForestDiffusion)
        diffusion_type: 'vp' (Variance Preserving) or 'flow' (Flow Matching)
        """
        super().__init__()
        self.n_t = n_t
        self.diffusion_type = diffusion_type
        self.model = None
        self.target_col = None
        self.train_data = None
        self.columns = None
        # ForestDiffusion specific metadata
        self.cat_indexes = []
        self.int_indexes = []

    def train(self, X: Union[pd.DataFrame, str], y: Optional[Union[pd.Series, pd.DataFrame]] = None, **kwargs):
        # 1. Load Data
        if isinstance(X, str):
            print(f"Loading ForestDiffusion data from: {X}")
            try:
                X_df = pd.read_csv(os.path.join(X, 'x_train.csv'), skipinitialspace=True)
                y_df = pd.read_csv(os.path.join(X, 'y_train.csv'), skipinitialspace=True)
            except FileNotFoundError:
                raise FileNotFoundError(f"Pipeline artifacts missing in {X}")
            
            if isinstance(y_df, pd.DataFrame) and y_df.shape[1] == 1:
                self.target_col = y_df.columns[0]
                data = pd.concat([X_df, y_df], axis=1)
            else:
                data = X_df.copy()
                self.target_col = 'target'
                data[self.target_col] = y_df.values
        else:
            data = X.copy()
            if y is not None:
                if isinstance(y, pd.Series):
                    self.target_col = y.name or 'target'
                    data[self.target_col] = y
                else:
                    self.target_col = y.columns[0]
                    data[self.target_col] = y.iloc[:,0]

        # Store metadata for recovery/export
        self.train_data = data
        self.columns = data.columns
        
        # 2. Pre-process for ForestDiffusion
        # It requires numpy input and indices for categorical/integer columns
        data_np = data.to_numpy()
        
        # Identify column types automatically
        self.cat_indexes = []
        self.int_indexes = []
        
        for idx, col in enumerate(data.columns):
            # If object or category, treat as categorical
            if pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_categorical_dtype(data[col]):
                self.cat_indexes.append(idx)
            # If integer, track it (ForestDiffusion handles rounding)
            elif pd.api.types.is_integer_dtype(data[col]):
                self.int_indexes.append(idx)

        # 3. Initialize & Train
        print(f"Training ForestDiffusion on {len(data)} rows (n_t={self.n_t})...")
        
        # FIX: Pass the lists directly. Do NOT convert to None if empty.
        self.model = ForestDiffusionModel(
            data_np, 
            label_y=None, 
            n_t=kwargs.get('n_t', self.n_t),
            diffusion_type=kwargs.get('diffusion_type', self.diffusion_type),
            duplicate_K=100, 
            cat_indexes=self.cat_indexes, # Changed: Always pass list
            int_indexes=self.int_indexes, # Changed: Always pass list
            n_jobs=-1
        )
        
        # 4. Generate & Save Artifacts
        synthetic_dir = kwargs.get('synthetic_dir')
        if synthetic_dir:
            n_samples = kwargs.get('n_samples', len(data))
            print(f"Generating synthetic data to: {synthetic_dir}")
            os.makedirs(synthetic_dir, exist_ok=True)
            
            synth_df = self.sample(n_samples)

            # Recover target col from config if missing
            if not self.target_col:
                fairness_cfg = kwargs.get('fairness_config', {})
                self.target_col = fairness_cfg.get('Y')

            # Split artifacts
            if self.target_col and self.target_col in synth_df.columns:
                y_synth = synth_df[self.target_col]
                x_synth = synth_df.drop(columns=[self.target_col])
                x_synth.to_csv(os.path.join(synthetic_dir, 'x_synth.csv'), index=False)
                y_synth.to_csv(os.path.join(synthetic_dir, 'y_synth.csv'), index=False)
            else:
                synth_df.to_csv(os.path.join(synthetic_dir, 'synthetic.csv'), index=False)
            
            # Save Model Pickle
            with open(os.path.join(synthetic_dir, 'forestdiffusion_model.pkl'), 'wb') as f:
                pickle.dump(self.model, f)
            print("Saved artifacts.")

    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        # Generate raw numpy array
        synth_np = self.model.generate(batch_size=n_samples)
        
        # Convert back to DataFrame
        synth_df = pd.DataFrame(synth_np, columns=self.columns)

        # Fix Data Types
        for idx in self.int_indexes:
            col_name = self.columns[idx]
            synth_df[col_name] = synth_df[col_name].round().astype(int)

        # --- Class Recovery ---
        if self.train_data is not None and self.target_col is not None:
            # Safely handle mixed types by converting to string for comparison
            real_classes = self.train_data[self.target_col].unique()
            synth_classes = synth_df[self.target_col].unique()
            
            missing_classes = set(map(str, real_classes)) - set(map(str, synth_classes))
            
            if missing_classes:
                print(f"Warning: Mode collapse detected. Missing classes: {missing_classes}. Injecting recovery rows.")
                recovery_rows = []
                for cls_str in missing_classes:
                    mask = self.train_data[self.target_col].astype(str) == cls_str
                    if mask.any():
                        row = self.train_data[mask].iloc[[0]]
                        recovery_rows.append(row)
                
                if recovery_rows:
                    synth_df = pd.concat([synth_df] + recovery_rows, ignore_index=True)

        return synth_df

    def evaluate(self, X, y=None, **kwargs):
        return {}