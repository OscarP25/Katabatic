import pandas as pd
import numpy as np  # Required for noise generation
import torch
import os
from typing import Optional, Dict, Union
from katabatic.models.base_model import Model as BaseModel
from katabatic.models.tabfairgan_alex.models import TFG

class KatabaticTabFairGAN(BaseModel):
    def __init__(
        self, 
        epochs: int = 100, 
        batch_size: int = 256, 
        fairness_config: Optional[Dict] = None, 
        device: str = None,
        **kwargs
    ):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.fairness_config = fairness_config or {}
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model: Optional[TFG] = None
        self.fitted = False
        self._dummy_col = "__dummy_noise__"  # Internal tag

    def train(self, X: Union[pd.DataFrame, str], y: Optional[Union[pd.Series, pd.DataFrame]] = None, **kwargs):
        """
        1. Loads data.
        2. Trains TFG.
        3. Generates AND splits synthetic data for TSTREvaluation (x_synth.csv, y_synth.csv).
        """
        # --- 1. Data Loading ---
        if isinstance(X, str):
            train_dir = X
            print(f"Loading training data from: {train_dir}")
            try:
                # skipinitialspace=True fixes " sex" -> "sex"
                X_df = pd.read_csv(os.path.join(train_dir, 'x_train.csv'), skipinitialspace=True)
                y_df = pd.read_csv(os.path.join(train_dir, 'y_train.csv'), skipinitialspace=True)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Pipeline failed to generate training files. {str(e)}")
            
            if y_df.shape[1] == 1:
                y_df = y_df.iloc[:, 0]
            
            self.fit(X_df, y_df, **kwargs)
        else:
            self.fit(X, y, **kwargs)

        # --- 2. Auto-Generation for Pipeline Support ---
        synthetic_dir = kwargs.get('synthetic_dir')
        if synthetic_dir:
            print(f"Pipeline mode: Generating split synthetic artifacts to {synthetic_dir}")
            
            # Generate samples (n_samples defaults to training set size)
            n_samples = kwargs.get('n_samples', len(self.model.df))
            synthetic_df = self.sample(n_samples)
            
            # Ensure directory exists
            os.makedirs(synthetic_dir, exist_ok=True)
            
            # --- CRITICAL FIX: Split X and y for TSTREvaluation ---
            target_col = self.fairness_config.get('Y', 'target')
            
            if target_col in synthetic_df.columns:
                y_synth = synthetic_df[target_col]
                x_synth = synthetic_df.drop(columns=[target_col])
                
                # Save as specific files expected by TSTREvaluation
                x_synth.to_csv(os.path.join(synthetic_dir, 'x_synth.csv'), index=False)
                y_synth.to_csv(os.path.join(synthetic_dir, 'y_synth.csv'), index=False)
                print(f"Saved split artifacts: x_synth.csv, y_synth.csv ({len(x_synth)} rows)")
            else:
                # Fallback if column name mismatch (unlikely with our previous force-rename fix)
                print(f"Warning: Target '{target_col}' not found in synthetic data. Saving as single file.")
                synthetic_df.to_csv(os.path.join(synthetic_dir, 'synthetic.csv'), index=False)

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, pd.DataFrame]] = None, **kwargs) -> None:
        # 1. Config Updates
        self.epochs = kwargs.get('epochs', self.epochs)
        self.batch_size = kwargs.get('batch_size', self.batch_size)
        self.fairness_config = kwargs.get('fairness_config', self.fairness_config)
        self.device = kwargs.get('device', self.device)

        # 2. Data Preparation
        data = X.copy()
        data.columns = data.columns.str.strip()
        
        # FIX A: Cast to String (Fixes TFG missing discrete cols)
        data = data.astype(str)

        # FIX B: Inject Dummy Continuous Column (Fixes TFG QuantileTransformer crash)
        # We add random noise so TFG finds exactly 1 continuous column.
        data[self._dummy_col] = np.random.randn(len(data))

        # 3. Merge Target
        if y is not None:
            target_col = self.fairness_config.get('Y', 'target')
            
            if isinstance(y, pd.Series):
                data[target_col] = y.astype(str).values
            elif isinstance(y, pd.DataFrame):
                data[target_col] = y.iloc[:, 0].astype(str).values
            else:
                data[target_col] = str(y)

        # 4. Initialize & Train
        self.model = TFG(
            df=data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            device=self.device,
            fairness_config=self.fairness_config
        )

        print(f"Training TabFairGAN on {self.device} with {len(data.columns)} columns (including dummy)...")
        self.model.train() #
        self.fitted = True

    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        if not self.fitted or self.model is None:
            raise RuntimeError("Model must be fitted before sampling.")
            
        # 1. Generate (Includes the dummy column)
        synthetic_data = self.model.generate_fake_df(n_samples)
        
        # 2. Cleanup: Remove the dummy noise column before returning
        if self._dummy_col in synthetic_data.columns:
            synthetic_data = synthetic_data.drop(columns=[self._dummy_col])
            
        return synthetic_data

    def evaluate(self, X, y=None, **kwargs): return {}
    
    def save(self, path): 
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.generator.state_dict(), path)
        
    def load(self, path): 
        # Note: This load is partial; it won't restore the dummy column logic 
        # unless fit() is called again.
        self.model.generator.load_state_dict(torch.load(path))
        self.model.generator.eval()
        self.fitted = True