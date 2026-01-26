import pandas as pd
import numpy as np
import os
import pickle
from typing import Union, Optional
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from katabatic.models.base_model import Model as BaseModel

class TVAE(BaseModel):
    def __init__(self, epochs=300, batch_size=500, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.metadata = None
        self.target_col = None
        self.train_data = None  # Store for class recovery

    def train(self, X: Union[pd.DataFrame, str], y: Optional[Union[pd.Series, pd.DataFrame]] = None, **kwargs):
        # 1. Load Data
        if isinstance(X, str):
            print(f"Loading TVAE data from: {X}")
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

        # Store training data to handle mode collapse in rare classes later
        self.train_data = data

        # 2. Detect Metadata
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data)

        # 3. Initialize & Train
        self.model = TVAESynthesizer(
            metadata=self.metadata,
            epochs=kwargs.get('epochs', self.epochs),
            batch_size=kwargs.get('batch_size', self.batch_size)
        )
        
        print(f"Training TVAE on {len(data)} rows...")
        self.model.fit(data)

        # 4. Generate & Save Artifacts
        synthetic_dir = kwargs.get('synthetic_dir')
        if synthetic_dir:
            n_samples = kwargs.get('n_samples', len(data))
            print(f"Generating synthetic data to: {synthetic_dir}")
            os.makedirs(synthetic_dir, exist_ok=True)
            
            synth_df = self.sample(n_samples)

            if not self.target_col:
                fairness_cfg = kwargs.get('fairness_config', {})
                self.target_col = fairness_cfg.get('Y')

            if self.target_col and self.target_col in synth_df.columns:
                y_synth = synth_df[self.target_col]
                x_synth = synth_df.drop(columns=[self.target_col])
                x_synth.to_csv(os.path.join(synthetic_dir, 'x_synth.csv'), index=False)
                y_synth.to_csv(os.path.join(synthetic_dir, 'y_synth.csv'), index=False)
            else:
                synth_df.to_csv(os.path.join(synthetic_dir, 'synthetic.csv'), index=False)
            
            self.model.save(filepath=os.path.join(synthetic_dir, 'tvae_model.pkl'))
            print("Saved artifacts.")

    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        
        synth = self.model.sample(num_rows=n_samples)

        # class recovery for model collapse
        if self.train_data is not None and self.target_col is not None:
            real_classes = self.train_data[self.target_col].unique()
            synth_classes = synth[self.target_col].unique()
            
            missing_classes = set(real_classes) - set(synth_classes)
            
            if missing_classes:
                print(f"Warning: Mode collapse detected. Missing classes: {missing_classes}. Injecting recovery rows.")
                recovery_rows = []
                for cls in missing_classes:
                    # Get one row from training data for this missing class
                    row = self.train_data[self.train_data[self.target_col] == cls].iloc[[0]]
                    recovery_rows.append(row)
                
                if recovery_rows:
                    synth = pd.concat([synth] + recovery_rows, ignore_index=True)

        return synth

    def evaluate(self, X, y=None, **kwargs):
        return {}