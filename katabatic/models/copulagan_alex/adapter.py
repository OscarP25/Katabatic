import pandas as pd
import os
from typing import Union, Optional
from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
from katabatic.models.base_model import Model as BaseModel

class CopulaGAN(BaseModel):
    def __init__(self, epochs=300, batch_size=500, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.metadata = None
        self.target_col = None

    def train(self, X: Union[pd.DataFrame, str], y: Optional[Union[pd.Series, pd.DataFrame]] = None, **kwargs):
        # 1. Load Data
        if isinstance(X, str):
            print(f"Loading CopulaGAN data from: {X}")
            try:
                X_df = pd.read_csv(os.path.join(X, 'x_train.csv'), skipinitialspace=True)
                y_df = pd.read_csv(os.path.join(X, 'y_train.csv'), skipinitialspace=True)
            except FileNotFoundError:
                raise FileNotFoundError(f"Pipeline artifacts missing in {X}")
            
            # Combine for SDV (Single Table)
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

        # 2. Detect metadata
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data)

        # 3. Initialise & train
        self.model = CopulaGANSynthesizer(
            metadata=self.metadata,
            epochs=kwargs.get('epochs', self.epochs),
            batch_size=kwargs.get('batch_size', self.batch_size),
            verbose=True
        )
        
        print(f"Training CopulaGAN on {len(data)} rows...")
        self.model.fit(data)

        # 4. Generate & save artifacts
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

            # Split X/Y
            if self.target_col and self.target_col in synth_df.columns:
                y_synth = synth_df[self.target_col]
                x_synth = synth_df.drop(columns=[self.target_col])
                x_synth.to_csv(os.path.join(synthetic_dir, 'x_synth.csv'), index=False)
                y_synth.to_csv(os.path.join(synthetic_dir, 'y_synth.csv'), index=False)
            else:
                synth_df.to_csv(os.path.join(synthetic_dir, 'synthetic.csv'), index=False)
            
            self.model.save(filepath=os.path.join(synthetic_dir, 'copulagan_model.pkl'))
            print("Saved artifacts.")

    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.sample(num_rows=n_samples)

    def evaluate(self, X, y=None, **kwargs):
        return {}