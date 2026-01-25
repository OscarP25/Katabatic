import pandas as pd
import numpy as np

from katabatic.models.base_model import Model
from katabatic.models.tabkde.models import TabKDEModel


class TabKDEAdapter(Model):
    """
    Katabatic adapter for TabKDE (unconditional KDE-based model).

    Design:
    - Trains on full table (X + y)
    - Samples full table
    - Splits X / y AFTER sampling
    - No dataset-specific logic
    - No hyperparameter tuning
    - Algorithm unchanged
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.label_col = None

    def train(self, output_dir: str, label_col: str = None):
      
        # Load training splits
        x_train = pd.read_csv(f"{output_dir}/x_train.csv")
        y_train_df = pd.read_csv(f"{output_dir}/y_train.csv")

        # Auto-detect label column
        if label_col is None:
            label_col = y_train_df.columns[0]

        self.label_col = label_col
        y_train = y_train_df[label_col]

        
        # Merge X + y (TabKDE expects full table)
        train_df = x_train.copy()
        train_df[self.label_col] = y_train.values

        
        # Fit TabKDE (paper defaults, unchanged)
        self.model = TabKDEModel()
        self.model.fit(train_df)

        self.is_fitted = True
        return self

    def sample(self, n_samples: int):
        if not self.is_fitted:
            raise RuntimeError("TabKDEAdapter must be trained before sampling.")

        
        # Sample full table
        synth_df, _ = self.model.sample(n_samples)

      
        # Split X / y
        y_synth = synth_df[self.label_col].values
        x_synth = synth_df.drop(columns=[self.label_col])

        return x_synth.to_numpy(), np.asarray(y_synth)

    def evaluate(self, *args, **kwargs):
        # Katabatic handles evaluation
        return None