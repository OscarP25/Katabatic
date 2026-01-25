from pathlib import Path
import pandas as pd

from katabatic.models.base_model import Model
from katabatic.models.ctabgan_plus.models import CTABGANPlus


class CTABGANPlusAdapter(Model):
    """
    CTAB-GAN+ Adapter — ADULT (FINAL, SAFE, VERIFIED)
    
    Notes:
    - split_dataset renames the label column to 'class'
    - categorical columns are passed as COLUMN INDICES
    - label column MUST be included in categorical list
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.model = CTABGANPlus(config=config)

    def train(self, dataset_dir: str, synthetic_dir: str, **kwargs):
        dataset_dir = Path(dataset_dir)

       
        # Load train split
      
        x_train = pd.read_csv(dataset_dir / "x_train.csv")
        y_train = pd.read_csv(dataset_dir / "y_train.csv")

        # Combine X + y (CTAB-GAN+ trains on full table)
        df_train = pd.concat([x_train, y_train], axis=1)

        
        # ADULT categorical columns (BY NAME)
        # Label column AFTER split_dataset is 'class'
        
        categorical_cols = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
            "class"          
        ]

        # Convert names → column indices
        categorical_idx = [
            df_train.columns.get_loc(col)
            for col in categorical_cols
        ]

        
        # Train CTAB-GAN+
        
        self.model.train(
            dataset_dir=str(dataset_dir),
            synthetic_dir=str(synthetic_dir),
            categorical=categorical_idx
        )

        return self

    
    # Required abstract methods
  
    def sample(self, n: int):
        raise NotImplementedError(
            "CTAB-GAN+ sampling is handled inside train()"
        )

    def evaluate(self):
        raise NotImplementedError(
            "CTAB-GAN+ evaluation is handled by TSTR pipeline"
        )
