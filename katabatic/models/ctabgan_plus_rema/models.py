"""
CTAB-GAN+ model integration for the Katabatic framework.
"""

import os
import pandas as pd
import logging
from typing import List, Optional, Dict

from katabatic.models.base_model import Model
from katabatic.models.ctabgan_plus.data_prep import DataPrep
from katabatic.models.ctabgan_plus.synthesizer import CTABGANSynthesizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CTABGANPlus(Model):

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or {}
        self.synthesizer = CTABGANSynthesizer()
        self.data_prep = None

    def train(
        self,
        dataset_dir: str,
        synthetic_dir: str,
        categorical: Optional[List[int]] = None,
        **kwargs
    ):
        logger.info("=" * 80)
        logger.info("Training CTAB-GAN+")
        logger.info("=" * 80)

        
        # Load training data
        
        x_path = os.path.join(dataset_dir, "x_train.csv")
        y_path = os.path.join(dataset_dir, "y_train.csv")

        X_train = pd.read_csv(x_path)

        if os.path.exists(y_path):
            y_train = pd.read_csv(y_path)
            target_col = y_train.columns[0]

            
            df_train = pd.concat([X_train, y_train], axis=1)

            problem_type = {"Classification": target_col}
        else:
            df_train = X_train.copy()
            problem_type = {}

        
        if categorical is None:
            raise ValueError(
                "categorical must be provided as a list of COLUMN INDICES "
                "(already aligned with x_train + y_train)"
            )

        if not all(isinstance(i, int) for i in categorical):
            raise TypeError("categorical must be a list of column indices (int)")

        max_index = df_train.shape[1] - 1
        if any(i < 0 or i > max_index for i in categorical):
            raise ValueError(
                f"categorical indices out of range (0–{max_index})"
            )

        logger.info(f"Using categorical indices: {categorical}")

        
        # Data preparation
        
        self.data_prep = DataPrep(
            raw_df=df_train,
            categorical=categorical,
            log=[],
            mixed={},
            general=[],
            non_categorical=[],
            integer=[],
            type=problem_type,
            test_ratio=None,
        )

        logger.info("Fitting CTAB-GAN+ synthesizer")

        self.synthesizer.fit(
            train_data=self.data_prep.df,
            categorical=self.data_prep.column_types["categorical"],
            mixed=self.data_prep.column_types["mixed"],
            general=self.data_prep.column_types["general"],
            non_categorical=self.data_prep.column_types["non_categorical"],
            type=problem_type,
        )

        
        # Generate synthetic data

        n_samples = len(self.data_prep.df)
        synth_data = self.synthesizer.sample(n_samples)
        synth_df = self.data_prep.inverse_prep(synth_data)

        os.makedirs(synthetic_dir, exist_ok=True)

        if os.path.exists(y_path):
            x_synth = synth_df.drop(columns=[target_col])
            y_synth = synth_df[[target_col]]

            x_synth.to_csv(
                os.path.join(synthetic_dir, "x_synth.csv"),
                index=False
            )
            y_synth.to_csv(
                os.path.join(synthetic_dir, "y_synth.csv"),
                index=False
            )
        else:
            synth_df.to_csv(
                os.path.join(synthetic_dir, "x_synth.csv"),
                index=False
            )

        logger.info(f"Synthetic data saved to {synthetic_dir}")
        return self

    def sample(self, n: int):
        raise NotImplementedError("Use train() to generate synthetic samples")

    def evaluate(self):
        pass
