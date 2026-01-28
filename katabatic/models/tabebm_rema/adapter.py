from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

import pandas as pd

from katabatic.models.base_model import Model
from katabatic.models.tabebm.models import TabEBMModel, TabEBMConfig


@dataclass
class TabEBMAdapter(Model):
    """
    Katabatic-compliant adapter for TabEBM.

    Responsibilities:
    - Load x_train / y_train produced by TrainTestSplitPipeline
    - Fit TabEBM preprocessing
    - Generate synthetic samples
    - Save tabebm.csv
    """

    target_col: str = "6"

    _model: Optional[TabEBMModel] = None
    _x_train: Optional[pd.DataFrame] = None
    _y_train: Optional[pd.Series] = None

    def train(self, output_dir: str, *args, **kwargs) -> None:
        output_dir = Path(output_dir)

        x_train_path = output_dir / "x_train.csv"
        y_train_path = output_dir / "y_train.csv"

        if not x_train_path.exists() or not y_train_path.exists():
            raise FileNotFoundError(
                "Expected x_train.csv and y_train.csv in output_dir"
            )

        x_train = pd.read_csv(x_train_path)
        y_train = pd.read_csv(y_train_path).iloc[:, 0]

        print(f"[TabEBM] Loaded x_train: {x_train.shape}")
        print(f"[TabEBM] Loaded y_train: {y_train.shape}")

        cfg = TabEBMConfig()

        self._model = TabEBMModel(
            target_col=self.target_col,
            config=cfg,
        )

        self._model.fit(x_train, y_train)

        self._x_train = x_train
        self._y_train = y_train

        x_synth, y_synth = self.sample(len(x_train))

        synth_csv = output_dir / "tabebm.csv"
        pd.concat([x_synth, y_synth], axis=1).to_csv(synth_csv, index=False)

        print(f"[TabEBM] Synthetic data saved to: {synth_csv}")

    def sample(self, n: int) -> Tuple[pd.DataFrame, pd.Series]:
        if self._model is None:
            raise RuntimeError("Model not trained yet")

        return self._model.sample(
            n=n,
            x_train=self._x_train,
            y_train=self._y_train,
        )

    def evaluate(self, *args, **kwargs):
        return None
