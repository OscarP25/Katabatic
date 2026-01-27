from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from .models import TabEBMConfig, TabEBMModel


@dataclass
class TabEBMAdapter:
    """
    Katabatic adapter pattern:
      - fit(x_train, y_train)
      - sample(n) -> (x_synth, y_synth)

    Notes:
    - TabEBM does not train epochs like GANs.
    - Generation is via SGLD steps (paper default: 200).
    - We keep paper-aligned params by default (no tuning).
    """
    target_col: str = "target"
    config: Optional[TabEBMConfig] = None

    model: Optional[TabEBMModel] = None
    _x_train: Optional[pd.DataFrame] = None
    _y_train: Optional[pd.Series] = None

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        self._x_train = x_train.copy()
        self._y_train = pd.Series(y_train).copy()

        cfg = self.config or TabEBMConfig()
        self.model = TabEBMModel(target_col=self.target_col, config=cfg)
        self.model.fit(self._x_train, self._y_train)

    def sample(self, num_rows: int) -> Tuple[pd.DataFrame, pd.Series]:
        if self.model is None or self._x_train is None or self._y_train is None:
            raise RuntimeError("TabEBMAdapter not fitted. Call fit() first.")

        x_synth, y_synth = self.model.sample(
            n=num_rows,
            x_train=self._x_train,
            y_train=self._y_train,
        )

        return x_synth, y_synth
