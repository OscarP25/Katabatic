from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal

import os
import pandas as pd
import re

from katebatic.models.tabddpm.scripts.TabDDPM import load_cfg, run_train, run_sample, run_eval
from katebatic.models.tabddpm.scripts.data_processing import prepare_dataset_for_tabddpm
from katebatic.models.tabddpm.scripts.analyze_synthetic_data import load_synthetic_data

def _stem_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r'[^0-9a-zA-Z]+', '_', stem).strip('_').lower()
    return stem or "dataset"

@dataclass
class TabDDPMPipeline:
    csv_path: str
    target_column: str
    dataset_name: Optional[str] = None
    normalization: Literal["quantile", "standard", "none"] = "quantile"
    exp_name: str = "ddpm_cb_best" 
    train_overrides: Dict[str, Any] = field(default_factory=dict) 
    _prepared: bool = field(default=False, init=False, repr=False)
    _cfg: Optional[dict] = field(default=None, init=False, repr=False)
    _config_path: Optional[str] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        if not self.dataset_name:
            self.dataset_name = _stem_from_path(self.csv_path)

    def fit(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None):
        # Load data to decide task if needed
        df = pd.read_csv(self.csv_path)

        prepare_dataset_for_tabddpm(
            csv_path=self.csv_path,
            dataset_name=self.dataset_name,
            target_column=self.target_column,
            task_type='auto',
            normalization=self.normalization
        )

        # Tabddom config saves at: katebatic/models/tabddpm/exp/<dataset_name>/<exp_name>/config.toml
        self._config_path = os.path.join(
            "katebatic", "models", "tabddpm", "exp",
            self.dataset_name, self.exp_name, "config.toml"
        )

        self._cfg = load_cfg(self._config_path)
        self._prepared = True

        if self.train_overrides:
            run_train(self._cfg, change_val=False, device=None, main_overrides=self.train_overrides)

        return self

    def sample(self, num_samples: int = 5000, batch_size: int = 1024) -> pd.DataFrame:
        self._ensure_prepared()

        so = {"num_samples": num_samples, "batch_size": batch_size}

        _ = run_sample(self._cfg, sample_overrides=so)

        syn_df = load_synthetic_data(
            parent_dir=self._cfg["parent_dir"],
            real_data_path=self._cfg["real_data_path"]
        )
        return syn_df

    def evaluate(self) -> Dict[str, Any]:
        self._ensure_prepared()
        out = run_eval(self._cfg)
        return out.get("result", out)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "csv_path": self.csv_path,
            "dataset_name": self.dataset_name,
            "target_column": self.target_column,
            "normalization": self.normalization,
            "exp_name": self.exp_name,
            "train_overrides": self.train_overrides,
            "sample_overrides": self.sample_overrides,
        }

    @property
    def config(self) -> dict:
        self._ensure_prepared()
        return self._cfg

    @property
    def config_path(self) -> str:
        self._ensure_prepared()
        return self._config_path

    # Internal guard
    def _ensure_prepared(self):
        if not self._prepared or self._cfg is None or self._config_path is None:
            raise RuntimeError("Estimator is not prepared. Call .fit() first.")
