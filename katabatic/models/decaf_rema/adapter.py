from __future__ import annotations

import os
import json
import pandas as pd
import torch
import pytorch_lightning as pl

from katabatic.models.base_model import Model as BaseModel
from .decaf import DECAF
from .data import DataModule


class DECAFModel(BaseModel):
    """
    DECAF: Debiasing Causal Fairness
    NeurIPS 2021
    """

    def __init__(
        self,
        *,
        epochs: int = 300,
        batch_size: int = 256,
        lr: float = 1e-3,
        seed: int = 42,
        device: str | None = None,
    ):
        super().__init__()

        self.cfg = {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "device": device,
        }

        self.model: DECAF | None = None
        self.datamodule: DataModule | None = None
        self.is_fitted = False

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["torch", "pytorch_lightning", "numpy", "pandas"]

    def train(
        self,
        data_dir: str,
        synthetic_dir: str | None = None,
        dag: list | None = None,
        *args,
        **kwargs,
    ) -> "DECAFModel":

        pl.seed_everything(self.cfg["seed"], workers=True)

        train_path = os.path.join(data_dir, "train_full.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError("DECAF requires train_full.csv")

        df = pd.read_csv(train_path)

        df_numeric = df.copy()
        for col in df_numeric.columns:
            if not pd.api.types.is_numeric_dtype(df_numeric[col]):
                df_numeric[col] = pd.factorize(df_numeric[col])[0]

        df_numeric = df_numeric.astype("float32")

        self.datamodule = DataModule(
            data=df_numeric.values,
            batch_size=self.cfg["batch_size"],
        )

        self.model = DECAF(
            input_dim=df_numeric.shape[1],
            dag_seed=dag or [],
            lr=self.cfg["lr"],
        )

        trainer = pl.Trainer(
            max_epochs=self.cfg["epochs"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )

        trainer.fit(self.model, self.datamodule)

        self.is_fitted = True

        with torch.no_grad():
            z = self.model.sample_z(len(df_numeric))
            x0 = torch.zeros(
                len(df_numeric),
                df_numeric.shape[1],
                device=self.model.device,
            )
            synth = self.model.generator.sequential(x0, z)

        synth_df = pd.DataFrame(
            synth.cpu().numpy(),
            columns=df_numeric.columns,
        )

        if synthetic_dir is None:
            synthetic_dir = os.path.join("synthetic", "decaf")

        os.makedirs(synthetic_dir, exist_ok=True)

        synth_df.to_csv(
            os.path.join(synthetic_dir, "synthetic.csv"),
            index=False,
        )

        with open(os.path.join(synthetic_dir, "metadata.json"), "w") as f:
            json.dump(
                {
                    "model": "DECAF",
                    "paper": "NeurIPS 2021",
                    "epochs": self.cfg["epochs"],
                    "note": "No hyperparameter or epoch tuning performed",
                },
                f,
                indent=2,
            )

        print("[DECAF] Synthetic data generated")

        return self

    def sample(self, n: int):
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Call train() first")

        with torch.no_grad():
            z = self.model.sample_z(n)
            x0 = torch.zeros(n, self.model.hparams.input_dim, device=self.model.device)
            return self.model.generator.sequential(x0, z)

    def evaluate(self, *args, **kwargs) -> float:
        return 0.0
