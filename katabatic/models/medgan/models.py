"""
Production-level MedGAN implementation for the Katabatic framework.

Based on "Generating Multi-label Discrete Patient Records using Generative Adversarial Networks"
by Choi et al. (2017) - https://arxiv.org/abs/1703.06490
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from dataclasses import dataclass


from katabatic.models.base_model import Model
from katabatic.models.medgan.utils import (
    Autoencoder,
    Generator,
    Discriminator,
    sample_noise
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MEDGAN(Model):
    """
    MedGAN for tabular synthesis (Katabatic-compatible).
    """

    def __init__(
        self,
        # Architecture hyperparameters
        encoder_dim: int = 128,
        latent_dim: int = 128,
        generator_hidden_dim: int = 128,
        discriminator_hidden_dim: int = 128,
        generator_num_layers: int = 2,
        discriminator_num_layers: int = 2,

        # Training hyperparameters (SAFE DEFAULTS — increase later)
        ae_pretrain_epochs: int = 20,
        gan_epochs: int = 50,
        batch_size: int = 512,
        ae_lr: float = 1e-3,
        generator_lr: float = 1e-3,
        discriminator_lr: float = 1e-3,

        # Regularization
        dropout: float = 0.1,
        bn_decay: float = 0.99,

        # Other
        random_state: int = 42,
        device: Optional[str] = None,

        # Stability
        max_synth: int = 5000,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.latent_dim = latent_dim
        self.generator_hidden_dim = generator_hidden_dim
        self.discriminator_hidden_dim = discriminator_hidden_dim
        self.generator_num_layers = generator_num_layers
        self.discriminator_num_layers = discriminator_num_layers

        self.ae_pretrain_epochs = ae_pretrain_epochs
        self.gan_epochs = gan_epochs
        self.batch_size = batch_size
        self.ae_lr = ae_lr
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr

        self.dropout = dropout
        self.bn_decay = bn_decay
        self.random_state = random_state
        self.max_synth = max_synth

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        self.autoencoder = None
        self.generator = None
        self.discriminator = None
        self.input_dim_ = None
        self.data_min_ = None
        self.data_max_ = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["torch", "numpy", "pandas"]

    def train(self, dataset_dir: str, synthetic_dir: str, **kwargs):
        """
        Katabatic pipeline hook.

        dataset_dir contains:
          - x_train.csv
          - y_train.csv
        synthetic_dir output:
          - x_synth.csv
          - y_synth.csv
        """
        logger.info("=" * 80)
        logger.info("Training MedGAN Model")
        logger.info("=" * 80)

        # Allow overrides from wrapper/pipeline
        self.ae_pretrain_epochs = int(kwargs.get("ae_pretrain_epochs", self.ae_pretrain_epochs))
        self.gan_epochs = int(kwargs.get("gan_epochs", self.gan_epochs))
        self.batch_size = int(kwargs.get("batch_size", self.batch_size))
        self.max_synth = int(kwargs.get("max_synth", self.max_synth))

        x_train_path = os.path.join(dataset_dir, "x_train.csv")
        y_train_path = os.path.join(dataset_dir, "y_train.csv")

        X_train = pd.read_csv(x_train_path)
        logger.info(f"Loaded X_train: {X_train.shape}")

        has_y = os.path.exists(y_train_path)
        if has_y:
            y_train = pd.read_csv(y_train_path)
            df_train = pd.concat([X_train, y_train], axis=1)
            y_name = y_train.columns[0]
        else:
            df_train = X_train
            y_name = None

        data = df_train.values.astype(np.float32)
        self.input_dim_ = data.shape[1]

        self.data_min_ = data.min(axis=0)
        self.data_max_ = data.max(axis=0)
        data_range = self.data_max_ - self.data_min_
        data_range[data_range == 0] = 1.0
        data_normalized = (data - self.data_min_) / data_range

        logger.info("Data normalized to [0, 1]")

        self._fit(data_normalized)

        # Sample size cap to avoid crashing
        n_real = len(data)
        n_synth = min(n_real, self.max_synth)
        logger.info(f"Generating {n_synth} synthetic samples (cap={self.max_synth})...")
        synth_data = self.sample(n_synth)

        # Round to discrete-ish values (your datasets are discretized)
        synth_data = np.round(synth_data)

        os.makedirs(synthetic_dir, exist_ok=True)

        if has_y:
            x_synth = pd.DataFrame(synth_data[:, :-1], columns=X_train.columns)
            y_synth = pd.DataFrame(synth_data[:, -1:], columns=[y_name])

            # ensure class coverage
            unique_train = np.unique(df_train[y_name].values)
            unique_synth = np.unique(y_synth[y_name].values)
            missing = set(unique_train) - set(unique_synth)
            if missing:
                logger.warning(f"Missing classes in synthetic data: {missing}")
                logger.info("Adding 1 dummy row for each missing class...")
                for cls in missing:
                    cls_idx = np.where(df_train[y_name].values == cls)[0][0]
                    dummy_row = df_train.iloc[cls_idx:cls_idx+1].values.astype(np.float32)
                    dummy_row = np.round(dummy_row)

                    dummy_x = pd.DataFrame(dummy_row[:, :-1], columns=X_train.columns)
                    dummy_y = pd.DataFrame(dummy_row[:, -1:], columns=[y_name])
                    x_synth = pd.concat([x_synth, dummy_x], ignore_index=True)
                    y_synth = pd.concat([y_synth, dummy_y], ignore_index=True)

            # cast to int
            for col in x_synth.columns:
                x_synth[col] = x_synth[col].astype(int)
            y_synth[y_name] = y_synth[y_name].astype(int)

            x_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            y_synth.to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)
        else:
            synth_df = pd.DataFrame(synth_data, columns=df_train.columns)
            for col in synth_df.columns:
                synth_df[col] = synth_df[col].astype(int)
            synth_df.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)

        logger.info(f"Synthetic data saved to: {synthetic_dir}")
        logger.info("Training complete!")
        return self

    def _fit(self, data: np.ndarray):
        self.autoencoder = Autoencoder(
            input_dim=self.input_dim_,
            encoder_dim=self.encoder_dim,
            latent_dim=self.latent_dim,
            bn_decay=self.bn_decay
        ).to(self.device)

        self.generator = Generator(
            latent_dim=self.latent_dim,
            hidden_dim=self.generator_hidden_dim,
            num_layers=self.generator_num_layers,
            bn_decay=self.bn_decay
        ).to(self.device)

        self.discriminator = Discriminator(
            latent_dim=self.latent_dim,
            hidden_dim=self.discriminator_hidden_dim,
            num_layers=self.discriminator_num_layers,
            dropout=self.dropout
        ).to(self.device)

        logger.info(f"\nPhase 1: Pretraining Autoencoder ({self.ae_pretrain_epochs} epochs)")
        self._pretrain_autoencoder(data)

        logger.info(f"\nPhase 2: Training GAN ({self.gan_epochs} epochs)")
        self._train_gan(data)

    def _pretrain_autoencoder(self, data: np.ndarray):
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.ae_lr)
        criterion = nn.BCELoss()

        dataset = torch.tensor(data, dtype=torch.float32)
        n_batches = (len(dataset) + self.batch_size - 1) // self.batch_size

        for epoch in range(self.ae_pretrain_epochs):
            self.autoencoder.train()
            total_loss = 0.0

            indices = torch.randperm(len(dataset))
            for i in range(n_batches):
                batch_idx = indices[i*self.batch_size:(i+1)*self.batch_size]
                batch = dataset[batch_idx].to(self.device)

                optimizer.zero_grad()
                x_recon, _ = self.autoencoder(batch)
                loss = criterion(x_recon, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == self.ae_pretrain_epochs:
                logger.info(f"[AE] epoch {epoch+1}/{self.ae_pretrain_epochs} loss={total_loss/n_batches:.6f}")

    def _train_gan(self, data: np.ndarray):
        optimizer_g = optim.Adam(self.generator.parameters(), lr=self.generator_lr)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr)
        criterion = nn.BCELoss()

        dataset = torch.tensor(data, dtype=torch.float32)
        n_batches = (len(dataset) + self.batch_size - 1) // self.batch_size

        self.autoencoder.eval()

        for epoch in range(self.gan_epochs):
            self.generator.train()
            self.discriminator.train()

            d_loss_total = 0.0
            g_loss_total = 0.0

            indices = torch.randperm(len(dataset))
            for i in range(n_batches):
                batch_idx = indices[i*self.batch_size:(i+1)*self.batch_size]
                real_data = dataset[batch_idx].to(self.device)
                batch_len = len(real_data)

                with torch.no_grad():
                    real_latent = self.autoencoder.encode(real_data)

                # ---- Discriminator ----
                optimizer_d.zero_grad()
                real_labels = torch.ones(batch_len, 1, device=self.device)
                fake_labels = torch.zeros(batch_len, 1, device=self.device)

                d_real = self.discriminator(real_latent)
                d_loss_real = criterion(d_real, real_labels)

                noise = sample_noise(batch_len, self.latent_dim, self.device)
                fake_latent = self.generator(noise)
                d_fake = self.discriminator(fake_latent.detach())
                d_loss_fake = criterion(d_fake, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_d.step()

                # ---- Generator ----
                optimizer_g.zero_grad()
                noise = sample_noise(batch_len, self.latent_dim, self.device)
                fake_latent = self.generator(noise)
                d_fake = self.discriminator(fake_latent)
                g_loss = criterion(d_fake, real_labels)
                g_loss.backward()
                optimizer_g.step()

                d_loss_total += d_loss.item()
                g_loss_total += g_loss.item()

            if (epoch + 1) % 25 == 0 or epoch == 0 or (epoch + 1) == self.gan_epochs:
                logger.info(
                    f"[GAN] epoch {epoch+1}/{self.gan_epochs} "
                    f"D={d_loss_total/n_batches:.6f} G={g_loss_total/n_batches:.6f}"
                )

    def sample(self, n: int) -> np.ndarray:
        if self.autoencoder is None or self.generator is None:
            raise RuntimeError("Model must be trained before sampling")

        self.autoencoder.eval()
        self.generator.eval()

        with torch.no_grad():
            noise = sample_noise(n, self.latent_dim, self.device)
            fake_latent = self.generator(noise)
            synthetic_norm = self.autoencoder.decode(fake_latent).cpu().numpy()

        data_range = self.data_max_ - self.data_min_
        return synthetic_norm * data_range + self.data_min_

    def evaluate(self):
        # handled by TSTREvaluation in the pipeline
        return None


# ---- Pipeline wrapper (recommended) ----

@dataclass
class MedGANModel:
    ae_pretrain_epochs: int = 20
    gan_epochs: int = 50
    batch_size: int = 512
    seed: int = 42
    device: str = "cpu"
    max_synth: int = 5000

    # architecture
    encoder_dim: int = 128
    latent_dim: int = 128
    generator_hidden_dim: int = 128
    discriminator_hidden_dim: int = 128
    generator_num_layers: int = 2
    discriminator_num_layers: int = 2
    dropout: float = 0.1
    bn_decay: float = 0.99

    def train(self, dataset: str, synthetic_dir: Optional[str] = None, *args, **kwargs):
        m = MEDGAN(
            encoder_dim=self.encoder_dim,
            latent_dim=self.latent_dim,
            generator_hidden_dim=self.generator_hidden_dim,
            discriminator_hidden_dim=self.discriminator_hidden_dim,
            generator_num_layers=self.generator_num_layers,
            discriminator_num_layers=self.discriminator_num_layers,
            ae_pretrain_epochs=self.ae_pretrain_epochs,
            gan_epochs=self.gan_epochs,
            batch_size=self.batch_size,
            dropout=self.dropout,
            bn_decay=self.bn_decay,
            random_state=self.seed,
            device=self.device,
            max_synth=self.max_synth,
        )
        m.train(dataset_dir=dataset, synthetic_dir=synthetic_dir, max_synth=self.max_synth)
        return m
