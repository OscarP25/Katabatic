# =========================
# Standard imports
# =========================
import os
import numpy as np
import pandas as pd
import torch
from torch import optim

# =========================
# Synthcity
# =========================
from synthcity.metrics import eval_detection, eval_performance, eval_statistical
from synthcity.plugins.core.schema import Schema

# =========================
# Katabatic / GOGGLE
# =========================
from katabatic.models.base_model import Model
from .utils import get_dataloader
from .core_goggle import Goggle, GoggleLoss


class GoggleModel(Model):
    """
    Katabatic adapter for GOGGLE.
    Architecture is defined in core_goggle.py
    """

    def __init__(
        self,
        ds_name,
        input_dim,
        encoder_dim=64,
        encoder_l=2,
        het_encoding=True,
        decoder_dim=64,
        decoder_l=2,
        threshold=0.1,
        decoder_arch="gcn",
        graph_prior=None,
        prior_mask=None,
        device=None,
        alpha=0.1,
        beta=0.1,
        seed=42,
        iter_opt=True,
        **kwargs,
    ):
        super().__init__()

        # --------------------------------------------------
        # Device (AUTO GPU)
        # --------------------------------------------------
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.ds_name = ds_name
        self.seed = seed
        torch.manual_seed(seed)

        # --------------------------------------------------
        # Training hyperparameters (paper defaults)
        # --------------------------------------------------
        self.learning_rate = kwargs.get("learning_rate", 5e-3)
        self.weight_decay = kwargs.get("weight_decay", 1e-3)
        self.epochs = kwargs.get("epochs", 1000)
        self.batch_size = kwargs.get("batch_size", 32)
        self.patience = kwargs.get("patience", 50)
        self.logging_epoch = kwargs.get("logging", 100)

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------
        self.loss = GoggleLoss(alpha, beta, graph_prior, self.device)

        # --------------------------------------------------
        # Model
        # --------------------------------------------------
        self.model = Goggle(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            encoder_l=encoder_l,
            het_encoding=het_encoding,
            decoder_dim=decoder_dim,
            decoder_l=decoder_l,
            threshold=threshold,
            decoder_arch=decoder_arch,
            graph_prior=graph_prior,
            prior_mask=prior_mask,
            device=self.device,
        ).to(self.device)

        # --------------------------------------------------
        # Optimisation strategy (paper-faithful)
        # --------------------------------------------------
        self.iter_opt = iter_opt
        if iter_opt:
            gl_params = ["learned_graph.graph"]

            graph_learner_params = [
                p for n, p in self.model.named_parameters() if n in gl_params
            ]
            graph_autoencoder_params = [
                p for n, p in self.model.named_parameters() if n not in gl_params
            ]

            self.optimiser_gl = optim.Adam(
                graph_learner_params,
                lr=self.learning_rate,
                weight_decay=0,
            )
            self.optimiser_ga = optim.Adam(
                graph_autoencoder_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimiser = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

    # ======================================================
    # Validation
    # ======================================================
    def evaluate(self, data_loader, epoch):
        self.model.eval()

        eval_loss = rec_loss = kld_loss = graph_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for data in data_loader:
                x = data[0].to(self.device)

                x_hat, adj, mu_z, logvar_z = self.model(x, epoch)
                loss, loss_rec, loss_kld, loss_graph = self.loss(
                    x_hat, x, mu_z, logvar_z, adj
                )

                eval_loss += loss.item()
                rec_loss += loss_rec.item()
                kld_loss += loss_kld.item()
                graph_loss += loss_graph.item() * x.size(0)
                num_samples += x.size(0)

        return (
            eval_loss / num_samples,
            rec_loss / num_samples,
            kld_loss / num_samples,
            graph_loss / num_samples,
        )

    # ======================================================
    # Training
    # ======================================================
    def fit(self, data):
        loaders = get_dataloader(data, self.batch_size, self.seed)
        train_loader = loaders["train"]
        val_loader = loaders["val"]

        os.makedirs("tmp", exist_ok=True)
        model_path = f"tmp/{self.ds_name}.pt"

        best_loss = np.inf
        patience = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss, num_samples = 0.0, 0

            for i, data in enumerate(train_loader):
                x = data[0].to(self.device)

                if self.iter_opt and i % 2 == 1:
                    self.optimiser_gl.zero_grad()
                else:
                    self.optimiser_ga.zero_grad()

                x_hat, adj, mu_z, logvar_z = self.model(x, epoch)
                loss, _, _, _ = self.loss(x_hat, x, mu_z, logvar_z, adj)

                loss.backward(retain_graph=True)

                if self.iter_opt and i % 2 == 1:
                    self.optimiser_gl.step()
                else:
                    self.optimiser_ga.step()

                train_loss += loss.item()
                num_samples += x.size(0)

            train_loss /= num_samples
            val_loss = self.evaluate(val_loader, epoch)

            if val_loss[1] < best_loss:
                best_loss = val_loss[1]
                patience = 0
                torch.save(self.model.state_dict(), model_path)
            else:
                patience += 1

            if (epoch + 1) % self.logging_epoch == 0:
                print(
                    f"[Epoch {epoch+1}/{self.epochs}] "
                    f"Train: {train_loss:.4f} | Val: {val_loss[0]:.4f}"
                )

            if patience >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                break

    # ======================================================
    # Sampling
    # ======================================================
    def sample(self, X_test):
        count = X_test.shape[0]
        X_synth = self.model.sample(count).cpu().numpy()
        X_synth = self.enforce_constraints(X_synth, X_test)
        return pd.DataFrame(X_synth, columns=X_test.columns)

    # ======================================================
    # Constraints
    # ======================================================
    def enforce_constraints(self, X_synth, X_test):
        schema = Schema(data=X_test)
        X_synth = pd.DataFrame(X_synth, columns=schema.features())

        for rule in schema.as_constraints().rules:
            if rule[1] == "in":
                X_synth[rule[0]] = X_synth[rule[0]].apply(
                    lambda x: min(rule[2], key=lambda z: abs(z - x))
                )

        return X_synth.values

    # ======================================================
    # Evaluation (TSTR)
    # ======================================================
    def evaluate_synthetic(self, X_synth, X_test):
        qual = eval_statistical.AlphaPrecision().evaluate(X_test, X_synth)
        qual_score = np.mean([v for k, v in qual.items() if "naive" in k])

        perf = (
            eval_performance.PerformanceEvaluatorXGB().evaluate(X_test, X_synth),
            eval_performance.PerformanceEvaluatorLinear().evaluate(X_test, X_synth),
            eval_performance.PerformanceEvaluatorMLP().evaluate(X_test, X_synth),
        )

        gt_perf = np.mean([p["gt"] for p in perf])
        synth_perf = np.mean([p["syn_ood"] for p in perf])

        det = (
            eval_detection.SyntheticDetectionXGB().evaluate(X_test, X_synth),
            eval_detection.SyntheticDetectionMLP().evaluate(X_test, X_synth),
            eval_detection.SyntheticDetectionGMM().evaluate(X_test, X_synth),
        )

        det_score = np.mean([d["mean"] for d in det])

        return qual_score, (gt_perf, synth_perf), det_score
