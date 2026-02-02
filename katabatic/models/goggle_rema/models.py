import os
import numpy as np
import pandas as pd
import torch
from torch import optim

from synthcity.metrics import eval_detection, eval_performance, eval_statistical
from synthcity.plugins.core.schema import Schema

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

    
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device is None else torch.device(device)

        self.ds_name = ds_name
        self.seed = seed
        torch.manual_seed(seed)

    
        self.learning_rate = kwargs.get("learning_rate", 5e-3)
        self.weight_decay = kwargs.get("weight_decay", 1e-3)
        self.epochs = kwargs.get("epochs", 1000)
        self.batch_size = kwargs.get("batch_size", 32)
        self.patience = kwargs.get("patience", 50)
        self.logging_epoch = kwargs.get("logging", 100)

        
        self.loss = GoggleLoss(alpha, beta, graph_prior, self.device)

        
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

       
        self.iter_opt = iter_opt

        if self.iter_opt:
            graph_params = []
            other_params = []

            for name, param in self.model.named_parameters():
                if "graph" in name.lower():
                    graph_params.append(param)
                else:
                    other_params.append(param)

            if len(graph_params) == 0:
                print("No graph parameters found — falling back to joint optimisation.")
                self.iter_opt = False
            else:
                self.optimiser_gl = optim.Adam(
                    graph_params,
                    lr=self.learning_rate,
                    weight_decay=0,
                )
                self.optimiser_ga = optim.Adam(
                    other_params,
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )

        if not self.iter_opt:
            self.optimiser = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

    def evaluate(self, data_loader, epoch):
        self.model.eval()

        eval_loss = rec_loss = kld_loss = graph_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for data in data_loader:
                x = data[0].to(self.device)

                x_hat, adj, mu, logvar = self.model(x, epoch)
                loss, rec, kld, graph = self.loss(x_hat, x, mu, logvar, adj)

                eval_loss += loss.item()
                rec_loss += rec.item()
                kld_loss += kld.item()
                graph_loss += graph.item() * x.size(0)
                num_samples += x.size(0)

        return (
            eval_loss / num_samples,
            rec_loss / num_samples,
            kld_loss / num_samples,
            graph_loss / num_samples,
        )


    def train(self, data_dir, **kwargs):
        _ = kwargs
        self._train(data_dir)

    def _train(self, data_dir):
        x_train_path = os.path.join(data_dir, "x_train.csv")
        if not os.path.exists(x_train_path):
            raise FileNotFoundError(f"x_train.csv not found in {data_dir}")

        x_train = pd.read_csv(x_train_path)
        print(f"Training GOGGLE on {x_train.shape}")
        self.fit(x_train)


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
            train_loss, n = 0.0, 0

            for i, batch in enumerate(train_loader):
                x = batch[0].to(self.device)

                if self.iter_opt and i % 2 == 1:
                    self.optimiser_gl.zero_grad()
                else:
                    self.optimiser_ga.zero_grad()

                x_hat, adj, mu, logvar = self.model(x, epoch)
                loss, _, _, _ = self.loss(x_hat, x, mu, logvar, adj)

                loss.backward(retain_graph=True)

                if self.iter_opt and i % 2 == 1:
                    self.optimiser_gl.step()
                else:
                    self.optimiser_ga.step()

                train_loss += loss.item()
                n += x.size(0)

            train_loss /= n
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
                    f"Train {train_loss:.4f} | Val {val_loss[0]:.4f}"
                )
            
            if patience >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                
            else:
                print(" No checkpoint found — using last epoch weights.")
                break

    
    def sample(self, X_test):
        n = X_test.shape[0]

        self.model.eval()
        with torch.no_grad():
            X_synth = self.model.sample(n).detach().cpu().numpy()

        X_synth = self.enforce_constraints(X_synth, X_test)
        return pd.DataFrame(X_synth, columns=X_test.columns)

    
    def enforce_constraints(self, X_synth, X_test):
        schema = Schema(data=X_test)
        X_synth = pd.DataFrame(X_synth, columns=schema.features())

        for rule in schema.as_constraints().rules:
            col, rule_type, allowed = rule

            if rule_type == "in":
                
                if X_test[col].dtype == object:
                    continue

                
                X_synth[col] = X_synth[col].apply(
                    lambda x: min(allowed, key=lambda z: abs(z - x))
                )

        return X_synth.values

  
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
