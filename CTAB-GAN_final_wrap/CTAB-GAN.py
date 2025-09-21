# CTAB-GAN.py

import os
import shutil
from copy import deepcopy
from typing import Optional, Union, List, Dict, Tuple

import numpy as np
import pandas as pd
import torch

from utils import (
    Transformations,
    make_dataset,
    dump_json,
    train as train_fn,
    sample as sample_fn,
    decode_synthetic_to_dataframe,
)


class CTABGAN:
    def __init__(
        self,
        # schedule
        epochs: int = 300,                   # ≈ steps/100 internally
        batch_size: int = 512,
        lr_G: float = 2e-4,                  # TTUR
        lr_D: float = 1e-4,                  # TTUR
        weight_decay: float = 1e-5,

        # model
        z_dim: int = 128,
        d_layers: Optional[List[int]] = None,
        dropout: float = 0.2,
        num_classes: int = 2,
        is_y_cond: bool = True,

        # regularizers / tricks
        corr_penalty_weight: float = 3e-3,
        lambda_gp: float = 12.0,
        n_critic: int = 5,
        emb_dim: int = 32,
        decode_temp: float = 0.7,
        inst_noise_std: float = 0.01,        # instance noise for D

        # preprocessing
        normalization: str = "quantile",
        num_nan_policy=None,
        cat_nan_policy=None,
        cat_min_frequency=None,
        cat_encoding: Optional[str] = "one-hot",
        y_policy: str = "default",

        # runtime
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 42,
        parent_dir: Optional[str] = None,

        # warm start
        warm_start_from: Optional[str] = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.weight_decay = weight_decay

        self.z_dim = z_dim
        self.d_layers = d_layers or [256, 256]
        self.dropout = dropout
        self.num_classes = num_classes
        self.is_y_cond = is_y_cond

        self.corr_penalty_weight = corr_penalty_weight
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.emb_dim = emb_dim
        self.decode_temp = decode_temp
        self.inst_noise_std = inst_noise_std

        self.normalization = normalization
        self.num_nan_policy = num_nan_policy
        self.cat_nan_policy = cat_nan_policy
        self.cat_min_frequency = cat_min_frequency
        self.cat_encoding = cat_encoding
        self.y_policy = y_policy

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        self.parent_dir = parent_dir or f"exp/ctabgan_{np.random.randint(1e9)}"
        self.warm_start_from = warm_start_from

        self.is_trained = False
        self.num_numerical_features_ = None
        self.real_data_path_ = None
        self.model_path_ = None
        self._feature_names_num: List[str] = []
        self._feature_names_cat: List[str] = []
        self._class_probs: Optional[np.ndarray] = None

        torch.manual_seed(self.seed); np.random.seed(self.seed)

    # ----------------- build real_data -----------------
    def _make_real_data_dir(self, X, y, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        info: Dict[str, object] = {}

        if isinstance(X, pd.DataFrame):
            is_cat = (X.dtypes == 'object') | (X.dtypes == 'category')
            X_num = X.loc[:, ~is_cat].values if (~is_cat).any() else None
            X_cat = X.loc[:, is_cat].values if is_cat.any() else None
            self._feature_names_num = list(X.loc[:, ~is_cat].columns)
            self._feature_names_cat = list(X.loc[:, is_cat].columns)
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            X_num = X; X_cat = None
            self._feature_names_num = [f"num_{i}" for i in range(X.shape[1])]
            self._feature_names_cat = []
        else:
            raise ValueError("X must be a DataFrame or 2D numpy array")

        if X_num is not None:
            np.save(os.path.join(output_dir, "X_num_train.npy"), X_num)
            info["n_num_features"] = X_num.shape[1]
            self.num_numerical_features_ = X_num.shape[1]
        else:
            info["n_num_features"] = 0
            self.num_numerical_features_ = 0

        if X_cat is not None:
            np.save(os.path.join(output_dir, "X_cat_train.npy"), X_cat)
            info["n_cat_features"] = X_cat.shape[1]
        else:
            info["n_cat_features"] = 0

        if y is not None:
            y_np = y.values if isinstance(y, pd.Series) else y
            np.save(os.path.join(output_dir, "y_train.npy"), y_np)
            info["y_dim"] = 1
            info["num_classes"] = int(self.num_classes) if self.num_classes else 0
            info["is_y_cond"] = bool(self.is_y_cond)
            info["y_policy"] = self.y_policy
            if self.num_classes and self.num_classes > 0:
                vals, counts = np.unique(y_np, return_counts=True)
                probs = counts / counts.sum()
                order = np.argsort(vals)
                self._class_probs = probs[order]
        else:
            info["y_dim"] = 0
            info["is_y_cond"] = False
            info["num_classes"] = int(self.num_classes) if self.num_classes else 0

        if self.num_classes == 0:
            info.update(dict(task_type="regression", is_regression=True, is_binclass=False, is_multiclass=False, n_classes=None))
        elif self.num_classes == 2:
            info.update(dict(task_type="binclass", is_regression=False, is_binclass=True, is_multiclass=False, n_classes=2))
        else:
            info.update(dict(task_type="multiclass", is_regression=False, is_binclass=False, is_multiclass=True, n_classes=self.num_classes))

        dump_json(info, os.path.join(output_dir, "info.json"))

        # mirror to val/test to satisfy downstream loaders
        for key in ["X_num", "X_cat", "y"]:
            src = os.path.join(output_dir, f"{key}_train.npy")
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(output_dir, f"{key}_val.npy"))
                shutil.copyfile(src, os.path.join(output_dir, f"{key}_test.npy"))
        return output_dir

    # ----------------- training -----------------
    def fit(self, X, y=None):
        self.real_data_path_ = os.path.join(self.parent_dir, "real_data")
        os.makedirs(self.real_data_path_, exist_ok=True)
        self._make_real_data_dir(X, y, self.real_data_path_)

        self.model_params_ = {
            'is_y_cond': self.is_y_cond,
            'num_classes': self.num_classes,
            'z_dim': self.z_dim,
            'rtdl_params': { 'd_layers': self.d_layers, 'dropout': self.dropout },
            'lambda_gp': float(self.lambda_gp),
            'spectral_d': True,
            'corr_penalty_weight': float(self.corr_penalty_weight),
            'warm_start_from': self.warm_start_from,
            'lr_G': float(self.lr_G),
            'lr_D': float(self.lr_D),
            'n_critic': int(self.n_critic),
            'emb_dim': int(self.emb_dim),
            'decode_temp': float(self.decode_temp),
            'inst_noise_std': float(self.inst_noise_std),
            # ACGAN/projection heads toggles handled internally
        }
        self._T_dict = {
            'seed': self.seed,
            'normalization': self.normalization,
            'num_nan_policy': self.num_nan_policy,
            'cat_nan_policy': self.cat_nan_policy,
            'cat_min_frequency': self.cat_min_frequency,
            'cat_encoding': self.cat_encoding,
            'y_policy': self.y_policy,
        }
        approx_steps = max(1, self.epochs * 100)

        train_fn(
            parent_dir=self.parent_dir,
            real_data_path=self.real_data_path_,
            steps=approx_steps,
            lr=self.lr_G,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            model_type='mlp',
            model_params=deepcopy(self.model_params_),
            num_timesteps=0,
            gaussian_loss_type='mse',
            scheduler='const',
            T_dict=self._T_dict,
            num_numerical_features=(X.shape[1] if isinstance(X, np.ndarray) else len(self._feature_names_num or [])),
            device=torch.device(self.device),
            seed=self.seed,
            change_val=False
        )
        self.is_trained = True
        self.model_path_ = os.path.join(self.parent_dir, "model.pt")
        return self

    # ----------------- generation -----------------
    def generate(self, n_samples: int, class_probs: Optional[np.ndarray] = None):
        if not self.is_trained:
            raise ValueError("Call fit() before generate().")

        sample_fn(
            parent_dir=self.parent_dir,
            real_data_path=self.real_data_path_,
            num_samples=n_samples,
            batch_size=min(10000, n_samples),
            model_type='mlp',
            model_params=deepcopy(self.model_params_),
            model_path=self.model_path_,
            num_timesteps=0,
            gaussian_loss_type='mse',
            scheduler='const',
            T_dict=self._T_dict,
            num_numerical_features=self.num_numerical_features_ or 0,
            disbalance=(class_probs if class_probs is not None else self._class_probs),
            device=torch.device(self.device),
            seed=self.seed,
            change_val=False
        )

        df_syn = decode_synthetic_to_dataframe(
            parent_dir=self.parent_dir,
            real_data_path=self.real_data_path_,
            feature_names_num=self._feature_names_num,
            feature_names_cat=self._feature_names_cat,
            T_dict=self._T_dict
        )

        # 🔹  save automatically to CSV
        csv_path = os.path.join(self.parent_dir, "synthetic_output.csv")
        df_syn.to_csv(csv_path, index=False)
        print(f"[CTAB-GAN] Synthetic data saved to {csv_path}")

        y_path = os.path.join(self.parent_dir, "y_train.npy")
        if os.path.exists(y_path):
            y_gen = np.load(y_path, allow_pickle=True)
            return df_syn.drop(columns=['__target__']).to_numpy(), y_gen
        else:
            return df_syn.to_numpy()


    # ----------------- benchmark eval -----------------
    def evaluate_benchmark(
        self,
        real_csv: str,
        out_dir: str,
        target: str = "income",
        n_synth: int = 10000,
        classifiers: List[str] = None,
        scaler_name: str = "Standard",
        test_ratio: float = 0.2,
    ):
        from evaluation import get_utility_metrics

        os.makedirs(out_dir, exist_ok=True)
        real = pd.read_csv(real_csv)
        assert target in real.columns, f"Target '{target}' not in real dataset"
        real_cols = list(real.drop(columns=[target]).columns)

        Xy = self.generate(n_synth)
        if isinstance(Xy, tuple):
            X_syn, y_syn = Xy
            df_syn = pd.DataFrame(X_syn, columns=real_cols[:X_syn.shape[1]])
            df_syn[target] = y_syn
        else:
            X_syn = Xy
            df_syn = pd.DataFrame(X_syn, columns=real_cols[:X_syn.shape[1]])

        syn_csv = os.path.join(out_dir, "synthetic.csv")
        df_syn.to_csv(syn_csv, index=False)

        scores = get_utility_metrics(
            real_path=real_csv,
            fake_paths=[syn_csv],
            scaler_name=scaler_name,
            classifiers_list=(classifiers or ["lr","dt","rf","mlp","svm"]),
            test_ratio=test_ratio,
            target=target,
        )
        return scores

    # convenience
    def generate_df(self, n_samples: int) -> pd.DataFrame:
        out = self.generate(n_samples)
        cols = (self._feature_names_num or []) + (self._feature_names_cat or [])
        if isinstance(out, tuple):
            X, y = out
            df = pd.DataFrame(X, columns=cols[:X.shape[1]])
            df['__target__'] = y
            return df
        return pd.DataFrame(out, columns=cols[:out.shape[1]])

    def clear_cache(self, remove_exp_folder: bool = False):
        if getattr(self, 'parent_dir', None) and self.parent_dir and os.path.exists(self.parent_dir):
            shutil.rmtree(self.parent_dir)
            print(f"Deleted model directory: {self.parent_dir}")
        if remove_exp_folder and os.path.exists("exp"):
            shutil.rmtree("exp")
            print("Deleted 'exp' folder")
        self.is_trained = False
        self.parent_dir = None
        self.real_data_path_ = None
        self.model_path_ = None


if __name__ == "__main__":
    try:
        df = pd.read_csv("Real_Datasets/adult.csv")
        X, y = df.drop(columns=["income"]), df["income"]
        model = CTABGAN()
        model.fit(X, y)
        try:
            print(model.evaluate_benchmark(
                real_csv="Real_Datasets/adult.csv",
                out_dir="out_bench",
                target="income",
                n_synth=10000,
                classifiers=["lr","dt","rf","mlp","svm"],
                scaler_name="Standard",
                test_ratio=0.2
            ))
        except ModuleNotFoundError:
            print("evaluation.py not found — trained; skipping benchmark.")
    except FileNotFoundError:
        print("Real_Datasets/adult.csv not found — training skipped.")
