# katabatic/models/ganblr/models.py
# COMPLETE FILE (drop-in replacement)
#
# Fixes:
# - Implements GANBLR.train() so GANBLR is NOT abstract
# - Pipeline wrapper GANBLRModel calls GANBLR.train()
# - Adds stability caps (max_synth) to avoid crashes
# - Self-contained: includes KdbHighOrderFeatureEncoder + DataUtils

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import TabularCPD

from katabatic.models.base_model import Model
from .kdb import build_graph, _add_uniform
from .utils import softmax_weight, get_lr, elr_loss, sample


# ============================================================
# kDB High-Order Feature Encoder (self-contained)
# ============================================================

def _get_dependencies_without_y(variables, y_name, kdb_edges):
    dependencies = {}
    kdb_edges_without_y = [edge for edge in kdb_edges if edge[0] != y_name]
    mi_desc_order = {t: i for i, (s, t) in enumerate(kdb_edges) if s == y_name}

    for x in variables:
        current_dependencies = [s for s, t in kdb_edges_without_y if t == x]
        if len(current_dependencies) >= 2:
            sort_dict = {t: mi_desc_order[t] for t in current_dependencies}
            dependencies[x] = sorted(sort_dict)
        else:
            dependencies[x] = current_dependencies
    return dependencies


def get_cross_table(*cols):
    if len(cols) == 0:
        raise TypeError("get_cross_table() requires at least one argument")

    cols = [np.asarray(c).reshape(-1) for c in cols]
    if not all(len(col) == len(cols[0]) for col in cols[1:]):
        raise ValueError("all arguments must be same size")

    uniq_vals_all_cols, idx = zip(*(np.unique(col, return_inverse=True) for col in cols))
    shape_xt = [uv.size for uv in uniq_vals_all_cols]
    xt = np.zeros(shape_xt, dtype="uint64")
    np.add.at(xt, idx, 1)
    return uniq_vals_all_cols, xt


def get_high_order_feature(X, col, evidence_cols, feature_uniques):
    X = np.asarray(X)
    if not evidence_cols:
        return X[:, [col]]

    base = [1, feature_uniques[col]] + [
        feature_uniques[_col] for _col in evidence_cols[::-1][:-1]
    ]
    cum_base = np.cumprod(base)[::-1]
    cols = evidence_cols + [col]
    return np.sum(X[:, cols] * cum_base, axis=1).reshape(-1, 1)


def get_high_order_constraints(X, col, evidence_cols, feature_uniques):
    X = np.asarray(X)
    if not evidence_cols:
        unique = feature_uniques[col]
        return np.ones(unique, dtype=bool), np.array([unique], dtype=int)

    cols = evidence_cols + [col]
    _, cross_table = get_cross_table(*[X[:, i] for i in cols])
    have_value = cross_table != 0
    have_value_reshape = have_value.reshape(-1, have_value.shape[-1])
    high_order_constraints = np.sum(have_value_reshape, axis=-1).astype(int)
    return have_value, high_order_constraints


class KdbHighOrderFeatureEncoder:
    """
    High-order feature encoder using kDB dependencies.

    Stability:
    - fit OrdinalEncoder once, reuse in transform
    - OneHotEncoder(handle_unknown="ignore")
    """

    def __init__(self):
        self.dependencies_: Dict[int, List[int]] = {}
        self.constraints_: np.ndarray = np.array([], dtype=int)
        self.have_value_idxs_: List[np.ndarray] = []
        self.feature_uniques_: List[int] = []
        self.high_order_feature_uniques_: List[int] = []
        self.edges_: List[Tuple[int, int]] = []
        self.ohe_ = None
        self._ord_ = None
        self.k: Optional[int] = None

    def fit(self, X, y, k=0):
        from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        self.k = k

        edges = build_graph(X, y, k)
        num_features = X.shape[1]

        if k > 0:
            dependencies = _get_dependencies_without_y(list(range(num_features)), num_features, edges)
        else:
            dependencies = {x: [] for x in range(num_features)}

        self.dependencies_ = dependencies
        self.feature_uniques_ = [len(np.unique(X[:, i])) for i in range(num_features)]
        self.edges_ = edges

        Xk_raw, constraints, have_value_idxs = self.transform(
            X, return_constraints=True, use_ohe=False
        )

        self._ord_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xk_ord = self._ord_.fit_transform(Xk_raw)

        self.ohe_ = OneHotEncoder(handle_unknown="ignore")
        self.ohe_.fit(Xk_ord)

        self.high_order_feature_uniques_ = [len(c) for c in self.ohe_.categories_]
        self.constraints_ = constraints
        self.have_value_idxs_ = have_value_idxs
        return self

    def transform(self, X, return_constraints=False, use_ohe=True):
        X = np.asarray(X)

        Xk = []
        have_value_idxs = []
        constraints = []

        for col, parents in self.dependencies_.items():
            Xk.append(get_high_order_feature(X, col, parents, self.feature_uniques_))
            if return_constraints:
                idx, constraint = get_high_order_constraints(X, col, parents, self.feature_uniques_)
                have_value_idxs.append(idx)
                constraints.append(constraint)

        Xk_raw = np.hstack(Xk)

        if self._ord_ is None:
            if return_constraints:
                conc = np.hstack(constraints) if constraints else np.array([], dtype=int)
                return Xk_raw, conc, have_value_idxs
            return Xk_raw

        Xk_ord = self._ord_.transform(Xk_raw)
        X_out = self.ohe_.transform(Xk_ord) if use_ohe else Xk_ord

        if return_constraints:
            conc = np.hstack(constraints) if constraints else np.array([], dtype=int)
            return X_out, conc, have_value_idxs
        return X_out


# ============================================================
# Data Utils
# ============================================================

class DataUtils:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = np.asarray(x)
        self.y = np.asarray(y).reshape(-1)

        self.data_size = len(self.x)
        self.num_features = self.x.shape[1]

        yunique, ycounts = np.unique(self.y, return_counts=True)
        self.num_classes = len(yunique)
        self.class_counts = ycounts

        self.feature_uniques = [len(np.unique(self.x[:, i])) for i in range(self.num_features)]

        self.constraint_positions = None
        self._kdbe: Optional[KdbHighOrderFeatureEncoder] = None
        self.__kdbe_cache: Dict[Tuple[int, bool], np.ndarray] = {}

    def get_categories(self):
        if self._kdbe is None or self._kdbe.ohe_ is None:
            raise RuntimeError("Kdb encoder not fit. Call get_kdbe_x() first.")
        return self._kdbe.ohe_.categories_

    def get_kdbe_x(self, k=0, dense_format=True):
        key = (k, dense_format)
        if key in self.__kdbe_cache:
            return self.__kdbe_cache[key]

        if self._kdbe is None or getattr(self._kdbe, "k", None) != k:
            self._kdbe = KdbHighOrderFeatureEncoder()
            self._kdbe.fit(self.x, self.y, k=k)

        Xk = self._kdbe.transform(self.x)
        if dense_format:
            Xk = Xk.toarray()

        self.constraint_positions = self._kdbe.constraints_
        self.__kdbe_cache[key] = Xk
        return Xk


# ============================================================
# GANBLR core model (NOW implements train())
# ============================================================

class GANBLR(Model):
    def __init__(self) -> None:
        super().__init__()
        self.check_dependencies()

        self._d: Optional[DataUtils] = None
        self.__gen_weights = None
        self.batch_size: Optional[int] = None
        self.k: Optional[int] = None
        self.constraints = None

        self._ordinal_encoder = OrdinalEncoder(
            dtype=int, handle_unknown="use_encoded_value", unknown_value=-1
        )
        self._label_encoder = LabelEncoder()

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["tensorflow", "pgmpy", "sklearn", "scipy", "pyitlib"]

    def fit(self, x, y, k=1, batch_size=32, epochs=20, warmup_epochs=1, verbose=1):
        epsilon = 1e-10

        x = self._ordinal_encoder.fit_transform(x)
        y = self._label_encoder.fit_transform(y).astype(int)

        self._d = DataUtils(x, y)
        self.k = int(k)
        self.batch_size = int(batch_size)

        if verbose:
            print("warmup run:")
        self._warmup_run(warmup_epochs, verbose=verbose)

        syn_data = self._sample(verbose=0)
        discriminator_label = np.hstack([np.ones(self._d.data_size), np.zeros(self._d.data_size)])

        for i in range(int(epochs)):
            discriminator_input = np.vstack([x, syn_data[:, :-1]])
            disc_input, disc_label = sample(
                discriminator_input, discriminator_label, frac=0.8, random_state=42
            )

            disc = self._discrim()
            disc.fit(disc_input, disc_label, batch_size=self.batch_size, epochs=1, verbose=0)

            prob_fake = disc.predict(x, verbose=0)
            ls = float(np.mean(-np.log(np.clip(1 - prob_fake, epsilon, 1.0))))

            self._run_generator(loss=ls)
            syn_data = self._sample(verbose=0)

            if verbose:
                print(f"Epoch {i+1}/{epochs} complete")

        return self

    # ✅ REQUIRED by abstract base Model
    def train(self, dataset, size_category="small", *args, **kwargs):
        """
        Pipeline passes:
          dataset = folder with x_train.csv, y_train.csv
          synthetic_dir = where to save x_synth.csv / y_synth.csv
        """

        k = int(kwargs.get("k", 1))
        epochs = int(kwargs.get("epochs", 20))
        batch_size = int(kwargs.get("batch_size", 32))
        warmup_epochs = int(kwargs.get("warmup_epochs", 1))
        seed = int(kwargs.get("seed", 42))
        max_synth = int(kwargs.get("max_synth", 5000))

        # reproducibility
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        X = pd.read_csv(os.path.join(dataset, "x_train.csv"))
        y = pd.read_csv(os.path.join(dataset, "y_train.csv")).values.ravel()

        self.fit(
            X, y,
            k=k,
            batch_size=batch_size,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            verbose=1
        )

        syn_size = min(len(X), max_synth)
        syn = self.sample(size=syn_size, verbose=0)

        syn_df = pd.DataFrame(syn)
        x_synth = syn_df.iloc[:, :-1]
        y_synth = syn_df.iloc[:, -1]

        x_synth.columns = list(X.columns)

        save_dir = kwargs.get("synthetic_dir")
        if not save_dir:
            save_dir = os.path.join("synthetic", os.path.basename(str(dataset)), "ganblr")
        os.makedirs(save_dir, exist_ok=True)

        x_synth.to_csv(os.path.join(save_dir, "x_synth.csv"), index=False)
        pd.Series(y_synth, name="target").to_csv(os.path.join(save_dir, "y_synth.csv"), index=False)

        print(f"✅ GANBLR synthetic saved to: {save_dir} (rows={syn_size})")
        return self

    def evaluate(self, x, y, model="lr") -> float:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import RandomForestClassifier

        models = dict(lr=LogisticRegression, rf=RandomForestClassifier, mlp=MLPClassifier)
        if model not in models:
            raise ValueError("model must be one of ['lr','rf','mlp']")
        eval_model = models[model]()

        synthetic = self._sample()
        syn_x, syn_y = synthetic[:, :-1], synthetic[:, -1]

        x_test = self._ordinal_encoder.transform(x)
        y_test = self._label_encoder.transform(y)

        categories = self._d.get_categories()
        pipeline = Pipeline([
            ("encoder", OneHotEncoder(categories=categories, handle_unknown="ignore")),
            ("model", eval_model),
        ])
        pipeline.fit(syn_x, syn_y)
        pred = pipeline.predict(x_test)
        return accuracy_score(y_test, pred)

    def sample(self, size=None, verbose=1) -> np.ndarray:
        ordinal = self._sample(size=size, verbose=verbose)
        origin_x = self._ordinal_encoder.inverse_transform(ordinal[:, :-1])
        origin_y = self._label_encoder.inverse_transform(ordinal[:, -1]).reshape(-1, 1)
        return np.hstack([origin_x, origin_y])

    def _sample(self, size=None, verbose=1) -> np.ndarray:
        d = self._d
        feature_cards = np.array(d.feature_uniques)

        _idxs = np.cumsum([0] + d._kdbe.constraints_.tolist())
        constraint_idxs = [(_idxs[i], _idxs[i + 1]) for i in range(len(_idxs) - 1)]

        probs = np.exp(self.__gen_weights[0])
        cpd_probs = [probs[start:end, :] for start, end in constraint_idxs]
        cpd_probs = np.vstack([p / p.sum(axis=0) for p in cpd_probs])

        idxs = np.cumsum([0] + d._kdbe.high_order_feature_uniques_)
        feature_idxs = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
        have_value_idxs = d._kdbe.have_value_idxs_

        full_cpd_probs = []
        for have_value, (start, end) in zip(have_value_idxs, feature_idxs):
            cpd_prob_ = cpd_probs[start:end, :]
            hv = have_value.ravel()
            hv_rep = np.hstack([hv] * d.num_classes)

            full_ravel = np.zeros_like(hv_rep, dtype=float)
            full_ravel[hv_rep] = cpd_prob_.T.ravel()

            full = full_ravel.reshape(-1, have_value.shape[-1]).T
            full = _add_uniform(full, noise=0)
            full_cpd_probs.append(full)

        node_names = [str(i) for i in range(d.num_features + 1)]
        edge_names = [(str(i), str(j)) for i, j in d._kdbe.edges_]
        y_name = node_names[-1]

        evidences = d._kdbe.dependencies_
        feature_cpds = [
            TabularCPD(
                str(name),
                feature_cards[name],
                table,
                evidence=[y_name, *[str(e) for e in evs]],
                evidence_card=[d.num_classes, *feature_cards[evs].tolist()],
            )
            for (name, evs), table in zip(evidences.items(), full_cpd_probs)
        ]

        y_probs = (d.class_counts / d.data_size).reshape(-1, 1)
        y_cpd = TabularCPD(y_name, d.num_classes, y_probs)

        bn = DiscreteBayesianNetwork(edge_names)
        bn.add_cpds(y_cpd, *feature_cpds)

        sample_size = d.data_size if size is None else int(size)
        result = BayesianModelSampling(bn).forward_sample(size=sample_size, show_progress=verbose > 0)
        return result[node_names].values

    def _warmup_run(self, epochs, verbose=None):
        d = self._d
        tf.keras.backend.clear_session()

        ohex = d.get_kdbe_x(self.k, dense_format=True)
        self.constraints = softmax_weight(d.constraint_positions)

        elr = get_lr(ohex.shape[1], d.num_classes, self.constraints)
        elr.fit(ohex, d.y, batch_size=self.batch_size, epochs=int(epochs), verbose=verbose)
        self.__gen_weights = elr.get_weights()

        tf.keras.backend.clear_session()

    def _run_generator(self, loss):
        d = self._d
        ohex = d.get_kdbe_x(self.k, dense_format=True)

        tf.keras.backend.clear_session()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            d.num_classes,
            input_dim=ohex.shape[1],
            activation="softmax",
            kernel_constraint=self.constraints
        ))
        model.compile(loss=elr_loss(loss), optimizer="adam", metrics=["accuracy"])
        model.set_weights(self.__gen_weights)

        model.fit(ohex, d.y, batch_size=self.batch_size, epochs=1, verbose=0)
        self.__gen_weights = model.get_weights()

        tf.keras.backend.clear_session()

    def _discrim(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(1, input_dim=self._d.num_features, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model


# ============================================================
# Pipeline Wrapper
# ============================================================

@dataclass
class GANBLRModel:
    k: int = 1
    epochs: int = 20
    batch_size: int = 32
    warmup_epochs: int = 1
    seed: int = 42
    max_synth: int = 5000  # cap synthetic rows to prevent crashes

    def train(self, dataset: str, synthetic_dir: Optional[str] = None, *args, **kwargs):
        m = GANBLR()
        m.train(
            dataset=dataset,
            synthetic_dir=synthetic_dir,
            k=self.k,
            epochs=self.epochs,
            batch_size=self.batch_size,
            warmup_epochs=self.warmup_epochs,
            seed=self.seed,
            max_synth=self.max_synth,
        )
        return m
