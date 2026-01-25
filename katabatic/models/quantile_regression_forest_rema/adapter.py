from __future__ import annotations
import numpy as np
import pandas as pd

from .generator import QRFSequentialGenerator


class QRFAdapter:
    """
    Class-conditional Quantile Regression Forest adapter.
    """

    def __init__(
        self,
        target_col: str,
        n_estimators=100,
        criterion="squared_error",
        bootstrap=True,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        default_quantiles=0.5,
        random_state=None,
    ):
        self.target_col = target_col

        self.qrf_params = dict(
            n_estimators=n_estimators,
            criterion=criterion,
            bootstrap=bootstrap,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            default_quantiles=default_quantiles,
            random_state=random_state,
        )

        self.generators = {}
        self.class_probs = None

    # required by TrainTestSplitPipeline
    def train(self, output_dir: str, label_col=None):
        return self

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        df = x_train.copy()
        df[self.target_col] = y_train.values

        # class distribution
        self.class_probs = df[self.target_col].value_counts(normalize=True)

        # train one generator per class
        for cls, group in df.groupby(self.target_col):
            gen = QRFSequentialGenerator(self.qrf_params)
            gen.fit(group.drop(columns=[self.target_col]))
            self.generators[cls] = gen

    def sample(self, n_samples: int):
        # sample class labels first
        classes = np.random.choice(
            self.class_probs.index,
            size=n_samples,
            p=self.class_probs.values,
        )

        synth_rows = []
        synth_labels = []

        for cls in np.unique(classes):
            n_cls = np.sum(classes == cls)
            gen = self.generators[cls]

            x_cls = gen.sample(n_cls)
            synth_rows.append(x_cls)
            synth_labels.extend([cls] * n_cls)

        x_synth = pd.concat(synth_rows, ignore_index=True)
        y_synth = pd.Series(synth_labels, name=self.target_col)

        return x_synth, y_synth