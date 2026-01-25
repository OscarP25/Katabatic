from __future__ import annotations
import numpy as np
import pandas as pd

from quantile_forest import RandomForestQuantileRegressor
from sklearn.preprocessing import OrdinalEncoder


class QRFSequentialGenerator:
    """
    Sequential feature generator using Quantile Regression Forests.
    """

    def __init__(self, qrf_params: dict):
        self.qrf_params = qrf_params
        self.columns = []
        self.models = {}
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

    def fit(self, data: pd.DataFrame):
        self.columns = list(data.columns)

        X_encoded = self.encoder.fit_transform(data.astype(str))
        X_encoded = pd.DataFrame(X_encoded, columns=self.columns)

        for i, col in enumerate(self.columns):
            X = X_encoded[self.columns[:i]].values if i > 0 else np.zeros((len(X_encoded), 1))
            y = X_encoded[col].values.astype(float)

            model = RandomForestQuantileRegressor(**self.qrf_params)
            model.fit(X, y)
            self.models[col] = model

    def sample(self, n: int) -> pd.DataFrame:
        synthetic = pd.DataFrame(index=range(n))

        for i, col in enumerate(self.columns):
            X = synthetic[self.columns[:i]].values if i > 0 else np.zeros((n, 1))
            q = float(np.random.uniform(0.1, 0.9))
            synthetic[col] = self.models[col].predict(X, quantiles=q)

        # inverse encoding
        synthetic[self.columns] = self.encoder.inverse_transform(synthetic[self.columns])

        return synthetic