# katabatic/models/ganblr/data_utils.py

import numpy as np
from .kdb import KdbHighOrderFeatureEncoder


class DataUtils:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = np.asarray(y).reshape(-1)

        self.data_size = len(x)
        self.num_features = x.shape[1]

        yunique, ycounts = np.unique(self.y, return_counts=True)
        self.num_classes = len(yunique)
        self.class_counts = ycounts

        self.feature_uniques = [len(np.unique(x[:, i])) for i in range(self.num_features)]

        self.constraint_positions = None
        self._kdbe = None
        self.__kdbe_cache = {}

    def get_categories(self):
        if self._kdbe is None or self._kdbe.ohe_ is None:
            raise RuntimeError("KDB encoder not fit. Call get_kdbe_x() first.")
        return self._kdbe.ohe_.categories_

    def get_kdbe_x(self, k=0, dense_format=True):
        key = (k, dense_format)
        if key in self.__kdbe_cache:
            return self.__kdbe_cache[key]

        if self._kdbe is None or getattr(self._kdbe, "k", None) != k:
            self._kdbe = KdbHighOrderFeatureEncoder()
            self._kdbe.fit(self.x, self.y, k=k)

        kdbex = self._kdbe.transform(self.x)
        if dense_format:
            kdbex = kdbex.toarray()

        self.constraint_positions = self._kdbe.constraints_
        self.__kdbe_cache[key] = kdbex
        return kdbex
