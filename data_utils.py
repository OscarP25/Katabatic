# katabatic/models/ganblr/data_utils.py

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from .kdb import KdbHighOrderFeatureEncoder


class DataUtils:
    """
    Utilities for preparing data before training GANBLR.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
        self.data_size = len(x)
        self.num_features = x.shape[1]

        yunique, ycounts = np.unique(y, return_counts=True)
        self.num_classes = len(yunique)
        self.class_counts = ycounts

        self.feature_uniques = [len(np.unique(x[:, i])) for i in range(self.num_features)]

        self.constraint_positions = None
        self._kdbe = None
        self.__kdbe_cache = {}  # cache by k + dense_format

    def get_categories(self):
        if self._kdbe is None or self._kdbe.ohe_ is None:
            raise RuntimeError("Kdb encoder not fit yet. Call get_kdbe_x() first.")
        return self._kdbe.ohe_.categories_

    def get_kdbe_x(self, k=0, dense_format=True) -> np.ndarray:
        key = (k, dense_format)
        if key in self.__kdbe_cache:
            return self.__kdbe_cache[key]

        # Fit encoder if needed or if k changed
        if self._kdbe is None or getattr(self._kdbe, "k", None) != k:
            self._kdbe = KdbHighOrderFeatureEncoder()
            self._kdbe.fit(self.x, self.y, k=k)

        kdbex = self._kdbe.transform(self.x)
        if dense_format:
            kdbex = kdbex.toarray()

        self.constraint_positions = self._kdbe.constraints_
        self.__kdbe_cache[key] = kdbex
        return kdbex

    def clear(self):
        self._kdbe = None
        self.__kdbe_cache = {}
