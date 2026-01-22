import pandas as pd
from .modules import arf

class ARF:
    def __init__(self, num_trees=30, delta=0.0, max_iters=10, min_node_size=5):
        self.num_trees = num_trees
        self.delta = delta
        self.max_iters = max_iters
        self.min_node_size = min_node_size
        self.model = None

    def train(self, data: pd.DataFrame):
        self.model = arf(
            x=data,
            num_trees=self.num_trees,
            delta=self.delta,
            max_iters=self.max_iters,
            min_node_size=self.min_node_size,
            verbose=False
        )
        # Estimate density
        self.model.forde(dist="truncnorm", alpha=0.1)

    def sample(self, n_samples: int):
        if self.model is None:
            raise RuntimeError("Model not fitted. Call train() first.")
        return self.model.forge(n=n_samples)

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.model = pickle.load(f)