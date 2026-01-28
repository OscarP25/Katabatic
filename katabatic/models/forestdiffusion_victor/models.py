from __future__ import annotations

from typing import Optional, Literal, List
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from .utils.diffusion import make_beta_schedule, q_sample, p_step, _alphas_from_betas
from .utils.utils_diffusion import TabularCodec


ModelType = Literal["xgboost", "rf"]


class ForestDiffusionCore:
    """
    Tree-based diffusion denoiser for mixed tabular data.

    Strategy:
    - Encode full training dataframe (X + y) into numeric matrix Z
    - For random t, create noisy Z_t and train model to predict eps
    - Sampling: start from N(0,1), run reverse steps, decode

    Improvements applied:
    - cache z_dim at fit time (no dummy row)
    - vectorized q_sample for speed (or per-rep batch)
    - timestep conditioning uses sqrt(alpha_bar[t]) and sqrt(1-alpha_bar[t]) (tree-friendly)
    """

    def __init__(
        self,
        n_t: int = 50,
        seed: int = 666,
        model: ModelType = "xgboost",
        n_estimators: int = 200,
        max_depth: int = 7,
        gpu_hist: bool = False,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        reps: int = 2,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
    ):
        self.n_t = int(n_t)
        self.seed = int(seed)
        self.model = model
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.gpu_hist = bool(gpu_hist)
        self.reps = int(reps)

        self.betas = make_beta_schedule(self.n_t, beta_start=beta_start, beta_end=beta_end)
        self.rng = np.random.default_rng(self.seed)

        self.codec = TabularCodec(categorical_cols=categorical_cols, numeric_cols=numeric_cols)

        self._denoiser = None
        self._colnames: Optional[List[str]] = None
        self._z_dim: Optional[int] = None

        # cached alpha_bar for timestep features
        _, self._alpha_bar = _alphas_from_betas(self.betas)

    @staticmethod
    def required_dependency() -> str:
        return "scikit-learn (and optionally xgboost)"

    def _build_denoiser(self, out_dim: int):
        """
        Multi-output regressor to predict eps for each column.
        """
        if self.model == "xgboost":
            try:
                from xgboost import XGBRegressor
                params = dict(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.seed,
                    tree_method=("gpu_hist" if self.gpu_hist else "hist"),
                    objective="reg:squarederror",
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                )
                base = XGBRegressor(**params)
                return MultiOutputRegressor(base, n_jobs=1)
            except Exception:
                # fallback
                self.model = "rf"

        base = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=-1,
        )
        return MultiOutputRegressor(base, n_jobs=-1)

    def fit(self, df_train: pd.DataFrame) -> "ForestDiffusionCore":
        if not isinstance(df_train, pd.DataFrame):
            df_train = pd.DataFrame(df_train)

        if len(df_train) < 2:
            raise ValueError("Need at least 2 rows to fit ForestDiffusionCore.")

        self._colnames = list(map(str, df_train.columns))
        df_train = df_train.copy()
        df_train.columns = self._colnames

        self.codec.fit(df_train)

        Z0 = self.codec.transform(df_train).astype(np.float32)  # (N, D)
        n, d = Z0.shape
        self._z_dim = int(d)

        reps = max(1, self.reps)
        X_list = []
        y_list = []

        # Training set:
        # input = [Z_t, sqrt(alpha_bar[t]), sqrt(1-alpha_bar[t])]
        # target = eps
        for _ in range(reps):
            t = self.rng.integers(low=0, high=self.n_t, size=n, dtype=np.int32)

            Zt, eps_all = q_sample(Z0, t, self.betas, self.rng)

            ab = self._alpha_bar[t].astype(np.float32).reshape(-1, 1)
            sqrt_ab = np.sqrt(ab).astype(np.float32)
            sqrt_1mab = np.sqrt(1.0 - ab).astype(np.float32)

            X_in = np.concatenate([Zt, sqrt_ab, sqrt_1mab], axis=1).astype(np.float32)

            X_list.append(X_in)
            y_list.append(eps_all.astype(np.float32))

        X_train = np.concatenate(X_list, axis=0)
        y_train = np.concatenate(y_list, axis=0)

        self._denoiser = self._build_denoiser(out_dim=d)
        self._denoiser.fit(X_train, y_train)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._denoiser is None or self._colnames is None or self._z_dim is None:
            raise RuntimeError("ForestDiffusionCore not fitted. Call fit(df_train) first.")

        n = int(n)
        z_dim = self._z_dim

        # Start from pure noise
        Zt = self.rng.normal(0.0, 1.0, size=(n, z_dim)).astype(np.float32)

        # Reverse diffusion (t -> t-1). We stop at t=1 because p_step(t<=0) returns x_t.
        for t in range(self.n_t - 1, 0, -1):
            ab = float(self._alpha_bar[t])
            sqrt_ab = np.full((n, 1), np.sqrt(ab), dtype=np.float32)
            sqrt_1mab = np.full((n, 1), np.sqrt(1.0 - ab), dtype=np.float32)

            X_in = np.concatenate([Zt, sqrt_ab, sqrt_1mab], axis=1).astype(np.float32)
            eps_pred = self._denoiser.predict(X_in).astype(np.float32)

            Zt = p_step(Zt, eps_pred, t=t, betas=self.betas, rng=self.rng)

        df = self.codec.inverse_transform(Zt)
        df = df[self._colnames].copy()
        return df
