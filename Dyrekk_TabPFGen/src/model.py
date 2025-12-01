import os
import numpy as np
import pandas as pd
import torch
from torch.autograd import Function 
from torch import nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from typing import Dict, List, Tuple, Any

from katabatic.models.base_model import Model
from tabpfn import TabPFNClassifier

# -----------------------------------------------------------------
# Custom Autograd Function
# -----------------------------------------------------------------
class PredictWithGrad(Function):
    @staticmethod
    def forward(ctx, x_synth, x_train, y_train, clf):
        ctx.save_for_backward(x_synth)
        ctx.clf = clf
        ctx.x_train = x_train
        ctx.y_train = y_train

        x_synth_np = x_synth.detach().cpu().numpy()
        x_train_np = x_train.detach().cpu().numpy()
        y_train_np = y_train.detach().cpu().numpy()

        clf.fit(x_train_np, y_train_np)
        probs = clf.predict_proba(x_synth_np)

        return torch.tensor(probs, device=x_synth.device, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        x_synth, = ctx.saved_tensors
        grad_x_synth = torch.zeros_like(x_synth)
        return grad_x_synth, None, None, None

# -----------------------------------------------------------------
# TabPFGen Class
# -----------------------------------------------------------------
class TabPFGen(Model):
    def __init__(
        self,
        n_steps: int = 5,
        step_size: float = 0.01,
        noise_scale: float = 0.01,
        device: str = None,
        use_adam: bool = True,
        adam_lr: float = 0.01,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        balance_classes: bool = True,
        use_full_energy: bool = True,
        sgld_batch_size: int = 1024,
        **kwargs
    ):
        super().__init__()
        self.n_sgld_steps = n_steps
        self.sgld_step_size = step_size
        self.sgld_noise_scale = noise_scale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_adam = use_adam
        self.adam_lr = adam_lr
        self.adam_betas = adam_betas
        self.balance_classes = balance_classes
        self.use_full_energy = use_full_energy
        self.sgld_batch_size = sgld_batch_size

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.optimizer = None
        self.is_fitted = False
        self._train_df = None
        self._target_col = None

        self.original_num_cols = []
        self.original_cat_cols = []
        self.final_feature_order = []
        self.cat_dummy_mapping: Dict[str, List[str]] = {}
        self.X_train_processed = None
        self.y_train_encoded = None
        self._tabpfn_cache = {}

    # -----------------------
    # Training / Preprocessing
    # -----------------------
    def train(self, output_dir: str, *args, **kwargs) -> 'TabPFGen':
        self.check_dependencies()
        train_csv_path = os.path.join(output_dir, "train_full.csv")
        print(f"[TabPFGen] Loading data from {train_csv_path}...")
        df = pd.read_csv(train_csv_path)
        self._train_df = df.copy()
        target_col = df.columns[-1]
        self._target_col = target_col

        # Separate numeric and categorical
        all_features = [c for c in df.columns if c != target_col]
        self.original_cat_cols, self.original_num_cols = [], []
        for col in all_features:
            is_cat = pd.api.types.is_object_dtype(df[col]) or \
                     pd.api.types.is_categorical_dtype(df[col]) or \
                     (pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() < 20)
            if is_cat:
                self.original_cat_cols.append(col)
            else:
                self.original_num_cols.append(col)

        # Numeric
        X_parts = []
        if self.original_num_cols:
            X_num = df[self.original_num_cols].copy()
            self.num_imputer = SimpleImputer(strategy='mean')
            X_num_vals = self.num_imputer.fit_transform(X_num)
            X_parts.append(pd.DataFrame(X_num_vals, columns=self.original_num_cols))

        # Categorical dummies
        self.cat_dummy_mapping = {}
        for cat_col in self.original_cat_cols:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col, dummy_na=True).astype(float)
            self.cat_dummy_mapping[cat_col] = dummies.columns.tolist()
            X_parts.append(dummies)

        if not X_parts:
            raise ValueError("No features found in dataset.")

        X_processed = pd.concat(X_parts, axis=1)
        self.final_feature_order = X_processed.columns.tolist()
        self.scaler = StandardScaler()
        self.X_train_processed = self.scaler.fit_transform(X_processed)
        self.y_train_encoded = self.label_encoder.fit_transform(df[target_col])
        self.is_fitted = True

        # Optional auto-generate synthetic data
        if kwargs.get('auto_generate_synthetic', True) and kwargs.get('synthetic_dir'):
            self.sample(
                num_samples=kwargs.get('num_synthetic_samples', len(df)), 
                synthetic_dir=kwargs['synthetic_dir']
            )
        return self

    # -----------------------
    # Core Energy Computation
    # -----------------------
    def _compute_energy_core(self, x_synth, y_synth, x_train, y_train, clf):
        batch_size = x_synth.shape[0]
        y_synth_clamped = torch.clamp(y_synth, 0, clf.n_classes_ - 1).long()
        probs_tensor = torch.tensor(
            clf.predict_proba(x_synth.cpu().numpy()),
            device=x_synth.device,
            dtype=torch.float32
        )
        class_probs = probs_tensor[torch.arange(batch_size), y_synth_clamped]
        return -torch.log(class_probs + 1e-10)

    def _compute_energy_full(self, x_synth, y_synth, x_train, y_train, clf):
        energy_core = self._compute_energy_core(x_synth, y_synth, x_train, y_train, clf)
        energy_swap = self._compute_energy_core(x_train, y_train, x_synth, y_synth, clf)
        return energy_core + energy_swap.mean()

    # -----------------------
    # SGLD Step
    # -----------------------
    def _sgld_step(self, x_batch, y_batch, x_train, y_train, clf):
        x_batch = x_batch.clone().requires_grad_(True)
        if self.use_full_energy:
            energy = self._compute_energy_full(x_batch, y_batch, x_train, y_train, clf)
        else:
            energy = self._compute_energy_core(x_batch, y_batch, x_train, y_train, clf)
        grad = torch.autograd.grad(energy.sum(), x_batch)[0]
        if grad is None: grad = torch.zeros_like(x_batch)

        if self.use_adam:
            optimizer = torch.optim.Adam([x_batch], lr=self.adam_lr, betas=self.adam_betas)
            optimizer.zero_grad()
            x_batch.grad = grad
            optimizer.step()
            x_new = x_batch.detach()
        else:
            x_new = x_batch - self.sgld_step_size * grad

        noise = torch.randn_like(x_new) * np.sqrt(2 * self.sgld_step_size)
        x_new += self.sgld_noise_scale * noise
        return x_new.detach()

    # -----------------------
    # Sampling
    # -----------------------
    def sample(self, num_samples: int, synthetic_dir: str = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not trained.")

        print(f"\n[TabPFGen] Generating {num_samples} samples...")
        x_train = torch.tensor(self.X_train_processed, device=self.device, dtype=torch.float32)
        y_train = torch.tensor(self.y_train_encoded, device=self.device, dtype=torch.long)

        # --- Class balancing init ---
        if self.balance_classes:
            classes, counts = np.unique(self.y_train_encoded, return_counts=True)
            max_count = counts.max()
            x_list, y_list = [], []
            for cls in classes:
                idx = np.where(self.y_train_encoded == cls)[0]
                sample_idx = np.random.choice(idx, size=max_count, replace=True)
                x_init = x_train[sample_idx] + torch.randn(max_count, x_train.shape[1], device=self.device) * 0.1
                y_init = torch.full((max_count,), cls, device=self.device, dtype=torch.long)
                x_list.append(x_init)
                y_list.append(y_init)
            x_synth = torch.cat(x_list, dim=0)[:num_samples]
            y_synth = torch.cat(y_list, dim=0)[:num_samples]
        else:
            x_synth = torch.randn(num_samples, x_train.shape[1], device=self.device)
            y_synth = torch.randint(0, len(self.label_encoder.classes_), (num_samples,), device=self.device)

        # --- Fit TabPFN once ---
        clf = TabPFNClassifier(device="cpu")
        clf.fit(self.X_train_processed, self.y_train_encoded)

        # --- SGLD Loop ---
        for step in tqdm(range(self.n_sgld_steps), desc="Sampling (SGLD)"):
            perm = torch.randperm(num_samples)
            x_synth, y_synth = x_synth[perm], y_synth[perm]

            x_batches = []
            for i in range(0, num_samples, self.sgld_batch_size):
                x_batch = x_synth[i:i+self.sgld_batch_size]
                y_batch = y_synth[i:i+self.sgld_batch_size]
                x_batch_new = self._sgld_step(x_batch, y_batch, x_train, y_train, clf)
                x_batches.append(x_batch_new)
            x_synth_new = torch.cat(x_batches, dim=0)
            x_synth[perm] = x_synth_new  # map back to original order

        # --- Final prediction ---
        clf_final = TabPFNClassifier(device="cpu")
        clf_final.fit(self.X_train_processed, self.y_train_encoded)
        y_synth_refined = clf_final.predict(x_synth.cpu().numpy())

        # --- Reconstruct DataFrame ---
        X_synth_descaled = self.scaler.inverse_transform(x_synth.cpu().numpy())
        df_synth_raw = pd.DataFrame(X_synth_descaled, columns=self.final_feature_order)
        output_data = {}
        for col in self.original_num_cols:
            output_data[col] = df_synth_raw[col].values
        for cat_col, dummy_cols in self.cat_dummy_mapping.items():
            subset = df_synth_raw[dummy_cols]
            recovered_vals = subset.idxmax(axis=1).str.split('_').str[-1]
            output_data[cat_col] = recovered_vals.values

        df_final = pd.DataFrame(output_data)
        df_final[self._target_col] = self.label_encoder.inverse_transform(y_synth_refined)
        df_final = df_final[self._train_df.columns]

        if synthetic_dir:
            os.makedirs(synthetic_dir, exist_ok=True)

            target_col = df_final.columns[-1]  # fixed df_synth -> df_final
            X_synth = df_final.drop(columns=[target_col])
            Y_synth = df_final[[target_col]]

            # Fallback for collapsed target distribution
            if Y_synth[target_col].nunique() <= 1 and hasattr(self, '_train_df') and self._train_df is not None:
                print("[WARNING] Target collapsed to single class, resampling from original")
                orig_y = self._train_df[target_col]
                sampled = np.random.choice(orig_y.values, size=len(Y_synth), replace=True)
                Y_synth = pd.DataFrame(sampled, columns=[target_col])

            X_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            Y_synth.to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)

            print("\n[TabSyn] Synthetic data saved:")
            print("  X ->", os.path.join(synthetic_dir, "x_synth.csv"))
            print("  y ->", os.path.join(synthetic_dir, "y_synth.csv"))
        return df_final

    # -----------------------
    # Placeholder Evaluate
    # -----------------------
    def evaluate(self, *args, **kwargs) -> Any:
        return 0.0

    
    