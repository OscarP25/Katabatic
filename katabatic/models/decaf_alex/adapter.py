import pandas as pd
import torch
import os
from typing import Union, Optional
from .models import DECAF 
from katabatic.models.base_model import Model as BaseModel

class KatabaticDECAF(BaseModel):
    def __init__(self, epochs=50, batch_size=64, dag=None, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.dag = dag
        self.model = None
        self.train_data = None
        self.target_col = None
        self.columns = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, X: Union[pd.DataFrame, str], y: Optional[Union[pd.Series, pd.DataFrame]] = None, **kwargs):
        # 1. Load Data
        if isinstance(X, str):
            print(f"Loading DECAF training data from: {X}")
            try:
                X_df = pd.read_csv(os.path.join(X, 'x_train.csv'), skipinitialspace=True)
                y_df = pd.read_csv(os.path.join(X, 'y_train.csv'), skipinitialspace=True)
            except FileNotFoundError:
                raise FileNotFoundError(f"Pipeline artifacts missing in {X}")
            
            if isinstance(y_df, pd.DataFrame) and y_df.shape[1] == 1:
                self.target_col = y_df.columns[0]
                data = pd.concat([X_df, y_df], axis=1)
            else:
                data = X_df.copy()
                self.target_col = 'target'
                data[self.target_col] = y_df.values
        else:
            data = X.copy()
            if y is not None:
                if isinstance(y, pd.Series):
                    self.target_col = y.name or 'target'
                    data[self.target_col] = y
                else:
                    self.target_col = y.columns[0]
                    data[self.target_col] = y.iloc[:,0]

        # Store for recovery
        self.train_data = data
        self.columns = data.columns.tolist()

        # 2. Configure DAG
        dag_input = kwargs.get('dag', self.dag)
        if dag_input is None:
            raise ValueError("DECAF requires a DAG structure (list of edges).")

        # Map string column names to integer indices
        col_to_idx = {name: i for i, name in enumerate(self.columns)}
        
        dag_idxs = []
        for edge in dag_input:
            src, dst = edge[0], edge[1]
            src_idx = col_to_idx[src] if isinstance(src, str) else src
            dst_idx = col_to_idx[dst] if isinstance(dst, str) else dst
            dag_idxs.append([src_idx, dst_idx])

        # 3. Initialise model 
        print(f"Initializing DECAF (Dims:{data.shape[1]}, DAG Edges:{len(dag_idxs)}) on {self.device}...")
        
        self.model = DECAF(
            input_dim=data.shape[1], 
            dag_seed=dag_idxs,
            batch_size=kwargs.get('batch_size', self.batch_size),
            lr=kwargs.get('lr', 1e-3),
            device=self.device
        )
        
        # 4. Train model
        epochs = kwargs.get('epochs', self.epochs)
        print(f"Training DECAF on {len(data)} rows for {epochs} epochs...")
        
        self.model.train(
            data.values, 
            epochs=epochs
        )

        # 5. Generate & save artifacts
        synthetic_dir = kwargs.get('synthetic_dir')
        if synthetic_dir:
            n_samples = kwargs.get('n_samples', 1000)
            print(f"Generating DECAF synthetic data to: {synthetic_dir}")
            os.makedirs(synthetic_dir, exist_ok=True)
            
            synth_df = self.sample(n_samples)

            if not self.target_col:
                fairness_cfg = kwargs.get('fairness_config', {})
                self.target_col = fairness_cfg.get('Y')

            if self.target_col and self.target_col in synth_df.columns:
                y_synth = synth_df[self.target_col]
                x_synth = synth_df.drop(columns=[self.target_col])
                
                print(f"Saved split artifacts: x_synth ({x_synth.shape}), y_synth ({y_synth.shape})")
                
                x_synth.to_csv(os.path.join(synthetic_dir, 'x_synth.csv'), index=False)
                y_synth.to_csv(os.path.join(synthetic_dir, 'y_synth.csv'), index=False)
            else:
                synth_df.to_csv(os.path.join(synthetic_dir, 'synthetic.csv'), index=False)

    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        
        # Generate raw numpy array
        synth_np = self.model.generate(self.train_data.values, n_samples)
        
        # Convert back to DataFrame
        synth_df = pd.DataFrame(synth_np, columns=self.columns)

        # Enforce int for discrete columns
        if self.train_data is not None:
            for col in self.columns:
                # Get the original dtype
                orig_dtype = self.train_data[col].dtype
                
                # If originally int
                if pd.api.types.is_integer_dtype(orig_dtype):
                    synth_df[col] = synth_df[col].round().astype(int)
                    # Clip [0, 1]
                    synth_df[col] = synth_df[col].clip(self.train_data[col].min(), self.train_data[col].max())
                
                # Target column
                elif col == self.target_col:
                    synth_df[col] = synth_df[col].round().astype(int)
                    synth_df[col] = synth_df[col].clip(self.train_data[col].min(), self.train_data[col].max())

        # Class recovery for mode collapse
        if self.train_data is not None and self.target_col is not None:
            real_classes = self.train_data[self.target_col].unique()
            synth_classes = synth_df[self.target_col].unique()
            
            missing_classes = set(map(str, real_classes)) - set(map(str, synth_classes))
            
            if missing_classes:
                print(f"Warning: Mode collapse detected in DECAF. Missing classes: {missing_classes}. Injecting recovery rows.")
                recovery_rows = []
                for cls_str in missing_classes:
                    mask = self.train_data[self.target_col].astype(str) == cls_str
                    if mask.any():
                        row = self.train_data[mask].iloc[[0]]
                        recovery_rows.append(row)
                
                if recovery_rows:
                    synth_df = pd.concat([synth_df] + recovery_rows, ignore_index=True)
                    synth_df = synth_df.sample(frac=1).reset_index(drop=True)

        return synth_df

    def evaluate(self, X, y=None, **kwargs):
        return {}