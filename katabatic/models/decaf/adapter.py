import pandas as pd
import numpy as np
import torch
import os
from typing import Union, Optional, Dict, List
from katabatic.models.base_model import Model as BaseModel
from .models import DECAF

class KatabaticDECAF(BaseModel):
    def __init__(self, epochs=50, batch_size=64, dag: Optional[List[List[str]]] = None, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.dag_config = dag or []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        
        # Internal storage
        self.columns = []     
        self.target_col = None 
        # FIX: Store constraints to prevent generation of invalid values (e.g. -1)
        self.col_constraints = {} 

    def train(self, X: Union[pd.DataFrame, str], y: Optional[Union[pd.Series, pd.DataFrame]] = None, **kwargs):
        """
        Loads data, trains DECAF, and generates valid split artifacts.
        """
        # 1. Load Data
        if isinstance(X, str):
            print(f"Loading DECAF training data from: {X}")
            try:
                X_df = pd.read_csv(os.path.join(X, 'x_train.csv'), skipinitialspace=True)
                y_df = pd.read_csv(os.path.join(X, 'y_train.csv'), skipinitialspace=True)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Pipeline artifacts missing in {X}. {e}")
            
            if y_df.shape[1] == 1:
                y_df = y_df.iloc[:, 0]
            
            self.fit(X_df, y_df, **kwargs)
        else:
            self.fit(X, y, **kwargs)

        # 2. Generate Artifacts
        synthetic_dir = kwargs.get('synthetic_dir')
        if synthetic_dir:
            print(f"Generating DECAF synthetic data to: {synthetic_dir}")
            os.makedirs(synthetic_dir, exist_ok=True)
            
            n_samples = kwargs.get('n_samples', 1000) 
            synth_df = self.sample(n_samples)
            
            # Split X and Y using the detected target column
            target_name = self.target_col
            
            if not target_name:
                fairness_cfg = kwargs.get('fairness_config', {})
                target_name = fairness_cfg.get('Y') or kwargs.get('target_col', 'target')

            if target_name and target_name in synth_df.columns:
                y_synth = synth_df[target_name]
                x_synth = synth_df.drop(columns=[target_name])
                
                x_synth.to_csv(os.path.join(synthetic_dir, 'x_synth.csv'), index=False)
                y_synth.to_csv(os.path.join(synthetic_dir, 'y_synth.csv'), index=False)
                print(f"Saved split artifacts: x_synth ({x_synth.shape}), y_synth ({y_synth.shape})")
            else:
                print(f"Warning: Target '{target_name}' not found. Saving single file.")
                synth_df.to_csv(os.path.join(synthetic_dir, 'synthetic.csv'), index=False)

    def evaluate(self, X, y=None, **kwargs):
        return {}

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame], **kwargs):
        # 1. Clean Headers
        X = X.copy()
        X.columns = X.columns.str.strip()
        
        # 2. Prepare Joint Data
        data = X.copy()
        
        y_name = 'target'
        if isinstance(y, pd.Series):
            y_name = y.name or 'target'
            data[y_name] = y.values
        elif isinstance(y, pd.DataFrame):
            y_name = y.columns[0]
            data[y_name] = y.iloc[:, 0].values
        else:
            data[y_name] = y
            
        self.target_col = y_name.strip()
        self.columns = data.columns.tolist()
        
        # FIX: Capture Min/Max constraints from Training Data
        # This allows us to clip generated values like -0.5 to 0.0
        for col in self.columns:
            self.col_constraints[col] = {
                'min': data[col].min(),
                'max': data[col].max()
            }
        
        data_np = data.values.astype(np.float32)
        
        # 3. Parse DAG
        dag_config = kwargs.get('dag', self.dag_config)
        dag_indices = []
        
        if dag_config:
            col_map = {name: i for i, name in enumerate(self.columns)}
            for parent, child in dag_config:
                p_clean = parent.strip()
                c_clean = child.strip()
                if p_clean in col_map and c_clean in col_map:
                    dag_indices.append([col_map[p_clean], col_map[c_clean]])
        
        # 4. Init & Train
        epochs = kwargs.get('epochs', self.epochs)
        print(f"Initializing DECAF (Dims:{data_np.shape[1]}, DAG Edges:{len(dag_indices)})...")
        
        self.model = DECAF(
            input_dim=data_np.shape[1],
            dag_seed=dag_indices,
            h_dim=200,
            batch_size=self.batch_size,
            lr=1e-3,
            device=self.device
        )
        
        print(f"Training DECAF on {len(data)} rows for {epochs} epochs...")
        self.model.train(data_np, epochs=epochs)

    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        if not self.model:
            raise RuntimeError("Model not fitted")
            
        gen_np = self.model.generate(None, n_samples)
        
        # Create DataFrame to handle columns safely
        df_gen = pd.DataFrame(gen_np, columns=self.columns)
        
        # FIX: Clip values to valid range [min, max] BEFORE rounding
        # This prevents -0.1 becoming -1, ensuring strict [0, 1] for binary targets.
        for col in self.columns:
            if col in self.col_constraints:
                c_min = self.col_constraints[col]['min']
                c_max = self.col_constraints[col]['max']
                df_gen[col] = df_gen[col].clip(lower=c_min, upper=c_max)

        # Now safe to round and cast
        return df_gen.round().astype(int)