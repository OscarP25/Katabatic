import pandas as pd
import os
from typing import Union, Optional
from katabatic.models.base_model import Model as BaseModel
from .models import ARF

class KatabaticARF(BaseModel):
    def __init__(self, num_trees=30, max_iters=10, **kwargs):
        super().__init__()
        self.num_trees = num_trees
        self.max_iters = max_iters
        self.model_wrapper = None
        self.target_col = None
        self.columns = []

    def train(self, X: Union[pd.DataFrame, str], y: Optional[Union[pd.Series, pd.DataFrame]] = None, **kwargs):
        if isinstance(X, str):
            print(f"Loading ARF data from: {X}")
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

        synthetic_dir = kwargs.get('synthetic_dir')
        if synthetic_dir:
            print(f"Generating synthetic data to: {synthetic_dir}")
            os.makedirs(synthetic_dir, exist_ok=True)
            
            n_samples = kwargs.get('n_samples', 1000)
            synth_df = self.sample(n_samples)
            
            target_name = self.target_col
            if not target_name:
                fairness_cfg = kwargs.get('fairness_config', {})
                target_name = fairness_cfg.get('Y')

            if target_name and target_name in synth_df.columns:
                y_synth = synth_df[target_name]
                x_synth = synth_df.drop(columns=[target_name])
                x_synth.to_csv(os.path.join(synthetic_dir, 'x_synth.csv'), index=False)
                y_synth.to_csv(os.path.join(synthetic_dir, 'y_synth.csv'), index=False)
                print("Saved artifacts.")
            else:
                synth_df.to_csv(os.path.join(synthetic_dir, 'synthetic.csv'), index=False)

    def evaluate(self, X, y=None, **kwargs):
        return {}

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame], **kwargs):
        X = X.copy().reset_index(drop=True)
        X.columns = X.columns.str.strip()
        
        y_name = 'target'
        if isinstance(y, pd.Series): 
            y_name = y.name or 'target'
            y = y.reset_index(drop=True)
        elif isinstance(y, pd.DataFrame): 
            y_name = y.columns[0]
            y = y.reset_index(drop=True)
        
        self.target_col = y_name.strip()
        
        data = X.copy()
        data[self.target_col] = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y
        self.columns = data.columns.tolist()

        # Convert to category for ARF
        for col in data.columns:
            data[col] = data[col].astype('category')

        self.model_wrapper = ARF(
            num_trees=kwargs.get('num_trees', self.num_trees),
            max_iters=kwargs.get('max_iters', self.max_iters),
            min_node_size=kwargs.get('min_node_size', 5)
        )
        
        print(f"Training ARF on {len(data)} rows...")
        self.model_wrapper.train(data)

    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        if not self.model_wrapper:
            raise RuntimeError("Model not fitted")
        df_gen = self.model_wrapper.sample(n_samples)
        
        # Cast categorical outputs back to integers
        for col in df_gen.columns:
            if df_gen[col].dtype.name == 'category':
                if df_gen[col].isnull().any():
                    # Fill NaNs with mode. If mode NaN, fill with 0.
                    modes = df_gen[col].mode()
                    fill_val = modes[0] if not modes.empty else 0
                    df_gen[col] = df_gen[col].fillna(fill_val)
                
                df_gen[col] = df_gen[col].astype(int)
        return df_gen