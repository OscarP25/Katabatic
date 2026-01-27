import pandas as pd
import numpy as np
import os
import builtins
from typing import Union, Optional

if not hasattr(builtins, 'np'):
    builtins.np = np

# DataSynthesizer imports
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

from katabatic.models.base_model import Model as BaseModel

class PrivBayes(BaseModel):
    def __init__(self, epsilon=1.0, degree_of_bayesian_network=2, **kwargs):
        """
        epsilon: Privacy budget (lower = more private, less utility).
        degree_of_bayesian_network (k): Max number of parents in BN. k=2 is standard.
        """
        super().__init__()
        self.epsilon = float(epsilon)
        self.k = int(degree_of_bayesian_network)
        
        self.description_file = None
        self.train_data = None
        self.target_col = None
        self.columns = None
        self.col_map = None 

    def _sanitize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Casts to string.
        2. Prefixes ALL values with 'v' (e.g. 0 -> 'v0') to ensure they are read as strings.
        3. Renames columns to 'col_0', etc.
        """
        df = df.copy()
        df = df.astype(str)
        
        for col in df.columns:
            df[col] = "v" + df[col]

        new_cols = {}
        for col in df.columns:
            if str(col).isdigit() or str(col) == self.target_col:
                new_cols[col] = f"col_{col}"
            else:
                new_cols[col] = f"col_{col}"
        
        self.col_map = new_cols
        df.rename(columns=new_cols, inplace=True)
        return df

    def _desanitize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reverses sanitization: strips 'v' prefix and restores column names/types.
        """
        df = df.copy()
        
        for col in df.columns:
            df[col] = df[col].str.slice(start=1)
            df[col] = pd.to_numeric(df[col], errors='ignore')

        # Restore column names
        if self.col_map:
            inv_map = {v: k for k, v in self.col_map.items()}
            df.rename(columns=inv_map, inplace=True)
            
        return df

    def train(self, X: Union[pd.DataFrame, str], y: Optional[Union[pd.Series, pd.DataFrame]] = None, **kwargs):
        pd.set_option('compute.use_numexpr', False)

        # 1. Load Data
        if isinstance(X, str):
            print(f"Loading PrivBayes data from: {X}")
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

        # Store metadata
        self.train_data = data
        self.columns = data.columns.tolist()

        # Sanitise data
        data_safe = self._sanitize_dataset(data)

        # 2. Setup paths
        synthetic_dir = kwargs.get('synthetic_dir', 'synthetic/privbayes_temp')
        os.makedirs(synthetic_dir, exist_ok=True)
        
        temp_csv_path = os.path.join(synthetic_dir, 'temp_train_data.csv')
        data_safe.to_csv(temp_csv_path, index=False)
        
        self.description_file = os.path.join(synthetic_dir, 'description.json')

        # 3. Train
        print(f"Training PrivBayes (epsilon={self.epsilon}, k={self.k}) on {len(data)} rows...")
        
        # Explicitly map all columns to categorical
        categorical_map = {col: True for col in data_safe.columns}
        
        describer = DataDescriber(category_threshold=0)
        try:
            describer.describe_dataset_in_correlated_attribute_mode(
                dataset_file=temp_csv_path,
                epsilon=self.epsilon,
                k=self.k,
                attribute_to_is_categorical=categorical_map
            )
        except Exception as e:
            if os.path.exists(temp_csv_path): os.remove(temp_csv_path)
            raise RuntimeError(f"DataSynthesizer failed: {str(e)}") from e
        
        describer.save_dataset_description_to_file(self.description_file)
        print(f"PrivBayes description saved to {self.description_file}")

        # 4. Generate & save
        if synthetic_dir:
            n_samples = kwargs.get('n_samples', len(data))
            print(f"Generating synthetic data to: {synthetic_dir}")
            
            synth_df = self.sample(n_samples)

            if not self.target_col:
                fairness_cfg = kwargs.get('fairness_config', {})
                tgt = fairness_cfg.get('Y')
                if tgt and tgt in self.col_map:
                    self.target_col = self.col_map[tgt]
                else:
                    self.target_col = tgt

            if self.target_col and self.target_col in synth_df.columns:
                y_synth = synth_df[self.target_col]
                x_synth = synth_df.drop(columns=[self.target_col])
                
                x_synth.to_csv(os.path.join(synthetic_dir, 'x_synth.csv'), index=False)
                y_synth.to_csv(os.path.join(synthetic_dir, 'y_synth.csv'), index=False)
            else:
                synth_df.to_csv(os.path.join(synthetic_dir, 'synthetic.csv'), index=False)
                
            if os.path.exists(temp_csv_path): os.remove(temp_csv_path)

    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        if self.description_file is None or not os.path.exists(self.description_file):
            raise RuntimeError("Model description not found. Train first.")
        
        pd.set_option('compute.use_numexpr', False)
        
        generator = DataGenerator()
        temp_synth_path = self.description_file.replace('description.json', 'temp_synth.csv')
        
        generator.generate_dataset_in_correlated_attribute_mode(
            n_samples,
            self.description_file
        )
        
        generator.save_synthetic_data(temp_synth_path)
        synth_df_safe = pd.read_csv(temp_synth_path)
        
        if os.path.exists(temp_synth_path): os.remove(temp_synth_path)

        # Desanitise
        synth_df = self._desanitize_dataset(synth_df_safe)
        
        # Ensure column order
        synth_df = synth_df[self.columns]

        # Class recovery in case of mode collapse
        if self.train_data is not None and self.target_col is not None:
            real_classes = self.train_data[self.target_col].unique()
            synth_classes = synth_df[self.target_col].unique()
            
            # Use string map for comparison
            missing_classes = set(map(str, real_classes)) - set(map(str, synth_classes))
            
            if missing_classes:
                print(f"Warning: Mode collapse in PrivBayes. Missing: {missing_classes}. Injecting rows.")
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