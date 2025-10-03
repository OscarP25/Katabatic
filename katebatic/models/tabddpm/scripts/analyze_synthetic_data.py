import numpy as np
import pandas as pd
import json
import os

def load_synthetic_data(parent_dir, real_data_path=None):
    syn_path = parent_dir
    if not os.path.exists(syn_path):
        raise FileNotFoundError(f"Synthetic data not found in {syn_path}. Make sure you've run sampling first.")
    
    X_num_syn = None
    if os.path.exists(os.path.join(syn_path, "X_num_train.npy")):
        X_num_syn = np.load(os.path.join(syn_path, "X_num_train.npy"))
        print(f"Loaded numerical features: shape {X_num_syn.shape}")
    
    X_cat_syn = None
    if os.path.exists(os.path.join(syn_path, "X_cat_train.npy")):
        X_cat_syn = np.load(os.path.join(syn_path, "X_cat_train.npy"), allow_pickle=True)
        print(f"Loaded categorical features: shape {X_cat_syn.shape}")
    
    y_syn = None
    if os.path.exists(os.path.join(syn_path, "y_train.npy")):
        y_syn = np.load(os.path.join(syn_path, "y_train.npy"))
        print(f"Loaded target: shape {y_syn.shape}")
    
    info_path = os.path.join(real_data_path, "info.json") if real_data_path else None
    
    if info_path and os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
            
        num_cols = info.get("num_columns", [])
        cat_cols = info.get("cat_columns", [])
        target_col = info.get("target_column", "target")
        
        print(f"\nColumn names from info.json:")
        print(f"  Numerical columns: {num_cols}")
        print(f"  Categorical columns: {cat_cols}")
        print(f"  Target column: {target_col}")
    else:
        print("\nWarning: info.json not found, using generic column names")
        # Use generic column names if info.json not available
        num_cols = []
        cat_cols = []
        target_col = "target"
    
    all_columns = []
    all_data = []
    
    # Add numerical columns
    if X_num_syn is not None:
        if num_cols and len(num_cols) == X_num_syn.shape[1]:
            for i, col_name in enumerate(num_cols):
                all_columns.append(col_name)
                all_data.append(X_num_syn[:, i])
        else:
            print(f"Warning: Expected {X_num_syn.shape[1]} numerical columns but found {len(num_cols)} names. Using generic names.")
            for i in range(X_num_syn.shape[1]):
                all_columns.append(f"num_{i}")
                all_data.append(X_num_syn[:, i])
    
    if X_cat_syn is not None:
        if cat_cols and len(cat_cols) == X_cat_syn.shape[1]:
            # Use original column names if they match
            for i, col_name in enumerate(cat_cols):
                all_columns.append(col_name)
                all_data.append(X_cat_syn[:, i])
        else:
            # Fall back to generic names
            print(f"Warning: Expected {X_cat_syn.shape[1]} categorical columns but found {len(cat_cols)} names. Using generic names.")
            for i in range(X_cat_syn.shape[1]):
                all_columns.append(f"cat_{i}")
                all_data.append(X_cat_syn[:, i])
    
    if all_data:
        df = pd.DataFrame(dict(zip(all_columns, all_data)))
    else:
        df = pd.DataFrame()
    
    if y_syn is not None:
        df[target_col] = y_syn
    
    return df
