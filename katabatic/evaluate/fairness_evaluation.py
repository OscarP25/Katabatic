import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Import the base class
from katabatic.evaluate.tstr.evaluation import TSTREvaluation

class FairnessEvaluation(TSTREvaluation):
    def __init__(self, synthetic_dir, real_test_dir, fairness_config=None, **kwargs):
        super().__init__(synthetic_dir, real_test_dir, **kwargs)
        self.fairness_config = fairness_config or {}
        
        # CRITICAL FIX: Clean x_test headers immediately upon loading
        # This ensures ' sex' becomes 'sex' to match your config
        self.x_test.columns = self.x_test.columns.str.strip()
        self.x_train.columns = self.x_train.columns.str.strip()

    def evaluate(self):
        # 1. Validation
        s_col = self.fairness_config.get('S')
        s_under = self.fairness_config.get('S_under')

        if not s_col or s_under is None:
            print(f"Skipping Fairness: Config missing 'S' or 'S_under'.")
            return super().evaluate()

        # Check if sensitive column exists
        if s_col not in self.x_test.columns:
            print(f"Skipping Fairness: Sensitive column '{s_col}' not found in test data.")
            print(f"Available columns: {self.x_test.columns.tolist()}")
            return super().evaluate()

        # 2. Setup Masks
        # Cast to string to ensure '0' (int) matches '0' (str config)
        sensitive_data = self.x_test[s_col].astype(str)
        s_under_str = str(s_under)
        
        unprivileged_mask = (sensitive_data == s_under_str)
        privileged_mask = ~unprivileged_mask
        
        # Debugging Counts
        n_unpriv = unprivileged_mask.sum()
        n_priv = privileged_mask.sum()
        
        if n_unpriv == 0 or n_priv == 0:
            print(f"Skipping Fairness: Group imbalance. Unprivileged='{s_under}' found {n_unpriv} times. Privileged found {n_priv} times.")
            print(f"Unique values in '{s_col}': {sensitive_data.unique()}")
            return super().evaluate()

        print(f"\n--- Fairness Evaluation Setup ---")
        print(f"Sensitive Attribute: '{s_col}'")
        print(f"Unprivileged Group : '{s_under}' (Count: {n_unpriv})")
        print(f"Privileged Group   : (Count: {n_priv})")
        print("-" * 30)

        results = {}
        
        # 3. Model Training Loop
        # Calculate class imbalance for XGBoost
        num_neg = np.sum(self.y_train == 0)
        num_pos = np.sum(self.y_train == 1)
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

        models = {
            "LR": LogisticRegression(max_iter=1000),
            "MLP": MLPClassifier(max_iter=1000),
            "RF": RandomForestClassifier(),
            "XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight)
        }

        for name, model in models.items():
            # Train
            if name in ["LR", "MLP"]:
                scaler = StandardScaler()
                x_train_scaled = scaler.fit_transform(self.x_train)
                x_test_scaled = scaler.transform(self.x_test)
                model.fit(x_train_scaled, self.y_train)
                y_pred = model.predict(x_test_scaled)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(x_test_scaled)[:, 1]
            else:
                model.fit(self.x_train, self.y_train)
                y_pred = model.predict(self.x_test)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(self.x_test)[:, 1]

            # --- Standard Metrics ---
            metrics = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'F1 Score': f1_score(self.y_test, y_pred, average='weighted')
            }
            if len(np.unique(self.y_test)) == 2:
                metrics['AUC'] = roc_auc_score(self.y_test, y_prob)

            # --- Fairness Metrics ---
            # 1. Demographic Parity Diff
            sr_unpriv = y_pred[unprivileged_mask].mean()
            sr_priv = y_pred[privileged_mask].mean()
            metrics['Demographic Parity Diff'] = sr_unpriv - sr_priv

            # 2. Equal Opportunity Diff (TPR Diff)
            y_test_array = self.y_test
            up_pos_mask = unprivileged_mask & (y_test_array == 1)
            p_pos_mask = privileged_mask & (y_test_array == 1)

            tpr_unpriv = y_pred[up_pos_mask].mean() if up_pos_mask.any() else 0.0
            tpr_priv = y_pred[p_pos_mask].mean() if p_pos_mask.any() else 0.0
            metrics['Equal Opportunity Diff'] = tpr_unpriv - tpr_priv

            results[name] = metrics

        # 4. Save and Print
        self.save_results_to_csv(results, self.synthetic_dir)
        self._print_results(results)
        
        return results

    def _print_results(self, results):
        print("\nTSTR + Fairness Evaluation Results:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")