# katabatic/evaluate/tstr/evaluation.py

import os
import csv
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from katabatic.evaluate.base_evaluation import Evaluation


# -----------------------------
# IO
# -----------------------------
def load_data(synthetic_dir, real_test_dir):
    x_synth = pd.read_csv(os.path.join(synthetic_dir, "x_synth.csv"))
    y_synth = pd.read_csv(os.path.join(synthetic_dir, "y_synth.csv")).values.ravel()

    x_test = pd.read_csv(os.path.join(real_test_dir, "x_test.csv"))
    y_test = pd.read_csv(os.path.join(real_test_dir, "y_test.csv")).values.ravel()

    return x_synth, y_synth, x_test, y_test


# -----------------------------
# Preprocessing
# -----------------------------
def _safe_onehot_encoder():
    """sklearn compatibility: sparse_output (new) vs sparse (old)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in x_train.columns if c not in numeric_cols]

    ohe = _safe_onehot_encoder()

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", ohe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def encode_y_from_train_only(y_train_raw, y_test_raw):
    """
    Encode y so that y_train is always contiguous: 0..K-1 based on TRAIN classes only.
    Filter test rows containing unseen classes (cannot be evaluated properly).
    """
    y_train_raw = np.asarray(y_train_raw).astype(str)
    y_test_raw = np.asarray(y_test_raw).astype(str)

    train_classes = np.unique(y_train_raw)
    mapping = {c: i for i, c in enumerate(train_classes)}

    y_train = np.array([mapping[c] for c in y_train_raw], dtype=int)

    mask = np.array([c in mapping for c in y_test_raw], dtype=bool)
    y_test = np.array([mapping[c] for c in y_test_raw[mask]], dtype=int)

    return y_train, y_test, mask, train_classes


# -----------------------------
# Evaluation
# -----------------------------
class TSTREvaluation(Evaluation):
    def __init__(self, synthetic_dir, real_test_dir, **kwargs):
        super().__init__(model=None, dataset=None, **kwargs)
        self.synthetic_dir = synthetic_dir
        self.real_test_dir = real_test_dir
        self.x_train, self.y_train, self.x_test, self.y_test = load_data(
            synthetic_dir, real_test_dir
        )

    def evaluate(self):
        # Ensure DF types
        if not isinstance(self.x_train, pd.DataFrame):
            self.x_train = pd.DataFrame(self.x_train)
        if not isinstance(self.x_test, pd.DataFrame):
            self.x_test = pd.DataFrame(self.x_test)

        # Encode y based on TRAIN ONLY
        y_train_enc, y_test_enc, test_mask, train_classes = encode_y_from_train_only(
            self.y_train, self.y_test
        )

        # Coverage diagnostic (real test labels found in synthetic labels)
        y_test_raw = np.asarray(self.y_test).astype(str)
        test_unique = set(np.unique(y_test_raw))
        train_unique = set(np.unique(np.asarray(self.y_train).astype(str)))
        coverage = (len(test_unique & train_unique) / max(len(test_unique), 1))

        # Align x_test columns to synthetic x_train columns (prevents feature mismatch)
        x_test_aligned = self.x_test.copy()
        for c in self.x_train.columns:
            if c not in x_test_aligned.columns:
                x_test_aligned[c] = np.nan
        x_test_aligned = x_test_aligned[self.x_train.columns]

        # Filter x_test to match y_test_enc
        x_test_filtered = x_test_aligned.iloc[test_mask].reset_index(drop=True)

        # Special case: no rows left after filtering
        if len(y_test_enc) == 0 or len(x_test_filtered) == 0:
            msg = (
                "⚠️ TSTR cannot run: 0 test rows remained after filtering unseen classes.\n"
                f"Train classes only (from synthetic): {list(train_classes)}\n"
                f"Coverage of real test labels by synthetic labels: {coverage:.4f}"
            )
            print("\n" + msg)
            self._save_special_case(msg, extra={"Coverage": coverage})
            self.results_ = {"SpecialCase": {"Message": msg, "Coverage": coverage}}
            return self.results_

        n_classes_train = len(train_classes)

        # Special case: synthetic y_train has only 1 class
        if n_classes_train < 2:
            y_pred = np.zeros_like(y_test_enc)
            acc = accuracy_score(y_test_enc, y_pred)
            f1w = f1_score(y_test_enc, y_pred, average="weighted")
            f1m = f1_score(y_test_enc, y_pred, average="macro")

            msg = (
                "⚠️ Synthetic y_train has ONLY ONE class -> skipped LR/MLP/RF/XGBoost.\n"
                "Baseline predicts class: 0\n"
                f"Accuracy: {acc:.4f}\n"
                f"F1 Weighted: {f1w:.4f}\n"
                f"F1 Macro: {f1m:.4f}"
            )

            print("\nTSTR Evaluation Results:")
            print(f"Train classes only: {list(train_classes)}")
            print(msg)

            out = {"Baseline": {"Accuracy": float(acc), "F1 Score": float(f1w), "F1 Macro": float(f1m), "Coverage": float(coverage)}}
            self.save_results_to_csv(out, self.synthetic_dir, note=msg)
            self.results_ = out
            return out

        # Build preprocessor + scale where needed
        preprocessor = build_preprocessor(self.x_train)
        X_train_enc = preprocessor.fit_transform(self.x_train)
        X_test_enc = preprocessor.transform(x_test_filtered)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_enc)
        X_test_scaled = scaler.transform(X_test_enc)

        # Models
        models = {
            "LR": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "MLP": MLPClassifier(max_iter=200),
            "RF": RandomForestClassifier(n_estimators=200, n_jobs=-1),
        }

        # Optional XGBoost (don’t crash if xgboost missing)
        try:
            from xgboost import XGBClassifier

            if n_classes_train == 2:
                num_neg = np.sum(y_train_enc == 0)
                num_pos = np.sum(y_train_enc == 1)
                scale_pos_weight = (num_neg / num_pos) if num_pos > 0 else 1.0
                models["XGBoost"] = XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    tree_method="hist",
                )
            else:
                models["XGBoost"] = XGBClassifier(
                    objective="multi:softprob",
                    num_class=n_classes_train,
                    eval_metric="mlogloss",
                    tree_method="hist",
                )
        except Exception as e:
            # record as unavailable rather than failing
            models["XGBoost"] = None

        results = {}
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_enc)

        for name, model in models.items():
            if model is None:
                results[name] = {"Error": "xgboost not available", "Coverage": float(coverage)}
                continue

            try:
                if name in ["LR", "MLP"]:
                    if name == "MLP":
                        model.fit(X_train_scaled, y_train_enc, sample_weight=sample_weight)
                    else:
                        model.fit(X_train_scaled, y_train_enc)
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)
                else:
                    model.fit(X_train_enc, y_train_enc)
                    y_pred = model.predict(X_test_enc)
                    y_prob = model.predict_proba(X_test_enc)

                metrics = {
                    "Accuracy": float(accuracy_score(y_test_enc, y_pred)),
                    "F1 Score": float(f1_score(y_test_enc, y_pred, average="weighted")),
                    "F1 Macro": float(f1_score(y_test_enc, y_pred, average="macro")),
                    "Coverage": float(coverage),
                }

                if n_classes_train == 2 and y_prob is not None and y_prob.shape[1] >= 2:
                    metrics["AUC"] = float(roc_auc_score(y_test_enc, y_prob[:, 1]))

                results[name] = metrics

            except Exception as e:
                results[name] = {"Error": str(e), "Coverage": float(coverage)}

        # Save
        self.save_results_to_csv(results, self.synthetic_dir)

        print("\nTSTR Evaluation Results:")
        print(f"Train classes only: {list(train_classes)}")

        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric_name}: {value:.4f}")
                else:
                    print(f"{metric_name}: {value}")

        self.results_ = results
        return results

    def _save_special_case(self, message: str, extra: dict | None = None):
        payload = {"SpecialCase": {"Message": message}}
        if extra:
            payload["SpecialCase"].update(extra)
        self.save_results_to_csv(payload, self.synthetic_dir, note=message)

    @staticmethod
    def save_results_to_csv(results, synthetic_dir, note: str | None = None):
        parts = os.path.normpath(synthetic_dir).split(os.sep)
        model_name = parts[-1]
        dataset_name = parts[-2]

        results_dir = os.path.join("Results", dataset_name)
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, f"{model_name}_tstr.csv")

        # force UTF-8 so Windows cp1252 doesn’t crash on ⚠️ etc.
        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(["Model", "Metric", "Value"])
            if note is not None:
                writer.writerow(["__NOTE__", "Message", note])

            for m_name, metrics in results.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            value_out = round(float(value), 4)
                        else:
                            value_out = value
                        writer.writerow([m_name, metric_name, value_out])
                else:
                    writer.writerow([m_name, "Value", metrics])

        print(f"\nResults saved to: {output_path}")
