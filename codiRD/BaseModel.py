import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import joblib

# === Load dataset ===
path = "Iris.csv"   # change if needed
df = pd.read_csv(path)

print("First 5 rows of the dataset:")
print(df.head())

# === Prepare features and labels ===
drop_cols = [c for c in ['Id', 'Unnamed: 0'] if c in df.columns]
X = df.drop(drop_cols + ['Species'], axis=1).values
y_raw = df['Species'].values

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y_raw)
num_classes = len(le.classes_)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Build and train neural net ===
clf = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=50,       # 50 epochs
    random_state=42,
    verbose=False
)

start_time = time.time()
clf.fit(X_train, y_train)
end_time = time.time()
train_time_sec = end_time - start_time

# === Predictions ===
y_pred = clf.predict(X_test)

# Probabilities (for AUC)
y_prob = clf.predict_proba(X_test)

# === Metrics ===
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')

try:
    auc_macro = roc_auc_score(
        pd.get_dummies(y_test), y_prob,
        average='macro', multi_class='ovr'
    )
except Exception:
    auc_macro = None

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

# === Results ===
print(f"\nTraining time: {train_time_sec:.4f} seconds for 50 epochs")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Macro F1 score: {f1_macro:.4f}")
print(f"Macro precision: {precision_macro:.4f}")
print(f"Macro recall: {recall_macro:.4f}")
print(f"Macro AUC (OVR): {auc_macro if auc_macro is not None else 'Could not compute'}\n")

print("Classification report:\n", report)
print("Confusion matrix:\n", cm)

# === Plot loss curve ===
if hasattr(clf, 'loss_curve_') and len(clf.loss_curve_) > 0:
    plt.figure()
    plt.plot(clf.loss_curve_)
    plt.title('MLPClassifier Loss Curve (50 epochs)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

# === Save model for reuse ===
joblib.dump({'model': clf, 'scaler': scaler, 'label_encoder': le}, "iris_mlp_model.joblib")
print("Saved trained model + scaler + label encoder to iris_mlp_model.joblib")
