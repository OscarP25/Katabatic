import pandas as pd
from katabatic.models.pategan import PATEGAN

# 1. Load pre-split training data
X_train = pd.read_csv("sample_data/car/x_train.csv")
y_train = pd.read_csv("sample_data/car/y_train.csv").iloc[:, 0]

# 2. Initialise PATE-GAN with README-style defaults
model = PATEGAN(
    epsilon=1.0,      # Privacy budget
    delta=1e-5,       # Privacy parameter
    num_teachers=10,  # Teacher discriminators
    niter=10000,      # Training iterations
    batch_size=128,
    random_state=42
)

# 3. Train
model.fit(X_train, y_train, verbose=1)

# 4. Generate synthetic data
synthetic_data = model.sample(n=1000)
synthetic_data.to_csv("synthetic/car/pategan/standalone_synth.csv", index=False)

print("Standalone PATE-GAN training + sampling complete.")
