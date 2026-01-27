# VFAE: The Variational Fair Autoencoder

VFAE is a variational autoencoder architecture that disentangles semantic features from protected attributes to generate bias-free representations for downstream tasks.

## Paper

**The Variational Fair Autoencoder**  
Louizos et al., 2015  
[arXiv:1511.00830](https://arxiv.org/abs/1511.00830)
[Repository](https://github.com/yolomeus/dm-lab-2020-vfae)

## Installation

```bash
poetry install --extras vfae
```

## Usage

```python

import torch
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.vfae_alex.adapter import KatabaticVFAE

# Device Config
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# select protected and target attributes
protected_col = input("Protected Attribute (S): ").strip() or "sex"
target_col = input("Target Attribute (Y): ").strip() or "class"

# VFAE parameters
model_config = {
    "epochs": 50,
    "batch_size": 64,
    "z_dim": 50,         # Latent dimension size
    
    # Fairness Config
    "fairness_config": {
        "S": protected_col,       # Sensitive Column Name
        "Y": target_col,          # Target Column Name
        "S_under": "0",           # Value representing unprivileged group (e.g., '0' for Female)
        "Y_desire": "1"           # Value representing positive outcome (optional context)
    }
}

pipeline = TrainTestSplitPipeline(model=KatabaticVFAE)

pipeline.run(
    input_csv='data/my_dataset.csv',
    output_dir='sample_data/my_dataset',
    synthetic_dir='synthetic/my_dataset/codi',
    real_test_dir='sample_data/my_dataset'
    **model_config
)
```

See `examples/vafe.ipynb` for more examples.
