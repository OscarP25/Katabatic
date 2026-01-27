# TabFairGDT: A Fast Fair Tabular Data Generator using Autoregressive Decision Trees

TabFairGDT is an autoregression gradient-boosted decision tree that synthesises fair tabular data, optimising for either utility or group fairness. Due to it being a decision tree, it is explainable.

## Paper

**TABFAIRGDT: A Fast Fair Tabular Data Generator using Autoregressive Decision Trees**
Panagiotou et al., IEEE ICDM 2025
[arXiv:2509.19927](https://arxiv.org/abs/2509.19927
)
[Repository](https://github.com/Panagiotou/TABFAIRGDT)

## Installation

```bash
poetry install --extras tabfairgdt
```

## Usage

```python

import torch
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.tabfairgdt_alex import TabFairGDT

# Device Config
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# set protected attribute
protected_col = input("Protected Attribute (S): ").strip()

# TabFairGDT parameters
model_config = {
    "protected_attribute": protected_col,
    "lambda_val": 1 #fairness constraint. 0 = max utility, 1 = max fairness
}

pipeline = TrainTestSplitPipeline(model=TabFairGDT)

pipeline.run(
    input_csv='data/my_dataset.csv',
    output_dir='sample_data/my_dataset',
    synthetic_dir='synthetic/my_dataset/codi',
    real_test_dir='sample_data/my_dataset'
    **model_config
)
```

See `examples/tabfairgdt.ipynb` for more examples.
