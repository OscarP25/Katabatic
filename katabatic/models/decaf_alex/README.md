# DECAF: Generating Fair Synthetic Data Using Causally-Aware Generative Networks

DECAF is a causal generative model that uses directed acyclic graphs to generate fair synthetic data and enable inference of counterfactual fairness.

## Paper

**DECAF: Generating Fair Synthetic Data Using Causally-Aware Generative Networks**  
Breugel et al., 2021
[arXiv:2110.12884](https://arxiv.org/abs/2110.12884)
[Repository](https://github.com/trentkyono/DECAF)

## Installation

```bash
poetry install --extras decaf
```

## Usage

```python
import torch
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.decaf_alex.adapter import KatabaticDECAF

# Device Config
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Select protected and target attributes
protected_col = input("Protected Attribute (S): ").strip() or "sex"
target_col = input("Target Attribute (Y): ").strip() or "class"

# Construct Causal DAG with edges [Parent -> Child]
# DECAF uses this to mask the generator weights.
adult_dag = [
    ['age', 'marital-status'],
    ['sex', 'marital-status'],
    ['sex', 'education'],
    ['race', 'education'],
    ['education', 'occupation'],
    ['education', 'hours-per-week'],
    ['marital-status', 'relationship'],
    ['occupation', 'class'],
    ['hours-per-week', 'class'],
    # edge for [protected attribute -> target] (to be debiased later)
    [protected_col, target_col] 
]

# DECAF parameters
model_config = {
    "epochs": 50,
    "batch_size": 64,
    "dag": adult_dag,

    "fairness_config": {
        "S": protected_col,
        "Y": target_col,
        "S_under": "0", #underrepresented class
        "Y_desire": "1" #class to align to
    }
}

pipeline = TrainTestSplitPipeline(model=KatabaticDECAF)

pipeline.run(
    input_csv='data/my_dataset.csv',
    output_dir='sample_data/my_dataset',
    synthetic_dir='synthetic/my_dataset/codi',
    real_test_dir='sample_data/my_dataset'
    **model_config
)

```

See `examples/decaf.ipynb` for more examples.
