# ARF: Adversarial Random Forests

ARF (Adversarial Random Forests) is a non-parametric, tree-based generative model which iteratively trains random forest models to distinguish from synthetic data. The main advantage being it does not require a GPU to run, acting as a lightweight alternative to GANs.

## Paper

**Adversarial Random Forests for Density Estimation and Generative Modeling**  
Watson et al., AISTATS (PRML) 2023
[Link to paper](https://proceedings.mlr.press/v206/watson23a.html)
[Repository](https://github.com/bips-hb/arfpy/tree/master)

## Installation

```bash
poetry install --extras arf
```

## Usage

```python

from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.arf_alex.adapter import KatabaticARF

# ARF Hyperparameters
model_config = {
    "num_trees": 50,        # Number of trees in the forest
    "max_iters": 10,        # Adversarial iterations
    "min_node_size": 5,     # Minimum samples per leaf (controls overfitting)
    "delta": 0.0,           # Convergence tolerance
}

pipeline = TrainTestSplitPipeline(model=KatabaticARF)

pipeline.run(
    input_csv='data/my_dataset.csv',
    output_dir='sample_data/my_dataset',
    synthetic_dir='synthetic/my_dataset/codi',
    real_test_dir='sample_data/my_dataset'
    **model_config
)
```

See `examples/arf.ipynb` for more examples.
