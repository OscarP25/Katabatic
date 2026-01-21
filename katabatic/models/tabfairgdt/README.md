# TabFairGDT

TabFairGDT is a pre-print autoregressive decision tree that claims to be state-of-the-art in fairness-utility trade-off benchmarks for tabular data generation.

## Paper

**TABFAIRGDT: A Fast Fair Tabular Data Generator
using Autoregressive Decision Trees**  
Panagiotou et al., arXiv pre-print  
[arXiv:2509.19927](https://arxiv.org/abs/2509.19927)

## Installation

<!-- ```bash
poetry install --extras codi
``` -->

## Usage

```python
from katabatic.models.tabfairgdt import TabFairGDT
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline

pipeline = TrainTestSplitPipeline(model=TabFairGDT)
pipeline.run(
    input_csv='data/my_dataset.csv',
    output_dir='sample_data/my_dataset',
    synthetic_dir='synthetic/my_dataset/codi',
    real_test_dir='sample_data/my_dataset'
)

#enter protected variable.
```

<!-- See `examples/tabfairgdt.ipynb` for more examples. --

## Key Features
TODO