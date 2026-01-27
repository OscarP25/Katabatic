# CopulaGAN: Modeling Tabular data using Conditional GAN

CopulaGAN is a variation of the CTGAN architecture that uses Gaussian Copulas to model column correlations, improving the synthesis of complex numerical distributions.

## Paper

**Modeling Tabular data using Conditional GAN**  
Xu et al., NeurIPS 2019
[arXiv:1907.00503](https://arxiv.org/abs/1907.00503)

## Installation

```bash
poetry install --extras copulagan
```

## Usage

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.copulagan_alex.adapter import CopulaGAN

#CopulaGAN parameters
model_config = {
    "model_type": "copulagan",  # Select CopulaGAN as SDV model
    "epochs": 300,
    "batch_size": 500,
}

pipeline = TrainTestSplitPipeline(model=CopulaGAN)

pipeline.run(
    input_csv='data/my_dataset.csv',
    output_dir='sample_data/my_dataset',
    synthetic_dir='synthetic/my_dataset/codi',
    real_test_dir='sample_data/my_dataset'
)
```

See `examples/copulagan.ipynb` for more examples.
