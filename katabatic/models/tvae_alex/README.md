# TVAE: Modeling Tabular data using Conditional GAN

TVAE (Tabular Variational Autoencoder) is a VAE-based model specifically adapted for mixed-type tabular data, often outperforming GANs in likelihood estimation and marginal distribution fidelity.

## Paper

**Modeling Tabular data using Conditional GAN**  
Xu et al., NeurIPS 2019  
[arXiv:1907.00503](https://arxiv.org/abs/1907.00503)

## Installation

```bash
poetry install --extras tvae
```

## Usage

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.tvae_alex.adapter import TVAE

#TVAE parameters
model_config = {
        "model_type": "tvae",  # Select TVAE as SDV model
        "epochs": 300,
        "batch_size": 500,
    }

pipeline = TrainTestSplitPipeline(model=TVAE)

pipeline.run(
    input_csv='data/my_dataset.csv',
    output_dir='sample_data/my_dataset',
    synthetic_dir='synthetic/my_dataset/codi',
    real_test_dir='sample_data/my_dataset'
)
```

See `examples/tvae.ipynb` for more examples.
