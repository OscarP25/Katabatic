# TabFairGAN: Fair Tabular Data Generation with Generative Adversarial Networks

TabFairGAN is a GAN-based framework designed to generate fair synthetic data by incorporating a second discriminator that specifically penalises discrimination against protected groups.

## Paper
**TabFairGAN: Fair Tabular Data Generation with Generative Adversarial Networks**  
Rajabi et al., 2021  
[arXiv:2109.00666](https://arxiv.org/abs/2109.00666)
[Repository](https://github.com/amirarsalan90/TabFairGAN)

## Installation

```bash
poetry install --extras tabfairgan
```

## Usage

```python

import torch
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.tabfairgan_alex.adapter import KatabaticTabFairGAN

# Device Config
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# TabFairGAN parameters
model_config = {
    'epochs': 50,
    'batch_size': 256,
    'device': device,
    'fairness_config': {
        'fair_epochs': 10,
        'lamda': 0.5,
        'S': 'sex',     # protected attribute
        'Y': 'class',   # target class
        'S_under': '0', # 0 = ' Female'
        'Y_desire': '1' # 1 = ' >50K'
    }
}

pipeline = TrainTestSplitPipeline(model=KatabaticTabFairGAN)

pipeline.run(
    input_csv='data/my_dataset.csv',
    output_dir='sample_data/my_dataset',
    synthetic_dir='synthetic/my_dataset/codi',
    real_test_dir='sample_data/my_dataset'
    **model_config
)
```

See `examples/tabfairgan.ipynb` for more examples.
