# PrivBayes: Private Data Release via Bayesian Networks

PrivBayes is a Bayesian network model that constructs a low-degree approximation of the dataset's distribution to generate synthetic data satisfying differential privacy guarantees.

## Paper

**PrivBayes: Private Data Release via Bayesian Networks**  
Zhang et al., ACM (TODS) 2017
[Research Paper](https://dl.acm.org/doi/10.1145/3134428)
[GitHub Repo](https://github.com/DataResponsibly/DataSynthesizer)

## Installation

```bash
poetry install --extras privbayes
```

## Usage

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.privbayes_alex.adapter import PrivBayes

#PrivBayes parameters
model_config = {
            "epsilon": 1.0,  # Privacy budget (lower is more private)
            "degree_of_bayesian_network": 2, # 'k' parameter in PrivBayes paper
        }
pipeline = TrainTestSplitPipeline(model=PrivBayes)

pipeline.run(
    input_csv='data/my_dataset.csv',
    output_dir='sample_data/my_dataset',
    synthetic_dir='synthetic/my_dataset/codi',
    real_test_dir='sample_data/my_dataset'
)
```

See `examples/privbayes.ipynb` for more examples.
