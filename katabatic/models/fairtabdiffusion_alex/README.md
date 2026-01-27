# TabFairDiffusion: Enhancing Equity in Latent Diffusion Models via Fair Bayesian Perturbation

TabFairDiffusion is a diffusion-based generative model that integrates fairness constraints directly into the diffusion process, enabling the synthesis of high-utility tabular data that satisfies specific fairness definitions.

## Paper

**FairDiffusion: Enhancing Equity in Latent Diffusion Models via Fair Bayesian Perturbation**  
Lou et al., Science Advances 2025
[arXiv:2412.20374](https://arxiv.org/abs/2412.20374)
[Repository](https://github.com/comp-well-org/fair-tab-diffusion)

## Installation

```bash
poetry install --extras fairtabdiffusion
```

## Usage

```python

from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.fairtabdiffusion_alex.adapter import KatabaticFairTabDiffusion

# select protected and target attributes.
protected_col = input("Protected Attribute (S): ").strip() or "sex"
target_col = input("Target Attribute (Y): ").strip() or "class"

#FairTabDiffusion parameters
model_config = {
    "epochs": 100,
    "batch_size": 256,
    
    # Fairness Config
    "fairness_config": {
        "S": protected_col, #protected attributed
        "Y": target_col,    #target attribute
        "S_under": "0",     #underrepresented class
        "Y_desire": "1"     #desired representation
    }
}

pipeline = TrainTestSplitPipeline(model=KatabaticFairTabDiffusion)

pipeline.run(
    input_csv='data/my_dataset.csv',
    output_dir='sample_data/my_dataset',
    synthetic_dir='synthetic/my_dataset/codi',
    real_test_dir='sample_data/my_dataset'
    **model_config
)
```

See `examples/arf.ipynb` for more examples.
