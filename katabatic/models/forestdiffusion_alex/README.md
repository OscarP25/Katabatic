# ForestDiffusion: Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees

ForestDiffusion is a diffusion-based model that utilises Extreme Gradient Boosting (XGBoost) for the denoising step. It is a CPU-efficient alternative to GPU-based deep learning diffusion models.

## Paper

**Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees**  
Jolicoeur-Martineau et al., 2024
[arXiv:2309.09968](https://arxiv.org/abs/2309.09968)

## Installation

```bash
poetry install --extras forestdiffusion
```

## Usage

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.forestdiffusion_alex.adapter import ForestDiffusion

#ForestDiffusion parameters
model_config = {
    "n_t": 50,
    "diffusion_type": "vp"
}

pipeline = TrainTestSplitPipeline(model=ForestDiffusion)

pipeline.run(
    input_csv='data/my_dataset.csv',
    output_dir='sample_data/my_dataset',
    synthetic_dir='synthetic/my_dataset/codi',
    real_test_dir='sample_data/my_dataset'
)
```

See `examples/forestdiffusion.ipynb` for more examples.
