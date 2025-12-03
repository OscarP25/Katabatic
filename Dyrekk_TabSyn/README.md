# TabSyn — Tabular Data Synthesis with Latent Diffusion

## Overview

TabSyn is a two-stage deep generative pipeline designed to synthesize
mixed-type tabular datasets (numerical + categorical). It first trains a
Variational Autoencoder (VAE) to map heterogeneous tabular rows into a
continuous latent space, then trains a latent diffusion model to learn
and sample from the latent distribution. Finally, decoded latents are
post-processed back into tabular form.

This repository component is a practical implementation with sensible
defaults and training utilities to reproduce the two-stage approach.

## Key Components

- `vae.py` / `Model_VAE` — Tokenizer + Transformer-based VAE that
  converts mixed-type rows into token embeddings and outputs a compact
  continuous latent representation.
- `diffusion.py` / `Diffusion` — A latent-space diffusion model (DDPM)
  implemented as a residual MLP that learns to denoise latent vectors.
- `tabsyn.py` / `TabSyn` — Orchestrator that runs preprocessing, VAE
  training, latent extraction, diffusion training, and sampling.

## Architecture (Concise)

1. Tokenizer: converts columns into tokens — numerical columns are
   projected into learned vectors and categorical columns use embeddings.
   A CLS token summarizes row-level information.

2. VAE (Encoder/Decoder): a Transformer encoder maps tokens to latent
   means and log-variances; the decoder reconstructs numerical and
   categorical outputs from latent tokens.

3. Latent Diffusion: a Residual MLP predicts noise in latent space with
   sinusoidal time embeddings; a cosine noise schedule is used for
   stable training.

## Training Pipeline

1. Preprocess dataset:
   - Detect numerical and categorical columns (categorical detection by
     cardinality heuristics).
   - Impute numerical missing values, standardize, and apply a
     QuantileTransformer (normal output distribution).
   - Record per-column category counts for categorical columns.

2. Phase 1 — Train VAE:
   - Train to minimize ELBO: reconstruction loss (MSE for numeric,
     cross-entropy for categorical) + weighted KL divergence.
   - Use KL weight annealing (warmup) to prevent posterior collapse.
   - Cosine annealing scheduler for learning rate and gradient clipping
     for stability.

3. Extract latent embeddings:
   - Encode training data through the VAE encoder and collect `z`.
   - Standardize latents (store mean/std) and set `latent_shape`.

4. Phase 2 — Train Diffusion on latents:
   - Use a cosine noise schedule (Nichol & Dhariwal style).
   - Train the DDPM MLP to predict noise at timesteps; use MSE loss.
   - Cosine LR scheduler and gradient clipping are used.

5. Sampling & Reconstruction:
   - Sample `z_T ~ N(0, I)` and run the reverse diffusion to obtain `z_0`.
   - Decode `z_0` through the VAE decoder and reconstructor to obtain
     numeric and categorical outputs.
   - Inverse-transform numeric outputs (QuantileTransformer + scaler)
     and sample categorical values from softmax logits.

## Configuration Parameters (selected)

- `epochs_vae`, `epochs_diffusion` — number of epochs for each phase.
- `batch_size`, `vae_lr`, `diffusion_lr` — training hyperparameters.
- `d_token` — token embedding dimension used by the VAE.
- `num_timesteps` — number of diffusion steps (default: 1000).
- `cat_loss_weight`, `num_loss_weight` — weighting for reconstruction
  losses.

Defaults are chosen for generality; tune them for specific datasets.

## Sample run (PowerShell)

Train TabSyn (example):

```powershell
# From repository root
python -c "from Dyrekk_TabSyn.src.tabsyn import TabSyn; ts=TabSyn(epochs_vae=20, epochs_diffusion=20, batch_size=256); ts.train('Dyrekk_TabSyn')"
```

Generate samples in Python after training:

```python
from Dyrekk_TabSyn.src.tabsyn import TabSyn
ts = TabSyn()
ts.train('Dyrekk_TabSyn', synthetic_dir='synthetic_out')
# or if already fitted:
synth_df = ts.sample(num_samples=1000, synthetic_dir='synthetic_out')
```

Notes: replace `'Dyrekk_TabSyn'` with the folder that contains
`train_full.csv` for your dataset; pass `synthetic_dir` to save outputs.

## Dependencies
- numpy
- pandas
- torch
- scikit-learn
- tqdm

## Practical Tips & Caveats

- Numeric scaling: the implementation uses `StandardScaler` followed by
  `QuantileTransformer` to produce inputs appropriate for the VAE.
  Inverse transforms are applied during sampling — check for NaNs and
  fall back to column means if needed.

- KL warmup: the VAE uses a linear schedule to grow KL weight from 0 to
  a small value to reduce posterior collapse risk.

- Cosine noise schedule: improves diffusion training for compact latent
  spaces vs linear schedules.

- If categorical outputs collapse (single class), the code includes a
  safeguard that can resample or warn; consider adjusting training
  balance and sampling temperature.

## Files

- `src/tabsyn.py` — orchestration of the pipeline and training.
- `src/vae.py` — VAE model and tokenizer (feature embedding logic).
- `src/diffusion.py` — diffusion model and sampling utilities.

