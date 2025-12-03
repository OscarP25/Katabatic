# PATE-GAN
author: Dyrekk (Dinh Bao Duy Pham)

**Introduction**

PATE-GAN is an implementation of "PATE-GAN: Generating Synthetic Data
with Differential Privacy Guarantees". It uses an ensemble of teacher
models trained on disjoint partitions of real data to provide differentially
private labels for a student model that learns to discriminate between
real and synthetic data. A generator is trained adversarially against the
student to produce synthetic data that preserves the statistical
structure of the original dataset while satisfying a differential privacy
budget.

**This folder**

- `src/pategan.py` — Main PATE-GAN pipeline: preprocessing, teacher
  partitioning, training loop, DP aggregation (Laplace noisy-max or
  Gaussian+RDP) and sampling.
- `src/utils.py` — Model building blocks (Generator and Discriminator).

**Design goals**

- Reproduce PATE-GAN style training logic (paper-faithful noisy-max
  aggregation is supported via the `laplace` mechanism).
- Offer an alternative Gaussian mechanism with Rényi-DP (RDP) accounting
  for tighter composition and practical tuning (`mechanism='gaussian'`).
- Support mixed categorical and numeric columns from label-encoded CSVs.
- Keep comments and documentation concise and algorithm-focused.

**Architecture (high-level)**

- Generator
  - Gumbel-Softmax categorical heads (one head per categorical feature)
  - Numeric head(s) producing continuous outputs
  - Temperature annealing for Gumbel-Softmax during training

- Teachers (ensemble of Discriminators)
  - Train on disjoint partitions of the real dataset
  - Each teacher outputs a binary vote for a sample (real vs. fake)

- Student (Discriminator)
  - Trained on synthetic samples labeled by noisy aggregation of teacher
    votes (the PATE mechanism).

- Privacy accounting
  - `mechanism='laplace'` (default): Laplace noisy-max aggregation and an
    approximate moments-style accumulator to track privacy (paper-style).
  - `mechanism='gaussian'`: Gaussian noise on vote differences with an
    RDP accountant (orders 2..100) that can be converted to (ε, δ).

**Algorithm (concise)**

1. Preprocess the input CSV (`train_full.csv`) to detect categorical and
   numeric features. Categorical columns are expected to be label-encoded
   integers; numeric columns are floats.
2. Partition the preprocessed dataset into `n_teachers` disjoint subsets.
3. Initialize the generator, student, and a teacher discriminator for
   each partition.
4. Repeat until privacy budget `target_epsilon` is consumed or
   `max_iterations` reached:
   - Train each teacher on its partition using both real samples and
     generator-produced samples.
   - Produce synthetic samples and obtain noisy labels from teacher
     aggregation via `pate_mechanism` (`laplace` or `gaussian`).
   - Train the student on synthetic samples + noisy labels.
   - Train the generator to fool the student.
   - Update the privacy accountant according to the selected mechanism.
5. After training, generate synthetic data by sampling the generator and
   postprocessing per-feature outputs to a pandas DataFrame.

**Key configuration parameters**

- `target_epsilon`: float — privacy budget to target.
- `delta`: float — target δ for (ε, δ)-DP (used in RDP→ε conversion).
- `batch_size`, `max_iterations`, `lr` — training hyperparameters.
- `latent_dim` — latent vector size (defaults to sum of categorical dims).
- `n_teachers` — number of teacher models (auto-estimated if omitted).
- `noise_lambda` — Laplace scale or Gaussian std (depends on `mechanism`).
- `mechanism` — `'laplace'` (paper) or `'gaussian'` (RDP accountant).
- `gumbel_temp` — initial temperature for categorical Gumbel-Softmax.

**Sample run (PowerShell)**

Run training (example; adjust paths and args as needed):

```powershell
# From repository root
python -c "from Dyrekk_PateGan.src.pategan import PATEGAN; pg=PATEGAN(target_epsilon=1.0, mechanism='laplace'); pg.train('Dyrekk_PateGan')"
```

Generate synthetic data after fitting (in code):

```python
# Python usage example
from Dyrekk_PateGan.src.pategan import PATEGAN
pg = PATEGAN(target_epsilon=1.0, mechanism='gaussian')
pg.train('Dyrekk_PateGan', synthetic_dir='synthetic_out')
# or, if already fitted
synth_df = pg.sample(num_samples=1000, synthetic_dir='synthetic_out')
```

Notes on choosing mechanisms

- Use `mechanism='laplace'` for fidelity to the original PATE-GAN paper
  (Laplace noisy-max aggregation). The repository contains an approximate
  Laplace moments accumulator similar to earlier implementations.
- Use `mechanism='gaussian'` for the Gaussian mechanism with RDP
  accounting; this often yields better composition behavior and allows
  converting accumulated RDP to (ε, δ) via `get_current_epsilon()`.

Dependencies

- torch
- numpy
- pandas
- scikit-learn 
- tqdm 

**Developer notes**

- `pategan.py` focuses on algorithmic clarity in docstrings. If you need
  the original mathematical derivation for the Laplace accounting to be
  precisely matched to a specific paper, we should verify constants and
  adapt the `_update_privacy_accountant_laplace` implementation.
- Numeric features are not normalized automatically — if the dataset
  contains wide-range continuous features, consider adding a
  simple normalizer (mean/std or min/max) in `_preprocess_data` and
  inverse-transform in `_postprocess_data`.

