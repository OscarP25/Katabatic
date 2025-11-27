# TabSyn: Tabular Data Synthesis with Latent Diffusion

## Overview

TabSyn is a deep generative model for tabular data synthesis. It combines a Variational Autoencoder (VAE) with a Score-based Diffusion Model. The architecture operates in two stages:

- Stage 1 (VAE): Compresses mixed-type tabular data (numerical and categorical) into a continuous, lower-dimensional latent space.

- Stage 2 (Diffusion): Learns the probability distribution of these latent embeddings using a diffusion model to generate high-fidelity synthetic data.

## Part 1: The Variational Autoencoder (VAE)

The VAE is responsible for handling the heterogeneous nature of tabular data and mapping it to a smooth Gaussian latent manifold.

### Tokenizer

Unlike images or text, tabular data consists of distinct columns with different data types. The Tokenizer converts these features into a unified sequence of embedding vectors $X \in \mathbb{R}^{L \times D_{token}}$, where $L$ is the number of features (plus a CLS token) and $D_{token}$ is the embedding dimension.

Let the input row be composed of numerical features $x_{num} \in \mathbb{R}^{N_{num}}$ and categorical features $x_{cat} \in \mathbb{R}^{N_{cat}}$.

#### Numerical Embeddings:
For a numerical feature $j$ with value $v_j$, the model does not simply project it linearly. Instead, it learns a specific embedding vector $w_j \in \mathbb{R}^{D_{token}}$ and a bias $b_j \in \mathbb{R}^{D_{token}}$ for that column. The embedding is computed as:

$$e_{num, j} = v_j \cdot w_j + b_j$$

where $(\cdot)$ denotes scalar-vector multiplication (broadcasting the scalar value $v_j$ across the vector $w_j$). This allows the model to learn the "direction" of a feature in the embedding space, scaled by its magnitude.

#### Categorical Embeddings:
For a categorical feature $k$ with discrete value $c_k$, we use a learnable lookup table (Embedding Matrix) $E_k$. To handle multiple columns sharing the same embedding space, we use offset indices:

$$e_{cat, k} = \text{Lookup}(E, \text{offset}_k + c_k)$$

#### Sequence Construction:
The final input to the Transformer is a sequence of tokens, including a special [CLS] token (initialized as a vector of ones) used to aggregate global information:

$$X_{input} = [e_{cls}, e_{num, 1}, ..., e_{num, N_{num}}, e_{cat, 1}, ..., e_{cat, N_{cat}}]$$

### Transformer Encoder & Decoder

The VAE uses a Transformer architecture to capture correlations between different columns (features).

Self-Attention Mechanism:
The core operation is Scaled Dot-Product Attention. For a query matrix $Q$, key matrix $K$, and value matrix $V$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where $d_k$ is the dimension of the attention heads. This allows every feature to attend to every other feature to understand dependencies (e.g., "Age" attending to "Income").

Latent Space Mapping (Encoder):
The encoder outputs two vectors for each token: a mean $\mu$ and a log-variance $\log(\sigma^2)$.

$$z \sim \mathcal{N}(\mu(X), \sigma^2(X))$$

We use the Reparameterization Trick to allow backpropagation:

$$z = \mu + \sigma \odot \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$

### VAE Loss Function

The VAE is trained to minimize the Evidence Lower Bound (ELBO):

$$\mathcal{L}_{VAE} = \mathcal{L}_{Recon} + \beta \cdot \mathcal{L}_{KL}$$

Reconstruction Loss ($\mathcal{L}_{Recon}$):

- Numerical: Mean Squared Error (MSE) between original $x_{num}$ and reconstructed $\hat{x}_{num}$.

- Categorical: Cross-Entropy Loss between original labels and predicted logits.

KL Divergence ($\mathcal{L}_{KL}$): Measures how much the learned latent distribution deviates from a standard Normal distribution $\mathcal{N}(0, 1)$.

## Part 2: Latent Diffusion Model

Once the VAE is trained, we freeze it. We extract the latent vectors $z_0$ from the training data. The Diffusion model learns to generate new latent vectors $z$ from pure noise.

### Forward Diffusion Process (Noise Injection)

We define a fixed Markov chain that gradually adds Gaussian noise to the data over $T$ timesteps.
Let $\beta_t$ be the variance schedule and $\alpha_t = 1 - \beta_t$. We define $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

The distribution of $z_t$ at any arbitrary timestep $t$ given $z_0$ is closed-form:

$$q(z_t | z_0) = \mathcal{N}(z_t; \sqrt{\bar{\alpha}_t} z_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

This allows us to sample a noisy latent $z_t$ directly:

$$z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

### Reverse Denoising Process (Generation)

The goal is to reverse this process: sample pure noise $z_T \sim \mathcal{N}(0, \mathbf{I})$ and gradually denoise it to get $z_0$.

We approximate the reverse conditional probability $p_\theta(z_{t-1} | z_t)$ using a neural network (DenoisingMLP) that predicts the noise $\epsilon_\theta(z_t, t)$.

The Sampling Step (DDPM):
To move from step $t$ to $t-1$:

$$z_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( z_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(z_t, t) \right) + \sigma_t \mathbf{z}$$

Where:

$\epsilon_\theta(z_t, t)$ is the noise predicted by the MLP.

$\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ is random noise (added for stochasticity, except at $t=0$).

$\sigma_t = \sqrt{\beta_t}$ (or related variance term).

### Denoising MLP Architecture

The diffusion model uses a Residual MLP (DenoisingMLP) instead of a U-Net (common in image diffusion), as latent tabular data has no spatial structure.

Time Embedding: Sinusoidal embeddings (like Transformer positional encodings) inject the timestep $t$ information.

Architecture:

$$h = \text{InputProj}(z_t) + \text{TimeEmb}(t)$$

$$h_{l+1} = h_l + \text{Layer}_l(h_l) \quad (\text{Residual Connections})$$

$$\epsilon_{pred} = \text{OutputProj}(h_{final})$$

### Diffusion Loss Function

The model is trained to predict the noise $\epsilon$ that was added to $z_0$ to create $z_t$:

$$\mathcal{L}_{Diff} = \mathbb{E}_{z_0, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \|^2 \right]$$

## Part 3: Training Process & Optimization

To ensure stable convergence and high-fidelity generation, the model employs specific optimization strategies during the training of both the VAE and the Diffusion model.

### Cosine Annealing Learning Rate

Both the VAE and Diffusion phases utilize Cosine Annealing for learning rate decay. This technique starts with a high learning rate to traverse the loss landscape quickly and gradually decreases it following a cosine curve to settle into a local minimum.

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)$$

Implementation: torch.optim.lr_scheduler.CosineAnnealingLR

Effect: This prevents the model from oscillating around the minimum in later epochs while allowing for rapid initial learning.

### VAE Specific Optimizations

#### KL Divergence Annealing (Warmup):
To prevent "posterior collapse" (where the decoder ignores the latent code and the KL loss dominates early), the weight $\beta$ of the KL term is linearly annealed.

$$\beta_t = \text{min}\left(\beta_{end}, \beta_{start} + \frac{t}{T_{warmup}} (\beta_{end} - \beta_{start})\right)$$

In the code, $\beta$ starts at $0.0$ and increases to $0.005$ over the first half of training.

#### Inverse Class Weighting:
To handle class imbalance in categorical columns, the Cross-Entropy loss is weighted by the inverse frequency of classes:


$$w_c \propto \frac{1}{\text{count}(c)}$$

### Diffusion Specific Optimizations

#### Cosine Noise Schedule:
Instead of a standard linear noise schedule, TabSyn uses a Cosine Noise Schedule (Nichol & Dhariwal, 2021) for the diffusion process parameters. This is crucial for small inputs (like latent vectors) to prevent the signal from being destroyed too early in the forward process.

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$$

where $s=0.008$ is a small offset.

#### Gradient Clipping:
Both models use clip_grad_norm_(max_norm=1.0) to prevent exploding gradients, ensuring numerical stability during the optimization of the deep MLP networks.

## Synthesis Pipeline

To generate synthetic tabular data:

1. Sample Noise: $z_T \sim \mathcal{N}(0, \mathbf{I})$.

2. Reverse Diffusion: Iteratively apply the sampling step $T$ times to get $z_0$.

3. Decode: Pass $z_0$ through the VAE Decoder to get hidden states $h$.

4. Reconstruct:
    - Numerical: $\hat{x}_{num} = \text{MLP}_{num}(h)$.
    - Categorical: $\hat{x}_{cat} = \text{Argmax}(\text{MLP}_{cat}(h))$.

5. Post-process: Apply inverse feature scaling (Inverse Quantile Transformer) to return to the original data scale.