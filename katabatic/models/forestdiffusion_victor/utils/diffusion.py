from __future__ import annotations
import numpy as np


def make_beta_schedule(n_t: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> np.ndarray:
    """
    Linear beta schedule (DDPM-like).
    """
    n_t = int(n_t)
    if n_t < 2:
        raise ValueError("n_t must be >= 2")
    return np.linspace(beta_start, beta_end, n_t, dtype=np.float32)


def _alphas_from_betas(betas: np.ndarray):
    betas = np.asarray(betas, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas, axis=0)
    return alphas.astype(np.float32), alpha_bar.astype(np.float32)


def q_sample(
    x0: np.ndarray,
    t: int | np.ndarray,
    betas: np.ndarray,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward diffusion (add noise), vectorized over batch and timesteps:

      x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1-alpha_bar[t]) * eps

    Args:
        x0: (N, D) float32
        t: int or (N,) array of ints
    Returns:
        (x_t, eps) both same shape as x0
    """
    _, alpha_bar = _alphas_from_betas(betas)

    x0 = np.asarray(x0, dtype=np.float32)
    t_arr = np.asarray(t, dtype=np.int32)

    if t_arr.ndim == 0:
        tt = int(t_arr)
        if tt < 0 or tt >= len(alpha_bar):
            raise ValueError("t out of range")
        ab = alpha_bar[tt].astype(np.float32)
        eps = rng.normal(0.0, 1.0, size=x0.shape).astype(np.float32)
        x_t = (np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps).astype(np.float32)
        return x_t, eps

    if t_arr.shape[0] != x0.shape[0]:
        raise ValueError(f"t must have shape (N,), got {t_arr.shape} for x0 shape {x0.shape}")

    if np.any(t_arr < 0) or np.any(t_arr >= len(alpha_bar)):
        raise ValueError("t contains out-of-range values")

    ab = alpha_bar[t_arr].astype(np.float32)  # (N,)
    ab = ab.reshape(-1, 1)                    # (N,1) broadcast to (N,D)

    eps = rng.normal(0.0, 1.0, size=x0.shape).astype(np.float32)
    x_t = (np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps).astype(np.float32)
    return x_t, eps


def p_step(
    x_t: np.ndarray,
    eps_pred: np.ndarray,
    t: int,
    betas: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    One reverse step (simplified):
      x0_hat = (x_t - sqrt(1-a_bar)*eps_pred) / sqrt(a_bar)
      x_{t-1} = sqrt(a_bar_{t-1})*x0_hat + sqrt(1-a_bar_{t-1})*z  (z~N(0,1))
    """
    _, alpha_bar = _alphas_from_betas(betas)
    t = int(t)

    x_t = np.asarray(x_t, dtype=np.float32)
    eps_pred = np.asarray(eps_pred, dtype=np.float32)

    if t <= 0:
        return x_t.astype(np.float32)

    ab_t = float(alpha_bar[t])
    ab_prev = float(alpha_bar[t - 1])

    x0_hat = (x_t - np.sqrt(1.0 - ab_t) * eps_pred) / (np.sqrt(ab_t) + 1e-8)
    z = rng.normal(0.0, 1.0, size=x_t.shape).astype(np.float32)
    x_prev = (np.sqrt(ab_prev) * x0_hat + np.sqrt(1.0 - ab_prev) * z).astype(np.float32)
    return x_prev
