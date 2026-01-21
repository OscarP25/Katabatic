import math
import torch
import torch.nn.functional as F
import numpy as np

def extract(a, t, x_shape):
    t = t.to(a.device)
    out = a.gather(-1, t)
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.2):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, max_beta=0.2):
    if schedule_name == 'linear':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'cosine':
        offset = 0.008
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + offset) / (1 + offset) * math.pi / 2) ** 2,
            max_beta=max_beta,
        )
    raise NotImplementedError(f'unknown beta schedule: {schedule_name}')

def timestep_embedding(timesteps, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def index_to_log_onehot(x, num_classes):
    onehots = []
    for i in range(len(num_classes)):
        onehots.append(F.one_hot(x[:, i], num_classes[i]))
    x_onehot = torch.cat(onehots, dim=1)
    log_onehot = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_onehot

def ohe_to_categories(ohe, k):
    k = torch.from_numpy(k)
    indices = torch.cat([torch.zeros((1,)), k.cumsum(dim=0)], dim=0).int().tolist()
    res = []
    for i in range(len(indices) - 1):
        res.append(ohe[:, indices[i]:indices[i + 1]].argmax(dim=1))
    return torch.stack(res, dim=1)

def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)