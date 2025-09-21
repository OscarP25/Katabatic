# utils.py
# Backend for compact CTAB-GAN:
# - Cat columns -> ordinal ids -> learned per-column embeddings (fixed layout)
# - D = projection discriminator + auxiliary classifier (ACGAN)
# - G feature-matching on numerics + correlation penalty
# - Instance noise + TTUR + stronger GP
# - Exact reuse of training transforms during decode

from __future__ import annotations
import os, json, enum
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple, Union, Literal, cast

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, OrdinalEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ JSON helpers ------------------
def load_json(path): 
    with open(path, "r") as f: 
        return json.load(f)

def dump_json(obj, path, **kwargs):
    kwargs.setdefault("indent", 2)
    with open(path, "w") as f:
        json.dump(obj, f, **kwargs)

# ------------------ dataset structures ------------------
ArrayDict = Dict[str, np.ndarray]
Normalization = Literal['standard', 'quantile', 'minmax']
NumNanPolicy = Literal['drop-rows', 'mean']
CatNanPolicy = Literal['most_frequent']
YPolicy = Literal['default', None]

class TaskType(enum.Enum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'
    def __str__(self) -> str: return self.value

@dataclass(frozen=False)
class Dataset:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]   # integer category indices per column
    y: ArrayDict
    y_info: Dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]

    num_transform: Any = None
    cat_transform: Any = None     # {'encoder': OrdinalEncoder, 'n_categories': List[int]}

    @classmethod
    def from_dir(cls, dir_: str) -> 'Dataset':
        splits = [k for k in ['train','val','test'] if os.path.exists(os.path.join(dir_, f'y_{k}.npy'))]
        def load(item) -> ArrayDict:
            return {x: np.load(os.path.join(dir_, f'{item}_{x}.npy'), allow_pickle=True) for x in splits}
        info = load_json(os.path.join(dir_, 'info.json')) if os.path.exists(os.path.join(dir_, 'info.json')) else {}
        return Dataset(
            load('X_num') if os.path.exists(os.path.join(dir_, 'X_num_train.npy')) else None,
            load('X_cat') if os.path.exists(os.path.join(dir_, 'X_cat_train.npy')) else None,
            load('y'),
            {},
            TaskType(info.get('task_type', 'regression')),
            info.get('n_classes'),
        )

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num['train'].shape[1]
    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat['train'].shape[1]
    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

@dataclass(frozen=True)
class Transformations:
    seed: int = 42
    normalization: Optional[Normalization] = 'quantile'
    num_nan_policy: Optional[NumNanPolicy] = None
    cat_nan_policy: Optional[CatNanPolicy] = None
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[str] = 'one-hot'
    y_policy: Optional[YPolicy] = 'default'

# ------------------ I/O ------------------
def read_pure_data(path, split='train'):
    y = np.load(os.path.join(path, f'y_{split}.npy'), allow_pickle=True)
    X_num = np.load(os.path.join(path, f'X_num_{split}.npy'), allow_pickle=True) if os.path.exists(os.path.join(path, f'X_num_{split}.npy')) else None
    X_cat = np.load(os.path.join(path, f'X_cat_{split}.npy'), allow_pickle=True) if os.path.exists(os.path.join(path, f'X_cat_{split}.npy')) else None
    return X_num, X_cat, y

# ------------------ transforms ------------------
def num_process_nans(dataset: Dataset, policy: Optional[NumNanPolicy]) -> Dataset:
    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()}
    if not any(x.any() for x in nan_masks.values()):
        return dataset
    assert policy is not None
    if policy == 'drop-rows':
        valid_masks = {k: ~v.any(1) for k, v in nan_masks.items()}
        new_data = {}
        for data_name in ['X_num', 'X_cat', 'y']:
            data_dict = getattr(dataset, data_name)
            if data_dict is not None:
                new_data[data_name] = {k: v[valid_masks[k]] for k, v in data_dict.items()}
        dataset = replace(dataset, **new_data)
    elif policy == 'mean':
        new_values = np.nanmean(dataset.X_num['train'], axis=0)
        X_num = {k: v.copy() for k, v in dataset.X_num.items()}
        for k, v in X_num.items():
            idx = np.where(nan_masks[k])
            v[idx] = np.take(new_values, idx[1])
        dataset = replace(dataset, X_num=X_num)
    else:
        raise ValueError(f'Unknown num_nan_policy={policy}')
    return dataset

def normalize(X: ArrayDict, normalization: Normalization, seed: Optional[int], return_normalizer: bool = False):
    X_train = X['train']
    if normalization == 'standard':
        normalizer = StandardScaler()
    elif normalization == 'minmax':
        normalizer = MinMaxScaler()
    elif normalization == 'quantile':
        normalizer = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X_train.shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
    else:
        raise ValueError(f'Unknown normalization={normalization}')
    normalizer.fit(X_train)
    out = {k: normalizer.transform(v) for k, v in X.items()}
    if return_normalizer:
        return out, normalizer
    return out

def cat_process_nans(X: ArrayDict, policy: Optional[CatNanPolicy]) -> ArrayDict:
    if policy is None:
        return X
    if policy == 'most_frequent':
        imputer = SimpleImputer(strategy='most_frequent')
        imputer.fit(X['train'])
        return {k: cast(np.ndarray, imputer.transform(v)) for k, v in X.items()}
    raise ValueError(f'Unknown cat_nan_policy={policy}')

def cat_ordinal_encode(X: ArrayDict) -> Tuple[ArrayDict, Dict[str, Any]]:
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int64)
    oe.fit(X['train'])
    X_ord = {k: oe.transform(v).astype(np.int64) for k, v in X.items()}
    n_cats: List[int] = []
    C = X_ord['train'].shape[1]
    for j in range(C):
        max_id = int(X_ord['train'][:, j].max())
        for part in X_ord:
            X_ord[part][:, j] = np.where(X_ord[part][:, j] < 0, max_id + 1, X_ord[part][:, j])
        n_cats.append(max_id + 2)  # +1 unknown bucket
    return X_ord, {'encoder': oe, 'n_categories': n_cats}

def build_target(y: ArrayDict, policy: Optional[YPolicy], task_type: TaskType):
    info: Dict[str, Any] = {'policy': policy}
    if policy == 'default' and task_type == TaskType.REGRESSION:
        mean, std = float(y['train'].mean()), float(y['train'].std())
        y = {k: (v - mean) / std for k, v in y.items()}
        info['mean'] = mean; info['std'] = std
    return y, info

# ------------------ dataset builder ------------------
def make_dataset(
    data_path: str,
    T: Transformations,
    num_classes: int,
    is_y_cond: bool,
    change_val: bool
) -> Dataset:
    X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
    X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
    y = {}

    for split in ['train', 'val', 'test']:
        X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)
        if X_num is not None:
            X_num[split] = X_num_t
        if X_cat is not None:
            X_cat[split] = X_cat_t
        y[split] = y_t

    info = load_json(os.path.join(data_path, 'info.json')) if os.path.exists(os.path.join(data_path, 'info.json')) else {}
    D = Dataset(X_num, X_cat, y, {}, TaskType(info.get('task_type', 'regression')), info.get('n_classes'))

    if D.X_num is not None:
        D = num_process_nans(D, T.num_nan_policy)
        D.X_num, D.num_transform = normalize(D.X_num, cast(Normalization, T.normalization), T.seed, return_normalizer=True)

    if D.X_cat is not None:
        Xc_obj = {k: D.X_cat[k].astype(str) for k in D.X_cat}
        Xc_ord, cat_transform = cat_ordinal_encode(Xc_obj)
        D = replace(D, X_cat=Xc_ord)
        D.cat_transform = cat_transform

    D.y, D.y_info = build_target(D.y, T.y_policy, D.task_type)
    return D

# ------------------ dataloader ------------------
class FastTensorDataLoader:
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert len([t for t in tensors if t is not None]) > 0
        base = [t for t in tensors if t is not None][0]
        assert all(t.shape[0] == base.shape[0] for t in tensors if t is not None)
        self.tensors = list(t for t in tensors if t is not None)
        self.dataset_len = base.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        self.n_batches = n_batches + (1 if remainder > 0 else 0)

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len, device=self.tensors[0].device)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

# ------------------ embeddings ------------------
class CatEmbedding(nn.Module):
    def __init__(self, n_categories: List[int], emb_dim: int = 16):
        super().__init__()
        self.emb_dim = emb_dim
        self.embs = nn.ModuleList([nn.Embedding(nc, emb_dim) for nc in n_categories])

    def forward(self, cat_idx: torch.Tensor) -> torch.Tensor:
        embs = [emb(cat_idx[:, j]) for j, emb in enumerate(self.embs)]
        return torch.cat(embs, dim=1)

def invert_cat_embeddings(cat_emb: CatEmbedding, x_emb: torch.Tensor, n_cats: List[int], temp: float = 0.7) -> torch.Tensor:
    B = x_emb.size(0); E = cat_emb.emb_dim; C = len(n_cats)
    outs = []
    for j in range(C):
        seg = x_emb[:, j*E:(j+1)*E]
        table = cat_emb.embs[j].weight
        d = torch.cdist(seg, table) / max(1e-6, temp)
        outs.append(torch.argmin(d, dim=1))
    return torch.stack(outs, dim=1)

# ------------------ models ------------------
def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None: nn.init.constant_(m.bias, 0.0)

class MLP(nn.Module):
    def __init__(self, sizes: List[int], dropout: float = 0.0, bn: bool = True, spectral: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            in_f, out_f = sizes[i], sizes[i+1]
            lin = nn.Linear(in_f, out_f)
            if spectral:
                lin = nn.utils.spectral_norm(lin)
            layers.append(lin)
            if i < len(sizes) - 2:
                if bn: layers.append(nn.BatchNorm1d(out_f))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0: layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class Generator(nn.Module):
    def __init__(self, z_dim:int, data_dim:int, y_dim:int=0, hidden:Tuple[int,...]=(256,256), dropout:float=0.2):
        super().__init__()
        sizes = [z_dim + y_dim] + list(hidden) + [data_dim]
        self.mlp = MLP(sizes, dropout=dropout, bn=True, spectral=False)
    def forward(self, z, y_onehot=None):
        if y_onehot is not None:
            z = torch.cat([z, y_onehot], dim=1)
        return self.mlp(z)

class ProjectionDiscriminator(nn.Module):
    """WGAN projection D + Aux classifier head (ACGAN)"""
    def __init__(self, data_dim:int, num_classes:int=0, hidden:Tuple[int,...]=(256,256), spectral:bool=True, dropout:float=0.2):
        super().__init__()
        sizes = [data_dim] + list(hidden)
        self.body = MLP(sizes + [1], dropout=dropout, bn=False, spectral=spectral)
        self.num_classes = num_classes
        if num_classes > 0:
            self.class_emb = nn.Embedding(num_classes, sizes[-1])  # project y into last hidden
            self.aux_head = nn.Linear(sizes[-1], num_classes)
        else:
            self.class_emb = None
            self.aux_head = None

    def forward(self, data_vec, y_idx: Optional[torch.Tensor]=None):
        # get last hidden before scalar; we reuse body layers except the last linear
        x = data_vec
        features = x
        # manually run through body except last layer to get features
        feats = []
        for layer in self.body.net[:-1]:
            features = layer(features)
            feats.append(features)
        h = features
        score = self.body.net[-1](h)  # scalar critic output

        if self.num_classes > 0 and y_idx is not None and self.class_emb is not None:
            proj = (h * self.class_emb(y_idx)).sum(dim=1, keepdim=True)
            score = score + proj
            aux = self.aux_head(h)
        else:
            aux = None
        return score, aux, h

# ------------------ losses ------------------
def gradient_penalty(critic_fn, real, fake, device, lambda_gp=12.0):
    alpha = torch.rand(real.size(0), 1, device=device)
    while alpha.dim() < real.dim(): alpha = alpha.unsqueeze(-1)
    inter = (alpha*real + (1-alpha)*fake).requires_grad_(True)
    pred, _, _ = critic_fn(inter)
    grad = torch.autograd.grad(pred, inter, torch.ones_like(pred), create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad = grad.view(grad.size(0), -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp

def corr_penalty(x_real: torch.Tensor, x_fake: torch.Tensor, weight: float = 3e-3):
    if weight <= 0: return x_fake.sum()*0
    def _corr(m):
        m = m - m.mean(0, keepdim=True)
        cov = (m.T @ m) / (m.size(0) - 1 + 1e-6)
        d = torch.sqrt(torch.diag(cov) + 1e-6)
        corr = cov / (d[:, None] * d[None, :] + 1e-6)
        return corr
    c_r = _corr(x_real.detach())
    c_f = _corr(x_fake)
    return weight * (c_r - c_f).pow(2).mean()

def feature_matching(real_num: Optional[torch.Tensor], fake_num: Optional[torch.Tensor], w_mean: float=1e-2, w_std: float=1e-2):
    if real_num is None or fake_num is None or real_num.size(1) == 0:
        return (fake_num.sum()*0) if fake_num is not None else 0.0
    m_r, s_r = real_num.mean(0), real_num.std(0)
    m_f, s_f = fake_num.mean(0), fake_num.std(0)
    return w_mean * F.mse_loss(m_f, m_r) + w_std * F.mse_loss(s_f, s_r)

# ------------------ train & sample ------------------
def train(
    parent_dir: str,
    real_data_path: str,
    steps: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    model_type: str,
    model_params: Dict[str, Any],
    num_timesteps: int,
    gaussian_loss_type: str,
    scheduler: str,
    T_dict: Dict[str, Any],
    num_numerical_features: int,
    device: torch.device,
    seed: int,
    change_val: bool = False,
):
    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(parent_dir, exist_ok=True)

    T = Transformations(**T_dict)
    Dset = make_dataset(real_data_path, T, num_classes=model_params.get('num_classes', 0), is_y_cond=model_params.get('is_y_cond', False), change_val=change_val)

    # tensors
    Xn = torch.from_numpy(Dset.X_num['train']).float().to(device) if Dset.X_num is not None else None
    Xc_idx = torch.from_numpy(Dset.X_cat['train']).long().to(device) if Dset.X_cat is not None else None
    y_tr = torch.from_numpy(Dset.y['train']).long().to(device)

    n_num = (Xn.size(1) if Xn is not None else 0)
    n_cats: List[int] = (Dset.cat_transform['n_categories'] if Dset.cat_transform is not None else [])
    emb_dim = int(model_params.get('emb_dim', 32))
    cat_emb = CatEmbedding(n_cats, emb_dim=emb_dim).to(device)

    data_dim = n_num + (len(n_cats) * emb_dim)
    is_y_cond = bool(model_params.get('is_y_cond', False))
    num_classes = int(model_params.get('num_classes', 0))
    y_dim = num_classes if (is_y_cond and num_classes > 0) else 0

    z_dim = int(model_params.get('z_dim', 128))
    g_hidden = tuple(model_params.get('rtdl_params', {}).get('d_layers', [256,256]))
    dropout = float(model_params.get('rtdl_params', {}).get('dropout', 0.0))
    spectral_d = bool(model_params.get('spectral_d', True))
    lambda_gp = float(model_params.get('lambda_gp', 12.0))
    corr_w = float(model_params.get('corr_penalty_weight', 3e-3))
    warm_start_from = model_params.get('warm_start_from', None)
    inst_noise_std = float(model_params.get('inst_noise_std', 0.01))

    G = Generator(z_dim, data_dim, y_dim=y_dim, hidden=g_hidden, dropout=dropout).to(device).apply(_weights_init)
    Dsc = ProjectionDiscriminator(data_dim, num_classes=y_dim, hidden=g_hidden, spectral=spectral_d, dropout=dropout).to(device).apply(_weights_init)

    # warm-start generator if provided
    if warm_start_from and os.path.exists(warm_start_from):
        try:
            ckpt_ws = torch.load(warm_start_from, map_location=device)
            G.load_state_dict(ckpt_ws['G_state_dict'], strict=False)
            print(f"[Warm start] Loaded generator from {warm_start_from}")
        except Exception as e:
            print(f"[Warm start] Failed to load: {e}")

    # TTUR
    lr_G = float(model_params.get('lr_G', 2e-4))
    lr_D = float(model_params.get('lr_D', 1e-4))
    opt_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.9), weight_decay=weight_decay)
    opt_D = torch.optim.Adam(Dsc.parameters(), lr=lr_D, betas=(0.5, 0.9), weight_decay=weight_decay)

    n_critic = int(model_params.get('n_critic', 5))
    N = (Xn.size(0) if Xn is not None else Xc_idx.size(0))

    def pack(cat_idx: Optional[torch.Tensor], num: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        parts = []
        if num is not None: parts.append(num)
        if cat_idx is not None: parts.append(cat_emb(cat_idx))
        return torch.cat(parts, dim=1) if parts else None

    for step in range(steps):
        # --- D updates
        for _ in range(n_critic):
            idx = torch.randint(0, N, (batch_size,), device=device)
            Xn_b = (Xn[idx] if Xn is not None else None)
            Xc_b = (Xc_idx[idx] if Xc_idx is not None else None)
            y_idx = (y_tr[idx] if y_dim>0 else None)

            real_vec = pack(Xc_b, Xn_b)
            # instance noise
            if real_vec is not None and inst_noise_std > 0:
                real_vec = real_vec + inst_noise_std * torch.randn_like(real_vec)

            z = torch.randn(batch_size, z_dim, device=device)
            y_onehot = (F.one_hot(y_idx, num_classes=y_dim).float() if y_dim>0 else None)
            fake_vec = G(z, y_onehot).detach()
            if inst_noise_std > 0:
                fake_vec = fake_vec + inst_noise_std * torch.randn_like(fake_vec)

            d_real, aux_real, _ = Dsc(real_vec, y_idx if y_dim>0 else None)
            d_fake, aux_fake, _ = Dsc(fake_vec, y_idx if y_dim>0 else None)

            gp = gradient_penalty(lambda v: Dsc(v, y_idx if y_dim>0 else None), real_vec, fake_vec, device, lambda_gp=lambda_gp)
            wd = (d_fake.mean() - d_real.mean()) + gp

            # Aux classifier (real only)
            aux_loss = 0.0
            if y_dim>0 and aux_real is not None:
                aux_loss = F.cross_entropy(aux_real, y_idx)

            loss_D = wd + 0.5 * aux_loss
            opt_D.zero_grad(set_to_none=True); loss_D.backward(); opt_D.step()

        # --- G update
        idx = torch.randint(0, N, (batch_size,), device=device)
        Xn_b = (Xn[idx] if Xn is not None else None)
        Xc_b = (Xc_idx[idx] if Xc_idx is not None else None)
        y_idx = (y_tr[idx] if y_dim>0 else None)
        y_onehot = (F.one_hot(y_idx, num_classes=y_dim).float() if y_dim>0 else None)

        real_vec = pack(Xc_b, Xn_b)
        z = torch.randn(batch_size, z_dim, device=device)
        gen_vec = G(z, y_onehot)

        d_gen, aux_gen, h_gen = Dsc(gen_vec, y_idx if y_dim>0 else None)
        loss_G = -d_gen.mean()

        # feature matching on numeric block
        if n_num > 0:
            loss_G = loss_G + feature_matching(real_vec[:, :n_num], gen_vec[:, :n_num], w_mean=1e-2, w_std=1e-2)
            loss_G = loss_G + corr_penalty(real_vec[:, :n_num], gen_vec[:, :n_num], weight=corr_w)

        # encourage correct class via aux head
        if y_dim>0 and aux_gen is not None:
            loss_G = loss_G + 0.5 * F.cross_entropy(aux_gen, y_idx)

        opt_G.zero_grad(set_to_none=True); loss_G.backward(); opt_G.step()

        if (step+1) % max(1, steps//10) == 0:
            print(f"[CTAB-GAN][{step+1}/{steps}] D: {loss_D.item():.4f} | G: {loss_G.item():.4f}")

    # save checkpoint
    torch.save({
        'G_state_dict': G.state_dict(),
        'D_state_dict': Dsc.state_dict(),
        'cat_emb_state_dict': cat_emb.state_dict(),
        'z_dim': z_dim,
        'data_dim': data_dim,
        'n_num': n_num,
        'n_cats': n_cats,
        'emb_dim': emb_dim,
        'y_dim': y_dim,
        'g_hidden': g_hidden,
        'dropout': dropout,
        'spectral_d': spectral_d,
        'is_y_cond': is_y_cond,
        'num_classes': num_classes
    }, os.path.join(parent_dir, "model.pt"))

    # keep last train blocks for decode alignment
    if Dset.X_num is not None:
        np.save(os.path.join(parent_dir, "X_num_train.npy"), Dset.X_num['train'].astype(np.float32))
    if Dset.X_cat is not None:
        np.save(os.path.join(parent_dir, "X_cat_train.npy"), Dset.X_cat['train'].astype(np.int64))
    np.save(os.path.join(parent_dir, "y_train.npy"), Dset.y['train'])

def sample(
    parent_dir: str,
    real_data_path: str,
    num_samples: int,
    batch_size: int,
    model_type: str,
    model_params: Dict[str, Any],
    model_path: str,
    num_timesteps: int,
    gaussian_loss_type: str,
    scheduler: str,
    T_dict: Dict[str, Any],
    num_numerical_features: int,
    disbalance: Optional[np.ndarray],
    device: torch.device,
    seed: int,
    change_val: bool = False,
):
    # rebuild dataset transforms (for inverse)
    T = Transformations(**T_dict)
    Dset = make_dataset(real_data_path, T, num_classes=model_params.get('num_classes', 0), is_y_cond=model_params.get('is_y_cond', False), change_val=change_val)

    ckpt = torch.load(model_path, map_location=device)
    z_dim = int(ckpt['z_dim'])
    data_dim = int(ckpt['data_dim'])
    n_num = int(ckpt['n_num'])
    n_cats = list(ckpt['n_cats'])
    emb_dim = int(ckpt['emb_dim'])
    y_dim = int(ckpt.get('y_dim', 0))
    g_hidden = tuple(ckpt['g_hidden'])
    dropout = float(ckpt.get('dropout', 0.0))
    is_y_cond = bool(ckpt.get('is_y_cond', False))
    num_classes = int(ckpt.get('num_classes', 0))

    from math import ceil
    G = Generator(z_dim, data_dim, y_dim=y_dim, hidden=g_hidden, dropout=dropout).to(device)
    G.load_state_dict(ckpt['G_state_dict']); G.eval()

    cat_emb = CatEmbedding(n_cats, emb_dim=emb_dim).to(device)
    if 'cat_emb_state_dict' in ckpt:
        cat_emb.load_state_dict(ckpt['cat_emb_state_dict'])

    rng = np.random.default_rng(seed)
    if is_y_cond and num_classes > 0:
        if disbalance is not None:
            probs = np.asarray(disbalance, dtype=float); probs = probs / probs.sum()
        else:
            y_train = np.load(os.path.join(real_data_path, "y_train.npy"), allow_pickle=True)
            vals, counts = np.unique(y_train, return_counts=True)
            probs = counts / counts.sum()
            order = np.argsort(vals); probs = probs[order]
        y_samples = rng.choice(np.arange(num_classes), size=num_samples, p=probs).astype(np.int64)
        y_onehot_all = F.one_hot(torch.from_numpy(y_samples), num_classes=num_classes).float().to(device)
    else:
        y_samples = None
        y_onehot_all = None

    outs = []
    with torch.no_grad():
        bs = min(batch_size, 10000)
        for i in range(0, num_samples, bs):
            k = min(bs, num_samples - i)
            z = torch.randn(k, z_dim, device=device)
            if y_onehot_all is not None:
                v = G(z, y_onehot_all[i:i+k]).cpu()
            else:
                v = G(z, None).cpu()
            outs.append(v.numpy())
    V = np.vstack(outs).astype(np.float32)  # [N, data_dim]

    num_part = V[:, :n_num] if n_num > 0 else None
    cat_emb_part = torch.from_numpy(V[:, n_num:]).float() if len(n_cats) > 0 else None

    if cat_emb_part is not None:
        with torch.no_grad():
            cat_idx = invert_cat_embeddings(cat_emb, cat_emb_part, n_cats, temp=float(model_params.get('decode_temp', 0.7))).cpu().numpy()
        for j in range(cat_idx.shape[1]):
            cat_idx[:, j] = np.clip(cat_idx[:, j], 0, n_cats[j]-1)
    else:
        cat_idx = None

    if num_part is not None:
        np.save(os.path.join(parent_dir, "X_num_train.npy"), num_part.astype(np.float32))
    if cat_idx is not None:
        np.save(os.path.join(parent_dir, "X_cat_train.npy"), cat_idx.astype(np.int64))
    if y_samples is not None:
        np.save(os.path.join(parent_dir, "y_train.npy"), y_samples.astype(np.int64))
    else:
        np.save(os.path.join(parent_dir, "y_train.npy"), np.zeros((V.shape[0],), dtype=np.float32))

# ------------------ decode to DataFrame ------------------
def decode_synthetic_to_dataframe(
    parent_dir: str,
    real_data_path: str,
    feature_names_num: List[str],
    feature_names_cat: List[str],
    T_dict: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    Xn = np.load(os.path.join(parent_dir, "X_num_train.npy"), allow_pickle=True) if os.path.exists(os.path.join(parent_dir, "X_num_train.npy")) else None
    Xc = np.load(os.path.join(parent_dir, "X_cat_train.npy"), allow_pickle=True) if os.path.exists(os.path.join(parent_dir, "X_cat_train.npy")) else None

    # reconstruct transforms from real_data using the SAME T_dict used in training
    T = Transformations(**(T_dict or {}))
    Dset = make_dataset(real_data_path, T, num_classes=0, is_y_cond=True, change_val=False)

    blocks = []

    if Xn is not None and Dset.num_transform is not None:
        Xn_inv = Dset.num_transform.inverse_transform(Xn)
        blocks.append(pd.DataFrame(Xn_inv, columns=feature_names_num[:Xn.shape[1]]))

    if Xc is not None and Dset.cat_transform is not None:
        enc: OrdinalEncoder = Dset.cat_transform['encoder']
        Xc_clip = Xc.copy()
        for j in range(Xc_clip.shape[1]):
            Xc_clip[:, j] = np.clip(Xc_clip[:, j], 0, enc.categories_[j].shape[0]-1)
        cats = enc.inverse_transform(Xc_clip)
        blocks.append(pd.DataFrame(cats, columns=feature_names_cat[:Xc.shape[1]]))

    if not blocks:
        return pd.DataFrame()

    df = pd.concat(blocks, axis=1)
    df['__target__'] = np.load(os.path.join(parent_dir, "y_train.npy"), allow_pickle=True) if os.path.exists(os.path.join(parent_dir, "y_train.npy")) else 0
    return df
