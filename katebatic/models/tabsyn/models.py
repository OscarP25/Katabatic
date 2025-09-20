"""
Katabatic TabSyn wrapper.

It will:
1) Ensure amazon-science/tabsyn is importable (auto-install if missing).
2) Load a checkpoint.
3) Provide a DataFrame-in DataFrame-out generate API.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch

from .utils import (
    seed_everything,
    pick_device,
    ensure_repo_on_path,
    ensure_tabsyn_core_available,
)


class TabSynModel:
    def __init__(self, tabsyn_core, device: Union[str, torch.device] = "cpu"):
        self.core = tabsyn_core
        self.device = torch.device(device) if isinstance(device, str) else device
        if hasattr(self.core, "to"):
            self.core.to(self.device)
        self.core.eval()

    @torch.no_grad()
    def generate(
        self,
        real_df: pd.DataFrame,
        num_rows: int,
        batch_size: int = 512,
    ) -> pd.DataFrame:
        """
        Generate synthetic rows using the fitted feature transform.
        """
        ft = getattr(self.core, "feature_transform", None)
        if ft is None:
            raise RuntimeError("feature_transform not found in core. The checkpoint must contain it.")

        # Encode real data to model space if needed by downstream utilities
        if hasattr(ft, "encode_dataframe"):
            _ = ft.encode_dataframe(real_df)

        samples = []
        remain = int(num_rows)
        while remain > 0:
            b = min(batch_size, remain)
            out = self.core.sample(b)
            samples.append(out.detach().cpu())
            remain -= b

        x_gen = torch.cat(samples, dim=0).numpy()

        if hasattr(ft, "decode_to_dataframe"):
            df_syn = ft.decode_to_dataframe(x_gen, template_df=real_df)
        elif hasattr(ft, "inverse_transform_dataframe"):
            df_syn = ft.inverse_transform_dataframe(x_gen, template_df=real_df)
        else:
            raise RuntimeError("No decode method found on feature_transform.")

        return df_syn.iloc[:num_rows].reset_index(drop=True)


def _import_tabsyn_core(repo_root: Optional[Union[str, Path]] = None):
    """
    Import TabSyn core after ensuring availability.
    """
    ensure_tabsyn_core_available()  # pip-install from GitHub if missing
    if repo_root:
        ensure_repo_on_path(repo_root)

    # Try common layouts
    try:
        from tabsyn.core.model import TabSyn  # type: ignore
        from tabsyn.core.checkpoint import load_checkpoint  # type: ignore
        return TabSyn, load_checkpoint
    except Exception:
        pass

    try:
        # Some forks place code under src/
        from tabsyn.src.core.model import TabSyn  # type: ignore
        from tabsyn.src.core.checkpoint import load_checkpoint  # type: ignore
        return TabSyn, load_checkpoint
    except Exception as e:
        raise ImportError("Could not import TabSyn core after bootstrap.") from e


def load_tabsyn(
    repo_root: Optional[Union[str, Path]],
    ckpt_path: Union[str, Path],
    device: str = "auto",
    seed: Optional[int] = 42,
) -> TabSynModel:
    """
    Load a TabSyn checkpoint. repo_root can be None. If None, the core is pip-installed.
    """
    if seed is not None:
        seed_everything(seed)

    ckpt_path = Path(ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    TabSyn, load_checkpoint = _import_tabsyn_core(repo_root)
    device_obj = pick_device(device)

    # Build model and load weights
    model = TabSyn()
    state = load_checkpoint(str(ckpt_path))
    sd = state.get("state_dict", state)
    model.load_state_dict(sd, strict=False)

    # Attach fitted feature transform if present
    ft = state.get("feature_transform")
    if ft is not None:
        model.feature_transform = ft

    return TabSynModel(model, device=device_obj)


def generate_synthetic(
    model: TabSynModel,
    real_df: pd.DataFrame,
    num_rows: int,
    batch_size: int = 512,
) -> pd.DataFrame:
    return model.generate(real_df, num_rows, batch_size=batch_size)
