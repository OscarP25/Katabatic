import importlib
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch


GIT_URL = "git+https://github.com/amazon-science/tabsyn.git"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref in {"cpu", "cuda"}:
        return torch.device(pref)
    return torch.device(pref)


def ensure_repo_on_path(repo_root: Union[str, Path]) -> None:
    repo_root = Path(repo_root).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"Repo path not found: {repo_root}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    src = repo_root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _pip_install(package: str) -> None:
    """
    Install a package with pip in the current interpreter.
    """
    cmd = [sys.executable, "-m", "pip", "install", package]
    subprocess.run(cmd, check=True)


def ensure_tabsyn_core_available() -> None:
    """
    Ensure that `tabsyn` core is importable. Auto-install from GitHub if missing.
    """
    try:
        importlib.import_module("tabsyn")
        return
    except Exception:
        pass

    # Try to install from GitHub
    try:
        _pip_install(GIT_URL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Failed to auto-install amazon-science/tabsyn from GitHub. "
            "Install it manually with: pip install -e /path/to/tabsyn or pip install git+https://github.com/amazon-science/tabsyn.git"
        ) from e

    # Validate import
    importlib.import_module("tabsyn")
