# TabSyn Wrapper for Katabatic

This module wraps [amazon-science/tabsyn](https://github.com/amazon-science/tabsyn) and auto-installs it if missing.

## Files
- `__init__.py` public API
- `models.py` load and generate
- `utils.py` seeding, device, path, auto-install
- `pyproject.toml` Poetry config
- `README.md` this file

## Quick start
```python
import pandas as pd
from tabsyn import load_tabsyn, generate_synthetic

# Set None to auto-install the core. Or pass a local clone path.
model = load_tabsyn(repo_root=None, ckpt_path="checkpoints/tabsyn.ckpt", device="auto", seed=42)

real_df = pd.read_csv("real.csv")
synthetic_df = generate_synthetic(model, real_df, num_rows=1000)
print(synthetic_df.head())