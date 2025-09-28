"""
Lightweight tabular synthesis & TSTR evaluation toolkit.

Public API:
- BaseSynthesizer
- GaussianSynthesizer
- FrequencySynthesizer
- encode_mixed_frame / decode_mixed_frame
- evaluate_tstr
- set_seed, safe_train_val_test_split
"""

from .models import (
    BaseSynthesizer,
    GaussianSynthesizer,
    FrequencySynthesizer,
)

from .utils import (
    set_seed,
    encode_mixed_frame,
    decode_mixed_frame,
    safe_train_val_test_split,
    evaluate_tstr,
)

__all__ = [
    "BaseSynthesizer",
    "GaussianSynthesizer",
    "FrequencySynthesizer",
    "set_seed",
    "encode_mixed_frame",
    "decode_mixed_frame",
    "safe_train_val_test_split",
    "evaluate_tstr",
]
