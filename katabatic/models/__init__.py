"""
Model package for Katabatic.

Exposes the abstract base Model class and the concrete PATEGAN implementation.
"""

from .base_model import Model
from .pategan import PATEGAN  # this refers to katabatic/models/pategan/__init__.py

__all__ = ["Model", "PATEGAN"]
