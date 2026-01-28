"""
Model package for Katabatic.

Exposes the abstract base Model class and the concrete PATEGAN implementation.
"""

from .base_model import Model
from .pategan import PATEGAN  # this refers to katabatic/models/pategan/__init__.py
# GAN-based
from .ctgan.adapter import CTGANAdapter
from .copulaGAN.adapter import CopulaGANAdapter
from .gaussianCopula.adapter import GaussianCopulaAdapter
from .pategan.adapter import PATEGANAdapter
from .ganblr.models import GANBLRModel
from .medgan.models import MedGANModel

# Diffusion / Transformer
from .codi.models import CoDiModel
from .forestdiffusion.adapter import ForestDiffusionAdapter
from .tvae.adapter import TVAEAdapter
from .great.models import GReaTModel


__all__ = ["Model", "PATEGAN"]
