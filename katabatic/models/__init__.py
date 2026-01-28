"""
Model package for Katabatic.

Exposes the abstract base Model class and the concrete PATEGAN implementation.
"""
from .copulagan_victor.adapter import CopulaGANAdapter
from .ctgan_victor.adapter import CTGANAdapter
from .codi_victor.adapter import CoDiAdapter
from .ganblr_victor.models import GANBLRModel
from .medgan_victor.models import MedGANModel
from .great_victor.models import GReaTModel
from .gaussiancopula_victor.adapter import GaussianCopulaAdapter
from .forestdiffusion_victor.adapter import ForestDiffusionAdapter
from .tvae_victor.adapter import TVAEAdapter
from .pategan_victor.adapter import PATEGANAdapter



__all__ = ["Model", "PATEGAN"]
