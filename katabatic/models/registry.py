"""Model registry for dynamic model loading."""

from typing import Dict, Type, Optional
import importlib
from .base_model import Model


class ModelRegistry:
    """Registry for managing available models and their dependencies."""

    _models: Dict[str, Dict] = {
        'ganblr': {
            'module': 'katabatic.models.ganblr.models',
            'class': 'GANBLR',
            'dependencies': ['tensorflow', 'pgmpy', 'pyitlib', 'tf_keras', 'scipy'],
            'extra': 'ganblr'
        },
        'great': {
            'module': 'katabatic.models.great.models',
            'class': 'GReaT',
            'dependencies': ['transformers', 'torch'],
            'extra': 'great'
        },
        'tabsyn': {
            'module': 'katabatic.models.tabsyn.models',
            'class': 'Tabsyn',
            'dependencies': [],
            'extra': 'tabsyn'
        },
        'tabddpm': {
            'module': 'katabatic.models.tabddpm.models',
            'class': 'Tabddpm',
            'dependencies': [],
            'extra': 'tabddpm'
        },
        'pategan': {
            'module': 'katabatic.models.pategan.models',
            'class': 'PATEGANSynthesizer',
            'dependencies': ['tensorflow', 'numpy', 'pandas'],
            'extra': 'pategan'
        },
        'ctgan': {
            'module': 'katabatic.models.ctgan.models',
            'class': 'CTGANModel',
            'dependencies': ['torch', 'sklearn'],
            'extra': 'ctgan'
        },
        'mtabgen': {
            'module': 'katabatic.models.mtabgen.models',
            'class': 'MTabGen',
            'dependencies': ['torch', 'sklearn'], # Assuming similar dependencies to cvae/tvae, or adjust as needed
            'extra': 'mtabgen'
        },
        'cvae': {
            'module': 'katabatic.models.cvae.models',
            'class': 'CVAE',
            'dependencies': ['torch', 'sklearn'],
            'extra': 'cvae'
        },
        'tvae': {
            'module': 'katabatic.models.tvae.models',
            'class': 'TVAE',
            'dependencies': ['torch', 'sklearn'],
            'extra': 'tvae'
        },
        'pacgan': {
            'module': 'katabatic.models.pacgan.models',
            'class': 'PacGANModel',
            'dependencies': ['torch', 'sklearn'],
            'extra': 'pacgan'
        },
        'vaegan': {
            'module': 'katabatic.models.vaegan.models',
            'class': 'VAEGANModel',
            'dependencies': ['torch', 'sklearn'],
            'extra': 'vaegan'
        },
        'copulagan': {
            'module': 'katabatic.models.copulagan.models',
            'class': 'CopulaGANModel',
            'dependencies': ['torch', 'sklearn', 'scipy'],
            'extra': 'copulagan'
        },
        'dpctgan': {
            'module': 'katabatic.models.dpctgan.models',
            'class': 'DPCTGANModel',
            'dependencies': ['torch', 'sklearn', 'opacus'],
            'extra': 'dpctgan'
        }

    }

    @classmethod
    def get_available_models(cls) -> list[str]:
        """Get list of all registered model names."""
        return list(cls._models.keys())

    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict]:
        """Get information about a specific model."""
        return cls._models.get(model_name.lower())

    @classmethod
    def load_model(cls, model_name: str) -> Type[Model]:
        """Dynamically load a model class."""
        model_name = model_name.lower()

        if model_name not in cls._models:
            available = ', '.join(cls.get_available_models())
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available}")

        model_info = cls._models[model_name]

        # Check dependencies
        missing_deps = []
        for dep in model_info['dependencies']:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            raise ImportError(
                f"Missing dependencies for {model_name}: {missing_deps}. "
                f"Install with: pip install katabatic[{model_info['extra']}]"
            )

        # Import and return the model class
        try:
            module = importlib.import_module(model_info['module'])
            model_class = getattr(module, model_info['class'])
            return model_class
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load model {model_name}: {e}")

    @classmethod
    def create_model(cls, model_name: str, *args, **kwargs) -> Model:
        """Create an instance of the specified model."""
        model_class = cls.load_model(model_name)
        return model_class(*args, **kwargs)


def get_model(model_name: str, *args, **kwargs) -> Model:
    """Convenience function to create a model instance."""
    return ModelRegistry.create_model(model_name, *args, **kwargs)


def list_models() -> list[str]:
    """Convenience function to list available models."""
    return ModelRegistry.get_available_models()
