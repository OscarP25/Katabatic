"""
SynthPop model integration for Katabatic.

Paper:
Nowok, B., Raab, G. M., & Dibben, C. (2016).
Flexible generation of synthetic data using CART.
Journal of Statistical Software, 74(11).

This implementation uses the official R package `synthpop`
and performs sequential conditional synthesis using CART.
"""

import subprocess
import logging
from pathlib import Path

from katabatic.models.base_model import Model
from .utils import write_r_script

logger = logging.getLogger(__name__)


class SynthPop(Model):
    """
    Katabatic wrapper for SynthPop (CART-based statistical generator).
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = config or {}

    def train(
        self,
        dataset_path: str,
        synthetic_path: str,
        seed: int = 42,
        **kwargs
    ):
        """
        Generate synthetic tabular data using SynthPop.

        Parameters
        ----------
        dataset_path : str
            Path to input CSV (training data, includes X and y)
        synthetic_path : str
            Path where synthetic CSV will be saved
        seed : int
            Random seed for reproducibility
        """

        dataset_path = Path(dataset_path)
        synthetic_path = Path(synthetic_path)
        synthetic_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Running SynthPop (CART-based synthesis)")
        logger.info(f"Input dataset   : {dataset_path}")
        logger.info(f"Synthetic output: {synthetic_path}")

        # Temporary R script
        r_script_path = synthetic_path.parent / "run_synthpop.R"

        write_r_script(
            r_script_path=r_script_path,
            input_csv=dataset_path,
            output_csv=synthetic_path,
            seed=seed
        )

        try:
            subprocess.run(
                ["Rscript", str(r_script_path)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError("SynthPop execution failed") from e

        logger.info(" SynthPop synthetic data generation completed")
