import subprocess
from pathlib import Path
import pandas as pd

from katabatic.models.base_model import Model


class SynthPopAdapter(Model):
    """
    Katabatic adapter for SynthPop (CART-based statistical generator).

    Generates full synthetic data (X + y) via R synthpop,
    then splits into x_synth.csv and y_synth.csv for TSTR.
    """

    def __init__(self, seed: int = 42):
        super().__init__()
        self.seed = seed
        self.is_fitted = False

    def train(self, dataset_dir: str, synthetic_dir: str, label_col: str):
        dataset_dir = Path(dataset_dir)
        synthetic_dir = Path(synthetic_dir)
        synthetic_dir.mkdir(parents=True, exist_ok=True)

    
        # Load training data
     
        x_train = pd.read_csv(dataset_dir / "x_train.csv")
        y_train = pd.read_csv(dataset_dir / "y_train.csv")

        if label_col not in y_train.columns:
            raise ValueError(f"Label column '{label_col}' not found")

        full_train = pd.concat([x_train, y_train], axis=1)

        input_csv = synthetic_dir / "train_full.csv"
        full_train.to_csv(input_csv, index=False)

        
        # Write R script
      
        r_script = f"""
        suppressMessages(library(synthpop))
        set.seed({self.seed})

        data <- read.csv("{input_csv.as_posix()}")

        syn_data <- syn(
          data,
          method = "cart",
          seed = {self.seed}
        )

        write.csv(
          syn_data$syn,
          "{(synthetic_dir / 'synthetic_full.csv').as_posix()}",
          row.names = FALSE
        )
        """

        r_script_path = synthetic_dir / "run_synthpop.R"
        r_script_path.write_text(r_script.strip())

       
        # Run SynthPop
      
        subprocess.run(
            ["Rscript", str(r_script_path)],
            check=True
        )

        
        # Split synthetic X / y (TSTR-compatible)
      
        synth_full = pd.read_csv(synthetic_dir / "synthetic_full.csv")

        x_synth = synth_full.drop(columns=[label_col])
        y_synth = synth_full[[label_col]]

        x_synth.to_csv(synthetic_dir / "x_synth.csv", index=False)
        y_synth.to_csv(synthetic_dir / "y_synth.csv", index=False)

        self.is_fitted = True
        return self
