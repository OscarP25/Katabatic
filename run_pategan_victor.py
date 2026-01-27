# run_pategan_victor.py

import sys
from pathlib import Path

from utils import discretize_preprocess
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.evaluate.tstr.evaluation import TSTREvaluation
from katabatic.models.pategan import PATEGAN




def resolve_root() -> Path:
    """Resolve project root (where pyproject.toml or raw_data/ lives)."""
    root = Path.cwd().resolve()
    for _ in range(5):
        if (root / "pyproject.toml").exists() or (root / "raw_data").exists():
            break
        root = root.parent
    return root


def main():
    ROOT = resolve_root()
    print("=== PATE-GAN on CAR dataset (Victor, fresh run) ===")
    print(f"Project root: {ROOT}")

    # 1. Preprocess (discretize) raw data
    raw_path = ROOT / "raw_data" / "car.csv"
    disc_path = ROOT / "discretized_data" / "car.csv"
    disc_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Discretising: {raw_path} -> {disc_path}")
    discretize_preprocess(str(raw_path), str(disc_path))

    # 2. Define pipeline paths
    input_csv = str(disc_path)
    output_dir = str(ROOT / "sample_data" / "car")        # Real train/test dir
    synthetic_dir = str(ROOT / "synthetic" / "car" / "pategan")

    print(f"Input CSV:     {input_csv}")
    print(f"Output dir:    {output_dir}")
    print(f"Synthetic dir: {synthetic_dir}")

    # 3. Create pipeline with REAL PATEGAN + TSTR evaluator
    pipeline = TrainTestSplitPipeline(
        model=PATEGAN,
        evaluations=[TSTREvaluation]   # This class reads x_synth/y_synth + x_test/y_test
    )

    # 4. Run pipeline: split -> train (PATEGAN.train) -> TSTR.evaluate
    results = pipeline.run(
        input_csv=input_csv,
        output_dir=output_dir,
        synthetic_dir=synthetic_dir,
        real_test_dir=output_dir,   # TSTREvaluation uses this
        # Optional: override training/privacy params (from README)
        epsilon=1.0,
        delta=1e-5,
        num_teachers=10,
        niter=10000,
        batch_size=128,
        random_state=42,
        verbose=1,
    )

    print("\n=== Pipeline finished ===")
    print("Returned results dict:")
    print(results)
    print("\nCheck these locations:")
    print(f"  Real train/test : {output_dir}")
    print(f"  Synthetic data  : {synthetic_dir}")
    print("  TSTR results    : Results/car/...")


if __name__ == "__main__":
    main()
