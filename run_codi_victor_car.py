# run_codi_victor_car.py

import sys
from pathlib import Path

from utils import discretize_preprocess
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.evaluate.tstr.evaluation import TSTREvaluation
from katabatic.models.codi.models import CODI


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
    print("=== CoDi on CAR dataset (Victor) ===")
    print(f"Project root:   {ROOT}")

    # 1. Preprocess (discretize) raw data
    raw_path = ROOT / "raw_data" / "car.csv"
    disc_path = ROOT / "discretized_data" / "car.csv"
    disc_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Discretising: {raw_path} -> {disc_path}")
    discretize_preprocess(str(raw_path), str(disc_path))

    # 2. Define pipeline paths
    input_csv = str(disc_path)
    output_dir = str(ROOT / "sample_data" / "car")        # Real train/test dir
    synthetic_dir = str(ROOT / "synthetic" / "car" / "codi")

    print(f"Input CSV:      {input_csv}")
    print(f"Output dir:     {output_dir}")
    print(f"Synthetic dir:  {synthetic_dir}")

    # 3. Create pipeline with CODI + TSTR evaluator
    #    epochs/batch_size lowered to keep training manageable
    pipeline = TrainTestSplitPipeline(
        model=lambda: CODI(
            epochs=100,
            batch_size=256,
            device='cpu',   # force CPU
        ),
        evaluations=[TSTREvaluation]
    )

    # 4. Run pipeline: split -> train (CODI.train) -> TSTR.evaluate
    results = pipeline.run(
        input_csv=input_csv,
        output_dir=output_dir,
        synthetic_dir=synthetic_dir,
        real_test_dir=output_dir,   # for TSTREvaluation
    )

    print("\n=== Pipeline finished (CAR + CoDi) ===")
    print("Returned results dict:")
    print(results)
    print("\nCheck these locations:")
    print(f"  Real train/test : {output_dir}")
    print(f"  Synthetic data  : {synthetic_dir}")
    print("  TSTR results    : Results/car/codi_tstr_victor.csv (from your TSTREvaluation)")


if __name__ == "__main__":
    main()
