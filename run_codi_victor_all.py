# run_codi_victor_all.py

import sys
from pathlib import Path

from utils import discretize_preprocess
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.evaluate.tstr.evaluation import TSTREvaluation
from katabatic.models.codi.models import CODI


def resolve_root() -> Path:
    root = Path.cwd().resolve()
    for _ in range(5):
        if (root / "pyproject.toml").exists() or (root / "raw_data").exists():
            break
        root = root.parent
    return root


def run_one_dataset(root: Path, dataset: str):
    print("\n" + "=" * 30)
    print(f"  Running dataset: {dataset}")
    print("=" * 30)

    # 1. Discretise raw -> discretized_data/<dataset>.csv
    raw_path = root / "raw_data" / f"{dataset}.csv"
    disc_path = root / "discretized_data" / f"{dataset}.csv"
    disc_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset:    {dataset}")
    print(f"Raw CSV:    {raw_path}")
    print(f"Disc CSV:   {disc_path}")

    discretize_preprocess(str(raw_path), str(disc_path))

    # 2. Paths for pipeline
    input_csv = str(disc_path)
    output_dir = str(root / "sample_data" / dataset)
    synthetic_dir = str(root / "synthetic" / dataset / "codi")

    print(f"Output dir: {output_dir}")
    print(f"Synth dir:  {synthetic_dir}")

    # 3. Build a CODI instance with slightly tuned hyperparams per dataset
    #    (You can tweak these later if needed)
    if dataset in ["car", "nursery"]:
        epochs = 100
        batch_size = 256
    elif dataset in ["adult", "magic"]:
        epochs = 150
        batch_size = 512
    else:  # shuttle is larger but simple
        epochs = 200
        batch_size = 512

    pipeline = TrainTestSplitPipeline(
        model=lambda: CODI(
            epochs=epochs,
            batch_size=batch_size,
            device='cpu',    # keep everything on CPU
        ),
        evaluations=[TSTREvaluation],
    )

    # 4. Run pipeline
    try:
        results = pipeline.run(
            input_csv=input_csv,
            output_dir=output_dir,
            synthetic_dir=synthetic_dir,
            real_test_dir=output_dir,   # TSTR reads x_test/y_test from here
        )
        print(f"\n✅ Finished dataset: {dataset}")
        print("Results dict:")
        print(results)
        print("TSTR CSV should be in:")
        print(f"  Results/{dataset}/codi_tstr_victor.csv")
    except Exception as e:
        print(f"\n❌ ERROR in dataset {dataset}: {e}")


def main():
    ROOT = resolve_root()
    print("=== Running Victor’s CoDi on ALL datasets ===")

    datasets = ["car", "adult", "magic", "nursery", "shuttle"]

    for ds in datasets:
        run_one_dataset(ROOT, ds)

    print("\n=== COMPLETED ALL DATASETS (CoDi) ===")


if __name__ == "__main__":
    main()
