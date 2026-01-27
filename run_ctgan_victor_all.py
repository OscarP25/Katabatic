# run_ctgan_victor_all.py

from pathlib import Path

from utils import discretize_preprocess
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.evaluate.tstr.evaluation import TSTREvaluation
from katabatic.models.ctgan.models import CTGANModel



def resolve_root() -> Path:
    root = Path.cwd().resolve()
    for _ in range(5):
        if (root / "pyproject.toml").exists() or (root / "raw_data").exists():
            break
        root = root.parent
    return root


def run_one_dataset(root: Path, dataset: str):
    print("\n" + "=" * 30)
    print(f"  Running dataset: {dataset} (CTGAN)")
    print("=" * 30)

    raw_path = root / "raw_data" / f"{dataset}.csv"
    disc_path = root / "discretized_data" / f"{dataset}.csv"
    disc_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset:    {dataset}")
    print(f"Raw CSV:    {raw_path}")
    print(f"Disc CSV:   {disc_path}")

    discretize_preprocess(str(raw_path), str(disc_path))

    input_csv = str(disc_path)
    output_dir = str(root / "sample_data" / dataset)
    synthetic_dir = str(root / "synthetic" / dataset / "ctgan")

    print(f"Output dir: {output_dir}")
    print(f"Synth dir:  {synthetic_dir}")

    if dataset in ["car", "nursery"]:
        epochs = 100
        batch_size = 256
    elif dataset in ["adult", "magic"]:
        epochs = 150
        batch_size = 512
    else:
        epochs = 200
        batch_size = 512

    pipeline = TrainTestSplitPipeline(
        model=lambda: CTGANModel(
            epochs=epochs,
            batch_size=batch_size,
            backend="torch",
            device="cpu",
            seed=42,
        ),
        evaluations=[TSTREvaluation],
    )

    try:
        results = pipeline.run(
            input_csv=input_csv,
            output_dir=output_dir,
            synthetic_dir=synthetic_dir,
            real_test_dir=output_dir,
        )
        print(f"\n✅ Finished dataset: {dataset} (CTGAN)")
        print("Results dict:")
        print(results)
        print("TSTR CSV should be in:")
        print(f"  Results/{dataset}/ctgan_tstr_victor.csv")
    except Exception as e:
        print(f"\n❌ ERROR in dataset {dataset} (CTGAN): {e}")


def main():
    ROOT = resolve_root()
    print("=== Running Victor’s CTGAN on ALL datasets ===")

    datasets = ["car", "adult", "magic", "nursery", "shuttle"]

    for ds in datasets:
        run_one_dataset(ROOT, ds)

    print("\n=== COMPLETED ALL DATASETS (CTGAN) ===")


if __name__ == "__main__":
    main()
