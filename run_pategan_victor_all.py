# run_pategan_victor_all.py

from pathlib import Path
from utils import discretize_preprocess
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.evaluate.tstr.evaluation import TSTREvaluation
from katabatic.models.pategan import PATEGAN


DATASETS = ["car", "adult", "magic", "nursery", "shuttle"]


def resolve_root() -> Path:
    root = Path.cwd().resolve()
    for _ in range(5):
        if (root / "pyproject.toml").exists() or (root / "raw_data").exists():
            break
        root = root.parent
    return root


def run_for_dataset(root: Path, name: str):
    print("\n" + "=" * 30)
    print(f"  Running dataset: {name}")
    print("=" * 30 + "\n")

    raw_path = root / "raw_data" / f"{name}.csv"
    disc_path = root / "discretized_data" / f"{name}.csv"
    disc_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Discretising: {raw_path} -> {disc_path}")
    discretize_preprocess(str(raw_path), str(disc_path))

    input_csv = str(disc_path)
    output_dir = str(root / "sample_data" / name)
    synthetic_dir = str(root / "synthetic" / name / "pategan")

    pipeline = TrainTestSplitPipeline(
        model=PATEGAN,
        evaluations=[TSTREvaluation]
    )

    results = pipeline.run(
        input_csv=input_csv,
        output_dir=output_dir,
        synthetic_dir=synthetic_dir,
        real_test_dir=output_dir,
        epsilon=1.0,
        delta=1e-5,
        num_teachers=10,
        niter=10000,
        batch_size=128,
        random_state=42,
        verbose=1,
    )

    print(f"✅ Finished dataset {name}")
    print(results)


def main():
    root = resolve_root()
    print("=== Running Victor’s PATE-GAN on ALL datasets ===")

    for ds in DATASETS:
        try:
            run_for_dataset(root, ds)
        except Exception as e:
            print(f"❌ ERROR in dataset {ds}: {e}")

    print("\n=== COMPLETED ALL DATASETS ===")


if __name__ == "__main__":
    main()
