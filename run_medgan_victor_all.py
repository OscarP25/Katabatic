# run_medgan_victor_all.py

from pathlib import Path

from utils import discretize_preprocess
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.evaluate.tstr.evaluation import TSTREvaluation
from katabatic.models.medgan.models import MedGANModel


def resolve_root() -> Path:
    root = Path.cwd().resolve()
    for _ in range(5):
        if (root / "pyproject.toml").exists() or (root / "raw_data").exists():
            break
        root = root.parent
    return root


def run_one_dataset(root: Path, dataset: str):
    print("\n" + "=" * 30)
    print(f"  Running dataset: {dataset} (MedGAN)")
    print("=" * 30)

    raw_path = root / "raw_data" / f"{dataset}.csv"
    disc_path = root / "discretized_data" / f"{dataset}.csv"
    disc_path.parent.mkdir(parents=True, exist_ok=True)

    discretize_preprocess(str(raw_path), str(disc_path))

    input_csv = str(disc_path)
    output_dir = str(root / "sample_data" / dataset)
    synthetic_dir = str(root / "synthetic" / dataset / "medgan")

    # SAFE DEFAULTS (increase later)
    pipeline = TrainTestSplitPipeline(
        model=lambda: MedGANModel(
            ae_pretrain_epochs=20,
            gan_epochs=50,
            batch_size=512,
            device="cpu",
            seed=42,
            max_synth=5000,
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
        print(f"\n✅ Finished dataset: {dataset} (MedGAN)")
        print(results)
    except Exception as e:
        print(f"\n❌ ERROR in dataset {dataset} (MedGAN): {e}")


def main():
    ROOT = resolve_root()
    datasets = ["car", "adult", "magic", "nursery", "shuttle"]

    print("=== Running Victor’s MedGAN on ALL datasets (PIPELINE) ===")
    print(f"Root: {ROOT}")

    for ds in datasets:
        run_one_dataset(ROOT, ds)

    print("\n=== COMPLETED ALL DATASETS (MedGAN) ===")


if __name__ == "__main__":
    main()
