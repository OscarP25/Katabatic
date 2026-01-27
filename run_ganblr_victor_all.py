# run_ganblr_victor_all.py
#
# Pipeline runner for GANBLR (same structure as your CTGAN runner).
# Uses:
#   - utils.discretize_preprocess
#   - TrainTestSplitPipeline
#   - TSTREvaluation
#   - katabatic.models.ganblr.models.GANBLRModel
#


from pathlib import Path

from utils import discretize_preprocess
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.evaluate.tstr.evaluation import TSTREvaluation
from katabatic.models.ganblr.models import GANBLRModel


def resolve_root() -> Path:
    root = Path.cwd().resolve()
    for _ in range(5):
        if (root / "pyproject.toml").exists() or (root / "raw_data").exists():
            break
        root = root.parent
    return root


def run_one_dataset(root: Path, dataset: str):
    print("\n" + "=" * 30)
    print(f"  Running dataset: {dataset} (GANBLR)")
    print("=" * 30)

    raw_path = root / "raw_data" / f"{dataset}.csv"
    disc_path = root / "discretized_data" / f"{dataset}.csv"
    disc_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset:    {dataset}")
    print(f"Raw CSV:    {raw_path}")
    print(f"Disc CSV:   {disc_path}")

    # 1) Discretize (raw -> discretized)
    discretize_preprocess(str(raw_path), str(disc_path))

    # 2) Pipeline paths
    input_csv = str(disc_path)
    output_dir = str(root / "sample_data" / dataset)              # pipeline split outputs
    synthetic_dir = str(root / "synthetic" / dataset / "ganblr")  # ganblr saves x_synth/y_synth here

    print(f"Output dir: {output_dir}")
    print(f"Synth dir:  {synthetic_dir}")

    # 3) SAFE hyperparams (edit only after stable)
    k = 1
    epochs = 20
    batch_size = 32
    warmup_epochs = 1
    max_synth = 5000
    seed = 42

    # If you want slightly more training for larger sets (still safe):
    if dataset in ["adult", "shuttle"]:
        epochs = 30
        batch_size = 32

    # 4) Build pipeline
    pipeline = TrainTestSplitPipeline(
        model=lambda: GANBLRModel(
            k=k,
            epochs=epochs,
            batch_size=batch_size,
            warmup_epochs=warmup_epochs,
            seed=seed,
            max_synth=max_synth,
        ),
        evaluations=[TSTREvaluation],
    )

    # 5) Run
    try:
        results = pipeline.run(
            input_csv=input_csv,
            output_dir=output_dir,
            synthetic_dir=synthetic_dir,
            real_test_dir=output_dir,
        )
        print(f"\n✅ Finished dataset: {dataset} (GANBLR)")
        print("Results dict:")
        print(results)
        print(f"Check synth files in: {synthetic_dir}")
        print(f"Check results in:     {root / 'Results' / dataset}")
    except Exception as e:
        print(f"\n❌ ERROR in dataset {dataset} (GANBLR): {e}")


def main():
    ROOT = resolve_root()
    print("=== Running Victor’s GANBLR on ALL datasets (PIPELINE) ===")
    print(f"Root: {ROOT}")

    # Start with ONE dataset to confirm it runs without crashing.
    # Then expand to all.
    datasets = ["car", "adult", "magic", "nursery", "shuttle"]

    for ds in datasets:
        run_one_dataset(ROOT, ds)

    print("\n=== COMPLETED ALL DATASETS (GANBLR) ===")


if __name__ == "__main__":
    main()
