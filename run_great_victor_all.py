# run_great_victor_all.py

from pathlib import Path

from utils import discretize_preprocess
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.evaluate.tstr.evaluation import TSTREvaluation
from katabatic.models.great import GReaTModel


def resolve_root() -> Path:
    root = Path.cwd().resolve()
    for _ in range(6):
        if (root / "pyproject.toml").exists() or (root / "raw_data").exists():
            break
        root = root.parent
    return root


def run_one_dataset(root: Path, dataset: str):
    print("\n" + "=" * 30)
    print(f"  Running dataset: {dataset} (GReaT)")
    print("=" * 30)

    raw_path = root / "raw_data" / f"{dataset}.csv"
    disc_path = root / "discretized_data" / f"{dataset}.csv"
    disc_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset:  {dataset}")
    print(f"Raw CSV:  {raw_path}")
    print(f"Disc CSV: {disc_path}")

    discretize_preprocess(str(raw_path), str(disc_path))

    input_csv = str(disc_path)
    output_dir = str(root / "sample_data" / dataset)
    synthetic_dir = str(root / "synthetic" / dataset / "great")

    # ---- SAFE LIMITS (prevents VSCode crashes) ----
    # Keep these small for week-9 demonstration runs.
    MAX_SYNTH = 2000          # cap number of synthetic rows written
    SAMPLE_K = 16             # generation batch size inside sample()
    SAMPLE_MAX_LENGTH = 160   # token length cap
    SAMPLE_TEMP = 0.7

    pipeline = TrainTestSplitPipeline(
        model=lambda: GReaTModel(
            llm="distilgpt2",         # small + fast
            epochs=1,                 # keep tiny
            batch_size=1,             # keep tiny
            experiment_dir=f"trainer_great_{dataset}",
            report_to=[],             # disable W&B etc
        ),
        evaluations=[TSTREvaluation],
    )

    try:
        results = pipeline.run(
            input_csv=input_csv,
            output_dir=output_dir,
            synthetic_dir=synthetic_dir,
            real_test_dir=output_dir,
            # Pass caps to the model through pipeline kwargs if your pipeline forwards **kwargs
            # If your pipeline DOES NOT forward kwargs, the model itself already caps epochs/batch_size in train().
            model_kwargs={
                "synthetic_dir": synthetic_dir,
                "max_synth": MAX_SYNTH,
                "sample_k": SAMPLE_K,
                "sample_max_length": SAMPLE_MAX_LENGTH,
                "sample_temperature": SAMPLE_TEMP,
                "device": "cpu",
            },
        )
        print(f"\n✅ Finished dataset: {dataset} (GReaT)")
        print(results)
    except TypeError:
        # If your TrainTestSplitPipeline doesn't support model_kwargs,
        # we still run with the model's internal caps (epochs/batch size) and default sampling.
        # This keeps your script compatible without editing the pipeline.
        try:
            results = pipeline.run(
                input_csv=input_csv,
                output_dir=output_dir,
                synthetic_dir=synthetic_dir,
                real_test_dir=output_dir,
            )
            print(f"\n✅ Finished dataset: {dataset} (GReaT)")
            print(results)
        except Exception as e:
            print(f"\n❌ ERROR in dataset {dataset} (GReaT): {e}")
    except Exception as e:
        print(f"\n❌ ERROR in dataset {dataset} (GReaT): {e}")


def main():
    ROOT = resolve_root()
    print("=== Running Victor’s GReaT on ALL datasets (PIPELINE) ===")
    print(f"Root: {ROOT}")

    datasets = ["car", "adult", "magic", "nursery", "shuttle"]
    for ds in datasets:
        run_one_dataset(ROOT, ds)

    print("\n=== COMPLETED ALL DATASETS (GReaT) ===")


if __name__ == "__main__":
    main()
