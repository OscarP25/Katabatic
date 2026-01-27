from pathlib import Path
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.gaussianCopula import GaussianCopulaModel
from katabatic.evaluate.tstr.evaluation import TSTREvaluation

ROOT = Path.cwd()
datasets = ["car", "adult", "magic", "nursery", "shuttle"]

for ds in datasets:
    print("\n" + "=" * 40)
    print("Running GaussianCopula on", ds)
    print("=" * 40)

    pipeline = TrainTestSplitPipeline(
        model=lambda: GaussianCopulaModel(
            seed=42,
            max_synth=2000,
        ),
        evaluations=[TSTREvaluation],
    )

    pipeline.run(
        input_csv=str(ROOT / "raw_data" / f"{ds}.csv"),
        output_dir=str(ROOT / "sample_data" / ds),
        synthetic_dir=str(ROOT / "synthetic" / ds / "gaussiancopula"),
        label_col="6",
        real_test_dir=str(ROOT / "sample_data" / ds),
    )

print("\nALL DATASETS COMPLETE (GaussianCopula)")
