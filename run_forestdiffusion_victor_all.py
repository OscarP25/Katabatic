from pathlib import Path

from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.forestdiffusion.adapter import ForestDiffusionAdapter
from katabatic.evaluate.tstr.evaluation import TSTREvaluation

ROOT = Path.cwd().resolve()

datasets = ["car", "adult", "magic", "nursery", "shuttle"]
label_map = {"car": "6", "adult": "class", "magic": "class", "nursery": "class", "shuttle": "class"}

for ds in datasets:
    print("\n" + "=" * 40)
    print("Running ForestDiffusion on", ds)
    print("=" * 40)

    sample_dir = ROOT / "sample_data" / ds
    synth_dir = ROOT / "synthetic" / ds / "forestdiffusion"
    sample_dir.mkdir(parents=True, exist_ok=True)
    synth_dir.mkdir(parents=True, exist_ok=True)

    pipeline = TrainTestSplitPipeline(
        model=lambda: ForestDiffusionAdapter(
            n_t=50,
            reps=2,
            model="xgboost",
            n_estimators=200,
            max_depth=7,
            seed=666,
            gpu_hist=False,
        ),
        evaluations=[TSTREvaluation],
        override_evaluations=True,  # IMPORTANT: prevents duplicate TSTR
    )

    pipeline.run(
        input_csv=str(ROOT / "raw_data" / f"{ds}.csv"),
        output_dir=str(sample_dir),
        synthetic_dir=str(synth_dir),
        label_col=label_map[ds],
    )

print("\nALL DATASETS COMPLETE")
