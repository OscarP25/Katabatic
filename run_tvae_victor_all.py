from pathlib import Path

from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.tvae.adapter import TVAEAdapter
from katabatic.evaluate.tstr.evaluation import TSTREvaluation

ROOT = Path.cwd().resolve()

datasets = ["car", "adult", "magic", "nursery", "shuttle"]
label_map = {
    "car": "6",
    "adult": "class",
    "magic": "class",
    "nursery": "8",   # ✅ Nursery label column is "8"
    "shuttle": "class",
}

for ds in datasets:
    print("\n" + "=" * 40)
    print("Running TVAE on", ds)
    print("=" * 40)

    sample_dir = ROOT / "sample_data" / ds
    synth_dir = ROOT / "synthetic" / ds / "tvae"
    sample_dir.mkdir(parents=True, exist_ok=True)
    synth_dir.mkdir(parents=True, exist_ok=True)

    pipeline = TrainTestSplitPipeline(
        model=lambda: TVAEAdapter(
            epochs=20,
            batch_size=256,
            enable_gpu=False,
            max_synth=2000,
        ),
        evaluations=[TSTREvaluation],
    )

    pipeline.run(
        input_csv=str(ROOT / "raw_data" / f"{ds}.csv"),
        output_dir=str(sample_dir),
        synthetic_dir=str(synth_dir),
        label_col=label_map[ds],      # ✅ correct per dataset
        real_test_dir=str(sample_dir),
    )

print("\nALL DATASETS COMPLETE (TVAE)")
