# run_copulagan_victor_all.py
# ✅ COMPLETE VERSION (includes share-format output: Model | Metric | victor nyabote)

from __future__ import annotations

from pathlib import Path
import pandas as pd
import re
from typing import Any, Dict, Optional

from utils import discretize_preprocess
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.copulaGAN.adapter import CopulaGANAdapter
from katabatic.evaluate.tstr.evaluation import TSTREvaluation

OWNER = "victor nyabote"


def resolve_root() -> Path:
    """Find project root (where pyproject.toml or raw_data exists)."""
    root = Path.cwd().resolve()
    for _ in range(6):
        if (root / "pyproject.toml").exists() or (root / "raw_data").exists():
            return root
        root = root.parent
    return Path.cwd().resolve()


def get_label_col(csv_path: Path) -> str:
    """Auto-detect label column as the last column of the CSV."""
    cols = list(pd.read_csv(csv_path, nrows=0).columns)
    if not cols:
        raise ValueError(f"No columns found in {csv_path}")
    return cols[-1]


def safe_hparams_for_dataset(ds: str) -> dict:
    """
    Reduce memory usage for big datasets.
    CopulaGAN/CTGAN can explode RAM on large/high-cardinality tables (e.g., adult).
    """
    ds = ds.lower()
    if ds == "adult":
        return dict(epochs=5, batch_size=128, max_synth=5000)
    if ds in ("shuttle",):
        return dict(epochs=5, batch_size=128, max_synth=5000)
    # smaller ones
    return dict(epochs=10, batch_size=256, max_synth=2000)


# -------------------------------------------------------------------
# ✅ Share-format extraction + saving
# -------------------------------------------------------------------

def _normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def _map_clf_name(name: str) -> Optional[str]:
    n = _normalize_key(name)
    if n in ("lr", "logisticregression", "logreg"):
        return "LR"
    if n in ("mlp", "multilayerperceptron"):
        return "MLP"
    if n in ("rf", "randomforest", "randomforestclassifier"):
        return "RF"
    if n in ("xgboost", "xgb"):
        return "XGBoost"
    return None


def _extract_from_dataframe(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    cols = {_normalize_key(c): c for c in df.columns}

    # long format: classifier/metric/value
    if "classifier" in cols and "metric" in cols and ("value" in cols or "score" in cols):
        c_classifier = cols["classifier"]
        c_metric = cols["metric"]
        c_value = cols.get("value") or cols.get("score")

        out: Dict[str, Dict[str, float]] = {}
        for _, row in df.iterrows():
            clf_key = _map_clf_name(str(row[c_classifier]))
            if not clf_key:
                continue
            met = _normalize_key(row[c_metric])
            val = row[c_value]
            if pd.isna(val):
                continue

            out.setdefault(clf_key, {})
            if met in ("accuracy", "acc"):
                out[clf_key]["accuracy"] = float(val)
            elif met in ("f1", "f1score", "f1_score"):
                out[clf_key]["f1"] = float(val)

        return out

    # fallback: convert to dict and try again
    return _extract_from_dict(df.to_dict())


def _extract_from_dict(d: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    wanted = {"LR", "MLP", "RF", "XGBoost"}
    out: Dict[str, Dict[str, float]] = {}

    # direct dict-of-dicts
    def capture(clf_name: str, sub: Any):
        clf_key = _map_clf_name(clf_name)
        if not clf_key or not isinstance(sub, dict):
            return

        sub_norm = {_normalize_key(k): v for k, v in sub.items()}
        acc = sub_norm.get("accuracy") or sub_norm.get("acc")
        f1 = sub_norm.get("f1") or sub_norm.get("f1score") or sub_norm.get("f1_score")

        if acc is not None:
            out.setdefault(clf_key, {})["accuracy"] = float(acc)
        if f1 is not None:
            out.setdefault(clf_key, {})["f1"] = float(f1)

    for k, v in d.items():
        capture(str(k), v)

    # flatten nested keys (e.g. tstr.lr.accuracy)
    flat: Dict[str, Any] = {}

    def flatten(prefix: str, obj: Any):
        if isinstance(obj, dict):
            for kk, vv in obj.items():
                flatten(f"{prefix}.{kk}" if prefix else str(kk), vv)
        else:
            flat[_normalize_key(prefix)] = obj

    flatten("", d)

    for clf in wanted:
        clf_norm = _normalize_key(clf)
        acc = flat.get(f"{clf_norm}accuracy") or flat.get(f"{clf_norm}acc")
        f1 = flat.get(f"{clf_norm}f1") or flat.get(f"{clf_norm}f1score") or flat.get(f"{clf_norm}f1_score")
        if acc is not None or f1 is not None:
            out.setdefault(clf, {})
            if acc is not None:
                out[clf]["accuracy"] = float(acc)
            if f1 is not None:
                out[clf]["f1"] = float(f1)

    return out


def extract_tstr_scores(results: Any) -> Dict[str, Dict[str, float]]:
    """
    Try hard to extract:
      {"LR":{"accuracy":..,"f1":..}, "MLP":..., "RF":..., "XGBoost":...}
    from pipeline.run() output.
    """
    if isinstance(results, pd.DataFrame):
        return _extract_from_dataframe(results)

    if isinstance(results, (list, tuple)):
        for item in results:
            got = extract_tstr_scores(item)
            if got:
                return got
        return {}

    if isinstance(results, dict):
        got = _extract_from_dict(results)
        if got:
            return got
        for v in results.values():
            got = extract_tstr_scores(v)
            if got:
                return got

    return {}


def save_share_format(
    out_dir: Path,
    dataset: str,
    model_name: str,
    owner: str,
    scores: Dict[str, Dict[str, float]],
):
    """
    Writes exactly your share format (your column only):
    Model | Metric | victor nyabote
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for clf in ["LR", "MLP", "RF", "XGBoost"]:
        if clf not in scores:
            continue
        if "accuracy" in scores[clf]:
            rows.append([clf, "Accuracy", round(scores[clf]["accuracy"], 4)])
        if "f1" in scores[clf]:
            rows.append([clf, "F1 Score", round(scores[clf]["f1"], 4)])

    df = pd.DataFrame(rows, columns=["Model", "Metric", owner])

    # Console view (easy to copy into chat/team sheet)
    print("\n" + f"{model_name} — {dataset}".center(55))
    print(df.to_string(index=False))

    # Per-dataset file
    per_ds = out_dir / f"{model_name.lower()}_{dataset}_{owner.replace(' ', '_')}.csv"
    df.to_csv(per_ds, index=False)

    # Master appendable file
    master_path = out_dir / f"{model_name.lower()}_{owner.replace(' ', '_')}_ALL.csv"
    if master_path.exists():
        old = pd.read_csv(master_path)
        merged = pd.concat([old, df.assign(Dataset=dataset)], ignore_index=True)
    else:
        merged = df.assign(Dataset=dataset)
    merged.to_csv(master_path, index=False)

    print(f"\n✅ Saved share CSV: {per_ds}")
    print(f"✅ Updated master CSV: {master_path}")


# -------------------------------------------------------------------
# ✅ Main run per dataset
# -------------------------------------------------------------------

def run_one_dataset(root: Path, ds: str):
    print("\n" + "=" * 40)
    print(f"Running CopulaGAN on {ds}")
    print("=" * 40)

    raw_csv = root / "raw_data" / f"{ds}.csv"
    if not raw_csv.exists():
        print(f"❌ Missing file: {raw_csv}")
        return

    # 1) Discretize
    disc_csv = root / "discretized_data" / f"{ds}.csv"
    disc_csv.parent.mkdir(parents=True, exist_ok=True)
    discretize_preprocess(str(raw_csv), str(disc_csv))

    # 2) Auto label col (last col)
    label_col_name = get_label_col(disc_csv)

    # 3) Output folders
    sample_dir = root / "sample_data" / ds
    synth_dir = root / "synthetic" / ds / "copulagan"
    sample_dir.mkdir(parents=True, exist_ok=True)
    synth_dir.mkdir(parents=True, exist_ok=True)

    hp = safe_hparams_for_dataset(ds)

    # 4) Pipeline
    pipeline = TrainTestSplitPipeline(
        model=lambda: CopulaGANAdapter(
            target_col="target",  # internal target name used by adapter
            epochs=hp["epochs"],
            batch_size=hp["batch_size"],
            max_synth=hp["max_synth"],
        ),
        evaluations=[TSTREvaluation],
    )

    try:
        results = pipeline.run(
            input_csv=str(disc_csv),
            output_dir=str(sample_dir),
            synthetic_dir=str(synth_dir),
            label_col=label_col_name,
            real_test_dir=str(sample_dir),
        )
        print(f"\n✅ Finished {ds}")
        print(results)

        # ✅ Extract TSTR scores and save in your team-share format
        scores = extract_tstr_scores(results)
        if not scores:
            print("⚠️ Could not auto-extract TSTR scores from pipeline results.")
            print("   Paste the printed `results` here and I’ll tailor extractor to your exact structure.")
        else:
            share_dir = root / "results" / "shareable" / "copulagan"
            save_share_format(
                out_dir=share_dir,
                dataset=ds,
                model_name="CopulaGAN",
                owner=OWNER,
                scores=scores,
            )

    except Exception as e:
        print(f"\n❌ ERROR in {ds}: {e}")


def main():
    root = resolve_root()
    print("=== Running Victor’s CopulaGAN on ALL datasets (PIPELINE) ===")
    print(f"Root: {root}")

    datasets = ["car", "adult", "magic", "nursery", "shuttle"]
    for ds in datasets:
        run_one_dataset(root, ds)

    print("\n=== ALL DATASETS COMPLETE (CopulaGAN) ===")


if __name__ == "__main__":
    main()
