# merge_sdv_results.py
# Merge TSTR results across datasets for SDV-based models into one Excel/CSV.

import os
import pandas as pd

DATASETS = ["car", "adult", "magic", "nursery", "shuttle"]
RESULTS_ROOT = "Results"

# model folder name -> expected results filename (as written by TSTREvaluation.save_results_to_csv)
MODELS = {
    "copulagan": "copulagan_tstr.csv",
    "gaussiancopula": "gaussiancopula_tstr.csv",
    # if your folder uses different names, add them here
    # "gaussianCopula": "gaussianCopula_tstr.csv",
}

def main():
    all_rows = []
    print("=== Merging SDV TSTR results (CopulaGAN + GaussianCopula) ===")

    for ds in DATASETS:
        for model_key, fname in MODELS.items():
            p = os.path.join(RESULTS_ROOT, ds, fname)

            if not os.path.exists(p):
                print(f"⚠️ Missing: {p}")
                continue

            df = pd.read_csv(p)

            # Normalize to consistent columns
            if "Dataset" not in df.columns:
                df["Dataset"] = ds

            # If your CSV uses "Model" to mean the classifier (LR/MLP/RF/XGBoost),
            # we store the generator name separately as "Generator".
            if "Generator" not in df.columns:
                df["Generator"] = model_key

            # Some older merges use different column naming:
            # ensure we have at least: Model, Metric, Value
            if "Model" not in df.columns and "Classifier" in df.columns:
                df.rename(columns={"Classifier": "Model"}, inplace=True)

            all_rows.append(df)
            print(f"✔ Loaded: {p}")

    if not all_rows:
        print("\n❌ No result files found — nothing to merge.")
        return

    final = pd.concat(all_rows, ignore_index=True)

    # Consistent column order
    preferred = ["Dataset", "Generator", "Model", "Metric", "Value"]
    ordered = [c for c in preferred if c in final.columns]
    ordered += [c for c in final.columns if c not in ordered]
    final = final[ordered]

    # Save Excel (preferred)
    out_xlsx = "Victor_TSTR_All_Datasets_SDV.xlsx"
    try:
        final.to_excel(out_xlsx, index=False)
        print(f"\n✅ Merged results saved to: {out_xlsx}")
    except ModuleNotFoundError:
        out_csv = "Victor_TSTR_All_Datasets_SDV.csv"
        final.to_csv(out_csv, index=False)
        print("\nopenpyxl not installed – saved CSV instead.")
        print(f"✅ Merged results saved to: {out_csv}")

if __name__ == "__main__":
    main()
