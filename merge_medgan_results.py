# merge_medgan_results.py
# Merges MedGAN TSTR CSV results across datasets into one Excel/CSV.

import os
import pandas as pd

DATASETS = ["car", "adult", "magic", "nursery", "shuttle"]
RESULTS_ROOT = "Results"

def main():
    all_rows = []

    print("=== Merging MedGAN TSTR results ===")

    for ds in DATASETS:
        # Common naming patterns in Katabatic results
        candidate_files = [
            "medgan_tstr.csv",
            "tstr.csv",
            "tstr_results.csv",
            "medgan_tstr_victor.csv",
        ]

        found_path = None
        for fname in candidate_files:
            p = os.path.join(RESULTS_ROOT, ds, fname)
            if os.path.exists(p):
                found_path = p
                break

        if found_path is None:
            print(f"⚠️ Missing results for {ds} in {os.path.join(RESULTS_ROOT, ds)}")
            continue

        df = pd.read_csv(found_path)

        if "Dataset" not in df.columns:
            df["Dataset"] = ds
        if "Generator" not in df.columns:
            df["Generator"] = "MedGAN"

        all_rows.append(df)
        print(f"✔ Loaded {found_path}")

    if not all_rows:
        print("\n❌ No MedGAN result files found — nothing to merge.")
        return

    final = pd.concat(all_rows, ignore_index=True)

    preferred_order = ["Dataset", "Generator", "Model", "Metric", "Value"]
    ordered = [c for c in preferred_order if c in final.columns]
    ordered += [c for c in final.columns if c not in ordered]
    final = final[ordered]

    out_xlsx = "Victor_TSTR_All_Datasets_MedGAN.xlsx"
    try:
        final.to_excel(out_xlsx, index=False)
        print(f"\n✅ Merged MedGAN results saved to: {out_xlsx}")
    except ModuleNotFoundError:
        out_csv = "Victor_TSTR_All_Datasets_MedGAN.csv"
        final.to_csv(out_csv, index=False)
        print("\nopenpyxl not installed – saved CSV instead.")
        print(f"✅ Merged MedGAN results saved to: {out_csv}")

if __name__ == "__main__":
    main()
