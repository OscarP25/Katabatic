# merge_great_results.py
# Merge GReaT TSTR results across datasets into one Excel/CSV.

import os
import pandas as pd

DATASETS = ["car", "adult", "magic", "nursery", "shuttle"]
RESULTS_ROOT = "Results"


def main():
    all_rows = []

    print("=== Merging GReaT TSTR results ===")

    for ds in DATASETS:
        # Common file names that might exist depending on your pipeline naming
        candidate_files = [
            "great_tstr.csv",
            "great_tstr_victor.csv",
            "tstr.csv",
            "tstr_results.csv",
        ]

        found_path = None
        for fname in candidate_files:
            p = os.path.join(RESULTS_ROOT, ds, fname)
            if os.path.exists(p):
                found_path = p
                break

        if found_path is None:
            print(f"⚠️ Missing results for {ds} in: {os.path.join(RESULTS_ROOT, ds)}")
            continue

        df = pd.read_csv(found_path)

        # Ensure Dataset column exists
        if "Dataset" not in df.columns:
            df["Dataset"] = ds

        # Ensure Generator column exists
        if "Generator" not in df.columns:
            df["Generator"] = "GReaT"

        all_rows.append(df)
        print(f"✔ Loaded: {found_path}")

    if not all_rows:
        print("\n❌ No GReaT result files found — nothing to merge.")
        return

    final = pd.concat(all_rows, ignore_index=True)

    # Consistent column order
    preferred_order = ["Dataset", "Generator", "Model", "Metric", "Value"]
    ordered = [c for c in preferred_order if c in final.columns]
    ordered += [c for c in final.columns if c not in ordered]
    final = final[ordered]

    # Save Excel (preferred)
    out_xlsx = "Victor_TSTR_All_Datasets_GReaT.xlsx"
    try:
        final.to_excel(out_xlsx, index=False)
        print(f"\n✅ Merged GReaT results saved to: {out_xlsx}")
    except ModuleNotFoundError:
        # Fallback CSV
        out_csv = "Victor_TSTR_All_Datasets_GReaT.csv"
        final.to_csv(out_csv, index=False)
        print("\nopenpyxl not installed – saved CSV instead.")
        print(f"✅ Merged GReaT results saved to: {out_csv}")


if __name__ == "__main__":
    main()
