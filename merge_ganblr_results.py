# merge_ganblr_results.py

import os
import pandas as pd

DATASETS = ["car", "adult", "magic", "nursery", "shuttle"]
RESULTS_ROOT = "Results"

def main():
    all_rows = []

    print("=== Merging GANBLR TSTR results ===")

    for ds in DATASETS:
        csv_path = os.path.join(RESULTS_ROOT, ds, "ganblr_tstr.csv")

        if not os.path.exists(csv_path):
            print(f"⚠️ Missing results for {ds}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Ensure Dataset column exists
        if "Dataset" not in df.columns:
            df["Dataset"] = ds

        # Ensure Generator column exists
        if "Generator" not in df.columns:
            df["Generator"] = "GANBLR"

        all_rows.append(df)
        print(f"✔ Loaded {csv_path}")

    if not all_rows:
        print("\n❌ No GANBLR result files found — nothing to merge.")
        return

    final = pd.concat(all_rows, ignore_index=True)

    preferred_order = ["Dataset", "Generator", "Model", "Metric", "Value"]
    ordered = [c for c in preferred_order if c in final.columns]
    ordered += [c for c in final.columns if c not in ordered]
    final = final[ordered]

    out_xlsx = "Victor_TSTR_All_Datasets_GANBLR.xlsx"

    try:
        final.to_excel(out_xlsx, index=False)
        print(f"\n✅ Merged GANBLR results saved to: {out_xlsx}")
    except ModuleNotFoundError:
        out_csv = "Victor_TSTR_All_Datasets_GANBLR.csv"
        final.to_csv(out_csv, index=False)
        print("\nopenpyxl not installed – saved CSV instead.")
        print(f"✅ Merged GANBLR results saved to: {out_csv}")

if __name__ == "__main__":
    main()
