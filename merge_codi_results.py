# merge_codi_results.py

import os
import pandas as pd

DATASETS = ["car", "adult", "magic", "nursery", "shuttle"]
RESULTS_ROOT = "Results"

def main():
    all_rows = []

    for ds in DATASETS:
        csv_path = os.path.join(RESULTS_ROOT, ds, "codi_tstr_victor.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ Missing results for {ds}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        # Ensure there's a Dataset column
        if "Dataset" not in df.columns:
            df["Dataset"] = ds

        all_rows.append(df)

    if not all_rows:
        print("No result files found, nothing to merge.")
        return

    final = pd.concat(all_rows, ignore_index=True)

    # Nice ordering: Dataset, Model, Metric, Value
    cols = final.columns.tolist()
    ordered = []
    for c in ["Dataset", "Model", "Metric", "Value"]:
        if c in cols:
            ordered.append(c)
    ordered += [c for c in cols if c not in ordered]
    final = final[ordered]

    # Save merged Excel
    out_xlsx = "Victor_TSTR_All_Datasets_CoDi.xlsx"
    try:
        final.to_excel(out_xlsx, index=False)
        print(f"✅ Merged results saved to: {out_xlsx}")
    except ModuleNotFoundError:
        print("openpyxl not installed – installing is recommended to create Excel files.")
        print("You can still save CSV instead:")
        out_csv = "Victor_TSTR_All_Datasets_CoDi.csv"
        final.to_csv(out_csv, index=False)
        print(f"✅ Merged results saved to: {out_csv}")

if __name__ == "__main__":
    main()
