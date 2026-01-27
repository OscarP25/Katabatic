import os
import pandas as pd

DATASETS = ["car", "adult", "magic", "nursery", "shuttle"]
RESULTS_ROOT = "Results"

def main():
    all_rows = []

    for ds in DATASETS:
        csv_path = os.path.join(RESULTS_ROOT, ds, "forestdiffusion_tstr.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ Missing results for {ds}: {csv_path}")
            continue

        df = pd.read_csv(csv_path, encoding="utf-8")

        # Ensure Dataset column exists
        if "Dataset" not in df.columns:
            df["Dataset"] = ds

        # Ensure Generator column exists
        if "Generator" not in df.columns:
            df["Generator"] = "ForestDiffusion"

        all_rows.append(df)

    if not all_rows:
        print("❌ No ForestDiffusion result files found, nothing to merge.")
        return

    final = pd.concat(all_rows, ignore_index=True)

    # Match CTGAN/CoDi column order
    preferred_order = ["Dataset", "Generator", "Model", "Metric", "Value"]
    ordered = [c for c in preferred_order if c in final.columns]
    ordered += [c for c in final.columns if c not in ordered]
    final = final[ordered]

    # Save Excel (preferred)
    out_xlsx = "Victor_TSTR_All_Datasets_ForestDiffusion.xlsx"
    try:
        final.to_excel(out_xlsx, index=False)
        print(f"✅ Merged ForestDiffusion results saved to: {out_xlsx}")
    except ModuleNotFoundError:
        out_csv = "Victor_TSTR_All_Datasets_ForestDiffusion.csv"
        final.to_csv(out_csv, index=False)
        print("openpyxl not installed – saved CSV instead.")
        print(f"✅ Merged ForestDiffusion results saved to: {out_csv}")

if __name__ == "__main__":
    main()
