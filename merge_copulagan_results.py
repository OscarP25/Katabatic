# merge_copulagan_results.py
# Merge CopulaGAN TSTR results across datasets into one Excel/CSV.

import os
import pandas as pd

DATASETS = ["car", "adult", "magic", "nursery", "shuttle"]
RESULTS_ROOT = "Results"


def main():
    all_rows = []

    print("=== Merging CopulaGAN TSTR results ===")

    for ds in DATASETS:
        # Your pipeline saves: Results\<dataset>\copulagan_tstr.csv
        candidate_files = [
            "copulagan_tstr.csv",
            "copulaGAN_tstr.csv",
            "copula_gan_tstr.csv",
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
            df["Generator"] = "CopulaGAN"

        all_rows.append(df)
        print(f"✔ Loaded: {found_path}")

    if not all_rows:
        print("\n❌ No CopulaGAN result files found — nothing to merge.")
        return

    final = pd.concat(all_rows, ignore_index=True)

    # Consistent column order
    preferred_order = ["Dataset", "Generator", "Model", "Metric", "Value"]
    ordered = [c for c in preferred_order if c in final.columns]
    ordered += [c for c in final.columns if c not in ordered]
    final = final[ordered]

    # Save Excel (preferred)
    out_xlsx = "Victor_TSTR_All_Datasets_CopulaGAN.xlsx"
    try:
        final.to_excel(out_xlsx, index=False)
        print(f"\n✅ Merged CopulaGAN results saved to: {out_xlsx}")
    except ModuleNotFoundError:
        # Fallback CSV
        out_csv = "Victor_TSTR_All_Datasets_CopulaGAN.csv"
        final.to_csv(out_csv, index=False)
        print("\nopenpyxl not installed – saved CSV instead.")
        print(f"✅ Merged CopulaGAN results saved to: {out_csv}")


if __name__ == "__main__":
    main()
