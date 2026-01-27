# merge_results.py

import os
import pandas as pd

DATASETS = ["car", "adult", "magic", "nursery", "shuttle"]

all_rows = []

for ds in DATASETS:
    path = os.path.join("Results", ds, "pategan_tstr.csv")
    if not os.path.exists(path):
        print(f"Skipping {ds} – file not found: {path}")
        continue

    df = pd.read_csv(path)
    df["Dataset"] = ds
    all_rows.append(df)

final = pd.concat(all_rows, ignore_index=True)

print("\nMerged TSTR results:")
print(final.head())

final.to_excel("Victor_TSTR_All_Datasets.xlsx", index=False)
print("\nSaved to Victor_TSTR_All_Datasets.xlsx")
