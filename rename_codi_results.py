from pathlib import Path

# Base results folder
base = Path("Results")

# All datasets you’ve run CoDi on
datasets = ["car", "adult", "magic", "nursery", "shuttle"]

for ds in datasets:
    orig = base / ds / "codi_tstr.csv"
    dest = base / ds / "codi_tstr_victor.csv"

    if orig.exists():
        if not dest.exists():
            orig.rename(dest)
            print(f"Renamed {orig} -> {dest}")
        else:
            print(f"Destination already exists, skipping: {dest}")
    else:
        print(f"Source missing for {ds}: {orig}")
