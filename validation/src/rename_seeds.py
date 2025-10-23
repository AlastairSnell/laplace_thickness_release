import os
from pathlib import Path

folder = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\folds")

for file in folder.glob("outer_seed_*.npy"):
    num = file.stem.split("_")[-1]  # grab the number part
    new_name = f"zipped_patch_{num}_os.npy"
    file.rename(file.with_name(new_name))
    print(f"Renamed: {file.name} -> {new_name}")
