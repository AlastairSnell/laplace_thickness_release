#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

# --- EDIT THESE ---
SRC1 = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\folds")
SRC2 = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\folds2")
DEST = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\folds_selected")

# Which IDs to take from each source (strings, zero-padded)
SELECT = {
    str(SRC1): ["000", "003", "004", "005", "006", "009"],
    str(SRC2): ["000", "003", "004", "005", "006", "007", "008"],
}
# --- END EDITS ---

PAD = 3  # zero-padding width for new filenames (000, 001, ...)

def find_pair(src: Path, tag: str) -> dict[str, Path]:
    pkl = src / f"zipped_patch_{tag}.pkl"
    seed = src / f"outer_seed_{tag}.npy"
    found = {}
    if pkl.exists():
        found["pickle"] = pkl
    if seed.exists():
        found["seed"] = seed
    return found

def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)

    # Build ordered list of (src_path, tag)
    queue: list[tuple[Path, str]] = []
    for src_str, tags in SELECT.items():
        src = Path(src_str)
        for tag in tags:
            queue.append((src, tag))

    manifest = {
        "destination": str(DEST),
        "pad": PAD,
        "copies": [],  # list of {new_idx, src, tag, files:{pickle?, seed?}}
    }

    new_idx = 0
    errors = 0
    for src, tag in queue:
        pair = find_pair(src, tag)
        if not pair:
            print(f"[SKIP] {src} tag={tag} -> no files found (expected zipped_patch_{tag}.pkl and/or outer_seed_{tag}.npy)")
            errors += 1
            continue

        new_tag = f"{new_idx:0{PAD}d}"
        entry = {
            "new_idx": new_idx,
            "new_tag": new_tag,
            "src": str(src),
            "tag": tag,
            "files": {}
        }

        # Copy pickle if present
        if "pickle" in pair:
            dst_pkl = DEST / f"zipped_patch_{new_tag}.pkl"
            shutil.copy2(pair["pickle"], dst_pkl)
            entry["files"]["pickle"] = {"from": str(pair["pickle"]), "to": str(dst_pkl)}
        else:
            print(f"[WARN] Missing pickle in {src} for tag {tag}")

        # Copy seed if present
        if "seed" in pair:
            dst_seed = DEST / f"outer_seed_{new_tag}.npy"
            shutil.copy2(pair["seed"], dst_seed)
            entry["files"]["seed"] = {"from": str(pair["seed"]), "to": str(dst_seed)}
        else:
            print(f"[WARN] Missing seed in {src} for tag {tag}")

        # Only advance index if at least one file was copied
        if entry["files"]:
            manifest["copies"].append(entry)
            print(f"[OK]  {src.name} {tag} -> {new_tag}")
            new_idx += 1
        else:
            print(f"[SKIP] {src.name} {tag} -> nothing copied")
            errors += 1

    # Save manifest
    man_path = DEST / "_manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest to: {man_path}")

    if errors:
        print(f"\nCompleted with {errors} warning(s).")
    else:
        print("\nCompleted without warnings.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
