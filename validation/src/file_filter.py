#!/usr/bin/env python3
"""
Copy a subset of phantom families from multiple folders into a single new folder,
renumbered consecutively across all sources.

VS Code usage:
1. Edit the CONFIG section just below.
2. Press the Run / Play button in VS Code.
"""

from pathlib import Path
import shutil

# ---------- CONFIG ----------

# List of sources:
# Each entry has:
#   - "dir":         the folder containing phantom files
#   - "family_ids":  the original family indices you want from that folder
#
# Example:
# SOURCE_CONFIG = [
#     {
#         "dir": Path(r"...\phantoms2"),
#         "family_ids": [2, 4, 6, 10, 13],
#     },
#     {
#         "dir": Path(r"...\phantoms3"),
#         "family_ids": [3, 5],
#     },
# ]
SOURCE_CONFIG = [
    {
        "dir": Path(
            r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\phantoms_selected2"
        ),
        "family_ids": [1,2,3,4,5,6,7,8,9],
    },
    {
        "dir": Path(
            r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\phantoms3"
        ),
        "family_ids": [3],
    },
]

# Output folder (new renumbered set, across ALL sources).
# If None, will be "<first_source_dir>_merged_renumbered"
OUT_DIR = Path(
    r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\phantoms_selected3"
)

# ----------------------------


def extract_family_index(path: Path) -> int | None:
    """
    Extract the family index from a filename.

    Expected patterns:
      - 3x10_5.pkl        -> index 5
      - 3x10_05.pkl       -> index 5
      - 3x10_5_os.npy     -> index 5
      - anything_else_12.ext -> index 12

    We look at the stem, split on '_', and take the last token that is all digits.
    If no such token exists, return None.
    """
    stem = path.stem  # e.g. "3x10_5", "3x10_05", "3x10_5_os"
    parts = stem.split("_")
    for token in reversed(parts):
        if token.isdigit():
            return int(token)
    return None


def make_new_name(path: Path, new_idx: int) -> str:
    """
    Build a new filename with the family index replaced by new_idx,
    preserving the number of digits (zero padding) of the original token.

    Example:
      stem = "3x10_5"        -> "3x10_1"
      stem = "3x10_005_os"   -> "3x10_001_os"
    """
    stem = path.stem
    parts = stem.split("_")

    # Find last numeric token from the right
    for i in range(len(parts) - 1, -1, -1):
        token = parts[i]
        if token.isdigit():
            width = len(token)  # preserve padding width (e.g. "005" -> width 3)
            parts[i] = f"{new_idx:0{width}d}"
            new_stem = "_".join(parts)
            return new_stem + path.suffix

    # If no numeric token, just return original name
    return path.name


def main():
    if not SOURCE_CONFIG:
        print("ERROR: SOURCE_CONFIG is empty. Add at least one source in the CONFIG section.")
        return

    # Validate sources and gather family keys
    family_keys_in_order: list[tuple[Path, int]] = []
    print("Source config:")
    for src in SOURCE_CONFIG:
        src_dir = src.get("dir")
        fam_ids = src.get("family_ids", [])

        if src_dir is None:
            print("  ERROR: missing 'dir' in one SOURCE_CONFIG entry.")
            return

        src_dir = Path(src_dir)
        if not src_dir.is_dir():
            print(f"  ERROR: Source directory does not exist or is not a directory:\n    {src_dir}")
            return

        if not fam_ids:
            print(f"  WARNING: No family_ids specified for {src_dir}; this source will be ignored.")
            continue

        # Normalise family_ids to ints
        fam_ids_int = [int(x) for x in fam_ids]
        print(f"  {src_dir} -> family_ids: {fam_ids_int}")

        # Keep order as given, but avoid duplicates of the same (dir, id) pair
        for fid in fam_ids_int:
            key = (src_dir.resolve(), fid)
            family_keys_in_order.append(key)

    if not family_keys_in_order:
        print("ERROR: No valid (dir, family_ids) pairs found in SOURCE_CONFIG.")
        return

    # Decide output directory
    if OUT_DIR is None:
        first_dir = family_keys_in_order[0][0]
        dst_dir = first_dir.with_name(first_dir.name + "_merged_renumbered")
    else:
        dst_dir = Path(OUT_DIR)

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicate while preserving order
    ordered_unique_keys: list[tuple[Path, int]] = []
    seen = set()
    for key in family_keys_in_order:
        if key not in seen:
            seen.add(key)
            ordered_unique_keys.append(key)

    # Map each (dir, old_idx) to a consecutive new index 1..N
    key_to_new_idx: dict[tuple[Path, int], int] = {
        key: new for new, key in enumerate(ordered_unique_keys, start=1)
    }

    print()
    print(f"Output directory: {dst_dir}")
    print("Renumbering mapping ((source_dir, old_idx) -> new_idx):")
    for key, new_idx in key_to_new_idx.items():
        src_dir, old_idx = key
        print(f"  ({src_dir}, {old_idx}) -> {new_idx}")
    print()

    copied = 0
    skipped = 0
    unmatched = 0

    # Walk each source directory
    for src in SOURCE_CONFIG:
        src_dir = Path(src.get("dir"))
        if not src_dir.is_dir():
            # Already warned above; skip
            continue

        resolved_src = src_dir.resolve()
        print(f"Scanning: {resolved_src}")

        for path in src_dir.iterdir():
            if not path.is_file():
                continue

            fam_idx = extract_family_index(path)

            if fam_idx is None:
                unmatched += 1
                continue

            key = (resolved_src, fam_idx)
            if key not in key_to_new_idx:
                skipped += 1
                continue

            new_idx = key_to_new_idx[key]
            new_name = make_new_name(path, new_idx)
            target = dst_dir / new_name

            shutil.copy2(path, target)
            copied += 1

    print("\nDone.")
    print(f"  Copied    : {copied} file(s)")
    print(f"  Skipped   : {skipped} file(s) (family index not requested for that source)")
    print(f"  Unmatched : {unmatched} file(s) (no numeric family index found in name)")


if __name__ == "__main__":
    main()
