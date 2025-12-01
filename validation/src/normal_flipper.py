#!/usr/bin/env python3
from __future__ import annotations
import pickle
from pathlib import Path
import shutil
import sys
import numpy as np

DEST = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\folds_selected")

CAP_NAMES = {"pial", "white"}  # flip these; leave 'side' alone

def unit_normal(tri: np.ndarray) -> np.ndarray:
    """Compute unit normal from a (3,3) array of vertices using right-hand rule."""
    v0, v1, v2 = tri[0], tri[1], tri[2]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n) + 1e-12
    return (n / ln).astype(np.float32)

def flip_winding(tri: np.ndarray) -> np.ndarray:
    """Flip triangle winding by swapping the last two vertices."""
    return np.array([tri[0], tri[2], tri[1]], dtype=np.float32)

def process_pickle(pkl_path: Path) -> tuple[int, int]:
    """Return (#caps_fixed, #sides_kept)."""
    with open(pkl_path, "rb") as f:
        pkg = pickle.load(f)

    tris = pkg.get("triangles", [])
    fixed_caps = 0
    kept_sides = 0

    # Modify in place
    for t in tris:
        name = t.get("name", "")
        verts = np.asarray(t["vertices"], dtype=np.float32)

        if name in CAP_NAMES:
            verts_flipped = flip_winding(verts)
            t["vertices"] = verts_flipped
            t["normal"] = unit_normal(verts_flipped)
            fixed_caps += 1
        else:
            # side triangles unchanged
            kept_sides += 1

    # One-time backup
    bak_path = pkl_path.with_suffix(pkl_path.suffix + ".bak")
    if not bak_path.exists():
        shutil.copy2(pkl_path, bak_path)

    # Atomic write
    tmp = pkl_path.with_suffix(pkl_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(pkg, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(pkl_path)

    return fixed_caps, kept_sides

def main() -> int:
    if not DEST.exists():
        print(f"Destination not found: {DEST}", file=sys.stderr)
        return 2

    files = sorted(DEST.glob("zipped_patch_*.pkl"))
    if not files:
        print("No zipped_patch_*.pkl files found.")
        return 0

    total_caps = total_sides = 0
    for p in files:
        caps, sides = process_pickle(p)
        total_caps += caps
        total_sides += sides
        tag = p.stem.split("_")[-1]
        print(f"[OK] {p.name}: flipped {caps} cap tris; kept {sides} side tris")

    print(f"\nDone. Total flipped cap triangles: {total_caps}. Side triangles left unchanged: {total_sides}.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
