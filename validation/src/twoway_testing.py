#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-validate path tracing across multiple surface sets (BEM SLP version).

Uses bem_laplace_slp.py:
  - assemble_and_solve(...) to get charges q
  - path_trace_simple_bem(...) for down/up tracing in the BEM field
"""

from __future__ import annotations
import os
from pathlib import Path
import pickle
import math
import csv
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# SciPy for sparse graph + KDTree
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

# Your local helpers (keep these)
from path_trace_copy import (
    preprocess_triangles,      # must populate centroid/normal & fix orientations
    pick_even_start_points,    # your even sampling on top surface
)

# BEM + tracer from your provided file
from path_trace_copy2 import (
    BEMConfig,
    assemble_and_solve,        # returns (q, A, b)
    path_trace_simple_bem,     # gradient-based path tracing in BEM field
)

# --------------------------- Config ---------------------------

ROOT = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data")
RESULTS_ROOT = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\hemispheres_results")
SUBFOLDERS = [
    #"phantoms_final",
    # r"analytical\disks",
    r"analytical\hemispheres",
    # "perturbed_families",
    # "folds_selected",
]

N_CPU = os.cpu_count() or 1
COMPUTE_UP_PATH = True

# Tracer knobs (wired straight into path_trace_simple_bem)
ALPHA_INITIAL = 0.05
FIRST_STEP = 0.05
MAX_ITER = 400
DEBUG_TRACE = False
ANGLE_MAX_DEG = 30.0

# Start-point selection
TARGET_N_PATHS = 100
INIT_PCT = 50
MAX_PCT = 50
PCT_STEP = 10
TARGET_SPACING = None

# Output
OUTDIR = RESULTS_ROOT
OUTDIR.mkdir(exist_ok=True, parents=True)
MASTER_CSV = OUTDIR / "_MASTER_paths.csv"

# --------------------------------------------------------------


def load_triangles(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return obj["triangles"] if isinstance(obj, dict) and "triangles" in obj else obj

def load_outer_seed(pkl_path: Path) -> np.ndarray:
    """
    For phantom files named like '0.5x10_05.pkl', expect a matching
    outer-seed file '0.5x10_05_os.npy' in the same folder.
    """
    os_path = pkl_path.with_name(f"{pkl_path.stem}_os.npy")
    if not os_path.exists():
        raise FileNotFoundError(f"Missing outer seed file: {os_path}")
    return np.load(os_path)


def _suffix_from_zipped_patch(p: Path) -> str:
    """
    'zipped_patch_003.pkl' -> '003'
    Raises if pattern not matched.
    """
    stem = p.stem  # 'zipped_patch_003'
    if not stem.startswith("zipped_patch_"):
        raise ValueError(f"Unexpected pkl name (want 'zipped_patch_###.pkl'): {p.name}")
    suf = stem.split("zipped_patch_")[-1]
    if not suf.isdigit():
        # allow e.g. 'zipped_patch_3' too
        if not all(ch.isdigit() for ch in suf):
            raise ValueError(f"Could not parse numeric suffix from: {p.name}")
    return suf


def top_surface_vertices_faces(triangles):
    """
    Return V(N,3), F(M,3) for the Dirichlet=1 ('top') surface.
    """
    vert_index = {}
    V = []
    F = []
    for tri in triangles:
        if str(tri.get("bc_type", "")).lower() == "dirichlet" and np.isclose(tri.get("bc_value", 0.0), 1.0):
            idxs = []
            for v in tri["vertices"]:
                key = tuple(np.asarray(v, dtype=np.float64))
                if key not in vert_index:
                    vert_index[key] = len(V)
                    V.append(np.asarray(key, dtype=np.float64))
                idxs.append(vert_index[key])
            F.append(idxs)
    if not V:
        raise ValueError("No Dirichlet=1 triangles (top surface) found.")
    return np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int32)


def build_geodesic_graph(V: np.ndarray, F: np.ndarray) -> csr_matrix:
    edges = set()
    for a, b, c in F:
        edges.add((min(a, b), max(a, b)))
        edges.add((min(b, c), max(b, c)))
        edges.add((min(c, a), max(c, a)))
    I, J, W = [], [], []
    for i, j in edges:
        w = np.linalg.norm(V[i] - V[j])
        I.append(i); J.append(j); W.append(w)
        I.append(j); J.append(i); W.append(w)
    n = len(V)
    return coo_matrix((W, (I, J)), shape=(n, n)).tocsr()


def geodesic_distance_on_top(V: np.ndarray, G: csr_matrix,
                             src_pt: np.ndarray, dst_pt: np.ndarray,
                             tree: cKDTree | None = None) -> float:
    tree = tree or cKDTree(V)
    i_src = int(tree.query(src_pt, k=1)[1])
    i_dst = int(tree.query(dst_pt, k=1)[1])
    dist = dijkstra(G, directed=False, indices=i_src, limit=np.inf)
    return float(dist[i_dst])


def hausdorff_bidirectional(A: np.ndarray, B: np.ndarray) -> float:
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    treeA = cKDTree(A)
    treeB = cKDTree(B)
    dA = treeB.query(A, k=1)[0]
    dB = treeA.query(B, k=1)[0]
    return float(max(np.max(dA), np.max(dB)))


def select_start_points(triangles, outer_seed: np.ndarray,
                        target_n: int = TARGET_N_PATHS,
                        init_pct: int = INIT_PCT,
                        max_pct: int = MAX_PCT,
                        pct_step: int = PCT_STEP,
                        target_spacing: float | None = TARGET_SPACING) -> np.ndarray:
    """
    Uses your pick_even_start_points helper; expands pct until enough points.
    """
    pct = init_pct
    while True:
        pool = pick_even_start_points(
            triangles,
            outer_seed_point=outer_seed,
            pct=pct,
            target_spacing=target_spacing,
            max_points=None
        )
        if len(pool) >= target_n or pct >= max_pct:
            break
        pct = min(max_pct, pct + pct_step)

    if len(pool) == 0:
        return pool

    if len(pool) <= target_n:
        return pool

    idxs = np.linspace(0, len(pool) - 1, num=target_n, dtype=int)
    return pool[idxs]


def trace_one(start_pt: np.ndarray,
              triangles: list[dict],
              q: np.ndarray,
              cfg: BEMConfig) -> tuple[list[np.ndarray], float, list[np.ndarray], float]:
    """
    Trace down and (optionally) up using the SAME tracer as bem_laplace_slp.py
    """
    path_down, len_down = path_trace_simple_bem(
        start_pt=start_pt,
        triangles=triangles,
        q=q,
        cfg=cfg,
        direction='down',
        max_iter=MAX_ITER,
        alpha_initial=ALPHA_INITIAL,
        first_step=FIRST_STEP,
        debug=DEBUG_TRACE,
        angle_max_deg=ANGLE_MAX_DEG,
    )

    if COMPUTE_UP_PATH:
        bottom_pt = path_down[-1]
        path_up, len_up = path_trace_simple_bem(
            start_pt=bottom_pt,
            triangles=triangles,
            q=q,
            cfg=cfg,
            direction='up',
            max_iter=MAX_ITER,
            alpha_initial=ALPHA_INITIAL,
            first_step=FIRST_STEP,
            debug=DEBUG_TRACE,
            angle_max_deg=ANGLE_MAX_DEG,
        )
    else:
        path_up, len_up = [], 0.0

    return path_down, len_down, path_up, len_up


def process_surface(pkl_path: Path):
    """
    Per-surface pipeline:
      - load triangles + outer seed
      - preprocess (centroids, normals, orientation, etc.)
      - solve BEM (SLP) → q
      - choose start points, trace down/up
      - compute metrics and write CSV
    """
    triangles = load_triangles(pkl_path)
    outer_seed = load_outer_seed(pkl_path)

    # Ensure fields: vertices, centroid, normal, bc_type, bc_value
    preprocess_triangles(triangles)

    # Solve BEM using your SLP implementation
    cfg = BEMConfig()  # tweak TAU_NEAR/TOL_NEAR here if needed
    q, A, b = assemble_and_solve(triangles, cfg)

    # Build top-surface graph for geodesic displacement
    V_top, F_top = top_surface_vertices_faces(triangles)
    G_top = build_geodesic_graph(V_top, F_top)
    tree_top = cKDTree(V_top)

    # Start points
    starts = select_start_points(triangles, outer_seed, target_n=TARGET_N_PATHS,
                                 init_pct=INIT_PCT, max_pct=MAX_PCT, pct_step=PCT_STEP,
                                 target_spacing=TARGET_SPACING)
    n_paths = len(starts)
    if n_paths == 0:
        print(f"[WARN] No start points for {pkl_path.name}")
        return [], {"surface": pkl_path.stem, "n_paths": 0}

    # Parallel tracing (q/cfg/triangles are shared read-only)
    rows = [None] * n_paths
    with ThreadPoolExecutor(max_workers=N_CPU) as pool:
        futs = {
            pool.submit(trace_one, starts[i], triangles, q, cfg): i
            for i in range(n_paths)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                path_down, len_down, path_up, len_up = fut.result()
            except Exception as e:
                rows[i] = {
                    "path_idx": i,
                    "len_down": float("nan"),
                    "len_up": float("nan"),
                    "len_asym_abs": float("nan"),
                    "len_ratio_up_over_down": float("nan"),
                    "geodesic_disp": float("nan"),
                    "hausdorff": float("nan"),
                }
                print(f"[ERROR] Path {i+1} failed on {pkl_path.name}: {e}")
                continue

            start_down = np.asarray(path_down[0], dtype=float)
            end_up = np.asarray(path_up[-1], dtype=float) if len(path_up) else np.asarray(path_down[-1], dtype=float)

            geo_disp = geodesic_distance_on_top(V_top, G_top, start_down, end_up, tree_top)
            len_asym_abs = abs(len_up - len_down)
            len_ratio = (len_up / len_down) if len_down > 0 else float("inf")

            arr_down = np.asarray(path_down, dtype=float)
            arr_up = np.asarray(path_up, dtype=float) if len(path_up) else np.empty((0, 3))
            hd = hausdorff_bidirectional(arr_down, arr_up)

            rows[i] = {
                "path_idx": i,
                "len_down": float(len_down),
                "len_up": float(len_up),
                "len_asym_abs": float(len_asym_abs),
                "len_ratio_up_over_down": float(len_ratio),
                "geodesic_disp": float(geo_disp),
                "hausdorff": float(hd),
            }

    # Write per-surface CSV
    surface_csv = OUTDIR / f"{pkl_path.stem}_paths.csv"
    with open(surface_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            if r is not None:
                w.writerow(r)

    # Summary stats
    def _stats(vals):
        a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
        if a.size == 0:
            return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
        return {
            "n": int(a.size),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    summary = {
        "surface": pkl_path.stem,
        "n_paths": int(np.sum([r is not None and np.isfinite(r["len_down"]) for r in rows])),
        "len_down": _stats([r["len_down"] for r in rows if r is not None]),
        "len_up": _stats([r["len_up"] for r in rows if r is not None]),
        "len_asym_abs": _stats([r["len_asym_abs"] for r in rows if r is not None]),
        "len_ratio": _stats([r["len_ratio_up_over_down"] for r in rows if r is not None]),
        "geodesic_disp": _stats([r["geodesic_disp"] for r in rows if r is not None]),
        "hausdorff": _stats([r["hausdorff"] for r in rows if r is not None]),
    }

    # Console summary
    def ps(name, s):
        print(f"  {name:<16} n={s['n']:>3}  mean={s['mean']:.4f}  med={s['median']:.4f}  "
              f"std={s['std']:.4f}  min={s['min']:.4f}  max={s['max']:.4f}")

    print(f"\n=== {pkl_path.stem} ({summary['n_paths']} paths) ===")
    ps("len_down", summary["len_down"])
    ps("len_up", summary["len_up"])
    ps("Δlen_abs", summary["len_asym_abs"])
    ps("len_ratio", summary["len_ratio"])
    ps("geo_disp", summary["geodesic_disp"])
    ps("hausdorff", summary["hausdorff"])

    return rows, summary


def ensure_master_header():
    if not MASTER_CSV.exists():
        with open(MASTER_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "folder", "surface", "path_idx",
                "len_down", "len_up",
                "len_asym_abs", "len_ratio_up_over_down",
                "geodesic_disp", "hausdorff"
            ])


def append_master(folder: str, surface: str, rows: list[dict]):
    with open(MASTER_CSV, "a", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            if r is None:
                continue
            w.writerow([
                folder, surface, r["path_idx"],
                r["len_down"], r["len_up"],
                r["len_asym_abs"], r["len_ratio_up_over_down"],
                r["geodesic_disp"], r["hausdorff"]
            ])


def main():
    ensure_master_header()

    for sub in SUBFOLDERS:
        folder = (ROOT / sub)
        if not folder.exists():
            print(f"[WARN] Missing folder: {folder}")
            continue

        #pkls = sorted(folder.glob("zipped_patch_*.pkl"))
        pkls = sorted(folder.glob("15mm_*.pkl"))

        if not pkls:
            print(f"[INFO] No .pkl files in {folder}")
            continue

        print(f"\n>>> Folder: {folder} ({len(pkls)} surfaces)")
        for pkl_path in pkls:
            try:
                rows, summary = process_surface(pkl_path)
                append_master(str(sub), pkl_path.stem, rows)
            except Exception as e:
                print(f"[ERROR] Failed {pkl_path.name}: {e}")


if __name__ == "__main__":
    main()
