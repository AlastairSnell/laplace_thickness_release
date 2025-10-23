#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-validate path tracing across multiple surface sets.

For every *.pkl in the specified subfolders, loads triangles and matching *_os.npy
outer seed, solves Laplace, samples up to 100 start points on the top surface,
traces down and up, and records:
  - geodesic displacement between start(down) and end(up) on the TOP surface
  - path-length asymmetry: |len_up - len_down| and (len_up / len_down)
  - bidirectional Hausdorff distance between up and down polylines (3D Euclidean)

Outputs:
  - results/<surface-stem>_paths.csv   (one row per traced path)
  - results/_MASTER_paths.csv          (accumulates all surfaces)
"""

from __future__ import annotations
import os
from pathlib import Path
import pickle
import math
import csv
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# SciPy for sparse graph + KDTree (install if missing)
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

# Your helpers
from path_trace_copy import (
    preprocess_triangles,
    assemble_system,
    solve_system,
    store_solution,
    pick_even_start_points,
    path_trace_simple,
)

# --------------------------- Config ---------------------------

ROOT = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data")
RESULTS_ROOT = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\folds_final2_results")
SUBFOLDERS = [
    #"phantoms_final2",
    #r"analytical\disks",
    #r"analytical\hemispheres",
    #"perturbed_families",
    "folds",
]

N_CPU = os.cpu_count() or 1
COMPUTE_UP_PATH = True
ALPHA_INITIAL = 0.05
FIRST_STEP = 0.05
MAX_ITER = 100
DEBUG_TRACE = False

# Start-point selection
TARGET_N_PATHS = 100
INIT_PCT = 50
MAX_PCT = 100
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
    # Accept either obj['triangles'] or obj itself if already the list
    if isinstance(obj, dict) and "triangles" in obj:
        return obj["triangles"]
    return obj


def load_outer_seed(pkl_path: Path) -> np.ndarray:
    # Match stem + '_os.npy'
    os_path = pkl_path.with_suffix("").with_name(pkl_path.stem + "_os.npy")
    if not os_path.exists():
        raise FileNotFoundError(f"Missing outer seed file: {os_path}")
    return np.load(os_path)


def top_surface_vertices_faces(triangles):
    """
    Extract unique vertex list V(N,3) and faces F(M,3) for the Dirichlet=1 ('top') surface.
    Vertices are deduplicated by exact coordinate tuples (assumes your pipeline uses exact copies).
    """
    vert_index = {}
    V = []
    F = []
    for tri in triangles:
        if tri.get("bc_type") == "dirichlet" and np.isclose(tri.get("bc_value", 0.0), 1.0):
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
    """
    Build an undirected edge-weighted graph (CSR) over the top surface:
    edge weights = Euclidean edge lengths between vertices.
    """
    # Collect all triangle edges (i,j) with i<j to avoid duplicates
    edges = set()
    for a, b, c in F:
        edges.add((min(a, b), max(a, b)))
        edges.add((min(b, c), max(b, c)))
        edges.add((min(c, a), max(c, a)))

    I, J, W = [], [], []
    for i, j in edges:
        w = np.linalg.norm(V[i] - V[j])
        I.append(i); J.append(j); W.append(w)
        I.append(j); J.append(i); W.append(w)  # undirected

    n = len(V)
    G = coo_matrix((W, (I, J)), shape=(n, n)).tocsr()
    return G


def nearest_vertex_idx(V: np.ndarray, pt: np.ndarray, kdtree: cKDTree | None = None) -> int:
    tree = kdtree or cKDTree(V)
    d, i = tree.query(pt, k=1)
    return int(i)


def geodesic_distance_on_top(V: np.ndarray, G: csr_matrix,
                             src_pt: np.ndarray, dst_pt: np.ndarray,
                             tree: cKDTree | None = None) -> float:
    """
    Approximate top-surface geodesic distance by mapping the 3D points
    to their nearest top-surface vertices, then running Dijkstra.
    """
    tree = tree or cKDTree(V)
    i_src = nearest_vertex_idx(V, src_pt, tree)
    i_dst = nearest_vertex_idx(V, dst_pt, tree)
    dist = dijkstra(G, directed=False, indices=i_src, limit=np.inf)
    return float(dist[i_dst])


def hausdorff_bidirectional(A: np.ndarray, B: np.ndarray) -> float:
    """
    Discrete bidirectional Hausdorff distance between two polyline point sets in 3D (Euclidean).
    """
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    treeA = cKDTree(A)
    treeB = cKDTree(B)
    # directed A->B
    dA, _ = treeB.query(A, k=1)
    # directed B->A
    dB, _ = treeA.query(B, k=1)
    return float(max(np.max(dA), np.max(dB)))


def select_start_points(triangles, outer_seed: np.ndarray,
                        target_n: int = TARGET_N_PATHS,
                        init_pct: int = INIT_PCT,
                        max_pct: int = MAX_PCT,
                        pct_step: int = PCT_STEP,
                        target_spacing: float | None = TARGET_SPACING) -> np.ndarray:
    """
    Use your pick_even_start_points helper to get a geodesically local, evenly spread pool.
    If the pool is larger than target_n, deterministically downsample to target_n by index slicing.
    If it's smaller, progressively widen pct up to max_pct.
    """
    pct = init_pct
    pool = None
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
        return pool  # empty

    if len(pool) <= target_n:
        return pool

    # Deterministic downsample to target_n: take evenly spaced indices
    idxs = np.linspace(0, len(pool) - 1, num=target_n, dtype=int)
    return pool[idxs]


def trace_one(start_pt, triangles):
    """
    Trace one down and up path from a start point.
    Returns (path_down, len_down, path_up, len_up).
    """
    path_down, len_down = path_trace_simple(
        start_pt, triangles, 'down',
        MAX_ITER, ALPHA_INITIAL, FIRST_STEP, debug=DEBUG_TRACE
    )

    if COMPUTE_UP_PATH:
        bottom_pt = path_down[-1]
        path_up, len_up = path_trace_simple(
            bottom_pt, triangles, 'up',
            MAX_ITER, ALPHA_INITIAL, FIRST_STEP, debug=DEBUG_TRACE
        )
    else:
        path_up, len_up = [], 0.0

    return path_down, len_down, path_up, len_up


def process_surface(pkl_path: Path):
    """
    Process a single surface file:
      - load triangles + outer seed
      - preprocess + solve Laplace + store φ
      - choose up to 100 start points
      - trace down/up for each start
      - compute metrics and write per-surface CSV
    Returns (rows, summary) where rows are per-path dicts, summary is dict of aggregates.
    """
    triangles = load_triangles(pkl_path)
    outer_seed = load_outer_seed(pkl_path)

    preprocess_triangles(triangles)  # mutate once

    A, b = assemble_system(triangles, parallel=True, max_workers=N_CPU)
    phi = solve_system(A, b)
    store_solution(triangles, phi)

    # Top surface graph for geodesic displacement metric
    V_top, F_top = top_surface_vertices_faces(triangles)
    G_top = build_geodesic_graph(V_top, F_top)
    tree_top = cKDTree(V_top)

    starts = pick_even_start_points(triangles, outer_seed, pct = 40, target_spacing = 1e-3)
    n_paths = len(starts)
    if n_paths == 0:
        print(f"[WARN] No start points for {pkl_path.name}")
        return [], {"surface": pkl_path.stem, "n_paths": 0}

    # Parallel tracing
    rows = [None] * n_paths
    with ThreadPoolExecutor(max_workers=N_CPU) as pool:
        futs = {
            pool.submit(trace_one, starts[i], triangles): i
            for i in range(n_paths)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                path_down, len_down, path_up, len_up = fut.result()
            except Exception as e:
                # Record a failed path with NaNs so downstream summaries still work
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

            # Metrics
            # 1) geodesic displacement (top surface) between start(down) and end(up)
            geo_disp = geodesic_distance_on_top(V_top, G_top, start_down, end_up, tree_top)

            # 2) path-length asymmetry: absolute Δ and ratio (protect divide-by-zero)
            len_asym_abs = abs(len_up - len_down)
            len_ratio = (len_up / len_down) if len_down > 0 else float("inf")

            # 3) Hausdorff between the polylines (3D)
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

    # Compute summary stats
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

        pkls = sorted(folder.glob("*.pkl"))
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
