#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-validate path tracing across multiple surface sets (BEM SLP version),
with gradual mesh deformations and PyVista mesh quality metrics.

Deformation strategy:
  - Jitter vertices along surface normals with increasing amplitude
    (scaled by mean edge length) to mimic segmentation noise.

Mesh quality:
  - Uses pyvista.DataObjectFilters.cell_quality() to compute:
      * aspect_ratio
      * min_angle
      * radius_ratio
    and summarizes each across the mesh.

Outputs:
  - Per-(surface, deformation) CSV of path metrics + mesh-quality stats.
  - Global _MASTER_paths.csv combining all folds and deformations.
"""

from __future__ import annotations
import os
from pathlib import Path
import pickle
import math
import csv
from copy import deepcopy

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# SciPy for sparse graph + KDTree
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import pyvista as pv

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
RESULTS_ROOT = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\perturbations_results")
SUBFOLDERS = [
    "folds_selected",
]
PERTURBED_ROOT = ROOT / "perturbed"

N_CPU = os.cpu_count() or 1
COMPUTE_UP_PATH = True

# Tracer settings
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

# Deformation & quality config
DEFORMATION_SIGMA_FACTORS = [0.80]
RNG_SEED = 12345

# Output
OUTDIR = RESULTS_ROOT
OUTDIR.mkdir(exist_ok=True, parents=True)
MASTER_CSV = OUTDIR / "_MASTER_paths.csv"

PATH_FIELDS = [
    "folder", "surface", "deformation", "sigma_factor",
    "path_idx",
    "len_down", "len_up",
    "len_asym_abs", "len_ratio_up_over_down",
    "geodesic_disp", "hausdorff",
]

# New per-mesh metrics (same for all paths for a given surface+deformation)
METRIC_FIELDS = [
    "shape_mean", "shape_cv",
    "gauss_mean", "gauss_cv",
]

MASTER_FIELDS = PATH_FIELDS + METRIC_FIELDS
# --------------------------------------------------------------


def load_triangles(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return obj["triangles"] if isinstance(obj, dict) and "triangles" in obj else obj


def load_outer_seed(pkl_path: Path) -> np.ndarray:
    """
    For files named like 'zipped_patch_011.pkl', expect a matching
    outer-seed file 'outer_seed_011.npy' in the same folder.
    """
    suf = _suffix_from_zipped_patch(pkl_path)  
    os_path = pkl_path.with_name(f"outer_seed_{suf}.npy")
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


def _stats(vals):
    """
    Generic stats helper: n, mean, median, std, min, max.
    """
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan,
                "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


# ---------------------- PyVista helpers -----------------------


def triangles_to_pyvista(triangles):
    """
    Build a PyVista PolyData from your triangles list.

    Returns:
        mesh: pv.PolyData
        tri_vertex_indices: (n_tri, 3) int array of vertex indices per triangle
    """
    vert_index: dict[tuple[float, float, float], int] = {}
    points = []
    faces = []
    tri_vertex_indices = []

    for tri in triangles:
        idxs = []
        for v in tri["vertices"]:
            key = tuple(np.asarray(v, dtype=float))
            if key not in vert_index:
                vert_index[key] = len(points)
                points.append(np.asarray(key, dtype=float))
            idxs.append(vert_index[key])
        faces.extend([3, *idxs])
        tri_vertex_indices.append(idxs)

    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    tri_vertex_indices = np.asarray(tri_vertex_indices, dtype=np.int32)

    mesh = pv.PolyData(points, faces)
    return mesh, tri_vertex_indices


def apply_pyvista_points_to_triangles(mesh: pv.PolyData,
                                      triangles: list[dict],
                                      tri_vertex_indices: np.ndarray):
    """
    Overwrite tri['vertices'] with the (possibly deformed) PyVista mesh points
    using the original vertex indices.
    """
    pts = np.asarray(mesh.points, dtype=float)
    for tri, idxs in zip(triangles, tri_vertex_indices):
        tri["vertices"] = [pts[int(i)].copy() for i in idxs]


def compute_mean_edge_length(mesh: pv.PolyData) -> float:
    """
    Compute a characteristic mean edge length for scaling jitter.
    """
    edges = mesh.extract_all_edges()
    lines = edges.lines.reshape(-1, 3)  # [n_edges, 3] with [2, i0, i1]
    if lines.size == 0:
        return 1.0
    i0 = lines[:, 1]
    i1 = lines[:, 2]
    vecs = edges.points[i1] - edges.points[i0]
    elen = np.linalg.norm(vecs, axis=1)
    L = float(np.mean(elen)) if elen.size > 0 else 1.0
    return L


def jitter_along_normals(mesh: pv.PolyData,
                         sigma_factor: float,
                         rng: np.random.Generator) -> pv.PolyData:
    """
    Jitter vertices along point normals with amplitude sigma_factor * mean_edge_length.
    """
    if sigma_factor == 0.0:
        return mesh.copy()

    m = mesh.copy()
    m.compute_normals(point_normals=True, cell_normals=False, inplace=True)
    normals = np.asarray(m.point_data["Normals"], dtype=float)

    L = compute_mean_edge_length(m)
    sigma = sigma_factor * L

    noise = rng.normal(0.0, sigma, size=normals.shape)
    disp = noise * normals
    m.points = m.points + disp
    return m


def compute_shape_and_gaussian_stats(mesh: pv.PolyData) -> dict[str, float]:
    """
    Compute:
      - Verdict 'shape' quality per triangle (via cell_quality('shape')),
      - Vertex-wise Gaussian curvature (via curvature('gaussian')),

    and summarise each with mean and coefficient of variation.

    Returns a flat dict:
      {
        "shape_mean": ...,
        "shape_cv": ...,
        "gauss_mean": ...,
        "gauss_cv": ...,
      }
    """

    # --- Shape quality ---
    qmesh = mesh.cell_quality(quality_measure="shape")

    # Robustly pick the array name that cell_quality produced
    if len(qmesh.cell_data) == 0:
        shape_mean = float("nan")
        shape_cv = float("nan")
    else:
        # Prefer 'CellQuality' if present, else take the first available array
        if "CellQuality" in qmesh.cell_data:
            arr_name = "CellQuality"
        else:
            arr_name = list(qmesh.cell_data.keys())[0]

        vals = np.asarray(qmesh.cell_data[arr_name], dtype=float)
        mask = np.isfinite(vals) & (vals >= 0.0)
        vals = vals[mask]

        if vals.size == 0:
            shape_mean = float("nan")
            shape_cv = float("nan")
        else:
            shape_mean = float(np.mean(vals))
            shape_std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            shape_cv = float(shape_std / shape_mean) if shape_mean != 0.0 else float("nan")

    # --- Gaussian curvature (vertex-wise) ---
    K = mesh.curvature(curv_type="gaussian")
    K = np.asarray(K, dtype=float)
    mask = np.isfinite(K)
    K = K[mask]

    if K.size == 0:
        gauss_mean = float("nan")
        gauss_cv = float("nan")
    else:
        gauss_mean = float(np.mean(K))
        gauss_std = float(np.std(K, ddof=1)) if K.size > 1 else 0.0
        gauss_cv = float(gauss_std / gauss_mean) if gauss_mean != 0.0 else float("nan")

    return {
        "shape_mean": shape_mean,
        "shape_cv": shape_cv,
        "gauss_mean": gauss_mean,
        "gauss_cv": gauss_cv,
    }


def compute_gaussian_curvature_stats(mesh: pv.PolyData) -> dict:
    """
    Compute vertex-wise Gaussian curvature (1-ring, from VTK/PyVista)
    and summarise with mean and coefficient of variation.

    Uses signed curvature (no absolute values).
    """
    K = mesh.curvature(curv_type="gaussian")
    K = np.asarray(K, dtype=float)
    mask = np.isfinite(K)
    K = K[mask]
    if K.size == 0:
        return {"gauss_mean": float("nan"), "gauss_cv": float("nan")}

    mean = float(np.mean(K))
    std = float(np.std(K, ddof=1)) if K.size > 1 else 0.0
    cv = float(std / mean) if mean != 0.0 else float("nan")

    return {"gauss_mean": mean, "gauss_cv": cv}

# ---------------------- Tracing helpers -----------------------


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


# --------------------- Per-surface pipeline -------------------


def process_surface(pkl_path: Path):
    """
    Per-surface pipeline with deformations:
      - load triangles + outer seed
      - preprocess (centroids, normals, orientation, etc.)
      - build PyVista mesh + vertex index mapping
      - for each deformation level:
          * jitter mesh along normals
          * compute mesh-quality stats
          * copy deformed coords back into triangles
          * solve BEM (SLP) → q
          * choose start points, trace down/up
          * compute metrics and write CSV
    Returns:
      all_rows: list[dict] across all deformations
      summaries: list[dict] per deformation (for console/logging)
    """
    print(f"\n>>> Surface file: {pkl_path}")
    triangles_orig = load_triangles(pkl_path)
    outer_seed = load_outer_seed(pkl_path)

    # Ensure fields: vertices, centroid, normal, bc_type, bc_value
    preprocess_triangles(triangles_orig)

    # Build PyVista mesh & vertex mapping (un-deformed)
    base_mesh, tri_vertex_indices = triangles_to_pyvista(triangles_orig)

    rng = np.random.default_rng(RNG_SEED)

    all_rows: list[dict] = []
    summaries: list[dict] = []

    for sigma_factor in DEFORMATION_SIGMA_FACTORS:
        deformation_label = f"jitter_{sigma_factor:.3f}".replace('.', 'p')
        print(f"\n  --- Deformation: {deformation_label} ---")

        # 1) Deform mesh via jitter
        deform_mesh = jitter_along_normals(base_mesh, sigma_factor, rng=rng)

        # 2) Shape + Gaussian curvature stats for this deformed mesh
        metric_rowdict = compute_shape_and_gaussian_stats(deform_mesh)

        # 3) Copy deformed coords back into a fresh triangles list
        triangles = deepcopy(triangles_orig)
        apply_pyvista_points_to_triangles(deform_mesh, triangles, tri_vertex_indices)
        preprocess_triangles(triangles)  # recompute centroids/normals after deformation

        # --- NEW: save deformed surface for reproducibility ---
        sub_rel = pkl_path.parent.relative_to(ROOT)  # e.g. "folds_selected"
        perturbed_dir = PERTURBED_ROOT / sub_rel
        perturbed_dir.mkdir(exist_ok=True, parents=True)

        perturbed_pkl = perturbed_dir / f"{pkl_path.stem}_{deformation_label}.pkl"
        with open(perturbed_pkl, "wb") as f_out:
            pickle.dump({"triangles": triangles}, f_out)

        # 4) Solve BEM using your SLP implementation
        cfg = BEMConfig()
        q, A, b = assemble_and_solve(triangles, cfg)

        # 5) Build top-surface graph for geodesic displacement
        V_top, F_top = top_surface_vertices_faces(triangles)
        G_top = build_geodesic_graph(V_top, F_top)
        tree_top = cKDTree(V_top)

        # 6) Start points
        starts = select_start_points(
            triangles, outer_seed,
            target_n=TARGET_N_PATHS,
            init_pct=INIT_PCT, max_pct=MAX_PCT, pct_step=PCT_STEP,
            target_spacing=TARGET_SPACING
        )
        n_paths = len(starts)
        if n_paths == 0:
            print(f"[WARN] No start points for {pkl_path.name} at {deformation_label}")
            continue

        # 7) Parallel tracing (q/cfg/triangles are shared read-only)
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
                    print(f"[ERROR] Path {i+1} failed on {pkl_path.name}, {deformation_label}: {e}")
                    rows[i] = {
                        "deformation": deformation_label,
                        "sigma_factor": float(sigma_factor),
                        "path_idx": i,
                        "len_down": float("nan"),
                        "len_up": float("nan"),
                        "len_asym_abs": float("nan"),
                        "len_ratio_up_over_down": float("nan"),
                        "geodesic_disp": float("nan"),
                        "hausdorff": float("nan"),
                        **metric_rowdict,
                    }


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
                    "deformation": deformation_label,
                    "sigma_factor": float(sigma_factor),
                    "path_idx": i,
                    "len_down": float(len_down),
                    "len_up": float(len_up),
                    "len_asym_abs": float(len_asym_abs),
                    "len_ratio_up_over_down": float(len_ratio),
                    "geodesic_disp": float(geo_disp),
                    "hausdorff": float(hd),
                    **metric_rowdict,
                }



        # Filter out any None rows
        rows = [r for r in rows if r is not None]
        if not rows:
            print(f"[WARN] All paths failed for {pkl_path.name} at {deformation_label}")
            continue

        # 8) Write per-(surface, deformation) CSV
        surface_csv = OUTDIR / f"{pkl_path.stem}_{deformation_label}_paths.csv"
        with open(surface_csv, "w", newline="") as f:
            fieldnames = list(rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        # 9) Summary stats for this deformation (paths + quality)
        summary = {
            "surface": pkl_path.stem,
            "deformation": deformation_label,
            "sigma_factor": float(sigma_factor),
            "n_paths": int(np.sum([np.isfinite(r["len_down"]) for r in rows])),
            "len_down": _stats([r["len_down"] for r in rows]),
            "len_up": _stats([r["len_up"] for r in rows]),
            "len_asym_abs": _stats([r["len_asym_abs"] for r in rows]),
            "len_ratio": _stats([r["len_ratio_up_over_down"] for r in rows]),
            "geodesic_disp": _stats([r["geodesic_disp"] for r in rows]),
            "hausdorff": _stats([r["hausdorff"] for r in rows]),
            "metrics": metric_rowdict,
        }

        summaries.append(summary)

        # Console summary
        def ps(name, s):
            print(f"    {name:<16} n={s['n']:>3}  mean={s['mean']:.4f}  med={s['median']:.4f}  "
                  f"std={s['std']:.4f}  min={s['min']:.4f}  max={s['max']:.4f}")

        print(f"  === {pkl_path.stem} | {deformation_label} ({summary['n_paths']} paths) ===")
        ps("len_down", summary["len_down"])
        ps("len_up", summary["len_up"])
        ps("Δlen_abs", summary["len_asym_abs"])
        ps("len_ratio", summary["len_ratio"])
        ps("geo_disp", summary["geodesic_disp"])
        ps("hausdorff", summary["hausdorff"])

        # Mesh metrics summary to console
        print(f"    shape_mean   = {metric_rowdict['shape_mean']:.4f}")
        print(f"    shape_cv     = {metric_rowdict['shape_cv']:.4f}")
        print(f"    gauss_mean   = {metric_rowdict['gauss_mean']:.4f}")
        print(f"    gauss_cv     = {metric_rowdict['gauss_cv']:.4f}")


        # Accumulate rows for master CSV
        all_rows.extend(rows)

    return all_rows, summaries


# --------------------- Master CSV helpers ---------------------


def ensure_master_header():
    if not MASTER_CSV.exists():
        with open(MASTER_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MASTER_FIELDS)
            w.writeheader()


def append_master(folder: str, surface: str, rows: list[dict]):
    if not rows:
        return
    with open(MASTER_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_FIELDS)
        for r in rows:
            row = {fn: "" for fn in MASTER_FIELDS}
            row.update(r)
            row["folder"] = folder
            row["surface"] = surface
            w.writerow(row)


# ----------------------------- Main ---------------------------


def main():
    ensure_master_header()

    for sub in SUBFOLDERS:
        folder = (ROOT / sub)
        if not folder.exists():
            print(f"[WARN] Missing folder: {folder}")
            continue

        pkls = sorted(folder.glob("zipped_patch_*.pkl"))

        if not pkls:
            print(f"[INFO] No .pkl files in {folder}")
            continue

        print(f"\n>>> Folder: {folder} ({len(pkls)} surfaces)")
        for pkl_path in pkls:
            try:
                rows, summaries = process_surface(pkl_path)
                append_master(str(sub), pkl_path.stem, rows)
            except Exception as e:
                print(f"[ERROR] Failed {pkl_path.name}: {e}")


if __name__ == "__main__":
    main()
