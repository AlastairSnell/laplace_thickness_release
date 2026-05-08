"""
process_folds_timed.py

Run Laplacian thickness on all fold patches in
  validation/data/folds_selected/
with parallelised path tracing (all CPUs) and per-patch wall-clock timing.

Usage (from repo root, with the package installed or src/ on the path):
    python validation/scripts/process_folds_timed.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree

# ── project imports ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.laplace_thickness_release.bem.core import BEMConfig, assemble_and_solve
from src.laplace_thickness_release.main.main import (
    DEFAULT_ALPHA,
    DEFAULT_FIRST_STEP,
    DEFAULT_MAX_ITER,
    DEFAULT_START_SPACING_TRACE,
    _degenerate_reasons,
    load_triangles_from_vtk_polydata,
    run_single_path,
)
from src.laplace_thickness_release.mesh.preprocess import preprocess_triangles
from src.laplace_thickness_release.trace.normal import build_surface_vertex_normals
from src.laplace_thickness_release.trace.start_points import pick_even_start_points

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "validation" / "data" / "folds_selected"
OUT_DIR  = DATA_DIR / "results"

# ── parameters (matching main.py defaults) ───────────────────────────────────────
START_BC_VALUE = 0.0        # white surface (φ = 0)
DIRECTION      = "up"       # ascend from white toward pial (φ = 1)
PCT            = 50.0       # reverse-cap percentage

ALPHA      = DEFAULT_ALPHA                # 0.05 mm fixed step
MAX_ITER   = DEFAULT_MAX_ITER             # 150
FIRST_STEP = DEFAULT_FIRST_STEP           # 0.05 mm seed step
SPACING    = DEFAULT_START_SPACING_TRACE  # 1.0 mm target FPS spacing

N_WORKERS  = os.cpu_count()


# ──────────────────────────────────────────────────────────────────────────────────
# Helpers (identical to process_folds.py)
# ──────────────────────────────────────────────────────────────────────────────────

def _build_surface_mesh(triangles: list[dict], bc_value: float):
    verts_map: dict[tuple, int] = {}
    faces: list[list[int]] = []

    for tri in triangles:
        if str(tri.get("bc_type", "")).lower() != "dirichlet":
            continue
        if not np.isclose(float(tri.get("bc_value", -999.0)), float(bc_value)):
            continue
        idx = []
        for v in tri["vertices"]:
            key = tuple(np.asarray(v, dtype=np.float64))
            if key not in verts_map:
                verts_map[key] = len(verts_map)
            idx.append(verts_map[key])
        faces.append(idx)

    if not verts_map:
        raise ValueError(f"No Dirichlet bc_value={bc_value} triangles found.")

    V = np.asarray(list(verts_map.keys()), dtype=np.float64)
    F = np.asarray(faces, dtype=np.int32)

    i_idx = F[:, [0, 1, 2]].ravel()
    j_idx = F[:, [1, 2, 0]].ravel()
    w = np.linalg.norm(V[i_idx] - V[j_idx], axis=1)
    nV = len(V)
    A = coo_matrix((w, (i_idx, j_idx)), shape=(nV, nV))
    A = A.maximum(A.T).tocsr()

    return V, F, A


def _reverse_cap_admissible(
    V: np.ndarray, F: np.ndarray, A, pct: float
) -> np.ndarray:
    edge_cnt: Counter = Counter()
    for a, b, c in F:
        for e in ((a, b), (b, c), (c, a)):
            edge_cnt[tuple(sorted(e))] += 1

    b_verts = np.unique(
        [v for e, k in edge_cnt.items() if k == 1 for v in e]
    ).astype(np.int64)

    if b_verts.size == 0:
        return np.arange(len(V), dtype=np.int64)

    D = dijkstra(A, directed=False, indices=b_verts)
    d_to_bnd = np.min(D, axis=0)
    finite_d = d_to_bnd[np.isfinite(d_to_bnd)]
    if finite_d.size == 0:
        return np.arange(len(V), dtype=np.int64)
    inradius = float(np.nanmax(finite_d))
    thresh = (1.0 - pct / 100.0) * inradius
    return np.where(d_to_bnd >= thresh)[0].astype(np.int64)


def _trim_mesh(
    V: np.ndarray, F: np.ndarray, keep_verts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keep_mask = np.zeros(len(V), dtype=bool)
    keep_mask[keep_verts] = True

    face_mask = keep_mask[F[:, 0]] & keep_mask[F[:, 1]] & keep_mask[F[:, 2]]
    F_kept = F[face_mask]

    if len(F_kept) == 0:
        raise ValueError("No faces remain after trimming.")

    used_verts = np.unique(F_kept)
    old_to_new = np.full(len(V), -1, dtype=np.int64)
    for new_idx, old_idx in enumerate(used_verts):
        old_to_new[old_idx] = new_idx

    V_trim = V[used_verts]
    F_trim = old_to_new[F_kept]

    assert np.all(F_trim >= 0), "Negative face index after reindexing"
    assert np.all(F_trim < len(V_trim)), "Face index out of range after reindexing"
    used_check = np.unique(F_trim)
    assert len(used_check) == len(V_trim), "Orphan vertices remain after reindexing"

    return V_trim, F_trim, old_to_new, used_verts


def _extract_boundary_loop(F: np.ndarray) -> list[int] | None:
    edge_count: dict[tuple[int, int], int] = {}
    for f in F:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            edge_count[key] = edge_count.get(key, 0) + 1

    boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]

    if not boundary_edges:
        return None

    adj: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    for v, neighbors in adj.items():
        if len(neighbors) != 2:
            LOG.warning(
                "Boundary vertex %d has %d neighbors (expected 2) — not a simple loop.", v, len(neighbors)
            )
            return None

    start = boundary_edges[0][0]
    loop = [start]
    prev = -1
    current = start

    while True:
        neighbors = adj[current]
        nxt = neighbors[0] if neighbors[0] != prev else neighbors[1]
        if nxt == start:
            break
        loop.append(nxt)
        prev = current
        current = nxt
        if len(loop) > len(adj):
            LOG.warning("Boundary walk exceeded vertex count — aborting.")
            return None

    if len(loop) != len(adj):
        LOG.warning(
            "Boundary loop covers %d/%d boundary vertices — multiple loops or disconnected boundary.",
            len(loop), len(adj),
        )
        return None

    return loop


def _geodesic_idw(
    V: np.ndarray,
    A,
    seed_xyz: np.ndarray,
    seed_vals: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    kd = KDTree(V)
    src_idx = kd.query(seed_xyz)[1].astype(np.int32)
    D = dijkstra(A, directed=False, indices=src_idx)
    W = 1.0 / np.maximum(D, 1e-6) ** power
    num = (W * seed_vals[:, None]).sum(axis=0)
    den = W.sum(axis=0)
    return num / np.maximum(den, 1e-12)


# ──────────────────────────────────────────────────────────────────────────────────
# Per-patch processing
# ──────────────────────────────────────────────────────────────────────────────────

def process_patch(vtk_path: Path, out_dir: Path) -> None:
    stem = vtk_path.stem
    LOG.info("=" * 60)
    LOG.info("Processing: %s", stem)

    triangles = load_triangles_from_vtk_polydata(vtk_path)
    preprocess_triangles(triangles)
    cfg = BEMConfig()

    LOG.info("  Solving BEM system ...")
    q, _, _ = assemble_and_solve(triangles, cfg)

    surface_vertices, surface_normals = build_surface_vertex_normals(
        triangles, bc_value=START_BC_VALUE
    )

    LOG.info("  Selecting seed points (pct=%.0f, spacing=%.1f) ...", PCT, SPACING)
    start_pts = pick_even_start_points(
        triangles,
        pct=PCT,
        target_spacing=SPACING,
        max_points=None,
        bc_value=START_BC_VALUE,
        seed_mode="vertex",
    )
    start_pts = np.asarray(start_pts, dtype=np.float64)
    num_seeds = len(start_pts)
    LOG.info("  %d seed points selected.", num_seeds)

    coord_to_idx = {tuple(v): i for i, v in enumerate(surface_vertices)}
    start_idx_arr = np.full(num_seeds, -1, dtype=np.int64)
    missing_i: list[int] = []
    for i, pt in enumerate(start_pts):
        idx = coord_to_idx.get(tuple(np.asarray(pt, dtype=np.float64)))
        if idx is None:
            missing_i.append(i)
        else:
            start_idx_arr[i] = int(idx)

    if missing_i:
        kd_surf = KDTree(surface_vertices)
        mi = np.asarray(missing_i, dtype=int)
        _, nn = kd_surf.query(start_pts[mi])
        start_idx_arr[mi] = nn.astype(np.int64)
        LOG.warning(
            "  %d seed points mapped to nearest vertex (exact match absent).", len(missing_i)
        )

    seed_dirs = -1.0 * surface_normals[start_idx_arr]

    # Numba JIT warm-up
    run_single_path(
        0,
        triangles=triangles,
        start_pt=start_pts[0],
        q=q,
        cfg=cfg,
        direction_down=DIRECTION,
        max_iter=MAX_ITER,
        alpha_initial=ALPHA,
        first_step=FIRST_STEP,
        seed_dir=seed_dirs[0],
        seed_face_idx=None,
        debug=False,
    )

    LOG.info("  Tracing %d paths (parallel, %d workers) ...", num_seeds, N_WORKERS)
    lengths    = np.zeros(num_seeds, dtype=np.float64)
    valid_mask = np.ones(num_seeds, dtype=bool)
    tri_by_idx = {i: tri for i, tri in enumerate(triangles)}

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(
                run_single_path,
                i,
                triangles=triangles,
                start_pt=start_pts[i],
                q=q,
                cfg=cfg,
                direction_down=DIRECTION,
                max_iter=MAX_ITER,
                alpha_initial=ALPHA,
                first_step=FIRST_STEP,
                seed_dir=seed_dirs[i],
                seed_face_idx=None,
                debug=False,
            ): i
            for i in range(num_seeds)
        }
        for fut in as_completed(futures):
            idx, _path, length, meta = fut.result()
            lengths[idx] = length
            reasons = _degenerate_reasons(length, meta, tri_by_idx, START_BC_VALUE)
            if reasons:
                valid_mask[idx] = False
                LOG.debug("  Path %d degenerate (%s).", idx, ", ".join(reasons))

    n_valid = int(valid_mask.sum())
    LOG.info("  %d / %d valid paths.", n_valid, num_seeds)
    if n_valid == 0:
        raise ValueError("All paths are degenerate — cannot proceed.")

    LOG.info("  Interpolating per-vertex thickness (geodesic IDW) ...")
    V_w, F_w, A_w = _build_surface_mesh(triangles, bc_value=0.0)
    thick_all = _geodesic_idw(V_w, A_w, start_pts[valid_mask], lengths[valid_mask])

    LOG.info("  Trimming white surface (pct=%.0f) ...", PCT)
    adm_w = _reverse_cap_admissible(V_w, F_w, A_w, PCT)
    valid_thick = np.isfinite(thick_all) & (thick_all > 0.0)
    retain_w = adm_w[valid_thick[adm_w]]

    if len(retain_w) == 0:
        raise ValueError("No valid white vertices remain after trimming.")

    V_wt, F_wt, _, used_w = _trim_mesh(V_w, F_w, retain_w)
    thick_trim = thick_all[used_w]
    LOG.info("  White trimmed: %d vertices, %d faces.", len(V_wt), len(F_wt))

    LOG.info("  Trimming pial surface (pct=%.0f) ...", PCT)
    V_p, F_p, A_p = _build_surface_mesh(triangles, bc_value=1.0)
    adm_p = _reverse_cap_admissible(V_p, F_p, A_p, PCT)

    if len(adm_p) == 0:
        raise ValueError("No pial vertices remain after trimming.")

    V_pt, F_pt, _, _ = _trim_mesh(V_p, F_p, adm_p)
    LOG.info("  Pial trimmed: %d vertices, %d faces.", len(V_pt), len(F_pt))

    if len(V_wt) != len(V_pt):
        LOG.warning(
            "  White (%d) and pial (%d) vertex counts differ after independent trimming.",
            len(V_wt), len(V_pt),
        )

    LOG.info("  Extracting boundary loops ...")
    white_loop = _extract_boundary_loop(F_wt)
    if white_loop is None:
        raise ValueError("White trimmed surface has no single boundary loop.")

    pial_loop = _extract_boundary_loop(F_pt)
    if pial_loop is None:
        raise ValueError("Pial trimmed surface has no single boundary loop.")

    white_bnd = np.asarray(white_loop, dtype=np.int64)
    pial_bnd = np.asarray(pial_loop, dtype=np.int64)

    LOG.info(
        "  White boundary: %d verts | Pial boundary: %d verts.",
        len(white_bnd), len(pial_bnd),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_result.npz"

    np.savez(
        out_path,
        white_vertices=V_wt,
        pial_vertices=V_pt,
        white_faces=F_wt,
        pial_faces=F_pt,
        white_boundary=white_bnd,
        pial_boundary=pial_bnd,
        gt_thickness=thick_trim,
    )
    LOG.info("  Saved: %s", out_path.name)

    data = np.load(out_path)
    vw = data["white_vertices"]
    vp = data["pial_vertices"]
    fw = data["white_faces"]
    fp = data["pial_faces"]
    wb = data["white_boundary"]
    pb = data["pial_boundary"]
    gt = data["gt_thickness"]

    ok = True
    if gt.shape[0] != vw.shape[0]:
        LOG.warning("  VAL FAIL: gt_thickness (%d) != white_vertices (%d)", gt.shape[0], vw.shape[0])
        ok = False
    if not np.all(gt > 0):
        LOG.warning("  VAL FAIL: some gt_thickness values <= 0")
        ok = False
    for name, arr in [("white_vertices", vw), ("pial_vertices", vp), ("gt_thickness", gt)]:
        if not np.all(np.isfinite(arr)):
            LOG.warning("  VAL FAIL: NaN/Inf in %s", name)
            ok = False
    if fw.size > 0 and (np.any(fw < 0) or np.any(fw >= vw.shape[0])):
        LOG.warning("  VAL FAIL: white_faces index out of range [0, %d)", vw.shape[0])
        ok = False
    if fp.size > 0 and (np.any(fp < 0) or np.any(fp >= vp.shape[0])):
        LOG.warning("  VAL FAIL: pial_faces index out of range [0, %d)", vp.shape[0])
        ok = False
    if wb.size > 0 and (np.any(wb < 0) or np.any(wb >= vw.shape[0])):
        LOG.warning("  VAL FAIL: white_boundary index out of range [0, %d)", vw.shape[0])
        ok = False
    if pb.size > 0 and (np.any(pb < 0) or np.any(pb >= vp.shape[0])):
        LOG.warning("  VAL FAIL: pial_boundary index out of range [0, %d)", vp.shape[0])
        ok = False

    t_min = float(gt.min())
    t_max = float(gt.max())
    status = "OK" if ok else "WARNINGS"
    print(
        f"  SUMMARY | {stem} | "
        f"n_white_verts={vw.shape[0]} | n_pial_verts={vp.shape[0]} | "
        f"n_white_faces={fw.shape[0]} | n_pial_faces={fp.shape[0]} | "
        f"n_white_boundary={wb.shape[0]} | n_pial_boundary={pb.shape[0]} | "
        f"thickness=[{t_min:.3f}, {t_max:.3f}] mm | {status}"
    )


# ──────────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────────

def main() -> int:
    vtk_files = sorted(DATA_DIR.glob("zipped_patch_*.vtk"))
    if not vtk_files:
        print(f"No VTK files found in {DATA_DIR}")
        return 1

    print(f"Found {len(vtk_files)} patches.")
    print(f"Output: {OUT_DIR}")
    print(f"Workers: {N_WORKERS}")
    print()

    n_ok = n_fail = 0
    patch_times: dict[str, float] = {}

    for vtk_path in vtk_files:
        stem = vtk_path.stem
        t0 = time.perf_counter()
        try:
            process_patch(vtk_path, OUT_DIR)
            elapsed = time.perf_counter() - t0
            patch_times[stem] = elapsed
            print(f"  Time: {elapsed:.3f} s")
            n_ok += 1
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            patch_times[stem] = elapsed
            LOG.error("FAILED %s: %s", vtk_path.name, exc)
            traceback.print_exc()
            print(f"  Time: {elapsed:.3f} s (failed)")
            n_fail += 1

    print()
    print(f"Done: {n_ok} succeeded, {n_fail} failed.")
    print()
    print(f"{'Patch':<25}  {'Time (s)':>10}")
    print("-" * 38)
    for stem, t in patch_times.items():
        print(f"{stem:<25}  {t:>10.3f}")
    if patch_times:
        total = sum(patch_times.values())
        print("-" * 38)
        print(f"{'TOTAL':<25}  {total:>10.3f}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
