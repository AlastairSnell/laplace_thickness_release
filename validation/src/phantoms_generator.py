#!/usr/bin/env python3
"""
Generate multiple closed "zipped" cortical ribbon surfaces from a FreeSurfer mid-surface.

For each step_mm in STEP_LIST_MM, create N_SAMPLES random patches:
  - Extract geodesic patch on lh.graymid
  - Offset ±step_mm to form white/pial
  - Geodesically crop inner % around the seed
  - Zip boundary loops to form a side wall
  - Package triangles with BCs and save as: {thickness_mm}x{radius}_{i}.pkl
    e.g. 1x15_4.pkl => thickness=1.0 mm (step_mm=0.5), radius=15, sample #4
"""

from __future__ import annotations
import os, time, tempfile, pickle
from pathlib import Path
from typing import Tuple, List, Dict
from collections import defaultdict

import numpy as np
import nibabel as nib
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from multiprocessing import get_context

# ---------------------------- config ----------------------------
SUBJ_MID = Path(r'\\wsl.localhost\Ubuntu\home\uqasnell\freesurfer_subjects\good_output\surf\lh.graymid')

# Geometry / cropping
GEO_RADIUS_MM = 15.0          # initial patch radius on mid-surface
KEEP_PC = 95.0                # keep closest % geodesically after offset
SMOOTH_LAMBDA = 0.5
SMOOTH_MU = -0.53

# Generation plan
STEP_LIST_MM = [0.25]  # per-side offsets (mm) → thickness = 2*step
N_SAMPLES     = 10              # patches per step size
MAX_ATTEMPTS_PER_SAMPLE = 60    # retries per sample inside worker

# Output
OUT_DIR = Path("./validation/data/phantoms")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# I/O pacing (helps when a cloud drive syncs the folder)
WRITE_PAUSE_SEC = 1.0

# ------------------------- utilities ----------------------------

def write_pickle_then_pause(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('wb', delete=False, dir=path.parent,
                                     prefix=path.name, suffix='.tmp') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush(); os.fsync(f.fileno())
        tmp = f.name
    os.replace(tmp, path)
    time.sleep(WRITE_PAUSE_SEC)

def read_fs_surf(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    v, f = nib.freesurfer.read_geometry(str(path))
    return v.astype(np.float64), f.astype(np.int32)

def unique_edges(faces: np.ndarray) -> np.ndarray:
    e01 = np.sort(faces[:, [0, 1]], axis=1)
    e12 = np.sort(faces[:, [1, 2]], axis=1)
    e20 = np.sort(faces[:, [2, 0]], axis=1)
    return np.unique(np.vstack((e01, e12, e20)), axis=0)

def csr_graph(points: np.ndarray, faces: np.ndarray) -> csr_matrix:
    edges = unique_edges(faces)
    w = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    i = np.concatenate([edges[:, 0], edges[:, 1]])
    j = np.concatenate([edges[:, 1], edges[:, 0]])
    d = np.concatenate([w, w])
    return csr_matrix((d, (i, j)), shape=(len(points), len(points)))

def geodesic_ball(points: np.ndarray, faces: np.ndarray, seed_idx: int, radius_mm: float) -> np.ndarray:
    G = csr_graph(points, faces)
    dist = dijkstra(G, directed=False, indices=seed_idx, limit=radius_mm)  # limit trims search
    mask = np.zeros(len(points), dtype=bool)
    mask[np.isfinite(dist) & (dist < radius_mm)] = True
    return mask

def largest_component_faces(faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return faces
    edges = unique_edges(faces)
    n = faces.max() + 1
    A = csr_matrix((np.ones(edges.shape[0] * 2),
                    (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                   shape=(n, n))
    visited = np.zeros(n, dtype=bool)
    comp_id = -np.ones(n, dtype=int)
    cid = 0
    for v in range(n):
        if visited[v]:
            continue
        stack = [v]
        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            comp_id[u] = cid
            nbrs = A.indices[A.indptr[u]:A.indptr[u + 1]]
            stack.extend([w for w in nbrs if not visited[w]])
        cid += 1
    sizes = np.bincount(comp_id[comp_id >= 0])
    keep_comp = sizes.argmax()
    return faces[np.all(comp_id[faces] == keep_comp, axis=1)]

def reindex_patch(verts: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    used = np.unique(faces.ravel())
    mapping = -np.ones(verts.shape[0], dtype=np.int64)
    mapping[used] = np.arange(used.size, dtype=np.int64)
    return verts[used], mapping[faces], mapping

def compute_face_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1e-12
    return n / ln

def compute_vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)  # area-weighted face normals
    vnorms = np.zeros_like(points)
    for k in range(3):
        np.add.at(vnorms, faces[:, k], cross)
    ln = np.linalg.norm(vnorms, axis=1, keepdims=True); ln[ln == 0] = 1e-12
    return vnorms / ln


def taubin_smooth_once(points: np.ndarray, faces: np.ndarray, lam=0.5, mu=-0.53, fixed: np.ndarray | None=None) -> np.ndarray:
    edges = unique_edges(faces)
    n = len(points)
    A = csr_matrix((np.ones(edges.shape[0] * 2),
                    (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                   shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel(); deg[deg == 0] = 1.0
    P = points.copy()
    fixed_mask = np.zeros(n, dtype=bool) if fixed is None else fixed
    avg = A.dot(P) / deg[:, None]
    P[~fixed_mask] += lam * (avg[~fixed_mask] - P[~fixed_mask])
    avg = A.dot(P) / deg[:, None]
    P[~fixed_mask] += mu * (avg[~fixed_mask] - P[~fixed_mask])
    return P

def move_vertices_in_out_once(points: np.ndarray, faces: np.ndarray, seed_point: np.ndarray, step_mm: float):
    seed_idx = int(np.argmin(np.linalg.norm(points - seed_point, axis=1)))
    vnorms = compute_vertex_normals(points, faces)
    outer = points + step_mm * vnorms
    inner = points - step_mm * vnorms
    fixed = np.zeros(len(points), dtype=bool); fixed[seed_idx] = True
    outer = taubin_smooth_once(outer, faces, SMOOTH_LAMBDA, SMOOTH_MU, fixed)
    inner = taubin_smooth_once(inner, faces, SMOOTH_LAMBDA, SMOOTH_MU, fixed)
    return outer, inner, seed_idx

def geodesic_keep_inner_percent(points: np.ndarray, faces: np.ndarray, seed_idx: int, keep_pc: float):
    G = csr_graph(points, faces)
    dist = dijkstra(G, directed=False, indices=seed_idx)
    finite = np.isfinite(dist)
    thr = np.quantile(dist[finite], keep_pc / 100.0)
    keep_mask = finite & (dist <= thr)
    new_idx = -np.ones(len(points), dtype=np.int64)
    new_idx[keep_mask] = np.arange(keep_mask.sum(), dtype=np.int64)
    faces_mapped = new_idx[faces]
    ok = np.all(faces_mapped >= 0, axis=1)
    return points[keep_mask], faces_mapped[ok], keep_mask

def find_boundary_loop(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    e01 = np.sort(faces[:, [0, 1]], axis=1)
    e12 = np.sort(faces[:, [1, 2]], axis=1)
    e20 = np.sort(faces[:, [2, 0]], axis=1)
    uniq, counts = np.unique(np.vstack((e01, e12, e20)), axis=0, return_counts=True)
    boundary_edges = uniq[counts == 1]
    if boundary_edges.size == 0:
        raise ValueError("No boundary edges found.")
    adj: Dict[int, List[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b); adj[b].append(a)
    start = next((v for v, nbrs in adj.items() if len(nbrs) == 2), None)
    if start is None:
        raise ValueError("Boundary is not a single 2-regular loop.")
    loop = [start]; prev = -1; cur = start
    while True:
        nbrs = adj[cur]
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
    return np.asarray(loop, dtype=int)

def robust_match_loops(loop_a: np.ndarray, loop_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(loop_a) <= len(loop_b):
        small, large = loop_a, loop_b
    else:
        small, large = loop_b, loop_a
    n_small, n_large = len(small), len(large)
    t_small, t_large = cKDTree(small), cKDTree(large)
    _, large_to_small = t_small.query(large)
    _, small_to_large = t_large.query(small)
    matches = np.column_stack((np.arange(n_large, dtype=int), large_to_small))
    used_small = set(large_to_small.tolist())
    missing_small = np.setdiff1d(np.arange(n_small, dtype=int),
                                 np.fromiter(used_small, int, count=len(used_small)))
    if missing_small.size:
        extra = np.column_stack((small_to_large[missing_small], missing_small))
        matches = np.vstack((matches, extra))
    matches = matches[np.argsort(matches[:, 0])]
    L = matches[:, 0]; S = matches[:, 1]
    Ln = np.roll(L, -1); Sn = np.roll(S, -1)
    side_points = np.vstack((large, small))
    s_off = n_large
    faces_upper = np.column_stack((L, Ln, S + s_off))
    faces_lower = np.column_stack((Ln, Sn + s_off, S + s_off))
    side_faces = np.vstack((faces_upper, faces_lower)).astype(int, copy=False)
    return side_points, side_faces

def faces_to_tri_dicts(points: np.ndarray, faces: np.ndarray, bc_type: str, bc_value: float,
                       atol: float = 1e-12) -> List[dict]:
    # store triangles as float32 to reduce memory/size; keep computations upstream in float64
    V = points[faces].astype(np.float32, copy=False)   # (m,3,3)
    dup01 = np.all(np.isclose(V[:, 0], V[:, 1], rtol=0.0, atol=atol), axis=1)
    dup12 = np.all(np.isclose(V[:, 1], V[:, 2], rtol=0.0, atol=atol), axis=1)
    dup20 = np.all(np.isclose(V[:, 2], V[:, 0], rtol=0.0, atol=atol), axis=1)
    dup_mask = dup01 | dup12 | dup20
    finite_mask = np.isfinite(V).all(axis=(1, 2))
    e1 = V[:, 1] - V[:, 0]; e2 = V[:, 2] - V[:, 0]
    cross = np.cross(e1, e2).astype(np.float32, copy=False)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    valid = finite_mask & (~dup_mask) & (area >= atol)
    Vv = V[valid]
    Cv = cross[valid]
    norms = np.linalg.norm(Cv, axis=1, keepdims=True); norms[norms == 0.0] = 1.0
    Nv = (Cv / norms).astype(np.float32, copy=False)
    return [{'vertices': v, 'normal': n, 'bc_type': bc_type, 'bc_value': bc_value}
            for v, n in zip(Vv, Nv)]

# --------------------------- core --------------------------------

def generate_one(verts: np.ndarray,
                 faces: np.ndarray,
                 step_mm: float,
                 rng: np.random.Generator) -> dict | None:
    """Return dict with 'triangles' and 'meta', or None if this seed fails."""
    # 1) choose seed and make mid patch
    seed_idx0 = int(rng.integers(0, len(verts)))
    seed_point0 = verts[seed_idx0]

    mask = geodesic_ball(verts, faces, seed_idx0, GEO_RADIUS_MM)
    patch_faces = faces[np.all(mask[faces], axis=1)]
    patch_verts, patch_faces, mapping = reindex_patch(verts, patch_faces)  # shrink to local indices
    patch_faces = largest_component_faces(patch_faces)    

    seed_idx_patch = mapping[seed_idx0]
    if seed_idx_patch < 0:
        d = np.linalg.norm(patch_verts - seed_point0, axis=1)
        seed_idx_patch = int(np.argmin(d))
        seed_point0 = patch_verts[seed_idx_patch]

    # 2) offset to white/pial
    outer_pts, inner_pts, seed_idx_sub = move_vertices_in_out_once(
        patch_verts, patch_faces, seed_point0, step_mm
    )

    # 3) crop inner % on both shells
    outer_pts_c, faces_outer_c, _ = geodesic_keep_inner_percent(
        outer_pts, patch_faces, seed_idx_sub, KEEP_PC
    )
    inner_pts_c, faces_inner_c, _ = geodesic_keep_inner_percent(
        inner_pts, patch_faces, seed_idx_sub, KEEP_PC
    )
    if faces_outer_c.size == 0 or faces_inner_c.size == 0:
        return None

    # 4) boundary loops + zipping
    try:
        white_loop_idx = find_boundary_loop(inner_pts_c, faces_inner_c)
        pial_loop_idx  = find_boundary_loop(outer_pts_c, faces_outer_c)
    except ValueError:
        return None

    white_coords = inner_pts_c[white_loop_idx]
    pial_coords  = outer_pts_c[pial_loop_idx]
    side_points, side_faces = robust_match_loops(white_coords, pial_coords)

    # 5) package triangles with BCs
    tris = []
    tris += faces_to_tri_dicts(outer_pts_c, faces_outer_c, 'dirichlet', 1.0)  # pial
    tris += faces_to_tri_dicts(inner_pts_c, faces_inner_c, 'dirichlet', 0.0)  # white
    tris += faces_to_tri_dicts(side_points, side_faces, 'neumann', 0.0)       # side

    meta = {
        'seed_idx_global': seed_idx0,
        'seed_point_patch': seed_point0.tolist(),
        'step_mm': float(step_mm),
        'thickness_mm': float(2.0 * step_mm),
        'geo_radius_mm': float(GEO_RADIUS_MM),
        'keep_pc': float(KEEP_PC),
        'n_tris': len(tris),
        'n_white_tris': int(faces_inner_c.shape[0]),
        'n_pial_tris':  int(faces_outer_c.shape[0]),
        'n_side_tris':  int(side_faces.shape[0]),
    }
    return {'triangles': tris, 'meta': meta}

def thickness_label(thickness_mm: float) -> str:
    s = f"{thickness_mm:.3f}".rstrip('0').rstrip('.')
    return s

# ----------------------- subprocess worker -----------------------

def _worker(step_mm: float, seed: int, out_path_str: str) -> int:
    """Child process: generate ONE pack and write it, then exit. Return 0 on success."""
    try:
        rng = np.random.default_rng(seed)
        verts, faces = read_fs_surf(SUBJ_MID)
        attempts = 0
        while attempts < MAX_ATTEMPTS_PER_SAMPLE:
            attempts += 1
            pack = generate_one(verts, faces, step_mm, rng)
            if pack is None:
                continue
            write_pickle_then_pause(pack, Path(out_path_str))
            # Let the process exit to free all memory
            return 0
        return 2  # couldn't find a valid patch after many tries
    except Exception:
        return 1

# ------------------------------ main -----------------------------

def main():
    print(f"Output directory: {OUT_DIR}")
    ctx = get_context("spawn")  # Windows-safe

    for step_mm in STEP_LIST_MM:
        thickness = 2.0 * step_mm
        tlabel = thickness_label(thickness)
        produced = 0
        print(f"\n=== step_mm={step_mm:.3f} → thickness={thickness:.3f} mm ===")

        while produced < N_SAMPLES:
            produced += 1
            out_path = OUT_DIR / f"{tlabel}x{int(GEO_RADIUS_MM)}_{produced}.pkl"
            seed = int(np.random.default_rng().integers(0, 1 << 61))

            p = ctx.Process(target=_worker, args=(step_mm, seed, str(out_path)))
            p.start()
            p.join()

            if p.exitcode == 0:
                print(f"  [{produced}/{N_SAMPLES}] saved {out_path.name}")
            else:
                print(f"  [{produced}/{N_SAMPLES}] failed (exit {p.exitcode}), retrying…")
                produced -= 1  # retry same index

    print("\nDone.")

if __name__ == "__main__":
    main()
