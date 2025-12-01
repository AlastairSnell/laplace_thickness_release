#!/usr/bin/env python3
"""
Generate *families* of zipped cortical ribbon patches from FreeSurfer lh.graymid.

Each family:
  • One geodesic mid-surface patch (same 'father').
  • Children from that same patch at steps in STEP_LIST_MM (thickness = 2*step).
  • If ANY child is rejected (self-intersection/loop failure), the ENTIRE family is rejected and retried.

Saves per child:
  {thickness_label}x{radius_mm}_{family_idx}.pkl
  {thickness_label}x{radius_mm}_{family_idx}_os.npy   (outer seed point)

BC tags:
  - pial faces:      dirichlet=1.0
  - white faces:     dirichlet=0.0
  - side-wall faces: neumann=0.0
"""

from __future__ import annotations
import os, gc, time, tempfile, pickle
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from collections import defaultdict

import numpy as np
import nibabel as nib
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from multiprocessing import get_context
import pymeshlab as ml  # preferred (with try/except fallbacks elsewhere)
from zipper2 import (
    find_boundary_loop,
    _edge_lengths,
    _loop_perimeter,
    _resample_loop_arclength,
    _enforce_same_ccw,
)

# --------------------- CONFIG ---------------------
VERBOSE = False

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

SUBJ_MID = Path(r'\\wsl.localhost\Ubuntu\home\uqasnell\freesurfer_subjects\good_output\surf\lh.graymid')

# Patch footprint
GEO_RADIUS_MM = 10.0     # geodesic radius of initial ball on mid-surface
KEEP_PC       = 95.0     # keep inner % after offset (rim trimming)

# ---------- PRE-STEP (GLOBAL MID-SURFACE) ----------
# Toggle pre-remeshing/smoothing of lh.graymid before any offsets
PRE_REMESH_ENABLE    = False
PRE_REMESH_ITERS     = 15
PRE_REMESH_TARGET_PC = 0.65  # 0.80 denser, 0.95 closer to original
PRE_TAUBIN_PASSES    = 0     # number of whole-surface Taubin passes

# ---------- PRE-STEP (LOCAL MID-SURFACE) ----------
# Patch-level smoothing (applied once on the mid-surface patch before offsets)
PATCH_SMOOTH_ENABLE = True
PATCH_TAUBIN_PASSES = 2
PATCH_SMOOTH_LAMBDA = 0.4
PATCH_SMOOTH_MU     = -0.55

# ---------- OFFSET SHAPING (raw shells) ------------
# One Taubin pass on inner/outer after stepping; seed fixed
SMOOTH_LAMBDA = 0.4
SMOOTH_MU     = -0.55

# ---------- PATCH-LEVEL TANGENTIAL SMOOTH ----------
TAN_SMOOTH_ITERS = 1
TAN_SMOOTH_LAM   = 0.6
TAN_SMOOTH_MU    = -0.6

# ---------- POST-STEP (PATCH REMESH) ---------------
# Rim-fixed isotropic remeshing on cropped shells (boundary loop fixed)
POST_REMESH_ENABLE    = True
POST_REMESH_ITERS     = 8
POST_REMESH_TARGET_PC = 0.85

# ---------- PLAN & OUTPUT --------------------------
STEP_LIST_MM = [2, 1.5, 1.0, 0.5, 0.25, 0.125]   # per-side offsets (thickness = 2*step)
N_FAMILIES   = 30                              # how many successful families to produce
MAX_ATTEMPTS_PER_FAMILY = 5000
OUT_DIR = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\phantoms3")
WRITE_PAUSE_SEC = 1.0

# --------------------- I/O ------------------------
def write_pickle_then_pause(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('wb', delete=False, dir=path.parent,
                                     prefix=path.name, suffix='.tmp') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush(); os.fsync(f.fileno()); tmp = f.name
    os.replace(tmp, path); time.sleep(WRITE_PAUSE_SEC)

def write_npy_then_pause(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('wb', delete=False, dir=path.parent,
                                     prefix=path.name, suffix='.tmp') as f:
        np.save(f, arr); f.flush(); os.fsync(f.fileno()); tmp = f.name
    os.replace(tmp, path); time.sleep(WRITE_PAUSE_SEC)

def read_fs_surf(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    v, f = nib.freesurfer.read_geometry(str(path))
    return v.astype(np.float64), f.astype(np.int32)

# --------------- CORE GEOMETRY UTILS --------------
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
    dist = dijkstra(G, directed=False, indices=seed_idx, limit=radius_mm)
    mask = np.zeros(len(points), dtype=bool)
    mask[np.isfinite(dist) & (dist < radius_mm)] = True
    return mask

def largest_component_faces(faces: np.ndarray) -> np.ndarray:
    if faces.size == 0: return faces
    edges = unique_edges(faces)
    n = faces.max() + 1
    A = csr_matrix((np.ones(edges.shape[0]*2),
                    (np.r_[edges[:,0],edges[:,1]], np.r_[edges[:,1],edges[:,0]])),
                   shape=(n,n))
    visited = np.zeros(n, dtype=bool); comp = -np.ones(n, dtype=int); cid = 0
    for v in range(n):
        if visited[v]: continue
        stack = [v]
        while stack:
            u = stack.pop()
            if visited[u]: continue
            visited[u] = True; comp[u] = cid
            nbrs = A.indices[A.indptr[u]:A.indptr[u+1]]
            stack.extend([w for w in nbrs if not visited[w]])
        cid += 1
    sizes = np.bincount(comp[comp>=0]); keep = sizes.argmax()
    return faces[np.all(comp[faces]==keep, axis=1)]

def reindex_patch(verts: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    used = np.unique(faces.ravel())
    mapping = -np.ones(verts.shape[0], dtype=np.int64)
    mapping[used] = np.arange(used.size, dtype=np.int64)
    return verts[used], mapping[faces], mapping

def compute_vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = points[faces[:,0]], points[faces[:,1]], points[faces[:,2]]
    cross = np.cross(v1 - v0, v2 - v0)
    vnorms = np.zeros_like(points)
    for k in range(3): np.add.at(vnorms, faces[:,k], cross)
    ln = np.linalg.norm(vnorms, axis=1, keepdims=True); ln[ln==0] = 1e-12
    return vnorms / ln

def taubin_smooth_once(points: np.ndarray, faces: np.ndarray, lam=0.5, mu=-0.53, fixed: np.ndarray|None=None) -> np.ndarray:
    edges = unique_edges(faces)
    n = len(points)
    A = csr_matrix((np.ones(edges.shape[0]*2),
                    (np.r_[edges[:,0],edges[:,1]], np.r_[edges[:,1],edges[:,0]])),
                   shape=(n,n))
    deg = np.asarray(A.sum(axis=1)).ravel(); deg[deg==0] = 1.0
    P = points.copy()
    fixed_mask = np.zeros(n, dtype=bool) if fixed is None else fixed
    avg = A.dot(P)/deg[:,None]; P[~fixed_mask] += lam*(avg[~fixed_mask] - P[~fixed_mask])
    avg = A.dot(P)/deg[:,None]; P[~fixed_mask] += mu *(avg[~fixed_mask] - P[~fixed_mask])
    return P

def taubin_smooth_patch(points: np.ndarray, faces: np.ndarray, passes: int,
                        lam: float, mu: float, seed_idx: Optional[int] = None) -> np.ndarray:
    """Apply Taubin smoothing on a mid-surface patch; optionally keep seed fixed."""
    V = points.copy()
    fixed = None
    if seed_idx is not None and 0 <= seed_idx < len(V):
        fixed = np.zeros(len(V), dtype=bool)
        fixed[seed_idx] = True
    for _ in range(max(0, passes)):
        V = taubin_smooth_once(V, faces, lam, mu, fixed)
    return V

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
    thr = np.quantile(dist[finite], keep_pc/100.0)
    keep_mask = finite & (dist <= thr)
    new_idx = -np.ones(len(points), dtype=np.int64); new_idx[keep_mask] = np.arange(keep_mask.sum())
    faces_mapped = new_idx[faces]; ok = np.all(faces_mapped >= 0, axis=1)
    return points[keep_mask], faces_mapped[ok], keep_mask

def robust_match_loops(loopA_pts: np.ndarray, loopB_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(loopA_pts) <= len(loopB_pts):
        small, large = loopA_pts, loopB_pts
    else:
        small, large = loopB_pts, loopA_pts
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
    matches = matches[np.argsort(matches[:,0])]
    L = matches[:,0]; S = matches[:,1]; Ln = np.roll(L, -1); Sn = np.roll(S, -1)
    side_points = np.vstack((large, small))
    s_off = n_large
    faces_upper = np.column_stack((L, Ln, S + s_off))
    faces_lower = np.column_stack((Ln, Sn + s_off, S + s_off))
    side_faces = np.vstack((faces_upper, faces_lower)).astype(int, copy=False)
    return side_points, side_faces

def faces_to_tri_dicts(points: np.ndarray, faces: np.ndarray, bc_type: str, bc_value: float, atol: float=1e-12) -> List[dict]:
    V = points[faces].astype(np.float32, copy=False)
    dup01 = np.all(np.isclose(V[:,0], V[:,1], rtol=0.0, atol=atol), axis=1)
    dup12 = np.all(np.isclose(V[:,1], V[:,2], rtol=0.0, atol=atol), axis=1)
    dup20 = np.all(np.isclose(V[:,2], V[:,0], rtol=0.0, atol=atol), axis=1)
    dup_mask = dup01 | dup12 | dup20
    finite_mask = np.isfinite(V).all(axis=(1,2))
    e1 = V[:,1] - V[:,0]; e2 = V[:,2] - V[:,0]
    cross = np.cross(e1, e2).astype(np.float32, copy=False)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    valid = finite_mask & (~dup_mask) & (area >= atol)
    Vv = V[valid]; Cv = cross[valid]
    norms = np.linalg.norm(Cv, axis=1, keepdims=True); norms[norms==0.0] = 1.0
    Nv = (Cv / norms).astype(np.float32, copy=False)
    return [{'vertices': v, 'normal': n, 'bc_type': bc_type, 'bc_value': bc_value}
            for v,n in zip(Vv, Nv)]

# ------------- SELF-INTERSECTION (pyigl / fallback) --------------
def _seg_tri_intersect(p0, p1, v0, v1, v2, eps=1e-12):
    dirv = p1 - p0; e1 = v1 - v0; e2 = v2 - v0
    h = np.cross(dirv, e2); a_ = np.dot(e1, h)
    if abs(a_) < eps: return False
    f = 1.0 / a_; s = p0 - v0
    u = f * np.dot(s, h);  qv = np.cross(s, e1); v = f * np.dot(dirv, qv)
    if u < -eps or u > 1+eps: return False
    if v < -eps or u + v > 1+eps: return False
    t = f * np.dot(e2, qv);  return (t >= -eps) and (t <= 1+eps)

def _tri_tri_intersect(P: np.ndarray, Q: np.ndarray, eps=1e-12) -> bool:
    pa, pb = P.min(0), P.max(0); qa, qb = Q.min(0), Q.max(0)
    if (pa > qb + eps).any() or (qa > pb + eps).any(): return False
    for i in range(3):
        if _seg_tri_intersect(P[i], P[(i+1)%3], Q[0], Q[1], Q[2], eps): return True
    for i in range(3):
        if _seg_tri_intersect(Q[i], Q[(i+1)%3], P[0], P[1], P[2], eps): return True
    return False

def has_self_intersections(V: np.ndarray, F: np.ndarray) -> bool:
    Vc = np.asarray(V, float, order='C'); Fc = np.asarray(F, np.int32, order='C')
    if Fc.size == 0: return False
    try:
        import igl
        res = igl.self_intersections(Vc, Fc)
        IF = res[0] if isinstance(res, (list, tuple)) else res
        return (IF is not None) and (len(IF) > 0)
    except Exception:
        # Fallback: voxel-culling + tri-tri tests (no shared-vertex pairs)
        e = Vc[Fc[:,[0,1,1,2,2,0]]].reshape(-1,2,3); edge_lens = np.linalg.norm(e[:,0]-e[:,1], axis=1)
        vox = (np.median(edge_lens)*1.5) if len(edge_lens) else 1.0
        mins = Vc[Fc].min(axis=1); maxs = Vc[Fc].max(axis=1); origin = mins.min(axis=0)
        def voxkeys(mn, mx):
            lo = np.floor((mn-origin)/vox).astype(int); hi = np.floor((mx-origin)/vox).astype(int)
            for x in range(lo[0], hi[0]+1):
                for y in range(lo[1], hi[1]+1):
                    for z in range(lo[2], hi[2]+1):
                        yield (x,y,z)
        grid: Dict[tuple, List[int]] = defaultdict(list)
        for idx in range(len(Fc)):
            for key in voxkeys(mins[idx], maxs[idx]): grid[key].append(idx)
        face_verts = [set(Fc[i]) for i in range(len(Fc))]
        seen = set()
        for idxs in grid.values():
            L = len(idxs)
            if L < 2: continue
            for a in range(L):
                ia = idxs[a]
                for b in range(a+1, L):
                    ib = idxs[b]; key = (min(ia,ib), max(ia,ib))
                    if key in seen: continue
                    seen.add(key)
                    if face_verts[ia] & face_verts[ib]: continue
                    if _tri_tri_intersect(Vc[Fc[ia]], Vc[Fc[ib]]): return True
        return False

def _csr_from_mesh(points: np.ndarray, faces: np.ndarray) -> csr_matrix:
    edges = unique_edges(faces)
    w = np.linalg.norm(points[edges[:,0]] - points[edges[:,1]], axis=1)
    i = np.concatenate([edges[:,0], edges[:,1]])
    j = np.concatenate([edges[:,1], edges[:,0]])
    d = np.concatenate([w, w])
    return csr_matrix((d, (i, j)), shape=(len(points), len(points)))

def _geodesic_mid_keep_mask(mid_V: np.ndarray, mid_F: np.ndarray, seed_idx: int, keep_pc: float) -> np.ndarray:
    """Distances on mid-surface; keep the inner keep_pc% of vertices."""
    G = _csr_from_mesh(mid_V, mid_F)
    dist = dijkstra(G, directed=False, indices=seed_idx)
    dist = np.asarray(dist).ravel()
    finite = np.isfinite(dist)
    if not finite.any():  # safety
        return np.zeros(len(mid_V), dtype=bool)
    thr = np.quantile(dist[finite], keep_pc/100.0)
    keep_mask = finite & (dist <= thr)
    return keep_mask

def _clip_by_mask_shared(verts: np.ndarray, faces: np.ndarray, keep_v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the same vertex mask to a mesh; returns (V_sub, F_sub, sub_to_old_idx)."""
    old_to_new = -np.ones(len(verts), dtype=np.int64)
    old_to_new[keep_v] = np.arange(int(keep_v.sum()), dtype=np.int64)
    Fm = old_to_new[faces]
    keepF = np.all(Fm >= 0, axis=1)
    V_sub = verts[keep_v]
    F_sub = Fm[keepF]
    sub_to_old = np.flatnonzero(keep_v)
    return V_sub, F_sub, sub_to_old

def _loop_longest(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return the vertex index loop (longest boundary)."""
    # reuse your zipper2.find_boundary_loop if you prefer:
    # return find_boundary_loop(points, faces)
    # or a local edge-count walk:
    from collections import defaultdict
    edge_count = defaultdict(int)
    for a,b,c in faces:
        for u,v in ((a,b),(b,c),(c,a)):
            if u>v: u,v=v,u
            edge_count[(u,v)] += 1
    b_edges = [(u,v) for (u,v),cnt in edge_count.items() if cnt==1]
    if not b_edges: return np.array([], dtype=int)
    adj = defaultdict(list)
    for u,v in b_edges:
        adj[u].append(v); adj[v].append(u)
    visited=set(); loops=[]
    for s in list(adj.keys()):
        if s in visited: continue
        loop=[s]; prev=None; cur=s
        while True:
            visited.add(cur)
            nxt = next((nb for nb in adj[cur] if nb!=prev), None)
            if nxt is None: break
            prev, cur = cur, nxt
            if cur==s: loops.append(np.array(loop, int)); break
            loop.append(cur)
    loops = [lp for lp in loops if lp.size>=3]
    loops.sort(key=lambda a:-a.size)
    return loops[0] if loops else np.array([], dtype=int)

def _ensure_same_ccw_robust(A_pts: np.ndarray, B_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # project both to a shared best-fit plane and enforce CCW on both
    X = np.vstack([A_pts, B_pts]); c = X.mean(0)
    U,_,Vt = np.linalg.svd(X - c, full_matrices=False)
    u,v = Vt[0], Vt[1]
    def signed_area(P):
        Q = P - c
        xy = np.c_[Q@u, Q@v]
        x,y = xy[:,0], xy[:,1]
        return 0.5 * float(np.sum(x*np.roll(y,-1) - y*np.roll(x,-1)))
    if signed_area(A_pts) < 0: A_pts = A_pts[::-1].copy()
    if signed_area(B_pts) < 0: B_pts = B_pts[::-1].copy()
    return A_pts, B_pts

def _best_circ_shift(A: np.ndarray, B: np.ndarray) -> int:
    """Shift of B that best matches A (min L2), coarse but robust."""
    n = len(A)
    best_s, best_val = 0, np.inf
    for s in range(n):
        diff = A - np.roll(B, -s, axis=0)
        val = float(np.sum(diff*diff))
        if val < best_val:
            best_val, best_s = val, s
    return best_s

def _build_side_indexed(A_pts: np.ndarray, B_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two triangles per segment with A[i] ↔ B[i]."""
    n = len(A_pts); off = n
    V = np.vstack([A_pts, B_pts])
    idx = np.arange(n, dtype=np.int32)
    upper = np.column_stack((idx, (idx+1)%n, off+idx))
    lower = np.column_stack(((idx+1)%n, off+(idx+1)%n, off+idx))
    F = np.vstack([upper, lower]).astype(np.int32)
    return V, F

# ------------- MORPHOLOGY-PRESERVING CLEANUP --------------------
def smooth_tangential_only(V: np.ndarray, F: np.ndarray, n_iter=3, lam=0.3, mu=-0.34, fixed_mask=None):
    V = V.copy()
    if fixed_mask is None: fixed_mask = np.zeros(len(V), dtype=bool)
    for _ in range(n_iter):
        N0 = compute_vertex_normals(V, F)
        V1 = taubin_smooth_once(V, F, lam, mu, fixed_mask)
        d  = V1 - V
        dot = np.einsum('ij,ij->i', d, N0)
        d_tan = d - dot[:,None] * N0
        V[~fixed_mask] += d_tan[~fixed_mask]
    return V

def _median_edge_length(V: np.ndarray, F: np.ndarray) -> float:
    E = V[F[:,[0,1,1,2,2,0]]].reshape(-1,2,3)
    if len(E)==0: return 1.0
    return float(np.median(np.linalg.norm(E[:,0]-E[:,1], axis=1)))

def remesh_isotropic_global(V: np.ndarray, F: np.ndarray,
                            target_len: float, iters: int) -> tuple[np.ndarray, np.ndarray]:
    """Whole-mesh isotropic remeshing. Prefers PyMeshLab; falls back to PyVista smooth (no remesh)."""
    try:
        ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(V, F), "mid")
        ms.apply_filter('meshing_isotropic_explicit_remeshing',
                        iterations=iters, targetlen=ml.AbsoluteValue(float(target_len)),
                        adaptivity=0.2, selectedonly=False, preserve_boundary=False)
        Vm = np.asarray(ms.current_mesh().vertex_matrix(), float)
        Fm = np.asarray(ms.current_mesh().face_matrix(),   int)
        print("PyMeshLab remesh complete.")
        return Vm, Fm
    except Exception:
        import pyvista as pv
        mesh = pv.PolyData(V, np.hstack([np.full((len(F),1),3), F]).ravel())
        mesh = mesh.smooth(n_iter=20, relaxation_factor=0.1, feature_smoothing=False)
        print("PyVista remash (backup) complete.")
        return np.asarray(mesh.points, float), F.copy()

def remesh_isotropic_fixed_rim(V: np.ndarray, F: np.ndarray, rim_idx: np.ndarray,
                               target_len: float|None=None, iters: int=4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remesh whole patch but preserve boundary vertices; then smooth ONLY interior (rim fixed).
    PyMeshLab preferred; PyVista fallback restores rim coords.
    """
    try:
        ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(V, F), "shell")
        if target_len is None: target_len = _median_edge_length(V,F) * POST_REMESH_TARGET_PC
        # 1) Remesh entire patch, but preserve boundary verts:
        ms.apply_filter('meshing_isotropic_explicit_remeshing',
                        iterations=iters, targetlen=target_len, adaptivity=0.0,
                        selectedonly=False, preserve_boundary=True)
        # 2) Smooth ONLY interior (invert rim selection):
        sel_interior = np.ones(len(V), dtype=bool); sel_interior[rim_idx] = False
        ms.set_vertex_selection_by_array(sel_interior)
        ms.apply_filter('surface_smooth_taubin', stepsmoothnum=1, lambda_=0.3, mu=-0.34, selectedonly=True)
        Vm = np.asarray(ms.current_mesh().vertex_matrix(), float)
        Fm = np.asarray(ms.current_mesh().face_matrix(), int)
        return Vm, Fm
    except Exception:
        import pyvista as pv
        mesh = pv.PolyData(V, np.hstack([np.full((len(F),1),3),F]).ravel())
        mesh = mesh.smooth(n_iter=10, relaxation_factor=0.1, feature_smoothing=True, feature_angle=50.0)
        V2 = np.asarray(mesh.points, float); V2[rim_idx] = V[rim_idx]
        return V2, F.copy()

def orient_outward_with_pymeshlab(
    ppoints: np.ndarray, pfaces: np.ndarray,
    wpoints: np.ndarray, wfaces: np.ndarray,
    spoints: np.ndarray, sfaces: np.ndarray,
    *,
    use_weld: bool = False,          # leave False to keep indices aligned with your arrays
    weld_tol: float = 1e-7,
    rays: int = 64,
    parity_sampling: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Re-orient faces for (pial, white, side) so that:
      1) face windings are coherent within each connected component
      2) normals point OUTWARD (geometry-based visibility)
    """
    Vp = ppoints.astype(float, copy=False)
    Vw = wpoints.astype(float, copy=False)
    Vs = spoints.astype(float, copy=False)

    off_w = Vp.shape[0]
    off_s = Vp.shape[0] + Vw.shape[0]

    Fp = pfaces.astype(np.int32, copy=True)
    Fw = (wfaces + off_w).astype(np.int32, copy=False)
    Fs = (sfaces + off_s).astype(np.int32, copy=False)

    V_cat = np.vstack([Vp, Vw, Vs])
    F_cat = np.vstack([Fp, Fw, Fs])

    nFp, nFw, nFs = Fp.shape[0], Fw.shape[0], Fs.shape[0]

    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(V_cat, F_cat), "zipped_shell")

    if use_weld:
        try:
            try:
                ms.apply_filter('meshing_merge_close_vertices', threshold=ml.AbsoluteValue(float(weld_tol)))
            except Exception:
                ms.apply_filter('meshing_merge_close_vertices', threshold=float(weld_tol))
        except Exception as e:
            vprint(f"[orient][warn] welding skipped: {e}")

    try:
        ms.apply_filter('meshing_re_orient_faces_coherently')
    except Exception as e:
        raise RuntimeError("PyMeshLab filter 'meshing_re_orient_faces_coherently' not available.") from e

    try:
        ms.apply_filter('meshing_re_orient_faces_by_geometry', rays=int(rays), parity_sampling=bool(parity_sampling))
    except Exception:
        try:
            ms.apply_filter('meshing_re_orient_faces_by_geometry', rays=int(rays))
        except Exception as e:
            vprint(f"[orient][warn] geometry-based outward orientation unavailable: {e}")

    F_or = np.asarray(ms.current_mesh().face_matrix(), dtype=np.int32)
    Fp_or = F_or[:nFp].copy()
    Fw_or = F_or[nFp:nFp+nFw].copy()
    Fs_or = F_or[nFp+nFw:nFp+nFw+nFs].copy()

    Fw_or -= off_w
    Fs_or -= off_s

    return Fp_or, Fw_or, Fs_or

# ---------- PREPROCESS WHOLE MID-SURFACE -----------
def preprocess_mid_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vprint(f"[pre] loading mid-surface: {path}")
    V, F = read_fs_surf(path)
    vprint(f"[pre] loaded: {len(V)} verts, {len(F)} tris")

    if PRE_REMESH_ENABLE:
        target = _median_edge_length(V, F) * PRE_REMESH_TARGET_PC
        vprint(f"[pre] global remesh → target_len={target:.4f}, iters={PRE_REMESH_ITERS}")
        V, F = remesh_isotropic_global(V, F, target_len=target, iters=PRE_REMESH_ITERS)
        vprint(f"[pre] remeshed: {len(V)} verts, {len(F)} tris")

    for i in range(PRE_TAUBIN_PASSES):
        V = taubin_smooth_once(V, F, SMOOTH_LAMBDA, SMOOTH_MU)
        vprint(f"[pre] Taubin pass {i+1}/{PRE_TAUBIN_PASSES}")

    if has_self_intersections(V, F):
        vprint("[pre][WARN] self-intersections after preprocessing; reverting to original geometry")
        V, F = read_fs_surf(path)

    vprint("[pre] done.")
    return V, F

# ---------------- FAMILY GENERATION ----------------
def prepare_patch(verts: np.ndarray, faces: np.ndarray, rng: np.random.Generator
                  ) -> Optional[tuple[np.ndarray, np.ndarray, int, np.ndarray]]:
    """Pick a seed on the mid-surface, cut a geodesic patch, keep largest component."""
    seed_idx0 = int(rng.integers(0, len(verts))); seed_point0 = verts[seed_idx0]
    vprint(f"[patch] seed_idx0={seed_idx0}  (geodesic radius={GEO_RADIUS_MM} mm)")
    mask = geodesic_ball(verts, faces, seed_idx0, GEO_RADIUS_MM)
    patch_faces = faces[np.all(mask[faces], axis=1)]
    if patch_faces.size == 0:
        vprint("[patch][fail] no faces inside geodesic ball")
        return None

    patch_verts, patch_faces, mapping = reindex_patch(verts, patch_faces)
    patch_faces = largest_component_faces(patch_faces)
    if patch_faces.size == 0:
        vprint("[patch][fail] largest connected component is empty")
        return None

    seed_idx_patch = mapping[seed_idx0]
    if seed_idx_patch < 0:
        d = np.linalg.norm(patch_verts - seed_point0, axis=1)
        seed_idx_patch = int(np.argmin(d)); seed_point0 = patch_verts[seed_idx_patch]
        vprint(f"[patch] snapped seed into patch: new seed_idx={seed_idx_patch}")

    vprint(f"[patch] patch: {len(patch_verts)} verts, {len(patch_faces)} tris")
    return patch_verts, patch_faces, seed_idx_patch, seed_point0

def generate_child_from_patch(patch_verts: np.ndarray, patch_faces: np.ndarray,
                              seed_idx_patch: int, seed_point0: np.ndarray, step_mm: float
                              ) -> Optional[tuple[dict, np.ndarray]]:
    """Generate one child (one thickness) from SAME mid patch."""
    vprint(f"[child] step={step_mm:.3f} mm → target thickness={2*step_mm:.3f} mm")

    # Offset once
    outer_pts, inner_pts, seed_idx_sub = move_vertices_in_out_once(
        patch_verts, patch_faces, seed_point0, step_mm
    )
    outer_seed_point = outer_pts[seed_idx_sub].astype(np.float32, copy=False)
    vprint(f"[child] offset done; seed_idx_sub={seed_idx_sub}")

    # --- Shared mid-surface trim (forces identical topology on pial & white) ---
    keep_mid = _geodesic_mid_keep_mask(
        mid_V=patch_verts, mid_F=patch_faces,
        seed_idx=seed_idx_patch, keep_pc=KEEP_PC
    )
    if not keep_mid.any():
        vprint("[child][fail] empty mid-mask")
        return None

    outer_pts_c, faces_outer_c, sub_to_old = _clip_by_mask_shared(outer_pts, patch_faces, keep_mid)
    inner_pts_c, faces_inner_c, _          = _clip_by_mask_shared(inner_pts, patch_faces, keep_mid)

    # Safety: faces must match 1–1 after shared clip
    if faces_outer_c.shape != faces_inner_c.shape or not np.all(faces_outer_c == faces_inner_c):
        vprint("[child][fail] subset_topology_mismatch after shared trim")
        return None
    if faces_outer_c.size == 0:
        vprint("[child][fail] empty crop after KEEP_PC trimming")
        return None

    # Pre-zip guards
    if has_self_intersections(outer_pts_c, faces_outer_c):
        vprint("[child][fail] self-intersections on pial shell")
        return None
    if has_self_intersections(inner_pts_c, faces_inner_c):
        vprint("[child][fail] self-intersections on white shell")
        return None

    # --- Boundary loops (use your zipper2 helper for longest rim) ---
    white_loop = find_boundary_loop(inner_pts_c, faces_inner_c)
    pial_loop  = find_boundary_loop(outer_pts_c, faces_outer_c)
    if len(white_loop) == 0 or len(pial_loop) == 0:
        vprint("[child][fail] could not find boundary loop(s)")
        return None
    vprint(f"[child] loop sizes → white={len(white_loop)}, pial={len(pial_loop)}")

    white_coords = inner_pts_c[white_loop]
    pial_coords  = outer_pts_c[pial_loop]

    # ---- TEL near loops (median edge length on both) ----
    w_edge_len = _edge_lengths(inner_pts_c, faces_inner_c, white_loop)
    p_edge_len = _edge_lengths(outer_pts_c, faces_outer_c, pial_loop)
    TEL = float(np.median(np.concatenate([w_edge_len, p_edge_len]))) if (len(w_edge_len) and len(p_edge_len)) else 1.0
    vprint(f"[child] TEL={TEL:.6f}")

    # ===========================================================
    #  Indexed pairing with circular-shift alignment (preferred)
    #  Fallback to arclength resampling if sets diverge
    # ===========================================================
    # Map cropped loop vertex ids back to original patch vertex ids (to align phases)
    pial_old  = sub_to_old[pial_loop]
    white_old = sub_to_old[white_loop]

    def _ensure_same_ccw_simple(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Use your existing robust CCW enforcer if present; otherwise a minimal one."""
        try:
            return _enforce_same_ccw(A, B)  # from zipper2
        except Exception:
            # Fallback: project to best-fit plane and flip if negative signed area
            X = np.vstack([A, B]); c = X.mean(0)
            _, _, Vt = np.linalg.svd(X - c, full_matrices=False)
            u, v = Vt[0], Vt[1]
            def sarea(P):
                Q = P - c
                xy = np.c_[Q@u, Q@v]
                x, y = xy[:,0], xy[:,1]
                return 0.5 * float(np.sum(x*np.roll(y,-1) - y*np.roll(x,-1)))
            A2 = A[::-1] if sarea(A) < 0 else A
            B2 = B[::-1] if sarea(B) < 0 else B
            return A2, B2

    # Can we do exact index-pairing?
    can_index = (len(pial_loop) == len(white_loop)) and (set(pial_old.tolist()) == set(white_old.tolist()))
    used_resample = False

    if can_index:
        n = len(pial_loop)

        # Build old-id -> position lookup for white loop
        pos_w = {int(vid): i for i, vid in enumerate(white_old.tolist())}
        # Sequence of white positions in pial order
        seq_w = np.array([pos_w[int(v)] for v in pial_old], dtype=np.int64)

        # Circular shift that aligns first pial old-id to position 0 in white
        s = (-int(seq_w[0])) % n
        white_loop_aligned = np.roll(white_loop, -s)
        white_coords_aligned = inner_pts_c[white_loop_aligned]

        # CCW consistency
        pial_ring, white_ring = _ensure_same_ccw_simple(pial_coords, white_coords_aligned)

        # Side band by exact index pairing
        idx = np.arange(n, dtype=np.int32); off = n
        side_points_init = np.vstack([pial_ring, white_ring])
        upper = np.column_stack((idx, (idx+1)%n, off+idx))
        lower = np.column_stack(((idx+1)%n, off+(idx+1)%n, off+idx))
        side_faces_init = np.vstack([upper, lower]).astype(np.int32)
        vprint(f"[child] side init (indexed) → {side_points_init.shape[0]} verts, {side_faces_init.shape[0]} tris")

    else:
        # --------- Fallback: arclength resampling + best circular shift ----------
        used_resample = True
        # pick n from perimeter/TEL, clamped to [64, 4096]
        Lp = _loop_perimeter(pial_coords)
        Lw = _loop_perimeter(white_coords)
        n = int(max(Lp, Lw) / max(TEL, 1e-9))
        n = int(np.clip(n, 64, 4096))

        pial_rs  = _resample_loop_arclength(pial_coords, n)
        white_rs = _resample_loop_arclength(white_coords, n)
        pial_rs, white_rs = _ensure_same_ccw_simple(pial_rs, white_rs)

        # Simple best shift: minimize L2 error
        best_s, best_val = 0, np.inf
        for s in range(n):
            diff = pial_rs - np.roll(white_rs, -s, axis=0)
            val = float(np.einsum('ij,ij->i', diff, diff).sum())
            if val < best_val:
                best_val, best_s = val, s
        white_aligned = np.roll(white_rs, -best_s, axis=0)

        idx = np.arange(n, dtype=np.int32); off = n
        side_points_init = np.vstack([pial_rs, white_aligned])
        upper = np.column_stack((idx, (idx+1)%n, off+idx))
        lower = np.column_stack(((idx+1)%n, off+(idx+1)%n, off+idx))
        side_faces_init = np.vstack([upper, lower]).astype(np.int32)
        vprint(f"[child] side init (resampled) → {side_points_init.shape[0]} verts, {side_faces_init.shape[0]} tris")

    # Early area sanity check (catches twist/degeneracy immediately)
    def _tri_area3(P3): return 0.5 * np.linalg.norm(np.cross(P3[1]-P3[0], P3[2]-P3[0]))
    _areas = np.array([_tri_area3(side_points_init[f]) for f in side_faces_init])
    if (not np.isfinite(_areas).all()) or (_areas <= 1e-14).any():
        vprint("[child][fail] zero/NaN areas in initial side strip")
        return None

    # ---- Optional: isotropic remesh (small & local) with seam locking ----
    # Save exact seam coordinates to snap back after remesh/smooth
    half = side_points_init.shape[0] // 2
    seam_pial_exact  = side_points_init[:half].copy()
    seam_white_exact = side_points_init[half:].copy()

    try:
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(side_points_init, side_faces_init), "side_init")
        side_target = float(TEL)
        ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=ml.PureValue(side_target))
        side_points = np.asarray(ms.current_mesh().vertex_matrix(), float)
        side_faces  = np.asarray(ms.current_mesh().face_matrix(),  int)
        vprint(f"[child] side remeshed → {side_points.shape[0]} verts, {side_faces.shape[0]} tris")
    except Exception as e:
        vprint(f"[child][warn] side remesh skipped ({e}); using initial strip")
        side_points, side_faces = side_points_init, side_faces_init

    # Optional: light Taubin on side (interior only)
    try:
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(side_points, side_faces), "side_remeshed")
        try:
            ms.apply_filter('meshing_surface_smooth_taubin', stepsmoothnum=2, lambda_=0.25, mu=-0.34)
        except Exception:
            ms.apply_filter('apply_coord_taubin_smoothing', stepsmoothnum=2, lambda_=0.25, mu=-0.34)
        side_points = np.asarray(ms.current_mesh().vertex_matrix(), float)
        side_faces  = np.asarray(ms.current_mesh().face_matrix(),  int)
        vprint("[child] side Taubin smooth applied")
    except Exception as e:
        vprint(f"[child][warn] side Taubin skipped ({e})")

    # ---- Snap seams back exactly (prevents boundary drift / reintroduced twist) ----
    if side_points.shape[0] >= 2*half:
        side_points[:half]  = seam_pial_exact
        side_points[half:half+half] = seam_white_exact
        vprint("[child] seam snap-back applied")

    # Final side self-intersection guard
    #if has_self_intersections(side_points, side_faces):
        #vprint("[child][fail] self-intersections on side band")
        #return None

    # ---- Face orientation (outward & coherent) ----
    try:
        pfaces_o, wfaces_o, sfaces_o = orient_outward_with_pymeshlab(
            ppoints=outer_pts_c, pfaces=faces_outer_c,
            wpoints=inner_pts_c, wfaces=faces_inner_c,
            spoints=side_points,  sfaces=side_faces,
            use_weld=False, weld_tol=1e-7, rays=64, parity_sampling=True
        )
        faces_outer_c = pfaces_o
        faces_inner_c = wfaces_o
        side_faces    = sfaces_o
        vprint("[child] face orientation: outward & coherent")
    except Exception as e:
        vprint(f"[child][fail] outward orientation failed ({e})")
        return None

    # ---- Package (BEM triangles) ----
    tris = []
    tris += faces_to_tri_dicts(outer_pts_c, faces_outer_c, 'dirichlet', 1.0)
    tris += faces_to_tri_dicts(inner_pts_c, faces_inner_c, 'dirichlet', 0.0)
    tris += faces_to_tri_dicts(side_points, side_faces, 'neumann', 0.0)

    meta = {
        'step_mm': float(step_mm),
        'thickness_mm': float(2.0 * step_mm),
        'geo_radius_mm': float(GEO_RADIUS_MM),
        'keep_pc': float(KEEP_PC),
        'n_tris': len(tris),
        'n_white_tris': int(faces_inner_c.shape[0]),
        'n_pial_tris':  int(faces_outer_c.shape[0]),
        'n_side_tris':  int(side_faces.shape[0]),
        'used_resample': bool(used_resample),
        'TEL': float(TEL),
    }
    vprint(f"[child] packaged: total_tris={meta['n_tris']} "
           f"(pial={meta['n_pial_tris']}, white={meta['n_white_tris']}, side={meta['n_side_tris']}) "
           f"resample_fallback={used_resample}")
    return ({'triangles': tris, 'meta': meta}, outer_seed_point)


def generate_family(verts: np.ndarray, faces: np.ndarray, step_list: List[float],
                    rng: np.random.Generator
                    ) -> Optional[List[tuple[float, dict, np.ndarray]]]:
    vprint("[family] preparing mid-surface patch")
    prep = prepare_patch(verts, faces, rng)
    if prep is None:
        vprint("[family][fail] patch preparation failed")
        return None

    patch_verts, patch_faces, seed_idx_patch, seed_point0 = prep
    vprint(f"[family] seed_idx_patch={seed_idx_patch}; steps={step_list}")

    # >>> NEW: patch-level Taubin smoothing (once per family)
    if PATCH_SMOOTH_ENABLE and PATCH_TAUBIN_PASSES > 0:
        vprint(f"[family] patch Taubin ×{PATCH_TAUBIN_PASSES} (λ={PATCH_SMOOTH_LAMBDA}, μ={PATCH_SMOOTH_MU})")
        patch_verts = taubin_smooth_patch(
            patch_verts, patch_faces,
            passes=PATCH_TAUBIN_PASSES,
            lam=PATCH_SMOOTH_LAMBDA,
            mu=PATCH_SMOOTH_MU,
            seed_idx=seed_idx_patch,   # keep seed stable (optional, but recommended)
        )

    results: List[tuple[float, dict, np.ndarray]] = []
    for step in step_list:
        vprint(f"[family] child begin (step={step})")
        # Children use the *smoothed* mid-surface patch
        child = generate_child_from_patch(patch_verts, patch_faces, seed_idx_patch, seed_point0, step)
        if child is None:
            vprint(f"[family][fail] child failed (step={step}); rejecting entire family")
            return None
        pack, os_point = child
        pack['meta'].update({
            'seed_idx_patch': int(seed_idx_patch),
            'seed_point_patch': seed_point0.tolist(),
        })
        results.append((step, pack, os_point))
        vprint(f"[family] child success (step={step})")

    vprint("[family] all children succeeded")
    return results

# -------------------- PROCESS WRAPPER ---------------------------
def thickness_label(thickness_mm: float) -> str:
    s = f"{thickness_mm:.3f}".rstrip('0').rstrip('.'); return s

def _worker_family(seed: int, family_idx: int, out_dir_str: str) -> int:
    try:
        rng = np.random.default_rng(seed)
        vprint(f"[worker] family={family_idx} seed={seed}")
        verts, faces = preprocess_mid_surface(SUBJ_MID)

        attempts = 0
        while attempts < MAX_ATTEMPTS_PER_FAMILY:
            attempts += 1
            vprint(f"[worker] attempt {attempts}/{MAX_ATTEMPTS_PER_FAMILY}")
            fam = generate_family(verts, faces, STEP_LIST_MM, rng)

            if fam is None:
                # family rejected; try again
                continue

            # success: write all siblings
            out_dir = Path(out_dir_str)
            for step, pack, outer_seed_point in fam:
                thickness = 2.0 * step
                tlabel = thickness_label(thickness)
                out_path = out_dir / f"{tlabel}x{int(GEO_RADIUS_MM)}_{family_idx}.pkl"
                vprint(f"[worker] writing {out_path.name}")
                write_pickle_then_pause(pack, out_path)
                write_npy_then_pause(outer_seed_point, out_path.with_name(out_path.stem + "_os.npy"))

            vprint(f"[worker] family {family_idx} written; exiting(0)")
            del fam, verts, faces
            gc.collect(); os._exit(0)

        vprint(f"[worker][fail] exhausted attempts for family {family_idx}")
        return 2  # exhausted attempts
    except Exception as e:
        vprint(f"[worker][EXC] family {family_idx} crashed: {e}")
        return 1

# -------------------------- MAIN -------------------------------
def _expected_family_files(family_idx: int) -> list[Path]:
    """Return the full list of files (pkl + _os.npy) we expect for a successful family."""
    labels = [thickness_label(2.0 * s) for s in STEP_LIST_MM]
    radius = int(GEO_RADIUS_MM)
    files = []
    for lab in labels:
        stem = f"{lab}x{radius}_{family_idx}"
        files.append(OUT_DIR / f"{stem}.pkl")
        files.append(OUT_DIR / f"{stem}_os.npy")
    return files

def _all_files_present_and_nonempty(paths: list[Path]) -> bool:
    for p in paths:
        try:
            if (not p.exists()) or (p.stat().st_size <= 0):
                return False
        except Exception:
            return False
    return True

def _cleanup_partial_family(family_idx: int) -> None:
    """Remove any files for this family index to avoid stale/partial output."""
    for p in _expected_family_files(family_idx):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass  # best-effort cleanup

def main():
    print(f"Output directory: {OUT_DIR}")  # always-on
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ctx = get_context("spawn")
    produced = 0

    while produced < N_FAMILIES:
        family_idx = produced + 1
        seed = int(np.random.default_rng().integers(0, 1 << 61))
        vprint(f"[main] start family {family_idx} with seed {seed}")

        p = ctx.Process(target=_worker_family, args=(seed, family_idx, str(OUT_DIR)))
        p.start()
        p.join()

        expected = _expected_family_files(family_idx)
        files_ok = _all_files_present_and_nonempty(expected)

        if p.exitcode == 0 and files_ok:
            labels = [thickness_label(2.0 * s) for s in STEP_LIST_MM]
            print(f"[{family_idx}/{N_FAMILIES}] saved siblings: "
                  + ", ".join(f"{lab}x{int(GEO_RADIUS_MM)}_{family_idx}.pkl" for lab in labels))
            produced += 1
        else:
            _cleanup_partial_family(family_idx)
            print(f"[{family_idx}/{N_FAMILIES}] failed (exit {p.exitcode}, files_ok={files_ok}), retrying…")

    print("\nDone.")

if __name__ == "__main__":
    main()
