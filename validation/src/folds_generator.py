#!/usr/bin/env python3
"""
Extract N cortical surface ribbon patches from FreeSurfer hemispheres, then *zip*
(their boundary loops) to form a closed triangulated patch suitable for BEM.

Changes vs previous version
---------------------------
- Loads FreeSurfer surfaces from FS_SURF_DIR (lh.pial, lh.white, rh.pial, rh.white).
- Loops to produce N_PATCHES random patches (or from a clicked voxel, if provided).
- Saves each patch in its own folder under OUT_DIR / f"patch_{k:03d}" with:
    - outer_seed.npy (RAS mm coords of the pial seed)
    - simulated_pial.vtk, simulated_white.vtk (cropped patches)
    - zipped_side.vtk (triangulated side-wall)
    - zipped_patch.pkl (triangle dicts + meta)

BC tags: pial=Dirichlet 1.0, white=Dirichlet 0.0, side=Neumann 0.0

Dependencies: numpy, scipy, nibabel, pyvista, (optional) pymeshlab
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

# ----------------------------- Config ----------------------------------------
# Geodesic + meshing
RADIUS_MM        = 15.0      # patch geodesic radius on the chosen hemisphere
SUBDIV_ITERS     = 0         # 0..2 loop subdivision after trim

# Reproducibility / clicked seed (optional)
CRITICAL_POINT_PATH = None   # e.g., "latest_clicked_coordinate.npy" (voxel coords)
RNG_SEED         = None

# --- FreeSurfer subject surfaces (input) -------------------------------------
FS_SURF_DIR = Path(r"\\wsl.localhost\Ubuntu\home\uqasnell\freesurfer_subjects\HCP103818_fs\surf")
LH_PIAL  = "lh.pial.T2"
LH_WHITE = "lh.white"
RH_PIAL  = "rh.pial.T2"
RH_WHITE = "rh.white"

# --- Output (Windows path) ---------------------------------------------------
OUT_DIR   = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\folds3")
N_PATCHES = 50

# Side remeshing/orientation
SIDE_REMESH_ITERS   = 3          # pymeshlab explicit isotropic remesh iters (if available)
OUTWARD_RAYS        = 64         # pymeshlab geometry orientation rays
PARITY_SAMPLING     = True

# ----------------------------- Imports ---------------------------------------
from pathlib import Path
from collections import defaultdict, deque
import os, pickle, tempfile, time
import numpy as np
import nibabel as nib
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import pyvista as pv

# ----------------------------- I/O helpers -----------------------------------
WRITE_PAUSE_SEC = 0.5

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_pickle_then_pause(obj, path: Path) -> None:
    path = Path(path); _ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile('wb', delete=False, dir=path.parent,
                                     prefix=path.name, suffix='.tmp') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush(); os.fsync(f.fileno()); tmp = f.name
    os.replace(tmp, path); time.sleep(WRITE_PAUSE_SEC)

def save_vtk(path: str | Path, verts: np.ndarray, faces: np.ndarray) -> None:
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32), faces])
    pv.PolyData(verts, faces_pv.ravel()).save(str(path))

# ----------------------------- Core utils ------------------------------------

def _triangle_area(P: np.ndarray) -> float:
    return 0.5 * np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))

def _dominant_axis(n: np.ndarray) -> int:
    return int(np.argmax(np.abs(n)))

def _tri2d_overlap(A: np.ndarray, B: np.ndarray) -> bool:
    """2D triangle–triangle overlap via SAT."""
    def edges(T): return np.array([T[1]-T[0], T[2]-T[1], T[0]-T[2]])
    def axes_from_edges(E): return np.stack([np.array([-e[1], e[0]]) for e in E], axis=0)
    def proj_interval(T, ax):
        dots = T @ ax
        return np.min(dots), np.max(dots)

    E_A, E_B = edges(A), edges(B)
    axes = np.vstack([axes_from_edges(E_A), axes_from_edges(E_B)])
    for ax in axes:
        n = np.linalg.norm(ax)
        if n == 0: 
            continue
        ax /= n
        a0, a1 = proj_interval(A, ax); b0, b1 = proj_interval(B, ax)
        if a1 < b0 or b1 < a0:
            return False
    return True

def _seg_tri_intersect(p0, p1, v0, v1, v2, eps=1e-12):
    # Möller–Trumbore with tolerances
    dirv = p1 - p0; e1 = v1 - v0; e2 = v2 - v0
    h = np.cross(dirv, e2); a_ = float(np.dot(e1, h))
    if abs(a_) < eps: return False
    f = 1.0 / a_; s = p0 - v0
    u = f * float(np.dot(s, h))
    if u < -eps or u > 1.0 + eps: return False
    qv = np.cross(s, e1); v = f * float(np.dot(dirv, qv))
    if v < -eps or u + v > 1.0 + eps: return False
    t = f * float(np.dot(e2, qv))
    return (-eps <= t <= 1.0 + eps)

def _tri_tri_intersect(P: np.ndarray, Q: np.ndarray, eps=1e-12) -> bool:
    """AABB reject → coplanar 2D SAT → seg–tri both ways."""
    # AABB reject
    pa, pb = P.min(0), P.max(0); qa, qb = Q.min(0), Q.max(0)
    if (pa > qb + eps).any() or (qa > pb + eps).any():
        return False

    # Coplanar?
    nP = np.cross(P[1] - P[0], P[2] - P[0]); nQ = np.cross(Q[1] - Q[0], Q[2] - Q[0])
    nP_n = np.linalg.norm(nP); nQ_n = np.linalg.norm(nQ)
    if nP_n > 0 and nQ_n > 0:
        nP_u = nP / nP_n
        dQ = (Q - P[0]) @ nP_u
        if np.max(np.abs(dQ)) < 1e-9:
            # project to dominant plane and do 2D overlap
            ax = _dominant_axis(nP_u); keep = [i for i in (0,1,2) if i != ax]
            return _tri2d_overlap(P[:, keep], Q[:, keep])

    # Non-coplanar: segment–triangle both ways
    for i in range(3):
        if _seg_tri_intersect(P[i], P[(i+1)%3], Q[0], Q[1], Q[2], eps): return True
    for i in range(3):
        if _seg_tri_intersect(Q[i], Q[(i+1)%3], P[0], P[1], P[2], eps): return True
    return False

def has_self_intersections(V: np.ndarray, F: np.ndarray, *, verbose: bool = True) -> bool:
    """
    Returns True if any non-adjacent triangles intersect.
    Tries libigl first; falls back to a BVH-based exact tester with coplanar handling.
    Prints which path was used + timings when verbose=True.
    """
    import time
    t0 = time.time()
    Vc = np.asarray(V, float, order="C"); Fc = np.asarray(F, np.int32, order="C")
    if Fc.size == 0:
        if verbose: print("[SELF-INT] empty faces -> False")
        return False

    # Cull degenerate faces (stabilizes everything)
    tri = Vc[Fc]  # (M,3,3)
    area = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1) * 0.5
    keep = area > 1e-15
    if not np.all(keep):
        tri = tri[keep]; Fc = Fc[keep]
        if tri.shape[0] == 0:
            if verbose: print("[SELF-INT] all degenerate -> False")
            return False

    # libigl fast path
    try:
        import igl
        res = igl.self_intersections(Vc, Fc)
        IF = res[0] if isinstance(res, (list, tuple)) else res
        hit = (IF is not None) and (len(IF) > 0)
        if verbose:
            dt = (time.time() - t0) * 1000
            nfaces = Fc.shape[0]
            print(f"[SELF-INT] libigl: faces={nfaces}, hit={hit} ({dt:.1f} ms)")
        return hit
    except Exception:
        pass

    # --------- BVH fallback (AABB tree) ----------
    t1 = time.time()

    # Build per-face bbox
    tri = Vc[Fc]  # ensure alignment
    mins = tri.min(axis=1); maxs = tri.max(axis=1)
    cent = (mins + maxs) * 0.5

    # Precompute vertex sets for adjacency skip
    face_sets = [set(Fc[i]) for i in range(len(Fc))]

    # BVH node struct
    class _Node:
        __slots__ = ("lo","hi","idx","left","right")
        def __init__(self, lo, hi, idx, left=None, right=None):
            self.lo=lo; self.hi=hi; self.idx=idx; self.left=left; self.right=right

    def _bbox_union(idxs):
        return np.min(mins[idxs], axis=0), np.max(maxs[idxs], axis=0)

    def _overlap(a_lo, a_hi, b_lo, b_hi, eps=0.0):
        return not ((a_lo > b_hi + eps).any() or (b_lo > a_hi + eps).any())

    # Build recursively
    LEAF = 16
    def build(indices: np.ndarray) -> _Node:
        lo, hi = _bbox_union(indices)
        if indices.size <= LEAF:
            return _Node(lo, hi, indices)
        # split along widest axis of bbox of centroids
        c = cent[indices]
        span = c.max(0) - c.min(0)
        ax = int(np.argmax(span))
        order = indices[np.argsort(c[:, ax], kind="mergesort")]
        mid = order.size // 2
        left = build(order[:mid]); right = build(order[mid:])
        # parent bbox as union
        lo_p = np.minimum(left.lo, right.lo); hi_p = np.maximum(left.hi, right.hi)
        return _Node(lo_p, hi_p, None, left, right)

    root = build(np.arange(Fc.shape[0], dtype=np.int32))

    # Traverse node pairs (self-overlap)
    checked = 0
    def traverse(A: _Node, B: _Node) -> bool:
        nonlocal checked
        if not _overlap(A.lo, A.hi, B.lo, B.hi):
            return False
        if A.idx is not None and B.idx is not None:
            idsA, idsB = A.idx, B.idx
            # if same leaf: only i<j to avoid duplicates
            if A is B:
                for a_i in range(idsA.size):
                    ia = int(idsA[a_i])
                    for b_i in range(a_i+1, idsB.size):
                        ib = int(idsB[b_i])
                        if face_sets[ia] & face_sets[ib]:
                            continue
                        # cheap per-tri bbox reject
                        if not _overlap(mins[ia], maxs[ia], mins[ib], maxs[ib]):
                            continue
                        checked += 1
                        if _tri_tri_intersect(tri[ia], tri[ib]):
                            return True
                return False
            else:
                # different leaves: full bipartite
                for ia in idsA.astype(int):
                    for ib in idsB.astype(int):
                        if ia >= ib:  # avoid double-testing symmetric pairs higher in tree
                            continue
                        if face_sets[ia] & face_sets[ib]:
                            continue
                        if not _overlap(mins[ia], maxs[ia], mins[ib], maxs[ib]):
                            continue
                        checked += 1
                        if _tri_tri_intersect(tri[ia], tri[ib]):
                            return True
                return False
        # internal nodes: recurse into the pair that is “tighter” first (slight pruning heuristic)
        if A.idx is None and B.idx is None:
            # visit the larger node split first
            sizeA = (A.hi - A.lo).sum(); sizeB = (B.hi - B.lo).sum()
            big, small = (A, B) if sizeA >= sizeB else (B, A)
            return (traverse(big.left, small) or traverse(big.right, small) or
                    traverse(A.left, B.left) or traverse(A.left, B.right) or
                    traverse(A.right, B.left) or traverse(A.right, B.right))
        elif A.idx is None:
            return traverse(A.left, B) or traverse(A.right, B)
        else:
            return traverse(A, B.left) or traverse(A, B.right)

    hit = traverse(root, root)
    if verbose:
        dt_bvh = (time.time() - t1) * 1000
        nfaces = Fc.shape[0]
        print(f"[SELF-INT] BVH: faces={nfaces}, pairs_checked={checked}, hit={hit} ({dt_bvh:.1f} ms)")
    return hit


def load_fs_surface(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read FreeSurfer surface (e.g., lh.pial) to (verts, faces)."""
    v, f = nib.freesurfer.read_geometry(str(path))
    return v.astype(np.float64, copy=False), f.astype(np.int32, copy=False)

def build_adjacency(verts: np.ndarray, faces: np.ndarray) -> csr_matrix:
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    i0 = np.concatenate([a, b, c]); j0 = np.concatenate([b, c, a])
    d0 = np.linalg.norm(verts[i0] - verts[j0], axis=1)
    i  = np.concatenate([i0, j0]); j = np.concatenate([j0, i0])
    d  = np.concatenate([d0, d0])
    return coo_matrix((d, (i, j)), shape=(len(verts), len(verts))).tocsr()

def nearest_vertex_index(verts: np.ndarray, point_xyz: np.ndarray) -> int:
    return int(np.linalg.norm(verts - point_xyz, axis=1).argmin())

def face_centroids(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return verts[faces].mean(axis=1)

def build_face_neighbours_vertex(faces: np.ndarray) -> list[set[int]]:
    v2faces: dict[int, list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        v2faces[a].append(fi); v2faces[b].append(fi); v2faces[c].append(fi)
    neigh = [set() for _ in range(len(faces))]
    for fi, (a, b, c) in enumerate(faces):
        s = set(v2faces[a]) | set(v2faces[b]) | set(v2faces[c])
        s.discard(fi); neigh[fi] = s
    return neigh

def clip_patch(verts: np.ndarray, faces: np.ndarray, seed_idx: int, radius_mm: float, adj: csr_matrix):
    dist = dijkstra(adj, indices=seed_idx, directed=False, unweighted=False)
    keep_v = np.isfinite(dist) & (dist <= radius_mm)
    remap = -np.ones(len(verts), dtype=int); remap[keep_v] = np.arange(keep_v.sum())
    mask = keep_v[faces].all(axis=1)
    return verts[keep_v], remap[faces[mask]]

# ---- Trim faces with <2 shared edges ----------------------------------------

def best_circular_shift(A: np.ndarray, B: np.ndarray) -> int:
    """
    Return s in [0,n) minimizing sum_i ||A[i] - B[(i+s)%n]||^2.
    A,B: (n,3)
    """
    n = len(A)
    # Vectorized brute-force over shifts: O(n^2) but fine for n<=4096
    # Expand A to (n,1,3), roll B for each shift to (n,n,3), then L2
    d2 = np.empty(n, dtype=np.float64)
    for s in range(n):
        C = np.roll(B, s, axis=0)
        diff = A - C
        d2[s] = np.einsum('ij,ij->', diff, diff)
    return int(d2.argmin())

def _edge_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)

def _count_shared_edges(faces: np.ndarray) -> np.ndarray:
    edge_cnt: dict[tuple[int, int], int] = defaultdict(int)
    for a, b, c in faces:
        edge_cnt[_edge_key(a, b)] += 1
        edge_cnt[_edge_key(b, c)] += 1
        edge_cnt[_edge_key(c, a)] += 1
    shared = np.zeros(len(faces), dtype=np.int32)
    for k, (a, b, c) in enumerate(faces):
        shared[k] = (
            (edge_cnt[_edge_key(a, b)] >= 2) +
            (edge_cnt[_edge_key(b, c)] >= 2) +
            (edge_cnt[_edge_key(c, a)] >= 2)
        )
    return shared

def trim_low_edge_sharing(verts: np.ndarray, faces: np.ndarray, min_shared_edges: int = 2) -> tuple[np.ndarray, np.ndarray]:
    f = faces
    while True:
        keep = _count_shared_edges(f) >= min_shared_edges
        if keep.all():
            break
        f = f[keep]
    unique_v = np.unique(f)
    remap = -np.ones(verts.shape[0], dtype=np.int32); remap[unique_v] = np.arange(len(unique_v))
    return verts[unique_v], remap[f]

# ---- White patch via inward dilation ----------------------------------------

def dilate_inwards(face_neigh: list[set[int]], face_dist: np.ndarray, seeds: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    accepted, visited = set(), set()
    q = deque(sorted(seeds, key=lambda f: face_dist[f], reverse=True))
    while q:
        f = q.popleft()
        if f in visited: continue
        visited.add(f); accepted.add(f)
        df = face_dist[f]
        for nb in face_neigh[f]:
            if nb not in visited and face_dist[nb] < df - eps:
                q.append(nb)
    return np.array(sorted(accepted), dtype=int)

# ---- Optional subdivision ----------------------------------------------------

def subdivide_mesh(verts: np.ndarray, faces: np.ndarray, iters: int = 1) -> tuple[np.ndarray, np.ndarray]:
    if iters <= 0: return verts, faces
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32), faces])
    mesh = pv.PolyData(verts, faces_pv.ravel()).subdivide(iters, subfilter="loop")
    return np.asarray(mesh.points), mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)

# ----------------------------- Zipping utils ---------------------------------

def boundary_loop_indices(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    # boundary edges appear once
    edges = defaultdict(int)
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            edges[e] += 1
    b_edges = [e for e, k in edges.items() if k == 1]
    if not b_edges:
        raise ValueError("No boundary edges found.")
    adj = defaultdict(list)
    for u, v in b_edges:
        adj[u].append(v); adj[v].append(u)
    visited = set(); loops: list[list[int]] = []
    for s in adj.keys():
        if s in visited: continue
        loop = [s]; visited.add(s)
        prev, cur = -1, s
        while True:
            nbrs = adj[cur]
            nxt = None
            for w in nbrs:
                if w == prev: continue
                nxt = w; break
            if nxt is None: break
            if nxt == loop[0]:
                loop.append(nxt); break
            loop.append(nxt); visited.add(nxt); prev, cur = cur, nxt
        if len(loop) > 2:
            if loop[0] == loop[-1]: loop = loop[:-1]
            loops.append(loop)
    if not loops: raise ValueError("Failed to assemble boundary loops.")
    return np.array(max(loops, key=len), dtype=int)

def loop_perimeter(coords: np.ndarray) -> float:
    dif = coords[(np.arange(len(coords))+1)%len(coords)] - coords
    return float(np.sum(np.linalg.norm(dif, axis=1)))

def resample_loop_arclength(coords: np.ndarray, n: int) -> np.ndarray:
    P = coords
    seg = np.linalg.norm(P[(np.arange(len(P))+1)%len(P)] - P, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    t = np.linspace(0.0, L, n+1)[:-1]
    out = np.empty((n, 3), dtype=float)
    j = 0
    for i, ti in enumerate(t):
        while j < len(P)-1 and s[j+1] < ti:
            j += 1
        a, b = s[j], s[j+1]
        u = 0.0 if b == a else (ti - a)/(b - a)
        p0, p1 = P[j], P[(j+1)%len(P)]
        out[i] = (1.0 - u)*p0 + u*p1
    return out

def _fit_plane_basis(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    return c, Vt[0], Vt[1]

def _signed_area_2d(XY: np.ndarray) -> float:
    x, y = XY[:,0], XY[:,1]
    return 0.5*float(np.sum(x*np.roll(y,-1) - y*np.roll(x,-1)))

def enforce_same_ccw(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c, u, v = _fit_plane_basis(np.vstack([A, B]))
    def proj(P):
        Q = P - c
        return np.stack([Q@u, Q@v], axis=1)
    a_ccw = _signed_area_2d(proj(A)) > 0
    b_ccw = _signed_area_2d(proj(B)) > 0
    if a_ccw != b_ccw:
        B = B[::-1].copy()
    return A, B

def build_side_strip(pial_loop: np.ndarray, white_loop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(pial_loop)
    pts = np.vstack([pial_loop, white_loop])
    off = n; idx = np.arange(n, dtype=int)
    upper = np.column_stack((idx, (idx+1)%n, off + idx))
    lower = np.column_stack(((idx+1)%n, off + (idx+1)%n, off + idx))
    faces = np.vstack([upper, lower]).astype(np.int32)
    return pts, faces

# ---- Pymeshlab helpers (optional) -------------------------------------------

def remesh_isotropic(verts: np.ndarray, faces: np.ndarray, target_len: float, iters: int = 3) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pymeshlab as ml
        ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(verts, faces), "side")
        try:
            target = ml.PureValue(float(target_len))
        except Exception:
            target = float(target_len)
        ms.apply_filter('meshing_isotropic_explicit_remeshing', iterations=int(iters), targetlen=target)
        V = np.asarray(ms.current_mesh().vertex_matrix(), float)
        F = np.asarray(ms.current_mesh().face_matrix(), int)
        return V, F
    except Exception:
        return verts, faces

def orient_outward_with_pymeshlab(ppoints: np.ndarray, pfaces: np.ndarray,
                                  wpoints: np.ndarray, wfaces: np.ndarray,
                                  spoints: np.ndarray, sfaces: np.ndarray,
                                  rays: int = 64, parity_sampling: bool = True,
                                  use_weld: bool = False, weld_tol: float = 1e-7
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import pymeshlab as ml
    except Exception:
        return pfaces, wfaces, sfaces
    Vp, Vw, Vs = ppoints, wpoints, spoints
    off_w = Vp.shape[0]; off_s = off_w + Vw.shape[0]
    Fp = pfaces.copy(); Fw = (wfaces + off_w).astype(np.int32); Fs = (sfaces + off_s).astype(np.int32)
    V = np.vstack([Vp, Vw, Vs]); F = np.vstack([Fp, Fw, Fs])
    ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(V, F), "zipped")
    if use_weld:
        try:
            try:
                ms.apply_filter('meshing_merge_close_vertices', threshold=ml.AbsoluteValue(float(weld_tol)))
            except Exception:
                ms.apply_filter('meshing_merge_close_vertices', threshold=float(weld_tol))
        except Exception:
            pass
    try:
        ms.apply_filter('meshing_re_orient_faces_coherently')
    except Exception:
        return pfaces, wfaces, sfaces
    try:
        ms.apply_filter('meshing_re_orient_faces_by_geometry', rays=int(rays), parity_sampling=bool(parity_sampling))
    except Exception:
        try:
            ms.apply_filter('meshing_re_orient_faces_by_geometry', rays=int(rays))
        except Exception:
            pass
    F_or = np.asarray(ms.current_mesh().face_matrix(), dtype=np.int32)
    nFp, nFw = Fp.shape[0], Fw.shape[0]
    Fp_or = F_or[:nFp].copy(); Fw_or = F_or[nFp:nFp+nFw].copy(); Fs_or = F_or[nFp+nFw:nFp+nFw+Fs.shape[0]].copy()
    Fw_or -= off_w; Fs_or -= off_s
    return Fp_or, Fw_or, Fs_or

# ---- Packaging ---------------------------------------------------------------

def faces_to_tri_dicts(points: np.ndarray, faces: np.ndarray, bc_type: str, bc_value: float, atol: float=1e-12) -> list[dict]:
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

# ----------------------------- Seeds -----------------------------------------

def choose_seeds(surfaces: dict[str, tuple[np.ndarray, np.ndarray]],
                 affine: np.ndarray | None,
                 clicked_voxel_path: str | Path | None,
                 rng: np.random.Generator) -> tuple[str, int, str, int]:
    hemi_map = {"lh_pial": "lh_white", "rh_pial": "rh_white"}
    pial_keys = ["lh_pial", "rh_pial"]
    if clicked_voxel_path and affine is not None and Path(clicked_voxel_path).exists():
        critical_voxel = np.load(clicked_voxel_path).astype(float)
        critical_point = nib.affines.apply_affine(affine, critical_voxel)
        choices = []
        for key in pial_keys:
            verts, _ = surfaces[key]
            idx = nearest_vertex_index(verts, critical_point)
            d = np.linalg.norm(verts[idx] - critical_point)
            choices.append((d, key, idx))
        _, outer_key, outer_idx = min(choices, key=lambda t: t[0])
    else:
        outer_key = rng.choice(pial_keys)
        outer_idx = int(rng.integers(surfaces[outer_key][0].shape[0]))
    inner_key = hemi_map[outer_key]
    inner_idx = outer_idx  # FS pial/white often share vertex ordering; ok as a starting assumption
    return outer_key, outer_idx, inner_key, inner_idx

# ----------------------------- One-patch pipeline ----------------------------

def build_one_patch(surfaces, adjacency, RADIUS_MM, SUBDIV_ITERS, rng) -> dict | None:
    affine = None
    outer_key, outer_idx, inner_key, inner_idx = choose_seeds(surfaces, affine, CRITICAL_POINT_PATH, rng)
    outer_seed = surfaces[outer_key][0][outer_idx]

    # Extract pial patch by radius
    outer_v, outer_f = clip_patch(*surfaces[outer_key], outer_idx, RADIUS_MM, adjacency[outer_key])

    # White patch via inward dilation
    white_v_all, white_f_all = surfaces[inner_key]
    pial_centroids  = face_centroids(outer_v, outer_f)
    white_centroids = face_centroids(white_v_all, white_f_all)
    _, nn_idx = cKDTree(white_centroids).query(pial_centroids, k=1)
    dist_inner_v  = dijkstra(adjacency[inner_key], indices=inner_idx, directed=False, unweighted=False)
    face_dist_in  = dist_inner_v[white_f_all].min(axis=1)
    seed_faces    = np.unique(nn_idx)
    seed_faces    = seed_faces[face_dist_in[seed_faces] <= RADIUS_MM]
    face_neigh    = build_face_neighbours_vertex(white_f_all)
    white_patch_id= dilate_inwards(face_neigh, face_dist_in, seed_faces)
    if white_patch_id.size == 0:
        return None

    uniq_wv = np.unique(white_f_all[white_patch_id])
    w_remap = -np.ones(len(white_v_all), dtype=int); w_remap[uniq_wv] = np.arange(len(uniq_wv))
    inner_v = white_v_all[uniq_wv]
    inner_f = w_remap[white_f_all[white_patch_id]]

    # Trim dangling triangles
    outer_v, outer_f = trim_low_edge_sharing(outer_v, outer_f, min_shared_edges=2)
    inner_v, inner_f = trim_low_edge_sharing(inner_v, inner_f, min_shared_edges=2)
    if outer_f.size == 0 or inner_f.size == 0:
        return None
    
    # After trim:
    if has_self_intersections(outer_v, outer_f): return None
    if has_self_intersections(inner_v, inner_f): return None

    # Optional refinement
    outer_v, outer_f = subdivide_mesh(outer_v, outer_f, iters=SUBDIV_ITERS)
    inner_v, inner_f = subdivide_mesh(inner_v, inner_f, iters=SUBDIV_ITERS)

    # Zipping
    try:
        p_loop_idx = boundary_loop_indices(outer_v, outer_f)
        w_loop_idx = boundary_loop_indices(inner_v, inner_f)
    except Exception:
        return None
    p_loop = outer_v[p_loop_idx]
    w_loop = inner_v[w_loop_idx]

    def _edge_lengths(P: np.ndarray) -> np.ndarray:
        dif = P[(np.arange(len(P))+1)%len(P)] - P
        return np.linalg.norm(dif, axis=1)
    el = np.concatenate([_edge_lengths(p_loop), _edge_lengths(w_loop)])
    tel = float(np.median(el)) if el.size else max(1e-3, float(loop_perimeter(p_loop)/max(len(p_loop),8)))

    Lp = loop_perimeter(p_loop); Lw = loop_perimeter(w_loop)
    n  = int(max(Lp, Lw) / max(tel, 1e-9)); n = int(np.clip(n, 64, 4096))
    p_rs = resample_loop_arclength(p_loop, n)
    w_rs = resample_loop_arclength(w_loop, n)
    p_rs, w_rs = enforce_same_ccw(p_rs, w_rs)

    s = best_circular_shift(p_rs, w_rs)
    w_rs = np.roll(w_rs, s, axis=0)
    # now build the side strip as before
    side_points, side_faces = build_side_strip(p_rs, w_rs)

    side_points, side_faces = build_side_strip(p_rs, w_rs)
    side_points, side_faces = remesh_isotropic(side_points, side_faces, target_len=tel, iters=SIDE_REMESH_ITERS)

    pfaces_o, wfaces_o, sfaces_o = orient_outward_with_pymeshlab(
        ppoints=outer_v, pfaces=outer_f,
        wpoints=inner_v, wfaces=inner_f,
        spoints=side_points, sfaces=side_faces,
        rays=OUTWARD_RAYS, parity_sampling=PARITY_SAMPLING,
        use_weld=False, weld_tol=1e-7,
    )

    tris = []
    tris += faces_to_tri_dicts(outer_v, pfaces_o, 'dirichlet', 1.0)
    tris += faces_to_tri_dicts(inner_v, wfaces_o, 'dirichlet', 0.0)
    tris += faces_to_tri_dicts(side_points, sfaces_o, 'neumann', 0.0)

    return {
        'outer_seed': outer_seed.astype(np.float32, copy=False),
        'outer_v': outer_v, 'outer_f': pfaces_o,
        'inner_v': inner_v, 'inner_f': wfaces_o,
        'side_v':  side_points, 'side_f': sfaces_o,
        'meta': {
            'radius_mm': float(RADIUS_MM),
            'n': int(n), 'tel_estimate': float(tel),
            'subdiv_iters': int(SUBDIV_ITERS),
        }
    }

# ----------------------------- Main ------------------------------------------

def main() -> None:
    out_root = Path(OUT_DIR); _ensure_dir(out_root)

    # Load FS surfaces
    surf_dir = Path(FS_SURF_DIR)
    lh_pial  = load_fs_surface(surf_dir / LH_PIAL)
    lh_white = load_fs_surface(surf_dir / LH_WHITE)
    rh_pial  = load_fs_surface(surf_dir / RH_PIAL)
    rh_white = load_fs_surface(surf_dir / RH_WHITE)

    surfaces = {
        'lh_pial': lh_pial,  'rh_pial': rh_pial,
        'lh_white': lh_white,'rh_white': rh_white,
    }
    adjacency = {k: build_adjacency(*surf) for k, surf in surfaces.items()}

    rng = np.random.default_rng(RNG_SEED)
    produced = 0; attempts = 0; MAX_ATTEMPTS = N_PATCHES * 20

    while produced < N_PATCHES and attempts < MAX_ATTEMPTS:
        attempts += 1
        pack = build_one_patch(surfaces, adjacency, RADIUS_MM, SUBDIV_ITERS, rng)
        if pack is None:
            continue
        k = produced
        out_dir = out_root
        _ensure_dir(out_dir)
        # seeds & meshes
        np.save(out_dir / f'outer_seed_{k:03d}.npy', pack['outer_seed'])

        # package triangles
        tris = []
        tris += faces_to_tri_dicts(pack['outer_v'], pack['outer_f'], 'dirichlet', 1.0)
        tris += faces_to_tri_dicts(pack['inner_v'], pack['inner_f'], 'dirichlet', 0.0)
        tris += faces_to_tri_dicts(pack['side_v'],  pack['side_f'],  'neumann', 0.0)
        meta = pack['meta'] | {
            'n_tris_total': int(len(tris)),
            'n_pial_tris': int(pack['outer_f'].shape[0]),
            'n_white_tris': int(pack['inner_f'].shape[0]),
            'n_side_tris': int(pack['side_f'].shape[0]),
        }
        write_pickle_then_pause({'triangles': tris, 'meta': meta}, out_dir / f'zipped_patch_{k:03d}.pkl')
        produced += 1
        print(f"[{produced}/{N_PATCHES}] -> {out_dir}")

    if produced < N_PATCHES:
        print(f"Finished early: produced {produced}/{N_PATCHES} after {attempts} attempts.")
    else:
        print("Done.")

if __name__ == '__main__':
    main()
