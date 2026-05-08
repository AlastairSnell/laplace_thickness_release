
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import nibabel as nib
import pyvista as pv
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

# Optional deps
try:
    import pymeshlab as _ms
    HAS_PYMESHLAB = True
except Exception:  # pragma: no cover
    _ms = None
    HAS_PYMESHLAB = False

try:
    import trimesh as _trimesh
    HAS_TRIMESH = True
except Exception:  # pragma: no cover
    _trimesh = None
    HAS_TRIMESH = False

# ------------------------------
# Small utilities & types
# ------------------------------

DEFAULT_SELF_INTERSECTION_TOL = 0.05

@dataclass
class PatchParams:
    radius_mm: float = 100.0
    subdiv_iters: int = 0
    min_n: int = 128
    max_n: int = 4096
    side_remesh_iters: int = 4
    outward_rays: int = 128
    parity_sampling: bool = True
    save_diags: bool = False
    protect_pw_intersect: bool = True
    allow_self_intersection: bool = True
    self_intersection_tolerance: float = DEFAULT_SELF_INTERSECTION_TOL


@dataclass
class PatchResult:
    success: bool
    reason: str
    seed_hemi: str
    seed_idx: int
    timings: Dict[str, float]
    counts: Dict[str, int]
    resample_n: int
    tel_estimate: float
    out_paths: Dict[str, str]


# ------------------------------
# IO & FS helpers
# ------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _repo_root() -> Path:
    # File is validation/scripts/patch_maker.py -> repo root is 2 parents up.
    return Path(__file__).resolve().parents[2]


# ------------------------------
# Surface loading & conversions
# ------------------------------

def load_fs_surface(fs_surf_dir: Path, hemi: str, kind: str = "pial") -> pv.PolyData:
    """Load FreeSurfer surface file (e.g., lh.pial) into pyvista PolyData.

    Tries standard and *.T2 variants. On failure, raises a FileNotFoundError
    with a helpful directory listing.
    """
    fname_candidates = [
        fs_surf_dir / f"{hemi}.{kind}",
        fs_surf_dir / f"{hemi}.{kind}.T2",
    ]
    for cand in fname_candidates:
        if cand.exists():
            coords, faces = nib.freesurfer.read_geometry(str(cand))
            poly = pv.PolyData(coords)
            faces_vtk = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
            poly.faces = faces_vtk
            poly.clear_data()
            return poly

    # Not found — build a helpful message
    try:
        listing = sorted(p.name for p in fs_surf_dir.iterdir())
    except Exception:
        listing = ["<unavailable>"]
    tried = ", ".join(str(c) for c in fname_candidates)
    raise FileNotFoundError(
        f"Could not find {hemi}.{kind} in {fs_surf_dir}. Tried: {tried}."
        f"Directory contents: {listing}"
    )


# ------------------------------
# Topology & graph utilities
# ------------------------------

def build_adjacency(verts: np.ndarray, faces: np.ndarray) -> csr_matrix:
    """Build symmetric CSR adjacency with edge weights = Euclidean edge length.

    Parameters
    ----------
    verts : (V,3) float
    faces : (F,3) int
    Returns
    -------
    csr_matrix shape (V,V)
    """
    I: List[int] = []
    J: List[int] = []
    W: List[float] = []

    def add_edge(a: int, b: int):
        I.append(a); J.append(b)
        W.append(float(np.linalg.norm(verts[a] - verts[b])))

    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        add_edge(a, b); add_edge(b, a)
        add_edge(b, c); add_edge(c, b)
        add_edge(c, a); add_edge(a, c)
    n = verts.shape[0]
    return csr_matrix((W, (I, J)), shape=(n, n))


def build_vertex_adjacency(faces: np.ndarray, n_verts: int) -> List[List[int]]:
    """Build unweighted vertex adjacency list from faces."""
    adj: List[List[int]] = [[] for _ in range(n_verts)]
    for a, b, c in faces:
        a = int(a); b = int(b); c = int(c)
        adj[a].extend([b, c])
        adj[b].extend([a, c])
        adj[c].extend([a, b])
    # Deduplicate while preserving deterministic order
    out: List[List[int]] = []
    for nbrs in adj:
        if not nbrs:
            out.append([])
            continue
        uniq = sorted(set(nbrs))
        out.append(uniq)
    return out


def _poly_to_numpy(mesh: pv.PolyData) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(mesh.points, dtype=np.float64)
    faces = mesh.faces.reshape(-1, 4)[:, 1:4].astype(np.int64)
    return pts, faces


def _print_patch_summary(path: Path) -> int:
    if not path.exists():
        print(f"Missing patch file: {path}", file=sys.stderr)
        return 2
    mesh = pv.read(str(path))
    print(f"Path: {path}")
    print(f"Points: {mesh.n_points}")
    print(f"Cells: {mesh.n_cells}")
    keys = list(mesh.cell_data.keys())
    print(f"Cell data arrays: {keys}")
    for key in ["part_id", "bc_type", "bc_value", "normal"]:
        if key in mesh.cell_data:
            arr = mesh.cell_data[key]
            print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
            if key in ("part_id", "bc_type"):
                print(f"  unique: {np.unique(arr)}")
    return 0


# ------------------------------
# Geodesic mask (Preferred A on mid)
# ------------------------------

def geodesic_mask_mid(vp: np.ndarray, vw: np.ndarray, faces: np.ndarray,
                      seed_idx: int, radius_mm: float) -> np.ndarray:
    """Compute vertex mask on mid-surface using Dijkstra geodesic radius.

    Returns boolean mask keep_v of shape (V,).
    """
    vm = 0.5 * (vp + vw)
    G = build_adjacency(vm, faces)
    dist = dijkstra(G, directed=False, indices=[seed_idx], unweighted=False, limit=radius_mm)
    d = dist[0]
    keep_v = np.isfinite(d) & (d <= radius_mm)
    return keep_v


def geodesic_mask_single(verts: np.ndarray, faces: np.ndarray,
                         seed_idx: int, radius_mm: float) -> np.ndarray:
    G = build_adjacency(verts, faces)
    dist = dijkstra(G, directed=False, indices=[seed_idx], unweighted=False, limit=radius_mm)
    d = dist[0]
    keep_v = np.isfinite(d) & (d <= radius_mm)
    return keep_v


# ------------------------------
# Subsetting, trimming, boundaries
# ------------------------------

def clip_by_mask(verts: np.ndarray, faces: np.ndarray, keep_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subset mesh by vertex mask, return (V_sub, F_sub, old_to_new_idx).

    Faces with any vertex outside mask are dropped.
    """
    keep_idx = np.flatnonzero(keep_v)
    map_old_to_new = -np.ones(len(keep_v), dtype=np.int64)
    map_old_to_new[keep_idx] = np.arange(keep_idx.size, dtype=np.int64)

    f_keep_mask = keep_v[faces].all(axis=1)
    faces_sub_old = faces[f_keep_mask]
    faces_sub = map_old_to_new[faces_sub_old]
    verts_sub = verts[keep_idx]
    return verts_sub, faces_sub, map_old_to_new


def trim_low_edge_sharing(faces: np.ndarray) -> np.ndarray:
    """Remove faces that share <2 edges with neighbors (dangling/whiskers).

    Returns a mask over faces to keep.
    """
    # Edge map: canonical (min,max) per undirected edge
    from collections import defaultdict
    edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        edges = [(a, b), (b, c), (c, a)]
        for u, v in edges:
            if u > v:
                u, v = v, u
            edge_to_faces[(u, v)].append(fi)

    # Count how many of a face's edges are shared by another face
    shared_counts = np.zeros(faces.shape[0], dtype=np.int32)
    for flist in edge_to_faces.values():
        if len(flist) >= 2:
            for fi in flist:
                shared_counts[fi] += 1
    keep = shared_counts >= 2
    return keep


def boundary_loop_indices(faces: np.ndarray) -> List[np.ndarray]:
    """Extract boundary loops as ordered vertex index arrays.

    Strategy: find all edges with count == 1 (boundary). Build adjacency on boundary
    edges and walk to form ordered closed loops.
    """
    from collections import defaultdict, deque

    edge_count: Dict[Tuple[int, int], int] = defaultdict(int)
    # Count undirected edges
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edge_count[(u, v)] += 1

    # Boundary edges are those seen exactly once
    boundary_edges = [(u, v) for (u, v), cnt in edge_count.items() if cnt == 1]
    if not boundary_edges:
        return []

    # Build directed adjacency for boundary walk (both directions)
    adj: Dict[int, List[int]] = defaultdict(list)
    for u, v in boundary_edges:
        adj[u].append(v)
        adj[v].append(u)

    visited = set()
    loops: List[np.ndarray] = []

    # Deterministic order for reproducibility and consistent pairing
    for k in adj:
        adj[k].sort()

    for start in sorted(adj.keys()):
        if start in visited:
            continue
        # Walk loop
        loop = []
        prev = None
        cur = start
        while True:
            loop.append(cur)
            visited.add(cur)
            nbrs = adj[cur]
            # Choose neighbor not equal to prev; if bifurcation, pick the one that continues boundary
            nxt = None
            for nb in nbrs:
                if nb != prev:
                    nxt = nb
                    break
            if nxt is None:
                break
            prev, cur = cur, nxt
            if cur == start:
                # Close the loop and store
                loops.append(np.array(loop, dtype=np.int64))
                break

    # Post-process: ensure loops are unique and not trivial
    loops = [lp for lp in loops if lp.size >= 3]
    # Keep the single longest loop first
    loops.sort(key=lambda a: -a.size)
    return loops


def largest_component_mask(mask: np.ndarray, adj: List[List[int]]) -> np.ndarray:
    """Return a mask of the largest connected component within the input mask."""
    n = mask.size
    visited = np.zeros(n, dtype=bool)
    best = []
    for start in np.flatnonzero(mask):
        if visited[start]:
            continue
        stack = [int(start)]
        comp = []
        visited[start] = True
        while stack:
            v = stack.pop()
            comp.append(v)
            for nb in adj[v]:
                if mask[nb] and not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(comp) > len(best):
            best = comp
    out = np.zeros(n, dtype=bool)
    if best:
        out[np.array(best, dtype=np.int64)] = True
    return out


def fill_mask_holes(mask: np.ndarray, adj: List[List[int]], max_iters: int = 10) -> np.ndarray:
    """Heuristic hole fill: iteratively add vertices whose neighbors are all in-mask."""
    filled = mask.copy()
    for _ in range(max_iters):
        changed = False
        for v in np.flatnonzero(~filled):
            nbrs = adj[int(v)]
            if not nbrs:
                continue
            if all(filled[n] for n in nbrs):
                filled[int(v)] = True
                changed = True
        if not changed:
            break
    return filled

def parse_surface_ras(s: str) -> np.ndarray:
    """
    Parse --surface-ras input. Expected form: "[x,y,z]".
    Accepts minor variations in whitespace.
    """
    try:
        v = json.loads(s)
        arr = np.asarray(v, dtype=float).reshape(3)
        return arr
    except Exception as e:
        raise ValueError(f'Could not parse --surface-ras. Use e.g. --surface-ras "[12.3,-45.6,78.9]". Got: {s!r}') from e

def snap_surface_ras_to_seed(surface_ras: np.ndarray,
                             hemi_pairs: Dict[str, Tuple[pv.PolyData, pv.PolyData]],
                             available_hemis: List[str]) -> Tuple[str, int, float]:
    """
    Return (hemi, seed_idx, dist_mm) by snapping to nearest pial vertex across hemis.
    """
    best = None  # (dist, hemi, idx)
    for hemi in available_hemis:
        pial, _ = hemi_pairs[hemi]
        pts = np.asarray(pial.points, dtype=np.float64)
        tree = cKDTree(pts)
        dist, idx = tree.query(surface_ras.astype(np.float64), k=1)
        cand = (float(dist), hemi, int(idx))
        if best is None or cand[0] < best[0]:
            best = cand
    assert best is not None
    return best[1], best[2], best[0]

# ------------------------------
# Loop orientation & resampling
# ------------------------------

def _loop_ccw_3d(points: np.ndarray) -> Tuple[bool, np.ndarray]:
    """Determine CCW of a closed 3D loop by projecting to best-fit plane.

    Returns (is_ccw, normal).
    """
    # PCA for plane
    P = points
    P0 = P.mean(axis=0)
    Q = P - P0
    U, S, Vt = np.linalg.svd(Q, full_matrices=False)
    # Normal is last right-singular vector
    n = Vt[-1]
    # Build 2D basis (first two columns of Vt)
    e1 = Vt[0]
    e2 = Vt[1]
    xy = np.c_[Q @ e1, Q @ e2]
    # Signed area via shoelace
    x, y = xy[:, 0], xy[:, 1]
    area2 = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    is_ccw = area2 > 0
    return bool(is_ccw), n


def ensure_same_ccw(loopA_pts: np.ndarray, loopB_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flip loop order so both are CCW in the same projection sense."""
    ccwA, nA = _loop_ccw_3d(loopA_pts)
    ccwB, nB = _loop_ccw_3d(loopB_pts)
    A = loopA_pts if ccwA else loopA_pts[::-1]
    B = loopB_pts if ccwB else loopB_pts[::-1]
    # If normals oppose strongly, flip B to match A
    if np.dot(nA, nB) < 0:
        B = B[::-1]
    return A, B


def resample_loop_arclength(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polyline (loop) by arclength to n points."""
    # Compute cumulative arclength for each segment between successive vertices
    seglens = np.linalg.norm(np.diff(points, axis=0, append=points[:1]), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglens)])
    L = cum[-1]
    targets = np.linspace(0, L, n + 1)[:-1]

    # Interpolate along segments
    pts = []
    i = 0
    for t in targets:
        while cum[i+1] < t:
            i += 1
        t0, t1 = cum[i], cum[i+1]
        alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        p = (1 - alpha) * points[i] + alpha * points[(i + 1) % len(points)]
        pts.append(p)
    return np.asarray(pts)


# ------------------------------
# Side wall builders
# ------------------------------

def build_side_strip_indexed(loopA_idx: np.ndarray, loopB_idx: np.ndarray,
                             A_pts: np.ndarray, B_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build side wall triangles by pairing A[k] with B[k] (indexed same length/order).

    Returns (verts, faces) where verts stacks A then B (or as unique rows), and faces are triangles.
    """
    assert loopA_idx.size == loopB_idx.size
    n = loopA_idx.size
    # Build quads between A[i]->A[i+1] and B[i]->B[i+1]; triangulate
    # We will produce 2*n triangles
    # To keep vertex count small, use unique concatenation of A and B separately
    verts = np.vstack([A_pts, B_pts])
    faces = []
    for i in range(n):
        a0 = i
        a1 = (i + 1) % n
        b0 = n + i
        b1 = n + (i + 1) % n
        # two triangles: (a0, a1, b0) and (a1, b1, b0)
        faces.append([a0, a1, b0])
        faces.append([a1, b1, b0])
    return verts, np.asarray(faces, dtype=np.int64)


def _best_circular_shift(A: np.ndarray, B: np.ndarray, max_check: Optional[int] = None) -> int:
    """Find circular shift s of B minimizing sum of squared distances |A[i] - B[(i+s)%n]|^2.
    Cap computations for large n.
    """
    n = A.shape[0]
    if max_check is None:
        max_check = min(n, 2048)
    # Sample evenly if n is huge
    stride = max(1, n // max_check)
    best_s = 0
    best_val = np.inf
    # Pre-center to reduce numeric errors
    for s in range(0, n, stride):
        diff = A - np.roll(B, s, axis=0)
        val = float(np.sum(np.einsum('ij,ij->i', diff, diff)))
        if val < best_val:
            best_val = val
            best_s = s
    return best_s


def _best_index_alignment(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, int, bool, int]:
    """Align index loops by shift/reverse to maximize exact matches.

    Returns (B_aligned, shift, reversed, match_count).
    """
    n = A.size
    best = (B, 0, False, -1)
    for rev in (False, True):
        Bb = B[::-1] if rev else B
        for s in range(n):
            matches = int(np.sum(A == np.roll(Bb, s)))
            if matches > best[3]:
                best = (np.roll(Bb, s), s, rev, matches)
            if matches == n:
                return best
    return best


def build_side_strip_resampled(loopA_pts: np.ndarray, loopB_pts: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resample both loops to n points and build paired strip (with best circular shift)."""
    A = resample_loop_arclength(loopA_pts, n)
    B = resample_loop_arclength(loopB_pts, n)
    s = _best_circular_shift(A, B)
    B = np.roll(B, s, axis=0)
    return build_side_strip_indexed(np.arange(n), np.arange(n), A, B)

# ------------------------------
# Optional: Remesh & orientation via PyMeshLab
# ------------------------------

def remesh_isotropic(verts: np.ndarray, faces: np.ndarray, target_len: float, iters: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isotropic explicit remeshing via PyMeshLab (optional).
    Uses version-agnostic targetlen (AbsoluteValue/PureValue/Percentage).
    Gracefully no-ops if PyMeshLab unavailable or filter not present.
    """
    if not HAS_PYMESHLAB or iters <= 0:
        return verts, faces
    try:
        import pymeshlab as ml
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(verts, faces), "side_init")
        targetlen = float(target_len)
        ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=ml.PureValue(targetlen))
        out = ms.current_mesh()
        V = np.asarray(out.vertex_matrix(), dtype=np.float64)
        F = np.asarray(out.face_matrix(), dtype=np.int64)
        return V, F

    except Exception as e:
        print(f"[remesh] PyMeshLab remesh skipped ({e}); keeping original side strip.")
        return verts, faces
    
def orient_outward(verts: np.ndarray, faces: np.ndarray, ref_normal_dir: Optional[np.ndarray] = None) -> np.ndarray:
    """Ensure faces are oriented so that the average normal aligns with ref_normal_dir if provided.
    Returns possibly flipped faces array.
    """
    # Compute average normal
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    n_mean = n.mean(axis=0)
    if ref_normal_dir is None:
        return faces
    if np.dot(n_mean, ref_normal_dir) < 0:
        faces = faces[:, [0, 2, 1]]
    return faces


# ------------------------------
# Self-intersection testing
# ------------------------------

def _tri_bbox(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.min(pts, axis=0), np.max(pts, axis=0)


def _bbox_overlap(a: Tuple[np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray]) -> bool:
    amin, amax = a; bmin, bmax = b
    return bool(np.all(amax >= bmin) and np.all(bmax >= amin))


def _seg_intersect(p: np.ndarray, p2: np.ndarray, q: np.ndarray, q2: np.ndarray) -> Tuple[bool, float, float]:
    """Segment intersection in 3D via closest points; returns (is_close, s, t)."""
    u = p2 - p
    v = q2 - q
    w0 = p - q
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)
    denom = a*c - b*b
    if denom == 0.0:
        return False, 0.0, 0.0
    s = (b*e - c*d) / denom
    t = (a*e - b*d) / denom
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    cp1 = p + s*u
    cp2 = q + t*v
    close = np.linalg.norm(cp1 - cp2) < 1e-6
    return close, float(s), float(t)


def _tri_tri_intersect(A: np.ndarray, B: np.ndarray) -> bool:
    """Robust-ish triangle-triangle intersection test using SAT + segment checks.
    A, B each (3,3).
    """
    # Based on Möller (1997)-style approach, simplified for small patches.
    # First quick bbox rejection
    if not _bbox_overlap(_tri_bbox(A), _tri_bbox(B)):
        return False

    # Plane of A
    nA = np.cross(A[1]-A[0], A[2]-A[0])
    nB = np.cross(B[1]-B[0], B[2]-B[0])
    if np.linalg.norm(nA) < 1e-12 or np.linalg.norm(nB) < 1e-12:
        return False

    def signed_dist(P, p):
        return float(np.dot(P - A[0], nA))

    # Distances of B vertices to plane of A
    dB = np.array([signed_dist(b, A[0]) for b in B])
    if np.all(dB > 1e-9) or np.all(dB < -1e-9):
        # Entire B on one side of A's plane
        pass

    # Check edges cross
    edgesA = [(A[0], A[1]), (A[1], A[2]), (A[2], A[0])]
    edgesB = [(B[0], B[1]), (B[1], B[2]), (B[2], B[0])]
    for p0, p1 in edgesA:
        for q0, q1 in edgesB:
            inter, _, _ = _seg_intersect(p0, p1, q0, q1)
            if inter:
                return True

    # As a fallback, test if a vertex is inside the other tri by barycentric coords
    def point_in_tri(P, T):
        v0 = T[2]-T[0]; v1 = T[1]-T[0]; v2 = P - T[0]
        d00 = np.dot(v0, v0); d01 = np.dot(v0, v1); d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0); d21 = np.dot(v2, v1)
        denom = d00*d11 - d01*d01
        if abs(denom) < 1e-18:
            return False
        v = (d11*d20 - d01*d21) / denom
        w = (d00*d21 - d01*d20) / denom
        u = 1.0 - v - w
        return (u >= -1e-8) and (v >= -1e-8) and (w >= -1e-8)

    if any(point_in_tri(a, B) for a in A) or any(point_in_tri(b, A) for b in B):
        return True

    return False


def self_intersection_details(verts: np.ndarray, faces: np.ndarray) -> Tuple[bool, str, int]:
    """Return (has_intersection, method, overlap_tri_count).

    method is "trimesh" when available, otherwise "tri_tri".
    overlap_tri_count is the number of unique triangles participating in any intersection.
    """
    if faces.size == 0:
        return False, "none", 0

    if HAS_TRIMESH:
        try:
            mesh = _trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            tris = mesh.triangles
            tree = mesh.triangles_tree  # requires rtree
            face_verts = [set(f.tolist()) for f in faces]
            overlapping = set()
            for i, tri in enumerate(tris):
                bounds = np.hstack([tri.min(axis=0), tri.max(axis=0)])
                for j in tree.intersection(bounds):
                    if j <= i:
                        continue
                    if not face_verts[i].isdisjoint(face_verts[j]):
                        continue
                    if _tri_tri_intersect(tri, tris[j]):
                        overlapping.add(i)
                        overlapping.add(j)
            return (len(overlapping) > 0), "trimesh", len(overlapping)
        except Exception:
            pass

    # Fallback: BVH via KDTree on centroids + tri-tri test ignoring adjacent faces
    tri_pts0 = verts[faces[:, 0]]
    tri_pts1 = verts[faces[:, 1]]
    tri_pts2 = verts[faces[:, 2]]
    tri_cent = (tri_pts0 + tri_pts1 + tri_pts2) / 3.0
    tree = cKDTree(tri_cent)

    # Build vertex adjacency per face to skip neighbors sharing a vertex
    face_verts = [set(f.tolist()) for f in faces]

    # Neighborhood radius: median triangle circumradius approximate
    edge_lens = np.linalg.norm(tri_pts1 - tri_pts0, axis=1)
    r = float(np.median(edge_lens)) * 2.5 if edge_lens.size else 1.0

    overlapping = set()
    for i in range(faces.shape[0]):
        idxs = tree.query_ball_point(tri_cent[i], r)
        Ai = np.vstack([tri_pts0[i], tri_pts1[i], tri_pts2[i]])
        for j in idxs:
            if j <= i:
                continue
            # Skip if share a vertex
            if not face_verts[i].isdisjoint(face_verts[j]):
                continue
            Bj = np.vstack([tri_pts0[j], tri_pts1[j], tri_pts2[j]])
            if _tri_tri_intersect(Ai, Bj):
                overlapping.add(i)
                overlapping.add(j)
    return (len(overlapping) > 0), "tri_tri", len(overlapping)

def has_cross_intersections(VA: np.ndarray, FA: np.ndarray,
                            VB: np.ndarray, FB: np.ndarray,
                            *, verbose: bool = True) -> bool:
    """
    Detect intersections BETWEEN two meshes A (VA,FA) and B (VB,FB).
    Uses a centroid KD-tree on B with a conservative radius and exact tri-tri tests.
    Treats 'touching' as intersecting (protective).
    """
    if FA.size == 0 or FB.size == 0:
        if verbose: print("[X-INT] one mesh empty -> False")
        return False

    # Cull degenerate faces
    TA = VA[FA]; TB = VB[FB]
    areaA = np.linalg.norm(np.cross(TA[:,1]-TA[:,0], TA[:,2]-TA[:,0]), axis=1) * 0.5
    areaB = np.linalg.norm(np.cross(TB[:,1]-TB[:,0], TB[:,2]-TB[:,0]), axis=1) * 0.5
    keepA = areaA > 1e-15; keepB = areaB > 1e-15
    if not keepA.any() or not keepB.any():
        if verbose: print("[X-INT] all degenerate -> False")
        return False
    FA = FA[keepA]; FB = FB[keepB]
    TA = VA[FA];    TB = VB[FB]

    # Centroids + KDTree on B
    from scipy.spatial import cKDTree
    centA = TA.mean(axis=1); centB = TB.mean(axis=1)
    treeB = cKDTree(centB)

    # Conservative neighborhood radius from median edge lengths
    def _edge_lengths(T):
        e0 = np.linalg.norm(T[:,1]-T[:,0], axis=1)
        e1 = np.linalg.norm(T[:,2]-T[:,1], axis=1)
        e2 = np.linalg.norm(T[:,0]-T[:,2], axis=1)
        return np.concatenate([e0, e1, e2])
    medA = float(np.median(_edge_lengths(TA))) if TA.shape[0] else 1.0
    medB = float(np.median(_edge_lengths(TB))) if TB.shape[0] else 1.0
    rad  = max(1e-6, 2.5 * max(medA, medB))  # generous to not miss near misses

    # Quick bboxes for B faces
    minsB = TB.min(axis=1); maxsB = TB.max(axis=1)

    checked = 0
    for i, Pa in enumerate(TA):
        # candidate B faces near centroid of A's tri
        cand = treeB.query_ball_point(centA[i], r=rad)
        if not cand:
            continue
        # further prune by bbox vs Pa's bbox
        minA, maxA = Pa.min(axis=0), Pa.max(axis=0)
        for j in cand:
            # AABB reject
            if (minA > maxsB[j]).any() or (minsB[j] > maxA).any():
                continue
            checked += 1
            if _tri_tri_intersect(Pa, TB[j]):
                if verbose:
                    print(f"[X-INT] hit after {checked} tests (A#{i} vs B#{j})")
                return True
    if verbose:
        print(f"[X-INT] no hit, pairs_checked={checked}")
    return False

# ------------------------------
# Packaging for BEM
# ------------------------------

def _face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n / ln


def faces_to_tri_dicts(name: str, verts: np.ndarray, faces: np.ndarray,
                       bc_type: str, bc_value: float) -> List[Dict[str, np.ndarray]]:
    normals = _face_normals(verts, faces)
    out = []
    for fi, f in enumerate(faces):
        tri = verts[f]
        out.append({
            "vertices": tri.astype(np.float32),
            "normal": normals[fi].astype(np.float32),
            "bc_type": bc_type,
            "bc_value": float(bc_value),
            "name": name,
        })
    return out


# ------------------------------
# Core pipeline per patch
# ------------------------------

def _pv_make(verts: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    faces_vtk = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    # Build with faces in the constructor to avoid implicit vertex cells.
    return pv.PolyData(verts.copy(), faces_vtk)


def _save_poly(mesh: pv.PolyData, path: Path) -> None:
    mesh.save(str(path))

def _append_polydata(meshes: Sequence[pv.PolyData]) -> pv.PolyData:
    if not meshes:
        raise ValueError("No meshes to append.")
    if hasattr(pv, "append_polydata"):
        return pv.append_polydata(list(meshes))
    merged = meshes[0].copy()
    for m in meshes[1:]:
        merged = merged.merge(m, merge_points=False)
    return merged


def build_one_patch(
    hemi: str,
    pial: pv.PolyData,
    white: pv.PolyData,
    seed_idx: int,
    params: PatchParams,
    patch_idx: int,
    out_dir: Path,
    out_file: Optional[Path],
    rng: random.Random,
    keep_v_override: Optional[np.ndarray] = None,
) -> PatchResult:
    t0 = time.time()
    timings: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    # Extract raw arrays
    Vp, F = _poly_to_numpy(pial)
    Vw, Fw = _poly_to_numpy(white)
    # By construction from _load_hemi_pair, faces are harmonised already.
    # Keep a light sanity check on vertex counts.
    if Vp.shape[0] != Vw.shape[0]:
        return PatchResult(False, 'vertex_count_mismatch', hemi, seed_idx, {}, {}, 0, 0.0, {})

    timings['load'] = time.time() - t0

    # A) mid-surface geodesic mask (unless overridden)
    tA = time.time()
    if keep_v_override is not None:
        keep_v = keep_v_override
        path_used = 'override_mask'
        timings['geodesic'] = time.time() - tA
    else:
        try:
            keep_v = geodesic_mask_mid(Vp, Vw, F, seed_idx, params.radius_mm)
            path_used = 'A_mid_geodesic'
        except Exception as e:
            keep_v = None  # type: ignore
            path_used = f'A_failed:{e}'
        timings['geodesic'] = time.time() - tA

        # Fallback B) pial-only geodesic mask
        if keep_v is None or not np.any(keep_v):
            tB = time.time()
            keep_v = geodesic_mask_single(Vp, F, seed_idx, params.radius_mm)
            path_used = 'B_pial_geodesic'
            timings['geodesic'] += time.time() - tB

    # Subset
    tS = time.time()
    Vp_sub, F_sub, map_old_to_new = clip_by_mask(Vp, F, keep_v)
    Vw_sub, Fw_sub, _ = clip_by_mask(Vw, F, keep_v)

    # Safety: faces must match after subsetting
    if F_sub.shape != Fw_sub.shape or not np.all(F_sub == Fw_sub):
        return PatchResult(False, 'subset_topology_mismatch', hemi, seed_idx, timings, counts, 0, 0.0, {})

    # Trim whiskers
    keep_faces = trim_low_edge_sharing(F_sub)
    Fp = F_sub[keep_faces]
    Fw2 = Fw_sub[keep_faces]
    Vp2 = Vp_sub
    Vw2 = Vw_sub
    counts['faces_after_trim'] = int(Fp.shape[0])
    if Fp.shape[0] == 0:
        return PatchResult(False, 'empty_after_trim', hemi, seed_idx, timings, counts, 0, 0.0, {})
    timings['trim'] = time.time() - tS

    # Reject if pial and white intersect
    if params.protect_pw_intersect and has_cross_intersections(Vp2, Fp, Vw2, Fw2, verbose=False):
        return PatchResult(False, 'pial_white_intersection', hemi, seed_idx, timings, counts, 0, 0.0, {})

    # Boundary loops
    tBdry = time.time()
    loops_p = boundary_loop_indices(Fp)
    loops_w = boundary_loop_indices(Fw2)
    if not loops_p or not loops_w:
        return PatchResult(False, 'no_boundary_loop', hemi, seed_idx, timings, counts, 0, 0.0, {})
    Lp = loops_p[0]  # longest
    Lw = loops_w[0]

    # Build boundary points (in submesh index space)
    Lp_pts = Vp2[Lp]
    Lw_pts = Vw2[Lw]

    # Try preferred indexed pairing by mapping via original vertex indices
    # Build reverse maps from submesh index to original
    # We kept Vp2=Vp_sub which was vertices of keep_v; original->sub map is map_old_to_new
    # To recover original indices for loop vertices, we need inverse mapping: sub->original
    sub_to_old = np.flatnonzero(keep_v)
    Lp_old = sub_to_old[Lp]
    Lw_old = sub_to_old[Lw]

    # If sets match exactly, attempt to align orders by circular shift using old ids
    resample_n = 0
    side_verts = None
    side_faces = None
    try_indexed = set(Lp_old.tolist()) == set(Lw_old.tolist()) and (Lp.size == Lw.size)

    if try_indexed:
        n = Lp.size
        Lw_old_aligned, shift, rev, match = _best_index_alignment(Lp_old, Lw_old)
        if match == n:
            # Align loop point order to matched indices
            Lw_aligned = np.roll(Lw, shift)
            if rev:
                Lw_aligned = Lw_aligned[::-1]
            Lw_pts_aligned = np.roll(Lw_pts, shift, axis=0)
            if rev:
                Lw_pts_aligned = Lw_pts_aligned[::-1]
            # Enforce CCW consistency
            Ap, Bw = ensure_same_ccw(Lp_pts, Lw_pts_aligned)
            side_verts, side_faces = build_side_strip_indexed(np.arange(n), np.arange(n), Ap, Bw)
            resample_n = n
        else:
            try_indexed = False

    if not try_indexed:
        # Resample route with clamped n
        # Choose n from median of both boundary lengths (edge counts)
        n0 = int((Lp.size + Lw.size) // 2)
        n0 = max(params.min_n, min(params.max_n, n0))
        Ap, Bw = ensure_same_ccw(Lp_pts, Lw_pts)
        side_verts, side_faces = build_side_strip_resampled(Ap, Bw, n0)
        resample_n = n0

    # Estimate target edge length from boundary median edge length
    be_A = np.linalg.norm(np.diff(Ap, axis=0, append=Ap[:1]), axis=1)
    be_B = np.linalg.norm(np.diff(Bw, axis=0, append=Bw[:1]), axis=1)
    tel = float(np.median(np.r_[be_A, be_B])) if be_A.size and be_B.size else 1.0

    # Optional isotropic remeshing of side wall
    tRem = time.time()
    sideV, sideF = remesh_isotropic(side_verts, side_faces, target_len=tel, iters=params.side_remesh_iters)
    timings['remesh_side'] = time.time() - tRem

    # Orient pial outward, white inward consistency
    # Use approximate outward direction = mean(Vp2 - Vw2)
    outward_dir = (Vp2.mean(axis=0) - Vw2.mean(axis=0))
    if np.linalg.norm(outward_dir) < 1e-9:
        outward_dir = np.array([0.0, 0.0, 1.0])

    # Ensure consistent orientation
    Fp_or = orient_outward(Vp2, Fp, ref_normal_dir=outward_dir)
    Fw_or = orient_outward(Vw2, Fw2, ref_normal_dir=-outward_dir)
    Fs_or = orient_outward(sideV, sideF, ref_normal_dir=outward_dir)

    # Build pyvista meshes for intersection checks and output
    mp = _pv_make(Vp2, Fp_or)
    mw = _pv_make(Vw2, Fw_or)
    ms = _pv_make(sideV, Fs_or)  # keep building; we just won't fail on it

    # Validate self-intersections (CAPS ONLY)
    if not params.allow_self_intersection:
        tInt = time.time()
        p_has, p_method, p_overlap = self_intersection_details(*_poly_to_numpy(mp))
        w_has, w_method, w_overlap = self_intersection_details(*_poly_to_numpy(mw))
        s_has, s_method, s_overlap = self_intersection_details(*_poly_to_numpy(ms))
        tol = max(0.0, min(1.0, params.self_intersection_tolerance))
        p_tris = int(Fp_or.shape[0])
        w_tris = int(Fw_or.shape[0])
        s_tris = int(Fs_or.shape[0])
        p_limit = int(np.floor(tol * p_tris))
        w_limit = int(np.floor(tol * w_tris))
        s_limit = int(np.floor(tol * s_tris))
        p_fail = p_has and (p_overlap > p_limit)
        w_fail = w_has and (w_overlap > w_limit)
        s_fail = s_has and (s_overlap > s_limit)
        si_caps = p_fail or w_fail or s_fail
        timings['self_intersections'] = time.time() - tInt

        if si_caps:
            failed = []
            if p_fail:
                failed.append("pial")
            if w_fail:
                failed.append("white")
            if s_fail:
                failed.append("side")
            failed_str = ",".join(failed) if failed else "none"
            print(
                f"[self_intersection] failed={failed_str} "
                f"pial(method={p_method}, overlap_tris={p_overlap}/{p_tris}, limit={p_limit}) "
                f"white(method={w_method}, overlap_tris={w_overlap}/{w_tris}, limit={w_limit}) "
                f"side(method={s_method}, overlap_tris={s_overlap}/{s_tris}, limit={s_limit})"
            )
            counts['pial_tris'] = p_tris
            counts['white_tris'] = w_tris
            counts['side_tris'] = s_tris
            return PatchResult(False, 'self_intersection', hemi, seed_idx, timings, counts, resample_n, tel, {})
    else:
        timings['self_intersections'] = 0.0


    # ------------------------------
    # VTK-only output (single labelled patch mesh)
    # part_id: 1=pial, 2=white, 3=side
    # ------------------------------
    counts['pial_tris'] = int(Fp_or.shape[0])
    counts['white_tris'] = int(Fw_or.shape[0])
    counts['side_tris'] = int(Fs_or.shape[0])

    patch_tag = f"{patch_idx:03d}"

    # Label each component by cell
    mp.cell_data['part_id'] = np.full(mp.n_cells, 1, dtype=np.int32)
    mw.cell_data['part_id'] = np.full(mw.n_cells, 2, dtype=np.int32)
    ms.cell_data['part_id'] = np.full(ms.n_cells, 3, dtype=np.int32)
    # Boundary conditions for main.py (string: "dirichlet" / "neumann")
    mp.cell_data['bc_type'] = np.full(mp.n_cells, "dirichlet", dtype="U9")
    mw.cell_data['bc_type'] = np.full(mw.n_cells, "dirichlet", dtype="U9")
    ms.cell_data['bc_type'] = np.full(ms.n_cells, "neumann", dtype="U7")
    # bc_value: pial=1.0, white=0.0, side=0.0 (no-flux Neumann)
    mp.cell_data['bc_value'] = np.full(mp.n_cells, 1.0, dtype=np.float32)
    mw.cell_data['bc_value'] = np.zeros(mw.n_cells, dtype=np.float32)
    ms.cell_data['bc_value'] = np.zeros(ms.n_cells, dtype=np.float32)
    # Cell normals for main.py (required CellData array)
    mp.cell_data['normal'] = _face_normals(Vp2, Fp_or).astype(np.float32)
    mw.cell_data['normal'] = _face_normals(Vw2, Fw_or).astype(np.float32)
    ms.cell_data['normal'] = _face_normals(sideV, Fs_or).astype(np.float32)

    patch_mesh = _append_polydata([mp, mw, ms])

    if out_file is None:
        vtk_path = out_dir / f"zipped_patch_{patch_tag}.vtk"
    else:
        vtk_path = out_file
        if vtk_path.suffix == "":
            vtk_path = vtk_path.with_suffix(".vtk")
    patch_mesh.save(str(vtk_path))

    out_paths = {'vtk': str(vtk_path)}

    timings['total'] = time.time() - t0
    return PatchResult(True, 'ok', hemi, seed_idx, timings, counts, resample_n, tel, out_paths)


# ------------------------------
# CLI / Main driver
# ------------------------------

def _load_hemi_pair(fs_surf_dir: Path, hemi: str) -> Tuple[pv.PolyData, pv.PolyData]:
    try:
        pial = load_fs_surface(fs_surf_dir, hemi, 'pial')
        white = load_fs_surface(fs_surf_dir, hemi, 'white')
    except Exception as e:
        raise RuntimeError(f"Failed to load {hemi} surfaces: {e}") from e

    Vp, Fp = _poly_to_numpy(pial)
    Vw, Fw = _poly_to_numpy(white)

    # Must have 1–1 vertex correspondence
    if Vp.shape[0] != Vw.shape[0]:
        raise RuntimeError(
            f"Vertex count mismatch for {hemi}: pial={Vp.shape[0]} vs white={Vw.shape[0]}"
        )

    # Force a common triangulation if faces differ (this happens in some pipelines)
    if Fp.shape != Fw.shape or not np.all(Fp == Fw):
        F_common = Fw  # prefer white’s faces as the common topology
        pial  = _pv_make(Vp, F_common)
        white = _pv_make(Vw, F_common)
    return pial, white


def _load_aparc_mask(aparc_dir: Path, hemi: str, region_name: str, n_verts: int) -> np.ndarray:
    annot_path = aparc_dir / f"{hemi}.aparc.annot"
    if not annot_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annot_path}")

    def _mask_from_labels(labels: np.ndarray, ctab: np.ndarray, names_in: Sequence[str]) -> np.ndarray:
        names_norm = [n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n) for n in names_in]
        name_map = {n.lower(): i for i, n in enumerate(names_norm)}
        key = region_name.lower()
        if key not in name_map:
            raise ValueError(f"Region name '{region_name}' not found in {annot_path}")
        name_idx = name_map[key]
        label_id = ctab[name_idx, -1]
        mask_idx = labels == name_idx
        mask_id = labels == label_id
        if mask_idx.any() or mask_id.any():
            return mask_idx if mask_idx.sum() >= mask_id.sum() else mask_id
        return mask_idx

    labels, ctab, names = nib.freesurfer.read_annot(str(annot_path), orig_ids=False)
    mask = _mask_from_labels(labels, ctab, names)
    if not mask.any():
        labels, ctab, names = nib.freesurfer.read_annot(str(annot_path), orig_ids=True)
        mask = _mask_from_labels(labels, ctab, names)
    if mask.size != n_verts:
        raise RuntimeError(
            f"Annotation vertex count mismatch: annot={mask.size} surf={n_verts}"
        )
    return mask

def _random_seed_idx(mesh: pv.PolyData, rng: random.Random) -> int:
    return int(rng.randrange(0, mesh.n_points))

# ---- Minimal, effective diagnostics ---------------------------------
from collections import Counter, defaultdict

REASON_CODE = {
    'ok': 'OK',
    'self_intersection': 'SI',
    'pial_white_intersection': 'PWX',
    'no_boundary_loop': 'NBL',
    'subset_topology_mismatch': 'STM',
    'empty_after_trim': 'EAT',
    'vertex_count_mismatch': 'VCM',
    'exception': 'EXC',
}

class Telemetry:
    def __init__(self, goal, summary_every=50):
        self.goal = goal
        self.summary_every = summary_every
        self.reasons = Counter()
        self.by_hemi = Counter()
        self.hot_seeds = defaultdict(Counter)
        self.attempts = 0
        self.produced = 0
        self.t0 = time.time()

    def _r(self, reason: str) -> str:
        return REASON_CODE.get(reason, reason[:3].upper())

    def _dom_stage(self, timings: dict) -> str:
        if not timings:
            return "-"
        k = max(timings, key=timings.get)
        return f"{k}:{timings[k]:.2f}s"

    def log_exception(self, hemi: str, seed_idx: int, exc: Exception):
        self.reasons['exception'] += 1
        self.by_hemi[hemi] += 1
        self.hot_seeds[hemi][seed_idx] += 1
        code = self._r('exception')
        print(f"[A#{self.attempts} P={self.produced}/{self.goal}] {hemi} seed={seed_idx} {code} err={exc}")

    def log_result(self, res):
        # counters
        self.reasons[res.reason] += 1
        self.by_hemi[res.seed_hemi] += 1
        if not res.success:
            self.hot_seeds[res.seed_hemi][res.seed_idx] += 1

        # one compact line
        code = self._r(res.reason)
        ttot = res.timings.get('total', 0.0)
        dom = self._dom_stage(res.timings)
        tris = f"{res.counts.get('pial_tris',0)}/{res.counts.get('white_tris',0)}/{res.counts.get('side_tris',0)}"
        rn = res.resample_n or 0
        tel = f"{res.tel_estimate:.3f}" if res.tel_estimate else "-"

        if res.success:
            self.produced += 1
            print(f"[A#{self.attempts} P={self.produced}/{self.goal}] {res.seed_hemi} seed={res.seed_idx} {code} "
                  f"t={ttot:.2f}s tris(p/w/s)={tris} n={rn} tel={tel} dom={dom}")
        else:
            print(f"[A#{self.attempts} P={self.produced}/{self.goal}] {res.seed_hemi} seed={res.seed_idx} {code} "
                  f"t={ttot:.2f}s n={rn} tel={tel} dom={dom} [REJ]")

        # periodic summary
        if (self.attempts % self.summary_every) == 0:
            self.summary(tag="PERIODIC")

    def summary(self, tag="FINAL"):
        dt = time.time() - self.t0
        succ = self.reasons.get('ok', 0)
        rate = (succ / max(1, self.attempts)) * 100.0
        # top 3 reject reasons (exclude 'ok')
        rejects = [(k, v) for k, v in self.reasons.items() if k != 'ok']
        rejects.sort(key=lambda kv: -kv[1])
        top3 = ", ".join(f"{self._r(k)}:{v}" for k, v in rejects[:3]) or "-"

        print(f"[{tag}] attempts={self.attempts} produced={self.produced}/{self.goal} "
              f"succ={succ} hitrate={rate:.1f}% elapsed={dt:.1f}s")
        print(f"[{tag}] top_rejects: {top3}")
        for h, cnt in self.hot_seeds.items():
            common = cnt.most_common(3)
            if common:
                topseeds = ", ".join(f"{sid}:{c}" for sid, c in common)
                print(f"[{tag}] worst_seeds {h}: {topseeds}")
# ---------------------------------------------------------------------

def _has_expert_flag(argv: Sequence[str]) -> bool:
    return "--expert" in argv


def build_argparser(*, expert: bool) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Cortical Ribbon Patch Zipper")
    ap.add_argument("--expert", action="store_true", help="Show/enable expert options in --help.")
    ap.add_argument(
        '--fs-surf-dir',
        type=Path,
        required=False,
        default=(_repo_root() / "validation/data/freesurfer_subjects/FCD_P04_fs/surf"),
        help='Directory containing lh.pial, lh.white, rh.pial, rh.white',
    )
    ap.add_argument(
        '--out-dir',
        type=Path,
        required=False,
        default=(_repo_root() / "validation/data/new_folds"),
    )
    ap.add_argument(
        '--out-file',
        type=Path,
        required=False,
        default=None,
        help='Optional explicit output VTK file path. If provided, overrides --out-dir.',
    )
    ap.add_argument('--radius-mm', type=float, default=50.0)
    ap.add_argument('--subdiv-iters', type=int, default=0)
    ap.add_argument('--rng-seed', type=int, default=None)
    ap.add_argument('--min-n', type=int, default=128)
    ap.add_argument('--max-n', type=int, default=4096)
    ap.add_argument('--side-remesh-iters', type=int, default=0)
    ap.add_argument('--outward-rays', type=int, default=128)
    ap.add_argument('--parity-sampling', action='store_true', default=False)
    ap.add_argument('--save-diags', action='store_true', default=False)
    si_group = ap.add_mutually_exclusive_group()
    si_group.add_argument(
        '--block-self-intersection',
        action='store_true',
        default=False,
        help='Reject patches with self-intersections (default: allow).',
    )
    si_group.add_argument(
        '--allow-self-intersection',
        action='store_true',
        default=False,
        help=argparse.SUPPRESS,
    )
    ap.add_argument('--region-name', type=str, default=None,
                    help='Use aparc.annotation region (e.g., supramarginal) instead of radius/seed')
    ap.add_argument('--aparc-dir', type=Path, default=None,
                    help='Directory containing lh.aparc.annot/rh.aparc.annot (defaults to ../label)')
    ap.add_argument('--hemi', type=str, choices=['lh', 'rh'], default=None,
                    help='Hemisphere to use when --region-name is set')
    ap.add_argument('--self-test', action='store_true', help='Run built-in acceptance tests')
    ap.add_argument('--surface-ras', type=str, default=None,
                help='Surface RAS coordinate copied from Freeview, format: "[x,y,z]"')
    ap.add_argument(
        '--summarize',
        action='store_true',
        default=False,
        help='Print a summary of the patch produced in this run',
    )
    if expert:
        g = ap.add_argument_group("Expert options")
        g.add_argument(
            '--self-intersection-tolerance',
            type=float,
            default=DEFAULT_SELF_INTERSECTION_TOL,
            help=(
                'Allowed fraction of triangles participating in self-intersections per surface (0-1). '
                'Only used with --block-self-intersection.'
            ),
        )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    expert = _has_expert_flag(argv)
    ap = build_argparser(expert=expert)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        if not expert:
            expert_only_tokens = {
                "--self-intersection-tolerance",
            }
            if any(tok in argv for tok in expert_only_tokens):
                print("\nNote: some advanced options are only available with --expert.")
        raise

    if args.self_test:
        return _self_test()

    if getattr(args, "allow_self_intersection", False):
        print(
            "Note: --allow-self-intersection is now the default; "
            "use --block-self-intersection to reject.",
            file=sys.stderr,
        )

    block_self_intersection = bool(getattr(args, "block_self_intersection", False))
    allow_self_intersection = not block_self_intersection
    self_intersection_tolerance = float(
        getattr(args, "self_intersection_tolerance", DEFAULT_SELF_INTERSECTION_TOL)
    )
    if allow_self_intersection and ("--self-intersection-tolerance" in argv):
        print(
            "Note: --self-intersection-tolerance only applies with --block-self-intersection.",
            file=sys.stderr,
        )

    
    fs_surf_dir: Path = args.fs_surf_dir
    if not fs_surf_dir.exists():
        print(f"Surface dir not found: {fs_surf_dir}", file=sys.stderr)
        return 2
    out_dir: Path = args.out_dir
    _ensure_dir(out_dir)
    out_file: Optional[Path] = args.out_file
    if out_file is not None:
        _ensure_dir(out_file.parent)

    rng = random.SystemRandom() if args.rng_seed is None else random.Random(args.rng_seed)

    # Load hemis
    hemis = ['lh', 'rh']
    # Attempt to load both hemispheres, but continue if only one exists
    load_errors = {}
    try:
        pial_lh, white_lh = _load_hemi_pair(fs_surf_dir, 'lh')
    except Exception as e:
        load_errors['lh'] = str(e)
        pial_lh = white_lh = None  # type: ignore
    try:
        pial_rh, white_rh = _load_hemi_pair(fs_surf_dir, 'rh')
    except Exception as e:
        load_errors['rh'] = str(e)
        pial_rh = white_rh = None  # type: ignore

    hemi_pairs = {
        'lh': (pial_lh, white_lh),
        'rh': (pial_rh, white_rh),
    }
    available_hemis = [h for h, (p, w) in hemi_pairs.items() if p is not None and w is not None]
    if not available_hemis:
        # Print helpful diagnostics and list directory contents
        try:
            contents = sorted(p.name for p in fs_surf_dir.iterdir())
        except Exception:
            contents = ["<unavailable>"]
        print(f"No valid hemis found in {fs_surf_dir}", file=sys.stderr)
        for h, err in load_errors.items():
            print(f"  {h}: {err}", file=sys.stderr)
        print(f"Directory contents: {contents}", file=sys.stderr)
        print("Expected files like 'lh.pial', 'lh.white', 'rh.pial', 'rh.white'", file=sys.stderr)
        return 2

    params = PatchParams(
        radius_mm=args.radius_mm,
        subdiv_iters=args.subdiv_iters,
        min_n=args.min_n,
        max_n=args.max_n,
        side_remesh_iters=args.side_remesh_iters,
        outward_rays=args.outward_rays,
        parity_sampling=args.parity_sampling,
        save_diags=args.save_diags,
        protect_pw_intersect=False,
        allow_self_intersection=allow_self_intersection,
        self_intersection_tolerance=self_intersection_tolerance,
    )

    # ----------------------------
    # Single-patch driver (VTK only)
    # ----------------------------
    MAX_ATTEMPTS = 2000

    if args.region_name is not None:
        if args.surface_ras is not None:
            print("Do not mix --region-name with --surface-ras.", file=sys.stderr)
            return 2
        if args.hemi is None:
            print("When using --region-name, you must supply --hemi (lh or rh).", file=sys.stderr)
            return 2
    else:
        if args.surface_ras is None:
            print("When not using --region-name, you must supply --surface-ras.", file=sys.stderr)
            return 2

    target_ras = None
    if args.surface_ras is not None:
        target_ras = parse_surface_ras(args.surface_ras)

    if args.region_name is not None:
        aparc_dir = args.aparc_dir
        if aparc_dir is None:
            aparc_dir = fs_surf_dir.parent / "label"

        hemi = args.hemi
        if hemi not in available_hemis:
            print(f"Hemisphere '{hemi}' not available in {fs_surf_dir}.", file=sys.stderr)
            return 2
        pial, white = hemi_pairs[hemi]
        Vp, F = _poly_to_numpy(pial)
        try:
            mask = _load_aparc_mask(aparc_dir, hemi, args.region_name, Vp.shape[0])
        except Exception as e:
            # Provide available region names to help diagnose label mismatches
            annot_path = (aparc_dir / f"{hemi}.aparc.annot")
            try:
                _, _, names = nib.freesurfer.read_annot(str(annot_path))
                names = [n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n) for n in names]
                print(f"[region] {hemi} available labels: {sorted(set(names))}", file=sys.stderr)
            except Exception:
                pass
            print(f"[region] {hemi} {e}", file=sys.stderr)
            return 2

        adj = build_vertex_adjacency(F, Vp.shape[0])
        raw_count = int(mask.sum())
        if raw_count == 0:
            print(f"[region] {hemi} '{args.region_name}' raw mask has 0 vertices", file=sys.stderr)
            try:
                annot_path = aparc_dir / f"{hemi}.aparc.annot"
                labels0, ctab0, names0 = nib.freesurfer.read_annot(str(annot_path), orig_ids=False)
                names0 = [n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n) for n in names0]
                name_map = {n.lower(): i for i, n in enumerate(names0)}
                key = args.region_name.lower()
                if key in name_map:
                    name_idx = name_map[key]
                    label_id = ctab0[name_idx, -1]
                    count_idx = int(np.sum(labels0 == name_idx))
                    count_id = int(np.sum(labels0 == label_id))
                    print(
                        f"[region] {hemi} label counts (orig_ids=False): "
                        f"name_idx={name_idx} count_idx={count_idx} label_id={label_id} count_id={count_id}",
                        file=sys.stderr,
                    )
                print(f"[region] {hemi} available labels: {sorted(set(names0))}", file=sys.stderr)
            except Exception:
                pass
        mask = largest_component_mask(mask, adj)
        comp_count = int(mask.sum())
        mask = fill_mask_holes(mask, adj)
        fill_count = int(mask.sum())
        print(
            f"[region] {hemi} '{args.region_name}' mask sizes: raw={raw_count} "
            f"largest_component={comp_count} filled={fill_count}",
            file=sys.stderr,
        )
        if not np.any(mask):
            print(f"Region '{args.region_name}' had no vertices after filtering.", file=sys.stderr)
            return 2

        keep_v = mask
        pial, white = hemi_pairs[hemi]
        seed_idx = int(np.flatnonzero(keep_v)[0]) if np.any(keep_v) else 0

        res = build_one_patch(
            hemi=hemi,
            pial=pial,
            white=white,
            seed_idx=seed_idx,
            params=params,
            patch_idx=0,
            out_dir=out_dir,
            out_file=out_file,
            rng=rng,
            keep_v_override=keep_v,
        )

        if not res.success:
            tri_note = ""
            if res.reason == "self_intersection":
                p = res.counts.get("pial_tris", 0)
                w = res.counts.get("white_tris", 0)
                s = res.counts.get("side_tris", 0)
                tri_note = f" (tris p/w/s={p}/{w}/{s})"
            print(
                f"Could not build a valid patch for region '{args.region_name}' (reason={res.reason}{tri_note}). "
                f"Try a different region or adjust settings."
            )
            return 3

        print(f"Done. Wrote: {res.out_paths.get('vtk', '<missing>')}")
        if args.summarize:
            vtk_path = res.out_paths.get('vtk')
            if vtk_path:
                _print_patch_summary(Path(vtk_path))
            else:
                print("No VTK output to summarize.", file=sys.stderr)
        return 0

    if target_ras is not None:
        # Deterministic: snap Surface RAS -> nearest pial vertex across hemis
        hemi, seed_idx, snap_dist = snap_surface_ras_to_seed(
            target_ras, hemi_pairs, available_hemis
        )

        # Short, human, precise sanity warning
        if snap_dist > 8.0:
            print(
                f"That point is far from the surface (nearest vertex {snap_dist:.1f} mm). "
                f"In Freeview, copy 'Surface RAS', not 'RAS'."
            )

        res = build_one_patch(
            hemi=hemi,
            pial=hemi_pairs[hemi][0],
            white=hemi_pairs[hemi][1],
            seed_idx=seed_idx,
            params=params,
            patch_idx=0,
            out_dir=out_dir,
            out_file=out_file,
            rng=rng,
        )

        if not res.success:
            tri_note = ""
            if res.reason == "self_intersection":
                p = res.counts.get("pial_tris", 0)
                w = res.counts.get("white_tris", 0)
                s = res.counts.get("side_tris", 0)
                tri_note = f" (tris p/w/s={p}/{w}/{s})"
            print(
                f"Could not build a valid patch at that location (reason={res.reason}{tri_note}). "
                f"Try a nearby Surface RAS point or adjust --radius-mm."
            )
            return 3

        print(f"Done. Wrote: {res.out_paths.get('vtk', '<missing>')}")
        if args.summarize:
            vtk_path = res.out_paths.get('vtk')
            if vtk_path:
                _print_patch_summary(Path(vtk_path))
            else:
                print("No VTK output to summarize.", file=sys.stderr)
        return 0

    return 0

# ------------------------------
# Self-test (minimal acceptance)
# ------------------------------

def _ico_sphere(radius: float = 50.0, subdivisions: int = 3) -> Tuple[pv.PolyData, pv.PolyData]:
    """Generate a pair of concentric spheres (pial ~ outer, white ~ inner) with identical topology."""
    sphere = pv.Sphere(radius=radius, theta_resolution=64, phi_resolution=64)
    V, F = _poly_to_numpy(sphere)
    # Project to exact radius to reduce irregularities
    V = V / np.linalg.norm(V, axis=1, keepdims=True) * radius
    pial = _pv_make(V, F)
    white = _pv_make(V * 0.96, F)  # ~2 mm thickness for radius ~50mm
    return pial, white


def _self_test() -> int:
    print("Running self-test...")
    pial, white = _ico_sphere()

    params = PatchParams(radius_mm=15.0, subdiv_iters=1, min_n=64, max_n=512,
                         side_remesh_iters=0, outward_rays=64, parity_sampling=True, save_diags=True)

    out_dir = Path('./_selftest_out')
    _ensure_dir(out_dir)

    res = build_one_patch(
        'lh',
        pial,
        white,
        seed_idx=0,
        params=params,
        patch_idx=0,
        out_dir=out_dir,
        out_file=None,
        rng=random.Random(0),
    )

    assert res.success, f"Self-test failed: {res.reason}"
    assert 'vtk' in res.out_paths, "Missing VTK output path in PatchResult"
    assert Path(res.out_paths['vtk']).exists(), "Missing VTK output file"

    print("Self-test passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
