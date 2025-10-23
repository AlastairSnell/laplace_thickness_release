import numpy as np
from concurrent.futures import ProcessPoolExecutor
from numba import njit, prange
import os
import multiprocessing as mp
import math
from math import sqrt
from collections import defaultdict
import statistics as stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from pathlib import Path
import csv
_CTX = mp.get_context("spawn")

# Global variable for sharing triangle data across worker processes
TRIANGLES = None
WORKER_PID = None
CALL_COUNT = 0
BC_CODE   = BC_VAL = None
UNK_IDX   = None
CENTROIDS = None
N_TRIS    = 0

def _init_worker(triangles_, bc_code_, bc_val_, unk_idx_, centroids_):
    """Initializer for multiprocessing workers."""
    global TRIANGLES, BC_CODE, BC_VAL, UNK_IDX, CENTROIDS, N_TRIS
    TRIANGLES  = triangles_
    BC_CODE    = bc_code_
    BC_VAL     = bc_val_
    UNK_IDX    = unk_idx_
    CENTROIDS  = centroids_
    N_TRIS     = len(triangles_)

def pick_vertex_start_points(triangles, outer_seed_point, target_spacing=1.0,
                             pct=30, max_points=None):
    """
    Return a list of (vertex_idx, coord) pairs that:
        • are on the pial (Dirichlet=1.0) surface
        • respect a minimum Euclidean spacing
    """
    # 1) gather all pial vertices
    verts = []
    for tri in triangles:
        if tri['bc_type'] == 'dirichlet' and abs(tri['bc_value'] - 1.0) < 1e-6:
            verts.extend([(i, v) for i, v in enumerate(tri['vertices'])])
    # drop duplicates by index
    verts_dict = dict(verts)
    vert_items  = list(verts_dict.items())

    # 2) keep only the densest `pct`%
    n_keep = int(len(vert_items) * pct / 100)
    rng    = np.random.default_rng(0)
    rng.shuffle(vert_items)
    vert_items = vert_items[:n_keep]

    # 3) greedy spacing filter
    chosen = []
    for idx, v in vert_items:
        if all(np.linalg.norm(v - c[1]) >= target_spacing for c in chosen):
            chosen.append((idx, v))
            if max_points and len(chosen) >= max_points:
                break
    return chosen

def pick_vertex_start_points_with_idx(triangles,
                                      outer_seed_point,
                                      pct=5,
                                      target_spacing=1,
                                      max_points=None):
    """
    Return a list of (vertex_idx, xyz) tuples.
    Works when `triangles` is a *list* of triangle dicts.

    • vertex_idx is the index in the de-duplicated pial-vertex array
      we build on the fly here (so it’s unique and stable for this run).
    """

    # 1) Collect *unique* pial-surface vertices
    pial_coords = []
    for tri in triangles:
        if tri['bc_value'] >= 0.99:              # pial face
            pial_coords.extend(tri['vertices'])  # three 3-D points

    # Deduplicate (within floating-point tolerance)
    pial_coords = np.asarray(pial_coords, dtype=float)
    pial_coords_unique, inverse = np.unique(
        pial_coords.round(decimals=6), axis=0, return_inverse=True
    )
    # inverse maps each row of pial_coords → unique-vertex index

    # 2) Build a KD-tree for nearest-neighbour queries
    tree = KDTree(pial_coords_unique)

    # 3) Seed-centred sampling
    #    • distance of every vertex from the outer_seed
    dists, _ = tree.query(outer_seed_point, k=len(pial_coords_unique))
    #    • keep the closest `pct` %
    k_keep = max(1, int(pct * 0.01 * len(pial_coords_unique)))
    keep_idx = np.argsort(dists)[:k_keep]

    # 4) Optionally thin by `target_spacing`
    #    Simple greedy thinning: keep a vertex only if it is at least
    #    `target_spacing` away from all already-kept vertices
    selected = []
    for idx in keep_idx:
        xyz = pial_coords_unique[idx]
        if all(np.linalg.norm(xyz - pial_coords_unique[j]) >= target_spacing
               for j, _ in selected):
            selected.append((idx, xyz))
            if max_points and len(selected) >= max_points:
                break

    return selected    # list of (vertex_idx, xyz)

def compare_down_up(all_lengths_down, all_lengths_up,
                    *, label="Down↔Up", debug=False):
    """
    Paired statistical comparison of down- and up-path lengths.

    Statistics reported
    -------------------
    • per-pair difference  Δ = up − down   (signed & absolute)
    • per-pair ratio       R = up / down   (avoids division by zero)
    • summary stats for Δ, |Δ| and R
    • Pearson correlation  r(length_down, length_up)
    • paired t-test for Δ  (H0: mean Δ == 0)

    Parameters
    ----------
    all_lengths_down / up : list[float]
        Must have same length; if not, the extra entries are ignored.
    label : str
        Header printed before the stats.
    debug : bool
        Print detailed per-pair rows in addition to the summary.
    save_txt : str or None
        If given, the summary (and optional detailed rows) are saved.
    """
    n_pairs = min(len(all_lengths_down), len(all_lengths_up))
    if n_pairs == 0:
        print("[WARN] No paired paths to compare.")
        return

    d = np.asarray(all_lengths_down[:n_pairs], dtype=float)
    u = np.asarray(all_lengths_up  [:n_pairs], dtype=float)

    delta      = u - d
    abs_delta  = np.abs(delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio  = np.where(d != 0.0, u / d, np.nan)

    # Paired t-test (manual, avoids scipy dependency)
    mean_delta = delta.mean()
    sd_delta   = delta.std(ddof=1)
    t_val      = (mean_delta / (sd_delta / sqrt(n_pairs))) if n_pairs > 1 else np.nan

    # Pearson r
    r_val = np.corrcoef(d, u)[0, 1]

    # ── assemble output ───────────────────────────────────────────────
    lines = [f"# === {label} comparison (n={n_pairs}) ===",
             f"Mean Δ (up−down):   {mean_delta:.3f}",
             f"SD   Δ:             {sd_delta:.3f}",
             f"Mean |Δ|:           {abs_delta.mean():.3f}",
             f"Median |Δ|:         {np.median(abs_delta):.3f}",
             f"Mean ratio up/down: {np.nanmean(ratio):.3f}",
             f"Median ratio:       {np.nanmedian(ratio):.3f}",
             f"Pearson r:          {r_val:.3f}",
             f"Paired t value:     {t_val:.3f} (df={n_pairs-1})"]

    if debug:
        lines.append("\n# per-pair rows = idx | down | up | Δ | |Δ| | ratio")
        for i in range(n_pairs):
            lines.append(f"{i+1:3d} {d[i]:8.3f} {u[i]:8.3f} "
                         f"{delta[i]:7.3f} {abs_delta[i]:7.3f} {ratio[i]:7.3f}")

    txt = "\n".join(lines)
    print("\n" + txt)

def summarise_paths(all_paths_down, all_paths_up, *, debug=False,
                    save_txt=None):
    """
    Dump all vertices that were visited along each path.

    Parameters
    ----------
    all_paths_down, all_paths_up : list[list[np.ndarray]]
        Lists returned by `trace_single_path` (one list per path).
    debug : bool, default False
        Print to stdout when True.
    save_txt : str or None
        If a filename is supplied, the same summary is written to disk.
    """
    lines = []

    def fmt(pt):
        return f"({pt[0]:.5f}, {pt[1]:.5f}, {pt[2]:.5f})"

    # ---- DOWN
    lines.append("# === DOWN paths ===")
    for i, path in enumerate(all_paths_down, 1):
        pts_str = ", ".join(fmt(p) for p in path)
        lines.append(f"Path {i:<3d}: {pts_str}")

    # ---- UP
    lines.append("# === UP paths ===")
    for i, path in enumerate(all_paths_up, 1):
        pts_str = ", ".join(fmt(p) for p in path)
        lines.append(f"Path {i:<3d}: {pts_str}")

    # ---- output
    txt = "\n".join(lines)
    if debug:
        print("\n" + txt)

    if save_txt is not None:
        with open(save_txt, "w") as fh:
            fh.write(txt + "\n")
        if debug:
            print(f"[INFO] Path summary written to {save_txt}")

def _print_length_stats(name: str, lengths: list[float]) -> None:
    """Pretty-print basic stats for a list of path lengths."""
    if not lengths:          # empty list → nothing to do
        return

    n        = len(lengths)
    mean_val = stats.mean(lengths)
    med_val  = stats.median(lengths)
    # stdev needs at least 2 values or it raises StatisticsError
    sd_val   = stats.stdev(lengths) if n > 1 else 0.0
    min_val  = min(lengths)
    max_val  = max(lengths)

    print(f"\n{name} statistics (n={n}):")
    print(f"  mean   : {mean_val:.3f}")
    print(f"  median : {med_val:.3f}")
    print(f"  stdev  : {sd_val:.3f}")
    print(f"  min    : {min_val:.3f}")
    print(f"  max    : {max_val:.3f}")

import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

def _find_boundary_vertices(F: np.ndarray) -> np.ndarray:
    # Boundary edges appear exactly once
    edges = []
    for a, b, c in F:
        for i, j in ((a, b), (b, c), (c, a)):
            if i < j: edges.append((i, j))
            else:     edges.append((j, i))
    cnt = Counter(edges)
    b_verts = np.unique(np.array([v for e,k in cnt.items() if k == 1 for v in e], dtype=int))
    return b_verts

def pick_even_start_points(
        triangles,
        outer_seed_point,        # kept for API compat; no longer used for the cap
        pct=50,                  # percentage of the inradius to keep (core size)
        target_spacing=None,     # desired *geodesic* spacing on-surface
        max_points=None
):
    """
    Reverse-cap selection: keep only vertices whose geodesic distance *to the boundary*
    is >= (1 - pct/100) * inradius, then run geodesic FPS within that admissible core.
    Returns an (M,3) array of start points on the Dirichlet=1 surface.
    """

    # 1) Collect top-surface vertices/faces (Dirichlet=1)
    top_verts = {}
    faces_top = []
    for tri in triangles:
        if tri['bc_type'] == 'dirichlet' and np.isclose(tri['bc_value'], 1.0):
            idxs = []
            for v in tri['vertices']:
                key = tuple(v)
                idxs.append(top_verts.setdefault(key, len(top_verts)))
            faces_top.append(idxs)

    if not top_verts:
        raise ValueError("No Dirichlet=1 triangles found.")

    V = np.asarray(list(top_verts.keys()), dtype=np.float64)  # (N,3)
    F = np.asarray(faces_top,             dtype=np.int32)     # (M,3)

    # 2) Build undirected weighted edge graph (edge length weights)
    rows, cols, wts = [], [], []
    for a, b, c in F:
        for i, j in ((a, b), (b, c), (c, a)):
            d = np.linalg.norm(V[i] - V[j])
            rows.append(i); cols.append(j); wts.append(d)
            rows.append(j); cols.append(i); wts.append(d)
    G_full = coo_matrix((wts, (rows, cols)), shape=(len(V), len(V))).tocsr()

    # 3) Reverse-cap: distance *from boundary*
    boundary = _find_boundary_vertices(F)
    if boundary.size == 0:
        # Fallback: no boundary detected → behave like full set (or you could
        # fall back to the old seed-cap if you prefer)
        d_to_boundary = np.full(len(V), np.inf)
        inradius = np.inf
        admissible = np.arange(len(V), dtype=int)
    else:
        # Multi-source Dijkstra from all boundary vertices
        D = dijkstra(G_full, directed=False, indices=boundary)   # (|B|, N)
        d_to_boundary = np.min(D, axis=0)                        # nearest-boundary distance
        finite = np.isfinite(d_to_boundary)
        if not finite.any():
            raise RuntimeError("No finite distance-to-boundary values.")
        inradius = float(np.max(d_to_boundary[finite]))          # geodesic inradius
        # Keep the inner core: >= (1 - pct)*inradius
        thresh = (1.0 - (pct / 100.0)) * inradius
        admissible = np.where(d_to_boundary >= thresh)[0]
        if admissible.size == 0:
            raise ValueError("Reverse-cap produced an empty admissible set. Consider increasing pct.")

    V_sub = V[admissible]
    G_sub = G_full[admissible][:, admissible]  # induced subgraph for the core

    # 4) Geodesic farthest-point sampling within the admissible core
    # Auto spacing: ~5% of core diameter if not provided
    if target_spacing is None:
        # Pick a deterministic start (index 0 in the core), sweep to farthest, then sweep again
        d0 = dijkstra(G_sub, directed=False, indices=0)
        a0 = int(np.argmax(d0)) if np.any(np.isfinite(d0)) else 0
        d_a0 = dijkstra(G_sub, directed=False, indices=a0)
        approx_diam = float(np.nanmax(d_a0[np.isfinite(d_a0)]))
        target_spacing = 0.05 * approx_diam if np.isfinite(approx_diam) and approx_diam > 0 else 0.0

    # Start deterministically at the core vertex with max distance-to-boundary (deepest core)
    first_sub_idx = int(np.argmax(d_to_boundary[admissible]))
    chosen_sub = [first_sub_idx]
    chosen_coords = [V_sub[first_sub_idx]]

    d_min = dijkstra(G_sub, directed=False, indices=first_sub_idx)
    d_min = np.asarray(d_min)

    while True:
        next_sub_idx = int(np.argmax(d_min))
        max_min_dist = float(d_min[next_sub_idx])

        if not np.isfinite(max_min_dist) or max_min_dist < target_spacing:
            break

        chosen_sub.append(next_sub_idx)
        chosen_coords.append(V_sub[next_sub_idx])

        if max_points is not None and len(chosen_sub) >= max_points:
            break

        d_new = dijkstra(G_sub, directed=False, indices=next_sub_idx)
        d_min = np.minimum(d_min, d_new)

    return np.asarray(chosen_coords, dtype=np.float64)


@njit
def distance(x, y):
    return np.linalg.norm(x - y)

@njit
def green_function(x, y):
    """
    G(x, y) = 1 / (4 pi |x - y|)
    """
    r = distance(x, y)
    if r < 1e-14:
        return 0.0
    return 1.0 / (4.0 * np.pi * r)

@njit
def green_function_normal_derivative(x, y, n_y):
    """
    dG/dn_y = ((x - y) dot n_y) / (4 pi |x - y|^3)
    """
    r_vec = x - y
    r = np.linalg.norm(r_vec)
    if r < 1e-14:
        return 0.0
    return np.dot(r_vec, n_y) / (4.0 * np.pi * r**3)

@njit
def grad_x_G(x, y):
    """
    grad_x G(x, y) = -(x - y) / (4 pi |x - y|^3)
    Returns a 3D vector
    """
    r_vec = x - y
    r = np.linalg.norm(r_vec)
    if r < 1e-14:
        return np.zeros(3)
    return -r_vec / (4.0 * np.pi * r**3)

@njit
def grad_x_dGdn(x, y, n_y):
    """
    grad_x[dG/dn_y](x, y) = derivative wrt x of dot((x-y), n_y)/(4 pi |x-y|^3).
    Uses quotient rule to handle the 1/|x-y|^3 factor.
    """
    r_vec = x - y
    r = np.linalg.norm(r_vec)
    if r < 1e-14:
        return np.zeros(3)

    A = np.dot(r_vec, n_y)
    B = r**3
    gradA = n_y
    gradB = 3.0 * r * r_vec
    out = (gradA * B - A * gradB) / (B**2)
    out /= (4.0 * np.pi)
    return out

INV4PI = 1.0 / (4.0 * math.pi)

_DUNAVANT_L = np.array([
    [1/3, 1/3, 1/3],                       # centroid
    [0.059715871789770, 0.470142064105115, 0.470142064105115],
    [0.470142064105115, 0.059715871789770, 0.470142064105115],
    [0.470142064105115, 0.470142064105115, 0.059715871789770],
    [0.797426985353087, 0.101286507323456, 0.101286507323456],
    [0.101286507323456, 0.797426985353087, 0.101286507323456],
    [0.101286507323456, 0.101286507323456, 0.797426985353087],
], dtype=np.float64)

_DUNAVANT_W = np.array([
    0.225000000000000,            # centroid
    0.132394152788506, 0.132394152788506, 0.132394152788506,  # 3 identical
    0.125939180544827, 0.125939180544827, 0.125939180544827   # 3 identical
], dtype=np.float64)

BARY_3PT = np.array([
    [1/2, 1/2, 0.0],
    [1/2, 0.0, 1/2],
    [0.0, 1/2, 1/2],
], dtype=np.float64)

BARY_7PT = np.array([
    [1/3,          1/3,          1/3        ],  # centroid
    [0.059715871789770, 0.470142064105115, 0.470142064105115],
    [0.470142064105115, 0.059715871789770, 0.470142064105115],
    [0.470142064105115, 0.470142064105115, 0.059715871789770],
    [0.797426985353087, 0.101286507323456, 0.101286507323456],
    [0.101286507323456, 0.797426985353087, 0.101286507323456],
    [0.101286507323456, 0.101286507323456, 0.797426985353087]
], dtype=np.float64)

@njit
def tri_area(vertices):
    e1 = vertices[1] - vertices[0]
    e2 = vertices[2] - vertices[0]
    return 0.5 * np.linalg.norm(np.cross(e1, e2))

def subdivide_triangle(vertices):
    """
    Subdivide 'vertices' into 4 sub-triangles by connecting midpoints of edges.
    """
    v0, v1, v2 = vertices
    m01 = 0.5*(v0 + v1)
    m12 = 0.5*(v1 + v2)
    m02 = 0.5*(v0 + v2)
    return [
        np.array([v0,  m01, m02]),
        np.array([m01, v1,  m12]),
        np.array([m02, m12, v2 ]),
        np.array([m01, m12, m02])
    ]

@njit
def _subdivide_triangle(verts):
    """
    Split into 4 children (Loop subdivision style).
    Returns (4,3,3) array.
    """
    m01 = 0.5 * (verts[0] + verts[1])
    m12 = 0.5 * (verts[1] + verts[2])
    m20 = 0.5 * (verts[2] + verts[0])
    out = np.empty((4, 3, 3), dtype=np.float64)
    # child 0
    out[0, 0], out[0, 1], out[0, 2] = verts[0], m01, m20
    # child 1
    out[1, 0], out[1, 1], out[1, 2] = m01, verts[1], m12
    # child 2
    out[2, 0], out[2, 1], out[2, 2] = m20, m12, verts[2]
    # centre child
    out[3, 0], out[3, 1], out[3, 2] = m01, m12, m20
    return out

def point_segment_distance(pt, s0, s1):
    v = s1 - s0
    w = pt - s0
    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(pt - s0)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(pt - s1)
    b = c1 / c2
    pb = s0 + b*v
    return np.linalg.norm(pt - pb)

def point_triangle_distance(pt, vertices):
    """
    Actual distance from 'pt' to the triangle.
    """
    v0, v1, v2 = vertices
    cross = np.cross(v1 - v0, v2 - v0)
    denom = np.linalg.norm(cross)
    if denom < 1e-14:
        # degenerate tri
        return min(np.linalg.norm(pt - v0),
                   np.linalg.norm(pt - v1),
                   np.linalg.norm(pt - v2))
    normal = cross / denom
    dist_plane = abs(np.dot(pt - v0, normal))

    proj = pt - (np.dot(pt - v0, normal))*normal

    v0p = proj - v0
    v1p = v1 - v0
    v2p = v2 - v0
    dot11 = np.dot(v1p, v1p)
    dot12 = np.dot(v1p, v2p)
    dot1p = np.dot(v1p, v0p)
    dot22 = np.dot(v2p, v2p)
    dot2p = np.dot(v2p, v0p)

    inv_denom = 1.0 / (dot11*dot22 - dot12*dot12)
    alpha = (dot22*dot1p - dot12*dot2p)*inv_denom
    beta  = (dot11*dot2p - dot12*dot1p)*inv_denom

    if alpha >= 0 and beta >= 0 and (alpha+beta)<=1:
        return dist_plane
    else:
        d0 = point_segment_distance(pt, v0, v1)
        d1 = point_segment_distance(pt, v1, v2)
        d2 = point_segment_distance(pt, v2, v0)
        return min(d0, d1, d2)

@njit(inline='always')
def _point_segment_distance(pt, a, b):
    """
    Distance from point `pt` to segment [a,b].
    """
    ab   = b - a
    ab2  = np.dot(ab, ab)
    if ab2 < 1e-14:
        return np.linalg.norm(pt - a)

    t = np.dot(pt - a, ab) / ab2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    closest = a + t * ab
    return np.linalg.norm(pt - closest)

@njit
def _point_triangle_distance(pt, vertices):
    """
    Mirror of original point_triangle_distance():
    • exact plane distance if projection inside
    • else min distance to the three edges
    """
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]

    # --- plane normal ---
    cross = np.cross(v1 - v0, v2 - v0)
    denom = np.linalg.norm(cross)
    if denom < 1e-14:           # degenerate triangle
        d0 = np.linalg.norm(pt - v0)
        d1 = np.linalg.norm(pt - v1)
        d2 = np.linalg.norm(pt - v2)
        return min(d0, d1, d2)

    normal = cross / denom
    dist_plane = abs(np.dot(pt - v0, normal))

    # --- projection of pt onto plane ---
    proj = pt - np.dot(pt - v0, normal) * normal

    # barycentric‐inside test
    v0p = proj - v0
    v1p = v1 - v0
    v2p = v2 - v0

    dot11 = np.dot(v1p, v1p)
    dot12 = np.dot(v1p, v2p)
    dot1p = np.dot(v1p, v0p)
    dot22 = np.dot(v2p, v2p)
    dot2p = np.dot(v2p, v0p)

    inv_den = 1.0 / (dot11 * dot22 - dot12 * dot12)
    alpha   = (dot22 * dot1p - dot12 * dot2p) * inv_den
    beta    = (dot11 * dot2p - dot12 * dot1p) * inv_den

    if alpha >= 0.0 and beta >= 0.0 and (alpha + beta) <= 1.0:
        # projection lies inside the triangle
        return dist_plane
    else:
        # distance to each edge
        d0 = _point_segment_distance(pt, v0, v1)
        d1 = _point_segment_distance(pt, v1, v2)
        d2 = _point_segment_distance(pt, v2, v0)
        return d0 if d0 < d1 and d0 < d2 else (d1 if d1 < d2 else d2)
    
@njit
def characteristic_size(vertices):
    v0, v1, v2 = vertices
    e01 = np.linalg.norm(v1 - v0)
    e12 = np.linalg.norm(v2 - v1)
    e02 = np.linalg.norm(v2 - v0)
    return (e01 + e12 + e02) / 3.0

def tri_integrate_coarse(func, vertices):
    """
    Coarse pass: 3-pt barycentric for scalar or vector integrands.
    """
    test_val = func(vertices[0])

    A = tri_area(vertices)
    if np.isscalar(test_val):
        # scalar
        val = 0.0
        for bc in BARY_3PT:
            y = bc[0]*vertices[0] + bc[1]*vertices[1] + bc[2]*vertices[2]
            val += func(y)
        w = 1.0/len(BARY_3PT)
        return val*(A*w)
    else:
        # vector
        val = np.zeros(3)
        for bc in BARY_3PT:
            y = bc[0]*vertices[0] + bc[1]*vertices[1] + bc[2]*vertices[2]
            val += func(y)
        w = 1.0/len(BARY_3PT)
        return val*(A*w)

def tri_integrate_fine(func, vertices):
    """
    Fine pass: 7-pt barycentric for scalar or vector integrands.
    """
    test_val = func(vertices[0])

    A = tri_area(vertices)
    if np.isscalar(test_val):
        # scalar
        val = 0.0
        for bc in BARY_7PT:
            y = bc[0]*vertices[0] + bc[1]*vertices[1] + bc[2]*vertices[2]
            val += func(y)
        w = 1.0/len(BARY_7PT)
        return val*(A*w)
    else:
        # vector
        val = np.zeros(3)
        for bc in BARY_7PT:
            y = bc[0]*vertices[0] + bc[1]*vertices[1] + bc[2]*vertices[2]
            val += func(y)
        w = 1.0/len(BARY_7PT)
        return val*(A*w)

def adaptive_triangle_integration_unified(func, vertices, x=None,
                                          tol=1e-5, max_refine=8):
    """
    A unified approach for both scalar & vector integrands.
    Forced subdivision, coarse vs. fine difference, recursion.
    """

    # forced subdiv if near
    if x is not None and max_refine > 0:
        d = point_triangle_distance(x, vertices)
        h = characteristic_size(vertices)
        if d < 0.1 * h:
            if max_refine==0:
                return tri_integrate_fine(func, vertices)
            test_val = func(vertices[0])
            val_sub = 0.0 if isinstance(test_val, float) else np.zeros(3)
            subs = subdivide_triangle(vertices)
            for st in subs:
                val_sub += adaptive_triangle_integration_unified(func, st, x, tol, max_refine-1)
            return val_sub

    if max_refine == 0:
        return tri_integrate_fine(func, vertices)

    # coarse vs fine check
    val_coarse = tri_integrate_coarse(func, vertices)

    val_fine   = tri_integrate_fine(func,   vertices)

    diff = val_fine - val_coarse
    if isinstance(diff, float):
        err_est = abs(diff)
    else:
        err_est = np.linalg.norm(diff)

    if err_est < tol:
        return val_fine
    else:
        test_val = func(vertices[0])
        val_sub = 0.0 if isinstance(test_val, float) else np.zeros(3)
        subs = subdivide_triangle(vertices)
        for st in subs:
            val_sub += adaptive_triangle_integration_unified(func, st, x, tol, max_refine-1)
        return val_sub

def preprocess_triangles(triangles):
    
    for tri in triangles:
        verts = tri['vertices']
        e1 = verts[1] - verts[0]
        e2 = verts[2] - verts[0]
        cross_prod = np.cross(e1, e2)
        area = 0.5*np.linalg.norm(cross_prod)
        normal_geom = cross_prod / (2.0*area)
        if np.dot(normal_geom, tri['normal'])<0:
            normal_geom = -normal_geom
        tri['area']     = area
        tri['normal']   = normal_geom
        tri['centroid'] = np.mean(verts, axis=0)

def plot_bumpy_top(ax, triangles, face_vals=None, cmap='hot'):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces_xyz = []
    colours   = []

    for tri in triangles:
        if tri['bc_value'] >= 0.99:
            verts = tri.get('vertices_xyz', tri['vertices'])
            faces_xyz.append(verts)
            if face_vals is not None:
                colours.append(face_vals[len(colours)])

    poly = Poly3DCollection(faces_xyz, linewidths=0)
    ax.add_collection3d(poly)               # ← add FIRST
    poly.set_label('Top surface')

    if face_vals is not None:
        norm = plt.Normalize(np.nanmin(face_vals), np.nanmax(face_vals))
        poly.set_facecolor(plt.cm.get_cmap(cmap)(norm(colours)))

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(face_vals)

        # pass ax=ax so Matplotlib knows which axes to shrink
        ax.figure.colorbar(mappable, ax=ax, shrink=0.5, pad=0.02,
                           label='Thickness (mm)')

    else:
        poly.set_facecolor('pink')
        poly.set_alpha(0.5)
        poly.set_edgecolors('k')

    return poly

def plot_bumpy_bottom(ax, triangles):
    """
    Collects triangles with bc_type='dirichlet' and bc_value≈0.0 => bottom face,
    then adds them as a Poly3DCollection for visualization.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    faces = []
    for tri in triangles:
        # The bottom face is presumed to have potential=0 => bc_value=0.0
        if tri['bc_type'] == 'dirichlet' and abs(tri['bc_value'] - 0.0) < 1e-10:
            faces.append(tri['vertices'])

    poly_bottom = Poly3DCollection(faces, facecolors='lightblue', alpha=0.5)
    ax.add_collection3d(poly_bottom)
    return poly_bottom

def subdivide_triangle_bary(vertices, n_sub=2):
    v0, v1, v2 = vertices
    subtris = []
    for i in range(n_sub):
        for j in range(n_sub-i):
            def bary_to_xyz(ib, jb):
                alpha, beta = ib, jb
                gamma = n_sub - alpha - beta
                return (alpha*v1 + beta*v2 + gamma*v0)/n_sub

            p0 = bary_to_xyz(i,  j)
            p1 = bary_to_xyz(i+1,j)
            p2 = bary_to_xyz(i,  j+1)
            subtris.append(np.array([p0,p1,p2]))
            if j<(n_sub-i-1):
                p3 = bary_to_xyz(i+1, j+1)
                p4 = bary_to_xyz(i+1, j)
                p5 = bary_to_xyz(i,   j+1)
                subtris.append(np.array([p3,p4,p5]))
    return subtris

def integrate_single_layer_self_3pt(x_i, tri_i,
                                    n_sub=2,
                                    adapt_tol=1e-5,
                                    adapt_refine=8):
    """
    I_ii = ∫ G(x_i, y) dS over T_i
    subdiv T_i => n_sub^2 smaller triangles => unify approach
    """
    sub_tris = subdivide_triangle_bary(tri_i['vertices'], n_sub)
    val = 0.0
    for st in sub_tris:
        def f_g(y):
            return green_function(x_i, y)
        increment = adaptive_triangle_integration_unified(
            f_g, st, x=x_i, tol=adapt_tol, max_refine=adapt_refine
        )
        old_val = val
        val += increment
    return val

def assign_unknown_indices(triangles):
    unknown_index = 0
    for tri in triangles:
        if tri['bc_type']=='dirichlet':
            tri['unknown_type'] = 'flux'
            tri['unknown_index'] = unknown_index
            unknown_index+=1
        else:
            tri['unknown_type'] = 'potential'
            tri['unknown_index'] = unknown_index
            unknown_index+=1
    return unknown_index

def triangle_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

def remove_degenerate_tris(verts: np.ndarray,
                           faces: np.ndarray,
                           eps: float = 1e-12,
                           dedup_faces: bool = True):
    """
    Returns (new_verts, new_faces, kept_face_mask, idx_old2new)
    - Drops faces with area <= eps or non-finite area
    - Optionally deduplicates identical faces (ignoring winding)
    - Reindexes faces to a compact vertex set
    """
    # 1) drop bad faces
    areas = triangle_areas(verts, faces)
    keep_face = np.isfinite(areas) & (areas > eps)
    faces = faces[keep_face]
    removed = int(np.count_nonzero(~keep_face))

    # 2) (optional) drop duplicate faces
    if dedup_faces and faces.size:
        f_sorted = np.sort(faces, axis=1)
        _, unique_idx = np.unique(f_sorted, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        removed_dups = faces.shape[0] - unique_idx.size
        faces = faces[unique_idx]
    else:
        removed_dups = 0

    # 3) reindex vertices
    if faces.size:
        used = np.unique(faces.ravel())
        idx_old2new = -np.ones(verts.shape[0], dtype=np.int64)
        idx_old2new[used] = np.arange(used.size)
        new_verts = verts[used]
        new_faces = idx_old2new[faces]
    else:
        # empty mesh fallback
        used = np.array([], dtype=np.int64)
        idx_old2new = -np.ones(verts.shape[0], dtype=np.int64)
        new_verts = verts[:0]
        new_faces = faces[:0]

    return new_verts, new_faces, keep_face, idx_old2new, removed, removed_dups



@njit(fastmath=True)
def _compute_IJ_numba(x, v0, v1, v2, n, order=1):
    """
    Compute single-layer (I) & double-layer (J) triangle integrals
    ∫_T G(x,y) dS  and  ∫_T ∂G/∂n_y (x,y) dS
    using 1-, 3-, or 7-point barycentric quadrature.
    If any quadrature point y coincides with x (r1=0), the triangle is skipped.
    """
    EPS = 1e-15

    # -- geometry --
    e1 = v1 - v0
    e2 = v2 - v0
    cr = np.cross(e1, e2)
    area = 0.5 * math.sqrt(cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2])
    if (not math.isfinite(area)) or (area <= EPS):
        return 0.0, 0.0  # skip degenerate triangle

    # -- choose quadrature rule --
    if order == 1:                       # centroid rule
        w = np.array([1.0], dtype=np.float64)
        l = np.array([[1.0/3.0, 1.0/3.0, 1.0/3.0]], dtype=np.float64)
    elif order == 3:                     # three points, equal weights
        w = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0], dtype=np.float64)
        l = np.array([[0.5, 0.5, 0.0],
                      [0.5, 0.0, 0.5],
                      [0.0, 0.5, 0.5]], dtype=np.float64)
    else:                                # 7-point Dunavant
        w = _DUNAVANT_W
        l = _DUNAVANT_L

    # -- accumulate integrals --
    I = 0.0
    J = 0.0
    for k in range(l.shape[0]):
        l1, l2, l3 = l[k, 0], l[k, 1], l[k, 2]
        y = l1 * v0 + l2 * v1 + l3 * v2  # quadrature point

        r0 = x[0] - y[0]
        r1_ = x[1] - y[1]
        r2_ = x[2] - y[2]
        r2sq = r0*r0 + r1_*r1_ + r2_*r2_

        # If r1 == 0 (i.e., r2sq == 0), skip the entire triangle
        if r2sq <= EPS*EPS:
            return 0.0, 0.0

        r1 = math.sqrt(r2sq)
        r3 = r2sq * r1

        # kernels
        G    = INV4PI / r1
        dGdn = INV4PI * (n[0]*r0 + n[1]*r1_ + n[2]*r2_) / r3

        I += w[k] * G
        J += w[k] * dGdn

    I *= area
    J *= area
    return I, J


def compute_Iij_Jij(x_i, tri_j, i, j, *, order=1):
    """
    Wrapper so existing code need not change.
    `order` propagates down to the Numba kernel (1/3/7).
    """
    if i == j:                                  # self-term (special treatment)
        I_ij = integrate_single_layer_self_3pt(x_i, tri_j)
        return I_ij, 0.0

    v0, v1, v2 = tri_j["vertices"]              # (3,3) ndarray float64
    n_j        = tri_j["normal"]                # (3,)  ndarray float64
    return _compute_IJ_numba(x_i, v0, v1, v2, n_j, order)

def _triangles_to_arrays(triangles):
    n = len(triangles)
    # 0 = Dirichlet, 1 = Neumann, 2 = other/unknown
    bc_code   = np.empty(n, dtype=np.int8)
    bc_value  = np.empty(n, dtype=np.float64)
    unk_index = np.empty(n, dtype=np.int32)
    centroids = np.empty((n, 3), dtype=np.float64)

    for i, t in enumerate(triangles):
        bc_code[i]  = 0 if t["bc_type"] == "dirichlet" else 1
        bc_value[i] = t["bc_value"]
        unk_index[i]= t["unknown_index"]
        centroids[i]= t["centroid"]

    return bc_code, bc_value, unk_index, centroids

def _row_task(i: int):
    """Return (i, row_I, row_J) for row i (no Python dict look-ups)."""
    row_I = np.empty(N_TRIS, dtype=np.float64)
    row_J = np.empty(N_TRIS, dtype=np.float64)

    x_i = CENTROIDS[i]            # centroid of triangle i

    for j in range(N_TRIS):
        # Still need full geometry of triangle j for the kernels:
        I_ij, J_ij = compute_Iij_Jij(x_i, TRIANGLES[j], i, j)
        row_I[j] = I_ij
        row_J[j] = J_ij

    return i, row_I, row_J
'''
def assemble_system(triangles, *, parallel=False, max_workers=None):
    n_unknowns = assign_unknown_indices(triangles)
    bc_code, bc_value, unk_index, centroids = _triangles_to_arrays(triangles)
    n_tris      = len(triangles)
    A = np.zeros((n_tris, n_unknowns))
    b = np.zeros(n_tris)

    HALF = 0.5
    for i in range(n_tris):
        if bc_code[i] == 0:          # Dirichlet
            b[i] -= HALF * bc_value[i]
        else:                        # Neumann / unknown
            A[i, unk_index[i]] += HALF

    if parallel:
        if max_workers is None:
            max_workers = mp.cpu_count() or 1
        chunksize = max(1, n_tris // (8 * max_workers))

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(triangles, bc_code, bc_value, unk_index, centroids),
            mp_context=_CTX,
        ) as pool:

            for count, (i, row_I, row_J) in enumerate(
                    pool.map(_row_task, range(n_tris), chunksize=chunksize), 1):

                # **vectorised consume – avoids inner Python loop**
                bcj   = bc_code
                valj  = bc_value
                alpha = unk_index

                # Dirichlet contribution to RHS
                b[i] -= np.where(bcj == 0, valj * row_J, 0.0).sum()

                # Neumann contribution to RHS
                b[i] -= np.where(bcj == 1, valj * row_I, 0.0).sum()

                # Unknowns contribution to A
                A[i, alpha] += np.where(bcj != 0, row_J, 0.0)   # J for non-Dirichlet
                A[i, alpha] -= np.where(bcj != 1, row_I, 0.0)   # I for non-Neumann
    else:
        # --- serial fallback, still array-based ---
        for i in range(n_tris):
            x_i = centroids[i]
            for j in range(n_tris):
                I_ij, J_ij = compute_Iij_Jij(x_i, triangles[j], i, j)

                if bc_code[j] == 0:        # Dirichlet
                    b[i] -= bc_value[j] * J_ij
                else:                      # Neumann / unknown
                    A[i, unk_index[j]] += J_ij

                if bc_code[j] == 1:        # Neumann
                    b[i] -= bc_value[j] * I_ij
                else:
                    A[i, unk_index[j]] -= I_ij

    return A, b
'''
def compute_Sij(x_i, tri_j, i, j):
    I_ij, J_ij = compute_Iij_Jij(x_i, tri_j, i, j)
    return J_ij

def compute_Kpij(x_i, n_i, tri_j, i, j):
    I_ij, J_ij = compute_Iij_Jij(x_i, tri_j, i, j)
    return I_ij

def assemble_system(triangles, *, parallel=False, max_workers=None):
    """
    Build the mixed-BC BEM system with a single-layer unknown q on *every* triangle.

    Dirichlet row i:    (S q)_i = phi_bc[i]
    Neumann row i:      (-0.5*I + K' q)_i = g_bc[i]     # interior limit; flip sign if your normals differ

    Notes:
      * Unknowns are now one-per-triangle (n_unknowns = n_tris).
      * We rely on your existing _row_task to return (i, row_I, row_J) where:
          row_I[j] = K'_{ij}  (principal-value; no jump inside!)
          row_J[j] = S_{ij}
      * compute_Iij_Jij / _row_task MUST handle singular / near-singular quadrature appropriately.
    """
    import numpy as np
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    # Pull arrays exactly as you already do
    # (we ignore unk_index here — q lives on all triangles)
    n_tris = len(triangles)
    bc_code, bc_value, unk_index, centroids = _triangles_to_arrays(triangles)

    # One unknown per triangle
    A = np.zeros((n_tris, n_tris), dtype=float)
    b = np.zeros(n_tris, dtype=float)

    # Interior limit with outward normals -> -0.5 jump.
    # If your K' or normals use opposite sign, change to +0.5.
    HALF_JUMP = -0.5

    if parallel:
        if max_workers is None:
            max_workers = mp.cpu_count() or 1
        chunksize = max(1, n_tris // (8 * max_workers))

        # Keep your existing worker wiring; we only change how we *consume* its outputs.
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(triangles, bc_code, bc_value, unk_index, centroids),
            mp_context=_CTX,
        ) as pool:

            for i, row_I, row_J in pool.map(_row_task, range(n_tris), chunksize=chunksize):
                if bc_code[i] == 0:        # Dirichlet row: S
                    A[i, :] = row_J
                    b[i]    = bc_value[i]
                elif bc_code[i] == 1:      # Neumann row: (-0.5*I + K')
                    A[i, :] = row_I
                    A[i, i] += HALF_JUMP
                    b[i]    = bc_value[i]  # 0 for your side wall
                else:
                    raise ValueError(f"Unknown bc_code at row {i}: {bc_code[i]}")

    else:
        # Serial path: compute rows directly from (I_ij, J_ij)
        for i in range(n_tris):
            x_i = centroids[i]

            # Build row_I (K') and row_J (S)
            row_I = np.empty(n_tris, dtype=float)
            row_J = np.empty(n_tris, dtype=float)
            for j in range(n_tris):
                I_ij, J_ij = compute_Iij_Jij(x_i, triangles[j], i, j)
                row_I[j] = I_ij
                row_J[j] = J_ij

            if bc_code[i] == 0:            # Dirichlet row: S
                A[i, :] = row_J
                b[i]    = bc_value[i]
            elif bc_code[i] == 1:          # Neumann row: (-0.5*I + K')
                A[i, :] = row_I
                A[i, i] += HALF_JUMP
                b[i]    = bc_value[i]
            else:
                raise ValueError(f"Unknown bc_code at row {i}: {bc_code[i]}")

    return A, b


def solve_system(A, b):
    return np.linalg.solve(A, b)

def store_solution(triangles, u):
    for tri in triangles:
        idx = tri['unknown_index']
        if tri['bc_type']=='dirichlet':
            tri['flux'] = u[idx]
            tri['potential']= tri['bc_value']
        else:
            tri['potential']=u[idx]
            tri['flux']     =tri['bc_value']

def evaluate_potential(x, triangles):
    val=0.0
    for tri_j in triangles:
        q_j   = tri_j['flux']
        phi_j = tri_j['potential']
        n_j   = tri_j['normal']
        verts_j= tri_j['vertices']

        def f_g(y):
            return green_function(x,y)
        def f_dg(y):
            return green_function_normal_derivative(x,y,n_j)

        I_xj= adaptive_triangle_integration_unified(f_g,  verts_j, x=x)
        J_xj= adaptive_triangle_integration_unified(f_dg, verts_j, x=x)
        val+= q_j*I_xj - phi_j*J_xj
    return val

def evaluate_gradient(x, triangles):
    grad = np.zeros(3)
    for tri_j in triangles:
        q_j   = tri_j['flux']
        phi_j = tri_j['potential']
        n_j   = tri_j['normal']
        verts_j= tri_j['vertices']

        def v_g(pt):
            return grad_x_G(x, pt)
        def v_dgdn(pt):
            return grad_x_dGdn(x, pt, n_j)

        I_vec= adaptive_triangle_integration_unified(v_g,   verts_j, x=x)
        J_vec= adaptive_triangle_integration_unified(v_dgdn,verts_j, x=x)
        grad+= q_j*I_vec - phi_j*J_vec
    return grad

def find_closest_triangle_by_plane(pt, triangles):
    """
    Return the triangle whose plane is closest to 'pt', using the
    perpendicular distance from 'pt' to each triangle's plane.
    
    Assumes:
      - tri['normal'] is roughly unit-length,
      - tri['vertices'] has shape (3,3).
    """
    best_tri = None
    best_dist = float('inf')
    
    for tri in triangles:
        normal = tri['normal']      # presumably a ~unit normal
        plane_pt = tri['vertices'][0]  # a point on the plane
        # distance from pt to the plane => abs( dot((pt - plane_pt), normal) )
        dist_to_plane = abs(np.dot(pt - plane_pt, normal))
        
        if dist_to_plane < best_dist:
            best_dist = dist_to_plane
            best_tri = tri
    
    return best_tri

def set_axes_equal(ax):
    """
    Sets equal scaling for a 3D plot so that the scale for x, y, and z axes are equal.
    This ensures that a cube appears as a cube rather than a rectangular prism.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot radius is half of the maximum range
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def pick_top_triangle_point_in_bounds(triangles):
    top_tris = [tri for tri in triangles 
                if tri['bc_type'] == 'dirichlet' and abs(tri['bc_value'] - 1.0) < 1e-10]
    if not top_tris:
        raise ValueError("No top-face triangles found.")
    
    chosen_tri = np.random.choice(top_tris)
    u, v = np.random.rand(), np.random.rand()
    if u + v > 1.0:
        u, v = 1.0 - u, 1.0 - v
    w = 1.0 - u - v
    
    verts = chosen_tri['vertices']
    return u*verts[0] + v*verts[1] + w*verts[2]

def pick_top_triangle_point_interior(triangles, tol=1e-10):
    """
    Return a random point inside a Dirichlet-top triangle that is
    *not* adjacent to any Neumann/side triangles.

    Parameters
    ----------
    triangles : list[dict]
        Your usual list-of-dict mesh.
    tol : float, default 1e-10
        Tolerance for matching vertex coordinates.

    Returns
    -------
    pt : (3,) ndarray[float64]
        A point (barycentric random) strictly inside an interior top triangle.
    """
    # --- 1. collect handles for faster comparison -----------------
    def key(v):
        # hashable vertex signature (rounded to tol)
        return tuple(np.round(v / tol).astype(np.int64))

    # map edge-key -> list of (tri_idx, is_top)
    edge_map = defaultdict(list)

    top_indices = []
    for idx, tri in enumerate(triangles):
        is_top = tri['bc_type'].lower() == 'dirichlet' and abs(tri['bc_value'] - 1.0) < 1e-10
        if is_top:
            top_indices.append(idx)

        v = tri['vertices']
        edges = ((v[0], v[1]),
                 (v[1], v[2]),
                 (v[2], v[0]))
        for a, b in edges:
            # canonical order so the same edge hashes identically
            ka, kb = key(a), key(b)
            ekey = (ka, kb) if ka < kb else (kb, ka)
            edge_map[ekey].append((idx, is_top))

    # --- 2. mark interior top triangles ---------------------------
    interior_top = []
    for idx in top_indices:
        tri = triangles[idx]
        v = tri['vertices']
        edges = ((v[0], v[1]),
                 (v[1], v[2]),
                 (v[2], v[0]))
        touching_neumann = False
        for a, b in edges:
            ka, kb = key(a), key(b)
            ekey = (ka, kb) if ka < kb else (kb, ka)
            # this edge may have 1 (boundary) or 2 (interior) adjacent tris
            # it's "safe" only if *every* triangle using the edge is a top tri
            for _, is_top in edge_map[ekey]:
                if not is_top:            # touches Neumann/side → skip
                    touching_neumann = True
                    break
            if touching_neumann:
                break
        if not touching_neumann:
            interior_top.append(tri)

    if not interior_top:
        raise ValueError("No interior top triangles found – "
                         "mesh too small or tol too strict.")

    # --- 3. sample barycentrically inside a chosen interior tri ---
    chosen = np.random.choice(interior_top)
    u, v = np.random.rand(), np.random.rand()
    if u + v > 1.0:          # fold into the triangle
        u, v = 1.0 - u, 1.0 - v
    w = 1.0 - u - v
    verts = chosen['vertices']
    return u * verts[0] + v * verts[1] + w * verts[2]

def compute_area_weighted_normal_at_point(pt, triangles, tol=1e-8):
    base_tri = find_closest_triangle_by_plane(pt, triangles)
    if base_tri is None:
        return None, []

    base_verts = base_tri['vertices']
    group = []
    for tri in triangles:
        shared = 0
        for v1 in tri['vertices']:
            for v2 in base_verts:
                if np.linalg.norm(v1 - v2) < tol:
                    shared += 1
                    break
        if shared >= 2:
            group.append(tri)

    weighted_normal = np.zeros(3)
    total_area = 0.0
    normals_and_weights = []
    for tri in group:
        weighted_normal += tri['area'] * tri['normal']
        total_area += tri['area']
        normals_and_weights.append((tri['normal'], tri['area']))
    if total_area > 0:
        avg_normal = weighted_normal / np.linalg.norm(weighted_normal)
        return avg_normal, normals_and_weights
    else:
        return None, []

def triangles_to_numeric_full(triangles):
    """
    Convert list-of-dict ‘triangles’ → contiguous numeric arrays.

    Returns
    -------
    verts      : (n, 3, 3) float64   – triangle vertices
    norms      : (n, 3)     float64   – unit normals
    areas      : (n,)       float64   – triangle areas
    flux       : (n,)       float64   – Neumann data  q_j
    phi        : (n,)       float64   – Dirichlet data φ_j
    centroids  : (n, 3)     float64   – pre-computed centroids
    """
    n = len(triangles)

    verts     = np.empty((n, 3, 3), dtype=np.float64)
    norms     = np.empty((n, 3),     dtype=np.float64)
    areas     = np.empty(n,          dtype=np.float64)
    flux      = np.empty(n,          dtype=np.float64)
    phi       = np.empty(n,          dtype=np.float64)
    centroids = np.empty((n, 3),     dtype=np.float64)

    for i, tri in enumerate(triangles):
        verts[i]     = tri['vertices']
        norms[i]     = tri['normal']
        areas[i]     = tri['area']
        flux[i]      = tri['flux']
        phi[i]       = tri['potential']
        centroids[i] = tri['centroid']      # already stored by your pre-process

    return verts, norms, areas, centroids, flux, phi

@njit
def _find_closest_triangle_idx_by_plane(pt, vertices, normals):
    """
    Return index of the triangle whose plane is closest to `pt`
    (perpendicular distance), matching the Python version you showed.
    Assumes `normals` are ~unit length.
    """
    best_idx  = -1
    best_dist = 1e30          # large number

    for i in range(vertices.shape[0]):
        v0 = vertices[i, 0]          # any vertex on the plane
        n  = normals[i]
        # signed distance from pt to plane: dot(pt - v0, n)
        dist = abs((pt[0] - v0[0]) * n[0] +
                   (pt[1] - v0[1]) * n[1] +
                   (pt[2] - v0[2]) * n[2])
        if dist < best_dist:
            best_dist = dist
            best_idx  = i

    return best_idx

@njit
def _compute_area_weighted_normal_jit(
    pt,
    vertices,
    normals,
    areas,
    centroids,
    tol=1e-8,
    debug=False,
):
    """
    JIT‐accelerated version of compute_area_weighted_normal_at_point.
    If debug=True, prints out detailed neighbor info.
    Returns the unit area‐weighted normal at pt.
    """
    # 1) find the triangle “under” pt
    base_idx = _find_closest_triangle_idx_by_plane(pt, vertices, centroids)
    if debug:
        print("  [debug] base_idx:", base_idx)
    if base_idx == -1:
        if debug:
            print("  [debug] no base triangle found → returning zero")
        return np.zeros(3, dtype=np.float64)

    base_verts = vertices[base_idx]
    if debug:
        print("  [debug] base triangle verts:\n", base_verts)

    # 2) accumulate area‐weighted normals of edge‐neighbors
    weighted_normal = np.zeros(3, dtype=np.float64)
    total_area      = 0.0

    for i in range(vertices.shape[0]):
        # count how many verts in common
        shared = 0
        for v1 in vertices[i]:
            for v2 in base_verts:
                if ((v1 - v2) ** 2).sum() < tol * tol:
                    shared += 1
                    break
        if shared >= 2:
            # edge‐neighbor found
            if debug:
                # Promote to Python scalars so Numba’s print can handle them
                a  = float(areas[i])
                nx = float(normals[i][0])
                ny = float(normals[i][1])
                nz = float(normals[i][2])

                # Plain concatenated print works in nopython mode
                print("  [debug] neighbour #", i,
                    ": shared=", shared,
                    ", area=", a,
                    ", normal=(", nx, ",", ny, ",", nz, ")")
            weighted_normal += areas[i] * normals[i]
            total_area      += areas[i]

    if total_area == 0.0:
        if debug:
            print("  [debug] no edge‐neighbors → returning zero")
        return np.zeros(3, dtype=np.float64)

    if debug:
        print("  [debug] total_area:", total_area)
        print("  [debug] weighted_normal (pre‐norm):", weighted_normal)

    # 3) normalize
    norm = np.sqrt((weighted_normal ** 2).sum())
    if debug:
        print("  [debug] norm of weighted_normal:", norm)
    if norm > 0.0:
        avg = weighted_normal / norm
        if debug:
            print("  [debug] returning unit normal:", avg)
        return avg
    else:
        if debug:
            print("  [debug] zero norm → returning weighted_normal as is:", weighted_normal)
        return weighted_normal

def compute_meyer_normal_jit(pt,
                             verts_arr,
                             norms_arr,
                             areas_arr,
                             cents_arr,
                             radius=-1.0,
                             eps=1e-12,
                             debug=False):
    """
    Meyer-style area-weighted pseudo-normal at arbitrary point `pt`.

    radius < 0  → base face + edge-sharing neighbours (“one-ring”)
    radius >= 0 → faces whose centroids lie within `radius` of `pt`
    """
    n_faces = cents_arr.shape[0]

    # ----------------------------------------------------------------------
    # 1) Find the face whose centroid is closest to `pt`
    # ----------------------------------------------------------------------
    base_idx = closest_centroid_idx(pt, cents_arr)
    if base_idx == -1:
        return norms_arr[0]          # degenerate fallback

    # ----------------------------------------------------------------------
    # 2) Build neighbour mask
    # ----------------------------------------------------------------------
    neigh_mask = np.zeros(n_faces, np.uint8)
    neigh_mask[base_idx] = 1

    if radius < 0.0:                 # literal one-ring
        v0, v1, v2 = verts_arr[base_idx]
        for j in range(n_faces):
            if j == base_idx:
                continue
            shared = 0
            for vv in verts_arr[j]:
                if (np.all(vv == v0) or np.all(vv == v1) or np.all(vv == v2)):
                    shared += 1
            if shared >= 2:          # shares an edge
                neigh_mask[j] = 1
    else:                            # centroid-radius set
        for j in range(n_faces):
            if np.linalg.norm(cents_arr[j] - pt) < radius:
                neigh_mask[j] = 1

    # ----------------------------------------------------------------------
    # 3) Area-weighted face-normal sum  (Meyer 2003, Eq. (11))
    # ----------------------------------------------------------------------
    n_sum = np.zeros(3, np.float64)

    for j in range(n_faces):
        if neigh_mask[j] == 0:
            continue

        # --- Weight choice -------------------------------------------------
        # Meyer pseudo-normal:           w = area_f
        # Angle-weighted pseudo-normal:  w = angle at pt in face f
        w = areas_arr[j]

        # Uncomment below for angle weighting (pt must coincide with a vertex)
        # if False:
        #     verts = verts_arr[j]
        #     # find which vertex indexes match pt
        #     vidx = -1
        #     for k in range(3):
        #         if np.all(np.abs(verts[k] - pt) < eps):
        #             vidx = k
        #             break
        #     if vidx >= 0:
        #         a = verts[(vidx + 1) % 3] - verts[vidx]
        #         b = verts[(vidx + 2) % 3] - verts[vidx]
        #         cos_ang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)
        #         w = np.arccos(np.clip(cos_ang, -1.0, 1.0))

        if debug:
            print(f"[dbg] face {j:4d}: area={areas_arr[j]:.4e}, w={w:.4e}, n={norms_arr[j]}")

        n_sum += norms_arr[j] * w

    # ----------------------------------------------------------------------
    # 4) Normalise and orient consistently with base face
    # ----------------------------------------------------------------------
    n_len = np.linalg.norm(n_sum)
    if n_len < eps:
        return norms_arr[base_idx]   # fallback if something went wrong

    n_sum /= n_len
    if np.dot(n_sum, norms_arr[base_idx]) < 0.0:
        n_sum *= -1.0

    return -n_sum

# --------------------- scalar adaptive ----------------------------------------
@njit
def _integrand_scalar(func_id, x, y, n_y):
    if func_id == 0:     # G
        return green_function(x, y)
    else:                # dG/dn_y
        return green_function_normal_derivative(x, y, n_y)

@njit
def _tri_integrate_scalar(func_id, x, verts, n_y, coarse):
    bset = BARY_3PT if coarse else BARY_7PT
    A = tri_area(verts)
    acc = 0.0
    for k in range(bset.shape[0]):
        bc = bset[k]
        y = bc[0]*verts[0] + bc[1]*verts[1] + bc[2]*verts[2]
        acc += _integrand_scalar(func_id, x, y, n_y)
    return acc * (A / bset.shape[0])

@njit
def _adaptive_scalar(func_id, x, verts, n_y,
                     tol=1e-5, max_ref=8):
    # forced subdivision if very close
    if max_ref > 0:
        d = _point_triangle_distance(x, verts)
        h = characteristic_size(verts)
        if d < 0.1*h:
            childs = _subdivide_triangle(verts)
            acc = 0.0
            for c in range(4):
                acc += _adaptive_scalar(func_id, x,
                                        childs[c], n_y,
                                        tol, max_ref-1)
            return acc

    if max_ref == 0:
        return _tri_integrate_scalar(func_id, x, verts, n_y, coarse=False)

    coarse = _tri_integrate_scalar(func_id, x, verts, n_y, coarse=True)
    fine   = _tri_integrate_scalar(func_id, x, verts, n_y, coarse=False)
    if abs(fine - coarse) < tol:
        return fine

    # recurse
    childs = _subdivide_triangle(verts)
    acc = 0.0
    for c in range(4):
        acc += _adaptive_scalar(func_id, x, childs[c], n_y,
                                tol, max_ref-1)
    return acc

# --------------------- vector adaptive ----------------------------------------
@njit
def _integrand_vec(func_id, x, y, n_y):
    if func_id == 2:          # ∇_x G
        return grad_x_G(x, y)
    else:                     # ∇_x dG/dn
        return grad_x_dGdn(x, y, n_y)

@njit
def _tri_integrate_vec(func_id, x, verts, n_y, coarse):
    bset = BARY_3PT if coarse else BARY_7PT
    A = tri_area(verts)
    acc = np.zeros(3)
    for k in range(bset.shape[0]):
        bc = bset[k]
        y = bc[0]*verts[0] + bc[1]*verts[1] + bc[2]*verts[2]
        acc += _integrand_vec(func_id, x, y, n_y)
    return acc * (A / bset.shape[0])

@njit
def _adaptive_vec(func_id, x, verts, n_y,
                  tol=1e-3, max_ref=3):
    if max_ref > 0:
        d = _point_triangle_distance(x, verts)
        h = characteristic_size(verts)
        if d < 0.1*h:
            childs = _subdivide_triangle(verts)
            acc = np.zeros(3)
            for c in range(4):
                acc += _adaptive_vec(func_id, x, childs[c], n_y,
                                     tol, max_ref-1)
            return acc

    if max_ref == 0:
        return _tri_integrate_vec(func_id, x, verts, n_y, coarse=False)

    coarse = _tri_integrate_vec(func_id, x, verts, n_y, coarse=True)
    fine   = _tri_integrate_vec(func_id, x, verts, n_y, coarse=False)
    if np.linalg.norm(fine - coarse) < tol:
        return fine

    childs = _subdivide_triangle(verts)
    acc = np.zeros(3)
    for c in range(4):
        acc += _adaptive_vec(func_id, x, childs[c], n_y,
                             tol, max_ref-1)
    return acc

# ------------------------------------------------------------------ φ(x)
@njit(parallel=True)
def evaluate_potential_numba(x, verts, normals, flux, phi,
                             tol=1e-5, max_ref=8):
    """
    Mirrors evaluate_potential() but runs fully in nopython mode.
    verts   : (n,3,3)
    normals : (n,3)
    flux    : (n,)
    phi     : (n,)
    """
    n_tris = verts.shape[0]
    acc = 0.0
    for j in prange(n_tris):
        I = _adaptive_scalar(0, x, verts[j], normals[j], tol, max_ref)   # G
        J = _adaptive_scalar(1, x, verts[j], normals[j], tol, max_ref)   # dG/dn
        acc += flux[j]*I - phi[j]*J
    return acc

# ------------------------------------------------------------------ ∇φ(x)
@njit(parallel=True)
def evaluate_gradient_numba(x, verts, normals, flux, phi,
                            tol=1e-3, max_ref=3):
    """
    Mirrors evaluate_gradient() in full nopython mode.
    Returns 3-vector gradient.
    """
    n_tris = verts.shape[0]
    grad = np.zeros(3)
    for j in prange(n_tris):
        I_vec = _adaptive_vec(2, x, verts[j], normals[j], tol, max_ref)  # ∇G
        J_vec = _adaptive_vec(3, x, verts[j], normals[j], tol, max_ref)  # ∇dG/dn
        grad += flux[j]*I_vec - phi[j]*J_vec
    return grad

def _find_closest_triangle_idx(pt, verts_arr, cents_arr):
    """
    Return index of the triangle whose plane is nearest to `pt`
    (minimum absolute signed distance).  If all faces are degenerate
    (area ≈ 0), returns -1.

    Parameters
    ----------
    pt         : (3,) float64  – query point.
    verts_arr  : (N,3,3) float64 – vertices of every face.
    cents_arr  : (N,3) float64   – centroids of every face
                                   (unused here but kept for drop-in
                                    compatibility).

    Returns
    -------
    idx : int64
        Index of the best face, or -1 if none found.
    """
    n_faces   = verts_arr.shape[0]
    best_idx  = -1
    best_dist = 1.0e308                     # effectively +∞

    for i in range(n_faces):
        # ------------------------------------------------------------------
        # 1) build unit normal of face i
        # ------------------------------------------------------------------
        v0 = verts_arr[i, 0]
        v1 = verts_arr[i, 1]
        v2 = verts_arr[i, 2]

        e1x = v1[0] - v0[0]
        e1y = v1[1] - v0[1]
        e1z = v1[2] - v0[2]
        e2x = v2[0] - v0[0]
        e2y = v2[1] - v0[1]
        e2z = v2[2] - v0[2]

        nx  =  e1y * e2z - e1z * e2y
        ny  =  e1z * e2x - e1x * e2z
        nz  =  e1x * e2y - e1y * e2x
        nlen = (nx*nx + ny*ny + nz*nz) ** 0.5

        # Skip degenerate (zero-area) triangles
        if nlen < 1e-12:
            continue

        inv_nlen = 1.0 / nlen
        nx *= inv_nlen
        ny *= inv_nlen
        nz *= inv_nlen

        # ------------------------------------------------------------------
        # 2) signed distance from pt to the plane of face i
        #    d = n · (pt − v0)
        # ------------------------------------------------------------------
        dx = pt[0] - v0[0]
        dy = pt[1] - v0[1]
        dz = pt[2] - v0[2]

        dist = nx*dx + ny*dy + nz*dz
        if dist < 0.0:                       # want absolute distance
            dist = -dist

        # ------------------------------------------------------------------
        # 3) keep the closest plane so far
        # ------------------------------------------------------------------
        if dist < best_dist:
            best_dist = dist
            best_idx  = i

    return best_idx

def closest_centroid_idx(pt, cents_arr):
    """
    Return the index of the triangle whose centroid is nearest to `pt`.

    Parameters
    ----------
    pt        : (3,) array_like
        Query point (x, y, z).
    cents_arr : (N, 3) float64
        Centroids of all N triangles.

    Returns
    -------
    idx : int
        Index of the closest centroid, or -1 if cents_arr is empty.
    """
    cents_arr = np.asarray(cents_arr, dtype=float)
    if cents_arr.size == 0:
        return -1

    pt        = np.asarray(pt, dtype=float)
    # Squared Euclidean distances, then argmin
    d2        = np.sum((cents_arr - pt)**2, axis=1)
    return int(np.argmin(d2))

def segment_triangle_intersection(p0, p1, tri_vertices):
    """
    Returns (hit_bool, intersection_point, tval)
    Möller–Trumbore intersection with segment p0->p1
    """
    v0, v1, v2= tri_vertices
    dir_seg= p1- p0
    epsilon=1e-14

    edge1= v1- v0
    edge2= v2- v0
    h= np.cross(dir_seg, edge2)
    a= np.dot(edge1,h)
    if abs(a)< epsilon:
        return (False,None,999.9)
    f=1.0/a
    s= p0- v0
    u= f* np.dot(s,h)
    if u<0.0 or u>1.0:
        return (False,None,999.9)
    q= np.cross(s,edge1)
    v= f* np.dot(dir_seg,q)
    if v<0.0 or (u+v)>1.0:
        return (False,None,999.9)
    t= f* np.dot(edge2,q)
    if t<0.0 or t>1.0:
        return (False,None,999.9)
    hit_point= p0+ t* dir_seg
    return (True, hit_point,t)

def find_exit_intersection(p0, p1, all_triangles):
    best_t= 999.9
    best_point= None
    for tri in all_triangles:
        tri_verts= tri['vertices']
        hit, xint, tval= segment_triangle_intersection(p0,p1,tri_verts)
        if hit and tval< best_t:
            best_t= tval
            best_point= xint
    return best_point

def curvature_at_point_meyer(
        pt,
        verts,        # (n,3,3)
        tri_areas,    # (n,)
        *,
        mixed_eps=1e-12,
        debug=False):
    """
    Meyer-style curvature at mesh vertex `pt`, with optional verbose tracing.
    """
    import numpy as np

    # ------------ helper for conditional printing ---------------------
    def dbg(msg):
        if debug:
            print(msg)

    dbg("\n===== curvature_at_point_meyer =====")
    dbg(f"pt = {pt}")

    # 1) collect incident faces
    incident = [(t, c)
                for t in range(verts.shape[0])
                for c in range(3)
                if np.allclose(verts[t, c], pt)]

    dbg(f"incident faces = {incident}")

    if not incident:
        dbg("  -> vertex not found in mesh; returning NaNs")
        return (np.nan,)*4

    A_mixed, angle_sum = 0.0, 0.0
    Hn = np.zeros(3)

    for face_idx, corner in incident:
        v0, v1, v2 = verts[face_idx]
        if corner == 1: v0, v1, v2 = v1, v2, v0
        if corner == 2: v0, v1, v2 = v2, v0, v1

        e1, e2 = v1 - v0, v2 - v0
        e0      = v2 - v1   # opposite edge
        l1, l2  = np.linalg.norm(e1), np.linalg.norm(e2)
        l0      = np.linalg.norm(e0)

        # lengths sanity
        l1 = max(l1, mixed_eps)
        l2 = max(l2, mixed_eps)
        l0 = max(l0, mixed_eps)

        # angle at v0
        cosθ = np.clip(np.dot(e1, e2)/(l1*l2), -1.0, 1.0)
        θ    = np.arccos(cosθ)
        angle_sum += θ

        # other two angles
        cosθ1 = np.clip(np.dot(-e0, e2)/(l0*l2), -1.0, 1.0)
        cosθ2 = np.clip(np.dot(-e1, e0)/(l1*l0), -1.0, 1.0)
        θ1, θ2 = np.arccos(cosθ1), np.arccos(cosθ2)

        # Voronoi / mixed area term
        if θ > np.pi/2:
            area_add = 0.5*tri_areas[face_idx]
        elif (θ1 > np.pi/2) or (θ2 > np.pi/2):
            area_add = 0.25*tri_areas[face_idx]
        else:
            area_add = ((l2**2)*np.tan(θ1) + (l1**2)*np.tan(θ2)) / 8.0
        A_mixed += area_add

        # cotangent weights
        cross1 = np.linalg.norm(np.cross(e1, -e0))
        cross2 = np.linalg.norm(np.cross(e2,  e0))
        cot_a  = np.dot(e1, -e0) / max(cross1, mixed_eps)
        cot_b  = np.dot(e2,  e0) / max(cross2, mixed_eps)
        Hn    += (cot_a + cot_b) * (v1 - v2)

        # ---- per‑face trace -----
        dbg(f"[face {face_idx}] corner={corner}")
        dbg(f"  θ0={np.degrees(θ):.2f}°, θ1={np.degrees(θ1):.2f}°, θ2={np.degrees(θ2):.2f}°")
        dbg(f"  area_add={area_add:.4e}, A_mixed running={A_mixed:.4e}")
        dbg(f"  cot_a={cot_a:.4e}, cot_b={cot_b:.4e}")
        dbg(f"  Hn running = {Hn}")

    dbg(f"\nFINAL A_mixed = {A_mixed:.6e}")
    dbg(f"angle_sum     = {angle_sum:.6e}")
    dbg(f"Hn            = {Hn}")

    if A_mixed < mixed_eps*10:
        dbg("  -> Mixed area too small, returning NaNs")
        return (np.nan,)*4

    K = (2*np.pi - angle_sum) / A_mixed
    H = 0.25*np.linalg.norm(Hn) / A_mixed
    dbg(f"K = {K}, H = {H}")

    disc = max(H*H - K, 0.0)
    sqrt_d = np.sqrt(disc)
    k1, k2 = H + sqrt_d, H - sqrt_d
    dbg(f"k1 = {k1}, k2 = {k2}")
    dbg("===== end curvature_at_point_meyer =====\n")
    return K, H, k1, k2

def trace_single_path(
    start_pt,
    triangles,
    direction='down',
    max_iter=200,
    alpha_initial=0.05,
    tolerance=1e-3,
    min_alpha=1e-3,
    first_step=0.1,
    *,
    debug=False,):
    """
    Follow the gradient of the Laplace potential on a stitched surface mesh
    until the path exits the domain.  Set debug=True for extremely verbose
    step-by-step logging.
    """
    # ------------------------------------------------------------------ utils
    def dbg(msg):
        if debug:
            print(msg)

    def forced_step(old_pt, g_prev, alpha_full):
        """Take an unconditional step along the trusted previous gradient."""
        step_dir  = (-g_prev if direction == 'down' else g_prev)
        step_dir /= np.linalg.norm(step_dir)
        new_pt    = old_pt + alpha_full * step_dir
        return new_pt

    # ----------------------------------------------------------------- setup
    verts_arr, norms_arr, areas_arr, cents_arr, q_arr, phi_arr = \
        triangles_to_numeric_full(triangles)

    path_points   = []
    total_length  = 0.0
    x_current     = start_pt.copy()
    alpha         = alpha_initial
    prev_grad     = None
    GRAD_NORM_MAX = 1.0

    dbg(f"[INIT] X0 = {x_current}, dir = {direction}, "
        f"α0 = {alpha_initial}, tol = {tolerance}")

    path_points.append(x_current.copy())

    # ----------------------------------------------------- 1) n-ring
    avg_normal = compute_meyer_normal_jit(
        x_current, verts_arr, norms_arr, areas_arr, cents_arr, radius=-1.0, eps=1e-12, debug=True)
    if direction == 'up':
        avg_normal = -avg_normal

    if avg_normal is not None:
        step_vec   = first_step * avg_normal
        new_pt     = x_current - step_vec
        d          = np.linalg.norm(step_vec)
        total_length += d
        x_current  = new_pt
        path_points.append(x_current.copy())
        dbg(f"[SEED] Avg-normal step {step_vec} len={d:.5f}")
    else:
        dbg("[SEED] No one-ring neighbourhood; skipping seed step")

    # --------------------------------------------------------- 2) iterations
    for it in range(1, max_iter + 1):
        phi_val  = evaluate_potential_numba(
            x_current, verts_arr, norms_arr, q_arr, phi_arr)
        grad_val = evaluate_gradient_numba(
            x_current, verts_arr, norms_arr, q_arr, phi_arr)
        grad_norm = np.linalg.norm(grad_val)

        angle_flip = False
        if prev_grad is not None and grad_norm > 0.0:
            cos_theta = np.dot(grad_val, prev_grad) / (grad_norm * np.linalg.norm(prev_grad))
            # numerically clamp
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta_deg = np.degrees(np.arccos(cos_theta))
            if theta_deg > 45.0:
                angle_flip = True
                dbg(f"[{it:03}] ∠(∇φᵢ, ∇φᵢ₋₁) = {theta_deg:.1f}°  ➜  FORCE full α step")

        dbg(f"[{it:03}] x={x_current}, φ={phi_val:.6f}, "
            f"∇φ={grad_val}, ||∇φ||={grad_norm:.4f}, α={alpha:.3e}")

        # ---------- termination on tiny gradient
        if grad_norm < tolerance:
            dbg(f"[{it:03}] Terminate: ||∇φ|| < tol")
            break

        # ---------- choose gradient to use
        huge_grad   = abs(grad_norm - 1.0) > GRAD_NORM_MAX
        phi_oob     = not (0.0 <= phi_val <= 1.0)
        use_prev = (huge_grad or phi_oob or angle_flip) and (prev_grad is not None)
        grad_use    = prev_grad if use_prev else grad_val

        if use_prev:
            dbg(f"[{it:03}] Using prev ∇φ  ➜  FORCE full α step")
            old_pt  = x_current.copy()
            new_pt  = forced_step(old_pt, prev_grad, alpha_initial)

            # --- intersection test for the forced step --------------------------
            X_int = find_exit_intersection(
                        old_pt, new_pt,
                        triangles)

            if X_int is not None:
                seg_len = np.linalg.norm(X_int - old_pt)
                total_length += seg_len
                path_points.append(X_int.copy())
                dbg(f"[{it:03}] EXIT at {X_int}, seg_len={seg_len:.5f}")
                break
            else:
                seg_len = np.linalg.norm(new_pt - old_pt)
                total_length += seg_len
                path_points.append(new_pt.copy())
                x_current = new_pt
                continue 

        # final safeguard
        if grad_use is None or np.allclose(grad_use, 0.0):
            grad_use = grad_val

        # ---------- step direction
        step_dir = (-grad_use if direction == 'down' else grad_use)
        step_dir /= np.linalg.norm(step_dir)

        old_pt = x_current.copy()
        new_pt = old_pt + alpha * step_dir

        # ---------- back-tracking line search
        phi_old = phi_val
        phi_new = evaluate_potential_numba(
            new_pt, verts_arr, norms_arr, q_arr, phi_arr)

        def improving(p_new, p_old):
            return (p_new < p_old) if direction == 'down' else (p_new > p_old)

        backtracks = 0
        while (not improving(phi_new, phi_old)) and alpha > min_alpha:
            alpha *= 0.5
            backtracks += 1
            new_pt = old_pt + alpha * step_dir
            phi_new = evaluate_potential_numba(
                new_pt, verts_arr, norms_arr, q_arr, phi_arr)

        dbg(f"[{it:03}] Line-search backtracks = {backtracks}, "
            f"φ_old={phi_old:.6f} → φ_new={phi_new:.6f}, α_final={alpha:.3e}")

        # ---------- tiny step fall-back
        if alpha <= min_alpha:
            if prev_grad is not None and not np.allclose(grad_use, prev_grad):
                dbg(f"[{it:03}] α<min  ➜  FORCE full α step along prev ∇φ")
                old_pt  = x_current.copy()
                new_pt  = forced_step(old_pt, prev_grad, alpha_initial)

                # intersection test for this forced step
                X_int = find_exit_intersection(old_pt, new_pt,
                                            triangles)

                if X_int is not None:                     # crossed sheet
                    seg_len = np.linalg.norm(X_int - old_pt)
                    total_length += seg_len
                    path_points.append(X_int.copy())
                    dbg(f"[{it:03}] EXIT at {X_int}, seg_len={seg_len:.5f}")
                    break
                else:                                     # remained inside
                    seg_len = np.linalg.norm(new_pt - old_pt)
                    total_length += seg_len
                    path_points.append(new_pt.copy())
                    x_current = new_pt
                    alpha     = alpha_initial      # reset for next iter
                    prev_grad = None               # avoid looping on same grad
                    continue                       # go to next iteration
            dbg(f"[{it:03}] α<min, giving up (no progress)")
            break

        # ---------- intersection test
        X_int = find_exit_intersection(
                    old_pt, new_pt,
                    triangles)
        if X_int is not None:
            seg_len = np.linalg.norm(X_int - old_pt)
            total_length += seg_len
            path_points.append(X_int.copy())
            dbg(f"[{it:03}] EXIT at {X_int}, seg_len={seg_len:.5f}")
            break
        else:
            seg_len = np.linalg.norm(new_pt - old_pt)
            total_length += seg_len
            path_points.append(new_pt.copy())
            x_current = new_pt
            prev_grad = grad_use.copy()
            alpha = alpha_initial

    dbg(f"[DONE] steps={len(path_points)-1}, total_len={total_length:.5f}")
    return path_points, total_length

def _log_thick_curv_pair(thick, curv_out, curv_in):
    """
    thick      : scalar thickness (mm)
    curv_out   : (K̄_out, H̄_out)
    curv_in    : (K̄_in , H̄_in )
    """
    K_avg = 0.5 * (curv_out[0] + curv_in[0])
    H_avg = 0.5 * (curv_out[1] + curv_in[1])

    csv_path   = Path("curv_thick_values.csv")
    write_head = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_head:
            w.writerow(["thickness_mm",
                        "mean_curvature_mm^-1",
                        "gaussian_curvature_mm^-2"])
        w.writerow([thick, H_avg, K_avg])

def average_curvature_one_ring(v_idx, verts, faces, areas):
    """Mean (K, H) over vertex + its 1‑ring neighbours."""
    ring_idx = {v_idx}
    # any triangle that contains v_idx ⇒ add all 3 vertices to ring
    for tri in faces:
        if v_idx in tri:
            ring_idx.update(tri)
    Ks, Hs = [], []
    for idx in ring_idx:
        v = verts.reshape(-1, 3)[idx]
        K, H, *_ = curvature_at_point_meyer(v, verts, areas)
        if not np.isnan(K):
            Ks.append(K); Hs.append(H)
    return float(np.mean(Ks)), float(np.mean(Hs))

def path_trace_simple(
    start_pt,
    triangles,
    direction='down',
    max_iter=200,
    alpha_initial=0.1,
    first_step=0.05,
    *,
    debug=True,
    angle_max_deg=35.0,
):
    def dbg(msg):
        if debug:
            print(msg, flush=True)

    def step_dir_from_grad(g):
        # unify sign: 'down' steps along -∇φ, 'up' along +∇φ
        d = (-g if direction == 'down' else g)
        n = np.linalg.norm(d)
        return d / max(n, 1e-12)

    # --- mesh → arrays
    verts_arr, norms_arr, areas_arr, cents_arr, q_arr, phi_arr = \
        triangles_to_numeric_full(triangles)
    faces = np.arange(verts_arr.size // 3).reshape(-1, 3)

    # --- state
    path_points  = [start_pt.copy()]
    total_length = 0.0
    x_current    = start_pt.copy()
    prev_step_dir = None

    # --- seed step (step 0) along average normal (NO pre-flip)
    avg_normal = compute_meyer_normal_jit(
        x_current, verts_arr, norms_arr, areas_arr, cents_arr,
        radius=-1.0, eps=1e-12, debug=False)

    if avg_normal is not None and np.linalg.norm(avg_normal) > 0:
        seed_dir    = avg_normal / np.linalg.norm(avg_normal)
        x_current = x_current + first_step * seed_dir
        total_length += first_step
        path_points.append(x_current.copy())
        prev_step_dir = seed_dir.copy()
        dbg(f"[SEED] moved {first_step:.5f}; seed_dir={seed_dir}")
    else:
        dbg("[SEED] no normal; skipping seed move")

    # --- main loop
    for it in range(1, max_iter + 1):
        g = evaluate_gradient_numba(x_current, verts_arr, norms_arr, q_arr, phi_arr)
        cand_dir = step_dir_from_grad(g)  # candidate step direction (unit)

        # angle guard (compare *step* directions, not raw gradients)
        angle_flip = False
        if prev_step_dir is not None:
            a = float(np.clip(np.dot(cand_dir, prev_step_dir), -1.0, 1.0))
            ang = float(np.degrees(np.arccos(a)))
            if ang > angle_max_deg:
                angle_flip = True
                dbg(f"[{it:03}] ∠flip {ang:.2f}° > {angle_max_deg}° → reuse prev_step_dir")

        used_dir = prev_step_dir if (angle_flip and prev_step_dir is not None) else cand_dir

        old_pt = x_current.copy()
        new_pt = old_pt + alpha_initial * used_dir

        X_int = find_exit_intersection(old_pt, new_pt, triangles)  # ensure tmin ε in this func
        if X_int is not None:
            seg = np.linalg.norm(X_int - old_pt)
            if seg <= 1e-8:
                dbg(f"[{it:03}] EXIT at start (ε) — stopping")
            else:
                total_length += seg
                path_points.append(X_int.copy())
                dbg(f"[{it:03}] EXIT (len={seg:.5f})")
            break

        # accept step
        seg = np.linalg.norm(new_pt - old_pt)
        total_length += seg
        path_points.append(new_pt.copy())
        x_current = new_pt
        prev_step_dir = used_dir.copy()
        dbg(f"[{it:03}] step ok len={seg:.5f}, total={total_length:.5f}")

    dbg(f"[DONE] steps={len(path_points)-1}, total_len={total_length:.5f}")

    if direction == 'down':
        def nearest_vertex_idx(pt, verts):
            flat = verts.reshape(-1, 3)
            return int(np.argmin(np.linalg.norm(flat - pt, axis=1)))

        v0_idx = nearest_vertex_idx(start_pt, verts_arr)
        vn_idx = nearest_vertex_idx(path_points[-1], verts_arr)
        K0, H0 = average_curvature_one_ring(v0_idx, verts_arr, faces, areas_arr)
        Kn, Hn = average_curvature_one_ring(vn_idx, verts_arr, faces, areas_arr)
        _log_thick_curv_pair(total_length, (K0, H0), (Kn, Hn))

    return path_points, total_length

