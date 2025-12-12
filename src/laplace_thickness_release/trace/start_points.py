from __future__ import annotations

import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

def _find_boundary_vertices(F: np.ndarray) -> np.ndarray:
    edges = []
    for a, b, c in F:
        for i, j in ((a, b), (b, c), (c, a)):
            if i < j:
                edges.append((i, j))
            else:
                edges.append((j, i))
    cnt = Counter(edges)
    b_verts = np.unique(np.array([v for e, k in cnt.items() if k == 1 for v in e], dtype=int))
    return b_verts

def pick_even_start_points(
    triangles: list[dict],
    pct: float = 50.0,
    target_spacing: float | None = None,
    max_points: int | None = None,
) -> np.ndarray:
    """
    Keep a "core" set of pial vertices far from the boundary, then do geodesic FPS.

    Returns (M,3) float64 start points on Dirichlet=1 surface.
    """
    # 1) Collect Dirichlet=1 surface vertices/faces
    top_verts = {}
    faces_top = []
    for tri in triangles:
        if tri["bc_type"] == "dirichlet" and np.isclose(tri["bc_value"], 1.0):
            idxs = []
            for v in tri["vertices"]:
                key = tuple(np.asarray(v, dtype=np.float64))
                idxs.append(top_verts.setdefault(key, len(top_verts)))
            faces_top.append(idxs)

    if not top_verts:
        raise ValueError("No Dirichlet=1 triangles found.")

    V = np.asarray(list(top_verts.keys()), dtype=np.float64)
    F = np.asarray(faces_top, dtype=np.int32)

    # 2) Weighted graph
    rows, cols, wts = [], [], []
    for a, b, c in F:
        for i, j in ((a, b), (b, c), (c, a)):
            d = float(np.linalg.norm(V[i] - V[j]))
            rows += [i, j]
            cols += [j, i]
            wts  += [d, d]
    G = coo_matrix((wts, (rows, cols)), shape=(len(V), len(V))).tocsr()

    # 3) Reverse cap: distance-to-boundary
    boundary = _find_boundary_vertices(F)
    if boundary.size == 0:
        d_to_boundary = np.full(len(V), np.inf, dtype=np.float64)
        admissible = np.arange(len(V), dtype=int)
    else:
        D = dijkstra(G, directed=False, indices=boundary)
        d_to_boundary = np.min(D, axis=0)
        inradius = float(np.nanmax(d_to_boundary[np.isfinite(d_to_boundary)]))
        thresh = (1.0 - pct / 100.0) * inradius
        admissible = np.where(d_to_boundary >= thresh)[0]
        if admissible.size == 0:
            raise ValueError("Reverse-cap produced empty admissible set. Increase pct.")

    Vc = V[admissible]
    Gc = G[admissible][:, admissible]

    # 4) Choose spacing if absent
    if target_spacing is None:
        d0 = dijkstra(Gc, directed=False, indices=0)
        a0 = int(np.nanargmax(d0))
        da = dijkstra(Gc, directed=False, indices=a0)
        diam = float(np.nanmax(da[np.isfinite(da)]))
        target_spacing = 0.05 * diam if np.isfinite(diam) and diam > 0 else 0.0

    # 5) FPS in geodesic metric
    first = int(np.argmax(d_to_boundary[admissible]))
    chosen = [first]
    coords = [Vc[first]]

    d_min = np.asarray(dijkstra(Gc, directed=False, indices=first))
    while True:
        nxt = int(np.nanargmax(d_min))
        if not np.isfinite(d_min[nxt]) or float(d_min[nxt]) < float(target_spacing):
            break
        chosen.append(nxt)
        coords.append(Vc[nxt])
        if max_points is not None and len(chosen) >= max_points:
            break
        d_new = np.asarray(dijkstra(Gc, directed=False, indices=nxt))
        d_min = np.minimum(d_min, d_new)

    return np.asarray(coords, dtype=np.float64)
