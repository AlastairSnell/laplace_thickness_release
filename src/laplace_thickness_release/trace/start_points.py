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

def _pick_even_start_points_vertices(
    triangles: list[dict],
    pct: float,
    target_spacing: float | None,
    max_points: int | None,
    *,
    bc_value: float,
) -> np.ndarray:
    # 1) Collect Dirichlet surface vertices/faces for the requested bc_value
    top_verts = {}
    faces_top = []
    for tri in triangles:
        if tri["bc_type"] == "dirichlet" and np.isclose(tri["bc_value"], float(bc_value)):
            idxs = []
            for v in tri["vertices"]:
                key = tuple(np.asarray(v, dtype=np.float64))
                idxs.append(top_verts.setdefault(key, len(top_verts)))
            faces_top.append(idxs)

    if not top_verts:
        raise ValueError(f"No Dirichlet={float(bc_value)} triangles found.")

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
        # Ignore disconnected vertices (inf distances) when choosing next seed.
        d_masked = np.where(np.isfinite(d_min), d_min, -np.inf)
        if not np.any(np.isfinite(d_masked)):
            break
        nxt = int(np.nanargmax(d_masked))
        if float(d_masked[nxt]) < float(target_spacing):
            break
        chosen.append(nxt)
        coords.append(Vc[nxt])
        if max_points is not None and len(chosen) >= max_points:
            break
        d_new = np.asarray(dijkstra(Gc, directed=False, indices=nxt))
        d_min = np.minimum(d_min, d_new)

    return np.asarray(coords, dtype=np.float64)


def _pick_even_start_points_centroids(
    triangles: list[dict],
    pct: float,
    target_spacing: float | None,
    max_points: int | None,
    *,
    bc_value: float,
    return_indices: bool = False,
) -> np.ndarray:
    # 1) Collect Dirichlet surface triangles for the requested bc_value
    tri_verts = []
    centroids = []
    tri_indices = []
    for tri_idx, tri in enumerate(triangles):
        if tri["bc_type"] == "dirichlet" and np.isclose(tri["bc_value"], float(bc_value)):
            verts = np.asarray(tri["vertices"], dtype=np.float64)
            tri_verts.append(verts)
            if "centroid" in tri:
                centroids.append(np.asarray(tri["centroid"], dtype=np.float64))
            else:
                centroids.append(np.mean(verts, axis=0))
            tri_indices.append(tri_idx)

    if not tri_verts:
        raise ValueError(f"No Dirichlet={float(bc_value)} triangles found.")

    C = np.asarray(centroids, dtype=np.float64)
    n_faces = len(tri_verts)

    # 2) Weighted graph over triangle adjacency (shared edges)
    edge_to_tris: dict[tuple[tuple[float, float, float], tuple[float, float, float]], list[int]] = {}
    for t_idx, verts in enumerate(tri_verts):
        keys = [tuple(np.asarray(v, dtype=np.float64)) for v in verts]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a = keys[i]
            b = keys[j]
            edge = (a, b) if a < b else (b, a)
            edge_to_tris.setdefault(edge, []).append(t_idx)

    rows, cols, wts = [], [], []
    boundary = set()
    for tris in edge_to_tris.values():
        if len(tris) == 1:
            boundary.add(tris[0])
            continue
        for i in range(len(tris)):
            for j in range(i + 1, len(tris)):
                a = tris[i]
                b = tris[j]
                d = float(np.linalg.norm(C[a] - C[b]))
                rows += [a, b]
                cols += [b, a]
                wts += [d, d]

    if rows:
        G = coo_matrix((wts, (rows, cols)), shape=(n_faces, n_faces)).tocsr()
    else:
        G = coo_matrix((n_faces, n_faces)).tocsr()

    # 3) Reverse cap: distance-to-boundary
    boundary_arr = np.asarray(sorted(boundary), dtype=int)
    if boundary_arr.size == 0:
        d_to_boundary = np.full(n_faces, np.inf, dtype=np.float64)
        admissible = np.arange(n_faces, dtype=int)
    else:
        D = dijkstra(G, directed=False, indices=boundary_arr)
        d_to_boundary = np.min(D, axis=0)
        inradius = float(np.nanmax(d_to_boundary[np.isfinite(d_to_boundary)]))
        thresh = (1.0 - pct / 100.0) * inradius
        admissible = np.where(d_to_boundary >= thresh)[0]
        if admissible.size == 0:
            raise ValueError("Reverse-cap produced empty admissible set. Increase pct.")

    Cc = C[admissible]
    Gc = G[admissible][:, admissible]

    # 4) Choose spacing if absent
    if target_spacing is None:
        d0 = dijkstra(Gc, directed=False, indices=0)
        a0 = int(np.nanargmax(d0))
        da = dijkstra(Gc, directed=False, indices=a0)
        diam = float(np.nanmax(da[np.isfinite(da)]))
        target_spacing = 0.05 * diam if np.isfinite(diam) and diam > 0 else 0.0

    # 5) FPS in geodesic metric (triangle-centroid graph)
    first = int(np.argmax(d_to_boundary[admissible]))
    chosen = [first]
    coords = [Cc[first]]

    d_min = np.asarray(dijkstra(Gc, directed=False, indices=first))
    while True:
        # Ignore disconnected faces (inf distances) when choosing next seed.
        d_masked = np.where(np.isfinite(d_min), d_min, -np.inf)
        if not np.any(np.isfinite(d_masked)):
            break
        nxt = int(np.nanargmax(d_masked))
        if float(d_masked[nxt]) < float(target_spacing):
            break
        chosen.append(nxt)
        coords.append(Cc[nxt])
        if max_points is not None and len(chosen) >= max_points:
            break
        d_new = np.asarray(dijkstra(Gc, directed=False, indices=nxt))
        d_min = np.minimum(d_min, d_new)

    coords_arr = np.asarray(coords, dtype=np.float64)
    if return_indices:
        chosen_full = [int(admissible[i]) for i in chosen]
        tri_idx_out = [int(tri_indices[i]) for i in chosen_full]
        return coords_arr, np.asarray(tri_idx_out, dtype=int)

    return coords_arr


def pick_even_start_points(
    triangles: list[dict],
    pct: float = 50.0,
    target_spacing: float | None = None,
    max_points: int | None = None,
    *,
    bc_value: float = 1.0,
    seed_mode: str = "vertex",
    return_indices: bool = False,
) -> np.ndarray:
    """
    Keep a "core" set of surface points far from the boundary, then do geodesic FPS.

    Returns (M,3) float64 start points on Dirichlet=bc_value surface.
    """
    mode = str(seed_mode).strip().lower()
    if mode == "vertex":
        if return_indices:
            raise ValueError("return_indices is only supported for seed_mode='centroid'.")
        return _pick_even_start_points_vertices(
            triangles,
            pct,
            target_spacing,
            max_points,
            bc_value=bc_value,
        )
    if mode == "centroid":
        return _pick_even_start_points_centroids(
            triangles,
            pct,
            target_spacing,
            max_points,
            bc_value=bc_value,
            return_indices=return_indices,
        )
    raise ValueError(f"Unknown seed_mode '{seed_mode}'. Use 'vertex' or 'centroid'.")
