from __future__ import annotations

from typing import Any, Sequence, Tuple, List

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


__all__ = ["compute_thickness_heatmap"]


def _build_vertex_graph(triangles: Sequence[dict[str, Any]]):
    """
    Build unique-vertex array V, face index array F, and a symmetric
    adjacency matrix A for geodesic distances on the surface.
    """
    if not triangles:
        raise ValueError("Cannot build vertex graph from empty triangle list.")

    all_verts = np.vstack([tri["vertices"] for tri in triangles])
    V, inverse = np.unique(all_verts, axis=0, return_inverse=True)
    F = inverse.reshape(-1, 3).astype(np.int32)

    # Undirected edges: (0->1, 1->2, 2->0) + symmetric counterpart
    i = F[:, [0, 1, 2]].ravel()
    j = F[:, [1, 2, 0]].ravel()
    w = np.linalg.norm(V[i] - V[j], axis=1)

    nV = len(V)
    A = coo_matrix((w, (i, j)), shape=(nV, nV))
    A = A.maximum(A.T).tocsr()
    return V, F, A


def _geodesic_idw(
    V: np.ndarray,
    A_sparse,
    sample_xyz: np.ndarray,
    sample_vals: np.ndarray,
    *,
    power: float = 2.0,
    logger: Any | None = None,
) -> np.ndarray:
    """
    Interpolate scalar values on a mesh using inverse-distance weighting
    with *geodesic* distance.
    """
    V = np.ascontiguousarray(V, dtype=np.float64)
    sample_xyz = np.ascontiguousarray(sample_xyz, dtype=np.float64)
    sample_vals = np.asarray(sample_vals, dtype=float)

    kd = KDTree(V)
    src_idx = kd.query(sample_xyz)[1].astype(np.int32)

    if logger is not None:
        logger.info(
            "Computing geodesic distance matrix (samples=%d, vertices=%d).",
            len(src_idx),
            len(V),
        )

    D = dijkstra(A_sparse, directed=False, indices=src_idx)

    W = 1.0 / np.maximum(D, 1e-6) ** power
    acc_num = (W * sample_vals[:, None]).sum(axis=0)
    acc_den = W.sum(axis=0)

    out = acc_num / np.maximum(acc_den, 1e-12)

    if logger is not None:
        logger.info("Geodesic IDW interpolation complete.")

    return out


def _clean_thickness_samples(
    sample_coords: np.ndarray,
    sample_thick: np.ndarray,
    *,
    lower_bound: float,
    k_neigh: int,
    logger: Any | None = None,
) -> np.ndarray:
    """
    Replace clearly-too-small path lengths using nearby good samples.
    """
    sample_coords = np.asarray(sample_coords, dtype=float)
    clean = np.asarray(sample_thick, dtype=float).copy()

    kd = KDTree(sample_coords)
    low_mask = clean < lower_bound
    good_mask = ~low_mask

    if logger is not None:
        n_bad = int(low_mask.sum())
        if n_bad > 0:
            logger.info(
                "Cleaning %d suspiciously small thickness samples ( < %.2f mm ).",
                n_bad,
                lower_bound,
            )

    for idx in np.where(low_mask)[0]:
        pt = sample_coords[idx]
        _, all_idx = kd.query(pt, k=len(sample_coords))

        neigh_idx = [j for j in all_idx if clean[j] >= lower_bound][:k_neigh]

        if len(neigh_idx) < k_neigh:
            if logger is not None:
                logger.warning(
                    "Index %d kept at %.3f mm (not enough good neighbours).",
                    idx,
                    clean[idx],
                )
            continue

        new_val = float(np.median(clean[neigh_idx]))
        if new_val < lower_bound:
            if logger is not None:
                logger.warning(
                    "Index %d upgrade still too thin (%.3f mm) – leaving original value %.3f mm.",
                    idx,
                    new_val,
                    clean[idx],
                )
            continue

        clean[idx] = new_val
        good_mask[idx] = True

    return clean


def compute_thickness_heatmap(
    *,
    triangles: Sequence[dict[str, Any]],
    sample_coords: np.ndarray,
    sample_thickness: np.ndarray,
    pct: float,
    lower_bound: float,
    power: float,
    k_neigh: int,
    logger: Any | None = None,
) -> Tuple[List[dict[str, Any]], np.ndarray, List[dict[str, Any]]]:
    """
    Build a cortical thickness heatmap on the pial surface using geodesic-IDW,
    and return:

        1) trimmed_top_tris     : list of pial (top) triangles (inner pct patch)
        2) trimmed_top_vals     : per-triangle thickness values for these pial faces
        3) trimmed_bottom_tris  : list of bottom (white) triangles cropped to the
                                  nearest pct% by geodesic distance on the white surface.

    Notes:
      - Geodesics for the heatmap are computed on the pial surface only.
      - Pial and white surfaces each get their own geodesic graph.
      - For both surfaces we keep the INNER pct% of faces (no rim / donut).
    """
    if logger is not None:
        logger.info(
            "Building mesh graph + geodesic IDW thickness heatmap (pct=%.1f, power=%.1f, lower_bound=%.2f).",
            pct,
            power,
            lower_bound,
        )

    if pct <= 0.0 or pct > 100.0:
        raise ValueError("--heatmap-pct must be in (0, 100].")

    if pct > 50.0 and logger is not None:
        logger.warning(
            "Heatmap pct %.1f > 50: thickness accuracy is less reliable close to the patch edges.",
            pct,
        )

    # ----------------- Partition triangles into pial / white / side -----------------
    top_tris: List[dict[str, Any]] = []
    bottom_tris: List[dict[str, Any]] = []

    for tri in triangles:
        bc_type = str(tri.get("bc_type", "")).lower()
        bc_val = float(tri.get("bc_value", 0.0))

        if bc_type == "neumann":
            # side wall: never part of the heatmap patch
            continue

        if bc_type == "dirichlet" and np.isclose(bc_val, 1.0):
            top_tris.append(tri)
        elif bc_type == "dirichlet" and np.isclose(bc_val, 0.0):
            bottom_tris.append(tri)

    if not top_tris:
        raise RuntimeError("No pial faces (Dirichlet bc_value≈1.0) found for heatmap creation.")

    if logger is not None and not bottom_tris:
        logger.warning("Heatmap: no white (Dirichlet bc_value≈0.0) faces found – bottom patch will be empty.")

    # ----------------- Clean sample thickness values -----------------
    sample_coords = np.asarray(sample_coords, dtype=float)
    sample_thickness = np.asarray(sample_thickness, dtype=float)

    clean_thick = _clean_thickness_samples(
        sample_coords,
        sample_thickness,
        lower_bound=lower_bound,
        k_neigh=k_neigh,
        logger=logger,
    )

    # ----------------- Build pial-only graph and do IDW -----------------
    V_top, F_top, A_top = _build_vertex_graph(top_tris)

    pial_thick_by_vert = _geodesic_idw(
        V_top,
        A_top,
        sample_coords,
        clean_thick,
        power=power,
        logger=logger,
    )

    # ----------------- Seed on pial surface (centre of samples) -----------------
    kd_top = KDTree(V_top)
    seed_xyz = sample_coords.mean(axis=0)
    seed_vert_idx_top = int(kd_top.query(seed_xyz)[1])

    if logger is not None:
        logger.info(
            "Computing seed-to-vertex geodesics on pial surface from vertex %d.",
            seed_vert_idx_top,
        )

    seed_dist_top = dijkstra(A_top, directed=False, indices=[seed_vert_idx_top])[0]

    # Per-face distance and value on pial surface
    face_dist_top = seed_dist_top[F_top].mean(axis=1)           # (F_top,)
    face_vals_top = pial_thick_by_vert[F_top].mean(axis=1)      # (F_top,)

    # ----------------- Keep INNER pct% of pial faces -----------------
    n_pial = face_dist_top.shape[0]
    keep_n_pial = max(1, int(round(n_pial * (pct / 100.0))))
    keep_n_pial = min(n_pial, keep_n_pial)

    order_top = np.argsort(face_dist_top)  # closest first
    keep_mask_top = np.zeros_like(face_dist_top, dtype=bool)
    keep_mask_top[order_top[:keep_n_pial]] = True

    if logger is not None:
        logger.info("Heatmap: kept %d pial faces (nearest %.1f%%).", int(keep_mask_top.sum()), pct)

    trimmed_top_tris: List[dict[str, Any]] = [top_tris[i] for i in np.where(keep_mask_top)[0]]
    trimmed_top_vals = face_vals_top[keep_mask_top]

    # ----------------- Build white-only graph and crop INNER pct% -----------------
    trimmed_bottom_tris: List[dict[str, Any]] = []

    if bottom_tris:
        V_bottom, F_bottom, A_bottom = _build_vertex_graph(bottom_tris)

        # Use same spatial seed (centre of samples), but mapped onto white surface
        kd_bottom = KDTree(V_bottom)
        seed_vert_idx_bottom = int(kd_bottom.query(seed_xyz)[1])

        if logger is not None:
            logger.info(
                "Computing seed-to-vertex geodesics on white surface from vertex %d.",
                seed_vert_idx_bottom,
            )

        seed_dist_bottom = dijkstra(A_bottom, directed=False, indices=[seed_vert_idx_bottom])[0]
        face_dist_bottom = seed_dist_bottom[F_bottom].mean(axis=1)  # (F_bottom,)

        n_bottom = face_dist_bottom.shape[0]
        keep_n_bottom = max(1, int(round(n_bottom * (pct / 100.0))))
        keep_n_bottom = min(n_bottom, keep_n_bottom)

        order_bottom = np.argsort(face_dist_bottom)  # closest first
        keep_mask_bottom = np.zeros_like(face_dist_bottom, dtype=bool)
        keep_mask_bottom[order_bottom[:keep_n_bottom]] = True

        n_kept = int(keep_mask_bottom.sum())
        if n_kept == 0:
            # Very defensive fallback – should basically never happen.
            if logger is not None:
                logger.warning(
                    "Heatmap: cropped white patch is empty at pct=%.1f; falling back to all white faces.",
                    pct,
                )
            keep_mask_bottom[:] = True
            n_kept = n_bottom

        trimmed_bottom_tris = [bottom_tris[i] for i in np.where(keep_mask_bottom)[0]]

        if logger is not None:
            logger.info("Heatmap: kept %d bottom faces (nearest %.1f%%).", n_kept, pct)
    else:
        if logger is not None:
            logger.warning("Heatmap: no white (bc_value≈0.0) faces found.")

    return trimmed_top_tris, trimmed_top_vals, trimmed_bottom_tris
