#!/usr/bin/env python3
import numpy as np
from typing import Tuple
from pathlib import Path
import pickle
from matplotlib.widgets import CheckButtons
import pymeshlab as ml
import pyvista as pv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt


# ----------------------------- Config ----------------------------------------
WHITE_VTK = "simulated_white.vtk"
PIAL_VTK = "simulated_pial.vtk"
OUT_PKL = "zipped_surface.pkl"
PLOT = True  # set False to skip plotting


# ----------------------------- Helpers ---------------------------------------
def _pv_faces_to_tris(mesh: pv.PolyData) -> Tuple[np.ndarray, np.ndarray]:
    """Return (points, faces) where faces is (m,3) int array."""
    points = mesh.points.astype(np.float64, copy=False)
    faces = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32, copy=False)
    return points, faces


def _build_poly(points: np.ndarray, faces: np.ndarray, **kwargs) -> Poly3DCollection:
    """Efficient Poly3DCollection from numpy arrays."""
    return Poly3DCollection(points[faces], **kwargs)


def find_boundary_loop(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Identify and return the single boundary loop as vertex indices.
    Assumes exactly one continuous boundary where each boundary vertex has degree 2.
    """
    e01 = np.sort(faces[:, [0, 1]], axis=1)
    e12 = np.sort(faces[:, [1, 2]], axis=1)
    e20 = np.sort(faces[:, [2, 0]], axis=1)

    uniq, counts = np.unique(
        np.vstack((e01, e12, e20)), axis=0, return_counts=True
    )
    boundary_edges = uniq[counts == 1]
    if boundary_edges.size == 0:
        raise ValueError("No boundary edges found.")

    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    # Start from any degree-2 boundary vertex
    start = next((v for v, nbrs in adj.items() if len(nbrs) == 2), None)
    if start is None:
        raise ValueError("Boundary is not a single 2-regular loop.")

    loop = [start]
    prev = -1
    cur = start
    while True:
        nbrs = adj[cur]
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt

    return np.asarray(loop, dtype=int)


def _edge_lengths(points: np.ndarray, faces: np.ndarray, ring_idx: np.ndarray) -> np.ndarray:
    """All edge lengths of faces incident to any ring vertex."""
    mask = np.isin(faces, ring_idx).any(axis=1)
    F = faces[mask]
    e01 = np.linalg.norm(points[F[:, 1]] - points[F[:, 0]], axis=1)
    e12 = np.linalg.norm(points[F[:, 2]] - points[F[:, 1]], axis=1)
    e20 = np.linalg.norm(points[F[:, 0]] - points[F[:, 2]], axis=1)
    return np.concatenate([e01, e12, e20])


def _loop_perimeter(coords: np.ndarray) -> float:
    return np.linalg.norm(np.roll(coords, -1, axis=0) - coords, axis=1).sum()


def _resample_loop_arclength(coords: np.ndarray, n_out: int) -> np.ndarray:
    seg = np.linalg.norm(np.roll(coords, -1, axis=0) - coords, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    t = np.linspace(0, L, n_out + 1)[:-1]
    res = np.empty((n_out, 3), float)
    j = 0
    for i, ti in enumerate(t):
        while s[j + 1] < ti:
            j += 1
        a = (ti - s[j]) / (s[j + 1] - s[j] + 1e-12)
        res[i] = (1 - a) * coords[j] + a * coords[(j + 1) % len(coords)]
    return res


def _loop_signed_area_like(coords: np.ndarray) -> float:
    # PCA to 2D then polygon area sign
    C = coords - coords.mean(0)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    uv = C @ Vt[:, :2]
    x, y = uv[:, 0], uv[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def _enforce_cw(coords: np.ndarray) -> np.ndarray:
    return coords if _loop_signed_area_like(coords) < 0 else coords[::-1].copy()


def _best_circular_shift(A: np.ndarray, B: np.ndarray) -> int:
    n = len(A)
    scores = np.empty(n)
    for k in range(n):
        scores[k] = np.linalg.norm(A - np.roll(B, -k, axis=0), axis=1).sum()
    return int(np.argmin(scores))

def _fit_plane_normal(coords: np.ndarray) -> np.ndarray:
    C = coords - coords.mean(0)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    # normal is the third singular vector (least variance direction)
    n = Vt[-1]
    # make sign deterministic: point normal roughly “outwards” using centroid->mean offset if you have one
    return n / (np.linalg.norm(n) + 1e-12)

def _signed_area_on_plane(coords: np.ndarray, n_ref: np.ndarray) -> float:
    # Build shared tangent basis (u,v) on the plane
    u = np.cross(n_ref, np.array([1.0,0.0,0.0]))
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(n_ref, np.array([0.0,1.0,0.0]))
    u /= np.linalg.norm(u)
    v = np.cross(n_ref, u)
    P = coords - coords.mean(0)
    x = P @ u
    y = P @ v
    return 0.5 * np.sum(x * np.roll(y,-1) - y * np.roll(x,-1))

def _enforce_same_ccw(pial_loop: np.ndarray, white_loop: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    all_pts = np.vstack([pial_loop, white_loop])
    n_ref = _fit_plane_normal(all_pts)
    if _signed_area_on_plane(pial_loop, n_ref) < 0:
        pial_loop = pial_loop[::-1].copy()
    if _signed_area_on_plane(white_loop, n_ref) < 0:
        white_loop = white_loop[::-1].copy()
    return pial_loop, white_loop


# ----------------------------- Main ------------------------------------------
def main() -> None:
    # 1) Read meshes
    white_mesh = pv.read(WHITE_VTK)
    pial_mesh = pv.read(PIAL_VTK)
    wpoints, wfaces = _pv_faces_to_tris(white_mesh)
    ppoints, pfaces = _pv_faces_to_tris(pial_mesh)
    print(f"White mesh: {wpoints.shape[0]} vertices, {wfaces.shape[0]} faces.")
    print(f"Pial mesh: {ppoints.shape[0]} vertices, {pfaces.shape[0]} faces.")

    # 2) Boundary loops
    white_loop = find_boundary_loop(wpoints, wfaces)
    pial_loop = find_boundary_loop(ppoints, pfaces)
    white_coords = wpoints[white_loop]
    pial_coords = ppoints[pial_loop]
    print(f"White boundary loop length: {len(white_loop)}")
    print(f"Pial boundary loop length: {len(pial_loop)}")

    # ---- STEP 1: measure TEL near the loops ----
    w_edge_len = _edge_lengths(wpoints, wfaces, white_loop)
    p_edge_len = _edge_lengths(ppoints, pfaces, pial_loop)
    TEL = float(np.median(np.concatenate([w_edge_len, p_edge_len])))
    print(f"Target edge length (TEL): {TEL:.6f}")

    # ---- STEP 2: resample both loops by arclength to a common n ----
    Lp = _loop_perimeter(pial_coords)
    Lw = _loop_perimeter(white_coords)
    n = int(max(Lp, Lw) / max(TEL, 1e-9))
    n = int(np.clip(n, 64, 4096))
    pial_rs  = _resample_loop_arclength(pial_coords, n)
    white_rs = _resample_loop_arclength(white_coords, n)
    pial_rs, white_rs = _enforce_same_ccw(pial_rs, white_rs)
    print(f"Resampled both loops to n={n} vertices.")


    # ---- STEP 4: build quad strip with best circular shift ----
    k = _best_circular_shift(pial_rs, white_rs)
    white_aligned = np.roll(white_rs, -k, axis=0)
    side_points_init = np.vstack([pial_rs, white_aligned])
    off = n
    idx = np.arange(n, dtype=int)
    upper = np.column_stack((idx, (idx + 1) % n, off + idx))
    lower = np.column_stack(((idx + 1) % n, off + (idx + 1) % n, off + idx))
    side_faces_init = np.vstack([upper, lower]).astype(np.int32)
    print(f"Initial side mesh: {side_points_init.shape[0]} verts, {side_faces_init.shape[0]} tris")

    # ---- STEP 6: isotropic remesh (boundary locked) to TEL ----
    try:
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(side_points_init, side_faces_init), "side_init")
        side_target = ml.PureValue(float(TEL))
        ms.apply_filter(
            'meshing_isotropic_explicit_remeshing',
            iterations=3,
            targetlen=side_target
        )
        side_points = np.asarray(ms.current_mesh().vertex_matrix(), float)
        side_faces = np.asarray(ms.current_mesh().face_matrix(), int)
        print(f"Remeshed side: {side_points.shape[0]} verts, {side_faces.shape[0]} tris")
    except Exception as e:
        print(f"[WARN] Remesh skipped ({e}); using initial side mesh.")
        side_points, side_faces = side_points_init, side_faces_init

    # ---- STEP 6: Taubin smoothing ----
    try:
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(side_points, side_faces), "side_remeshed")
        try:
            ms.apply_filter(
                'meshing_surface_smooth_taubin',
                stepsmoothnum=2,
                lambda_=0.25,
                mu=-0.34
            )
        except Exception:
            ms.apply_filter(
                'apply_coord_taubin_smoothing',
                stepsmoothnum=2,
                lambda_=0.25,
                mu=-0.34
            )
        side_points = np.asarray(ms.current_mesh().vertex_matrix(), float)
        side_faces = np.asarray(ms.current_mesh().face_matrix(), int)
        print("Applied Taubin smoothing to side mesh.")
    except Exception as e:
        print(f"[WARN] Smoothing skipped ({e}).")

    # ---- STEP 7: orient all faces outward wrt interior ----
    pfaces_o, wfaces_o, sfaces_o = (pfaces, wfaces, side_faces)  # placeholder

    # ---- STEP 9: pack and save BEM triangles (same format as before) ----
    all_triangles = []
    all_triangles += []  # placeholder
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(
            {'triangles': all_triangles, 'TEL': TEL, 'meta': {'n_resampled': int(n)}},
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )
    print(f"Saved BEM-ready pickle: {OUT_PKL}")

    # ------------------------- Optional plotting ------------------------------
    if PLOT:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        white_poly = _build_poly(wpoints, wfaces_o, facecolor='silver', edgecolor='k', alpha=1.0, label='White')
        pial_poly = _build_poly(ppoints, pfaces_o, facecolor='pink', edgecolor='k', alpha=1.0, label='Pial')
        side_poly = _build_poly(side_points, sfaces_o, facecolor='lime', edgecolor='k', alpha=1.0, label='Side')
        for poly in (white_poly, pial_poly, side_poly):
            ax.add_collection3d(poly)
        all_coords = np.vstack((wpoints, ppoints, side_points))
        mins = all_coords.min(axis=0)
        maxs = all_coords.max(axis=0)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_box_aspect(maxs - mins)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("White, Pial, and Side (Remeshed & Oriented)")
        ax.legend(loc='upper right')
        plt.show()


if __name__ == "__main__":
    main()
