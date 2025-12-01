#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pyvista as pv
import pickle
from collections import defaultdict
from scipy.spatial import cKDTree
import pymeshlab as ml  # <-- added


# ---------------------------- outward orientation ---------------------------- #

def orient_outward_with_pymeshlab(
    ppoints: np.ndarray, pfaces: np.ndarray,
    wpoints: np.ndarray, wfaces: np.ndarray,
    spoints: np.ndarray, sfaces: np.ndarray,
    *,
    use_weld: bool = False,          # keep False to preserve your per-part indexing
    weld_tol: float = 1e-7,
    rays: int = 64,
    parity_sampling: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Re-orient faces for (pial, white, side) so that:
      1) face windings are coherent within each connected component
      2) normals point OUTWARD (geometry-based visibility)

    Returns: (pfaces_oriented, wfaces_oriented, sfaces_oriented)
    Face arrays have the SAME shapes as inputs. When use_weld=False (default),
    indices remain aligned with the original point arrays.
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
                ms.apply_filter('meshing_merge_close_vertices',
                                threshold=ml.AbsoluteValue(float(weld_tol)))
            except Exception:
                ms.apply_filter('meshing_merge_close_vertices',
                                threshold=float(weld_tol))
        except Exception as e:
            print(f"[WARN] Vertex welding skipped ({e}). Proceeding without weld.")

    try:
        ms.apply_filter('meshing_re_orient_faces_coherently')
    except Exception as e:
        raise RuntimeError(
            "PyMeshLab filter 'meshing_re_orient_faces_coherently' not available on this build."
        ) from e

    try:
        ms.apply_filter('meshing_re_orient_faces_by_geometry',
                        rays=int(rays), parity_sampling=bool(parity_sampling))
    except Exception:
        try:
            ms.apply_filter('meshing_re_orient_faces_by_geometry', rays=int(rays))
        except Exception as e:
            print(f"[WARN] Geometry-based outward orientation unavailable ({e}). "
                  "Normals will be coherent, but outward vs inward may be arbitrary.")

    F_or = np.asarray(ms.current_mesh().face_matrix(), dtype=np.int32)

    Fp_or = F_or[:nFp].copy()
    Fw_or = F_or[nFp:nFp+nFw].copy()
    Fs_or = F_or[nFp+nFw:nFp+nFw+nFs].copy()

    Fw_or -= off_w
    Fs_or -= off_s

    return Fp_or, Fw_or, Fs_or


# ---------------------------- geometry builders ---------------------------- #

def _hexagonal_grid_points(radius: float, edge_len: float) -> np.ndarray:
    """Roughly hex-packed interior points for a disc of given radius."""
    dx = edge_len
    dy = edge_len * np.sqrt(3) / 2.0

    nx = int(np.ceil(radius / dx))
    ny = int(np.ceil(radius / dy))

    pts = []
    for j in range(-ny, ny + 1):
        row_shift = (j & 1) * 0.5 * dx  # shift every second row
        y = j * dy
        for i in range(-nx, nx + 1):
            x = i * dx + row_shift
            if x * x + y * y < (radius - 1e-6) ** 2:
                pts.append((x, y))
    return np.asarray(pts, dtype=float)


def make_disc(
    radius: float,
    n_boundary: int,
    z: float = 0.0,
    edge_len: float | None = None,
) -> tuple[pv.PolyData, np.ndarray]:
    """Create a flat triangulated disc at height z, with hex-packed interior."""
    if n_boundary < 3:
        raise ValueError("n_boundary must be ≥ 3")

    theta = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
    xy_boundary = np.stack((radius * np.cos(theta), radius * np.sin(theta)), axis=1)

    if edge_len is None:
        edge_len = 2.0 * np.pi * radius / n_boundary  # boundary chord length
    xy_interior = _hexagonal_grid_points(radius, edge_len)

    boundary_pts = np.hstack((xy_boundary, np.full((n_boundary, 1), z, dtype=float)))
    interior_pts = np.hstack((xy_interior, np.full((xy_interior.shape[0], 1), z, dtype=float)))
    points = np.vstack((boundary_pts, interior_pts))

    cloud = pv.PolyData(points)
    mesh = cloud.delaunay_2d()

    centre = np.array([0.0, 0.0, z], dtype=float)
    return mesh, centre


# ---------------------------- zipper helpers ---------------------------- #

def _pv_faces_to_tris(mesh: pv.PolyData) -> Tuple[np.ndarray, np.ndarray]:
    """Return (points, faces) where faces is (m,3) int array."""
    points = mesh.points.astype(np.float64, copy=False)
    faces = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32, copy=False)
    return points, faces


def find_boundary_loop(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Identify the single boundary loop (vertex indices)."""
    e01 = np.sort(faces[:, [0, 1]], axis=1)
    e12 = np.sort(faces[:, [1, 2]], axis=1)
    e20 = np.sort(faces[:, [2, 0]], axis=1)
    all_edges = np.vstack((e01, e12, e20))

    uniq, counts = np.unique(all_edges, axis=0, return_counts=True)
    boundary_edges = uniq[counts == 1]
    if boundary_edges.size == 0:
        raise ValueError("No boundary edges found.")

    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    start = next((v for v, nbrs in adj.items() if len(nbrs) == 2), None)
    if start is None:
        raise ValueError("Boundary is not a single 2-regular loop.")

    loop = [start]
    prev = -1
    current = start
    while True:
        nbrs = adj[current]
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        if nxt == start:
            break
        loop.append(nxt)
        prev, current = current, nxt

    return np.asarray(loop, dtype=int)


def robust_match_loops(loop_small_coords: np.ndarray, loop_large_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bidirectional NN loop matching. Returns (side_points, side_faces)."""
    if len(loop_small_coords) <= len(loop_large_coords):
        small, large = loop_small_coords, loop_large_coords
    else:
        small, large = loop_large_coords, loop_small_coords

    n_small, n_large = len(small), len(large)
    t_small, t_large = cKDTree(small), cKDTree(large)

    _, large_to_small = t_small.query(large)   # (n_large,)
    _, small_to_large = t_large.query(small)   # (n_small,)

    matches = np.column_stack((np.arange(n_large, dtype=int), large_to_small))
    used_small = set(large_to_small.tolist())
    missing_small = np.setdiff1d(np.arange(n_small, dtype=int), np.fromiter(used_small, int))

    if missing_small.size:
        extra = np.column_stack((small_to_large[missing_small], missing_small))
        matches = np.vstack((matches, extra))

    matches = matches[np.argsort(matches[:, 0])]
    L = matches[:, 0]
    S = matches[:, 1]

    L_next = np.roll(L, -1)
    S_next = np.roll(S, -1)

    side_points = np.vstack((large, small))
    s_offset = n_large

    faces_upper = np.column_stack((L, L_next, S + s_offset))
    faces_lower = np.column_stack((L_next, S_next + s_offset, S + s_offset))
    side_faces = np.vstack((faces_upper, faces_lower)).astype(int, copy=False)
    return side_points, side_faces


def faces_to_tri_dicts(points: np.ndarray, faces: np.ndarray, bc_type: str, bc_value: float,
                       atol: float = 1e-12) -> list[dict]:
    """Convert triangles to dicts with vertices, unit normal, and BC tags."""
    V = points[faces].astype(np.float64, copy=False)  # (m,3,3)

    dup01 = np.all(np.isclose(V[:, 0], V[:, 1], rtol=0.0, atol=atol), axis=1)
    dup12 = np.all(np.isclose(V[:, 1], V[:, 2], rtol=0.0, atol=atol), axis=1)
    dup20 = np.all(np.isclose(V[:, 2], V[:, 0], rtol=0.0, atol=atol), axis=1)
    dup_mask = dup01 | dup12 | dup20

    finite_mask = np.isfinite(V).all(axis=(1, 2))

    e1 = V[:, 1] - V[:, 0]
    e2 = V[:, 2] - V[:, 0]
    cross = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    nondeg_mask = area >= atol

    valid = finite_mask & (~dup_mask) & nondeg_mask
    Vv = V[valid]
    Cv = cross[valid]

    norms = np.linalg.norm(Cv, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    Nv = (Cv / norms).astype(np.float64, copy=False)

    tris = [{'vertices': v, 'normal': n, 'bc_type': bc_type, 'bc_value': bc_value}
            for v, n in zip(Vv, Nv)]
    return tris


# ---------------------------- utils & CLI ---------------------------- #

def _fmt_num_for_filename(x: float) -> str:
    s = f"{x:.6g}"
    return s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate zipped disc(s) for given thicknesses and save PKL + outer seed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--radius", type=float, default=15.0, help="Disc radius [mm]")
    parser.add_argument("--resolution", type=int, default=60, help="Number of rim vertices")
    parser.add_argument("--edge-len", type=float, default=None, help="Approx. interior edge length [mm] (defaults to boundary chord length)")
    parser.add_argument(
        "--thicknesses", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 3, 4],
        help="List of thickness values [mm], e.g. --thicknesses 0.25 0.5 1 2 4"
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(r"C:/Users/uqasnell/Documents/GitHub/laplace_thickness/validation/data/analytical/disks"),
        help="Output directory for .pkl and seed .npy files"
    )
    return parser.parse_args()


# ---------------------------- main pipeline ---------------------------- #

def build_and_save_one(radius: float, thickness: float, n_boundary: int, edge_len: float | None, out_dir: Path) -> None:
    """Create two discs separated by 'thickness', zip them, orient outward, and write outputs."""
    # Build discs
    white_mesh, white_seed = make_disc(radius, n_boundary, z=0.0, edge_len=edge_len)
    pial_mesh,  pial_seed  = make_disc(radius, n_boundary, z=thickness, edge_len=edge_len)

    # Extract (points, faces)
    wpoints, wfaces = _pv_faces_to_tris(white_mesh)
    ppoints, pfaces = _pv_faces_to_tris(pial_mesh)

    # Boundary loops -> coords
    white_loop = find_boundary_loop(wpoints, wfaces)
    pial_loop  = find_boundary_loop(ppoints, pfaces)
    white_coords = wpoints[white_loop]
    pial_coords  = ppoints[pial_loop]

    # Zipper side wall
    side_points, side_faces = robust_match_loops(white_coords, pial_coords)

    # ---- NEW: orient all faces outward for the closed shell ----
    pfaces_or, wfaces_or, sfaces_or = orient_outward_with_pymeshlab(
        ppoints, pfaces, wpoints, wfaces, side_points, side_faces,
        use_weld=False, weld_tol=1e-7, rays=128, parity_sampling=True
    )

    # Pack triangles with BCs using *oriented* faces
    all_triangles = []
    all_triangles += faces_to_tri_dicts(ppoints, pfaces_or, 'dirichlet', 1.0)
    all_triangles += faces_to_tri_dicts(wpoints, wfaces_or, 'dirichlet', 0.0)
    all_triangles += faces_to_tri_dicts(side_points, sfaces_or, 'neumann', 0.0)

    # Filenames
    r_str = _fmt_num_for_filename(radius)
    t_str = _fmt_num_for_filename(thickness)
    pkl_path = out_dir / f"{r_str}mm_{t_str}.pkl"
    outer_seed_path = out_dir / f"{r_str}mm_{t_str}_os.npy"

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, 'wb') as f:
        pickle.dump({'triangles': all_triangles}, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Outer seed = pial centre [0,0,thickness]
    np.save(outer_seed_path, pial_seed)

    # Console summary
    print(f"Saved: {pkl_path.name} (white V={wpoints.shape[0]}, pial V={ppoints.shape[0]})")
    print(f"Saved: {outer_seed_path.name} (outer seed = {pial_seed.tolist()})")


def main() -> None:
    args = parse_args()
    for t in args.thicknesses:
        build_and_save_one(
            radius=args.radius,
            thickness=float(t),
            n_boundary=args.resolution,
            edge_len=args.edge_len,
            out_dir=args.out_dir,
        )
    print("\nAll done.")


if __name__ == "__main__":
    main()
