#!/usr/bin/env python3
"""Generate zipped upper-hemisphere phantoms at multiple thicknesses.

For each thickness t this writes:
  - Pickle with BC-tagged triangles + meshes:
        {R}mm_{t}.pkl           e.g., 3mm_0.5.pkl
  - OUTER seed point at the pole:
        {R}mm_{t}_os.npy        e.g., 3mm_0.5_os.npy

BC encoding:
  - Outer (pial) hemisphere  : Dirichlet 1.0
  - Inner (white) hemisphere : Dirichlet 0.0
  - Side band at equator     : Neumann  0.0
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pickle
import pyvista as pv
from scipy.spatial import cKDTree
import pymeshlab as ml  # <-- added

# ---------------------------------------------------------------------
# Outward orientation (pial, white, side as a closed shell)
# ---------------------------------------------------------------------

def orient_outward_with_pymeshlab(
    ppoints: np.ndarray, pfaces: np.ndarray,
    wpoints: np.ndarray, wfaces: np.ndarray,
    spoints: np.ndarray, sfaces: np.ndarray,
    *,
    use_weld: bool = False,          # keep False to preserve per-part indexing
    weld_tol: float = 1e-7,
    rays: int = 64,
    parity_sampling: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make windings coherent and normals point OUTWARD for the combined shell.
    Returns oriented (pfaces, wfaces, sfaces) with original shapes/indexing.
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

# ---------------------------------------------------------------------
# Mesh builders and utilities
# ---------------------------------------------------------------------

def make_hemisphere(radius: float, theta_res: int, phi_res: int) -> pv.PolyData:
    """Create an upper (z>=0) triangulated hemisphere by clipping a sphere at the equator."""
    sph = pv.Sphere(radius=radius, theta_resolution=theta_res, phi_resolution=2 * phi_res)
    hemi = sph.clip(normal=(0.0, 0.0, -1.0), origin=(0.0, 0.0, 0.0), invert=False)  # keep upper
    hemi = hemi.clean()
    if not hemi.is_all_triangles:
        hemi = hemi.triangulate()
    return hemi

def _pv_faces_to_tris(mesh: pv.PolyData) -> Tuple[np.ndarray, np.ndarray]:
    points = mesh.points.astype(np.float64, copy=False)
    faces = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32, copy=False)
    return points, faces

def _find_boundary_loop(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    e01 = np.sort(faces[:, [0, 1]], axis=1)
    e12 = np.sort(faces[:, [1, 2]], axis=1)
    e20 = np.sort(faces[:, [2, 0]], axis=1)
    all_edges = np.vstack((e01, e12, e20))
    uniq, counts = np.unique(all_edges, axis=0, return_counts=True)
    boundary_edges = uniq[counts == 1]
    if boundary_edges.size == 0:
        raise ValueError("No boundary edges found (mesh may be closed).")

    from collections import defaultdict
    adj: dict[int, List[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

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

def _robust_match_loops(loop_a_pts: np.ndarray, loop_b_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bidirectional NN matching for two boundary loops (points arrays, possibly different sizes)."""
    A, B = loop_a_pts, loop_b_pts
    nA, nB = len(A), len(B)

    tA, tB = cKDTree(A), cKDTree(B)
    _, B_to_A = tA.query(B)
    _, A_to_B = tB.query(A)

    pairs = np.column_stack((np.arange(nB, dtype=int), B_to_A))
    used_A = set(B_to_A.tolist())
    missing_A = np.setdiff1d(np.arange(nA, dtype=int), np.fromiter(used_A, int))
    if missing_A.size:
        extra = np.column_stack((A_to_B[missing_A], missing_A))  # (B_idx, A_idx)
        pairs = np.vstack((pairs, extra))

    pairs = pairs[np.argsort(pairs[:, 0])]
    Bidx, Aidx = pairs[:, 0], pairs[:, 1]
    Bnext, Anext = np.roll(Bidx, -1), np.roll(Aidx, -1)

    side_points = np.vstack((B, A))               # [B first, then A]
    a_off = nB
    faces_upper = np.column_stack((Bidx, Bnext, Aidx + a_off))
    faces_lower = np.column_stack((Bnext, Anext + a_off, Aidx + a_off))
    side_faces = np.vstack((faces_upper, faces_lower)).astype(int, copy=False)
    return side_points, side_faces

def _faces_to_tri_dicts(points: np.ndarray, faces: np.ndarray, bc_type: str, bc_value: float,
                        atol: float = 1e-12) -> List[Dict]:
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

    return [{'vertices': v, 'normal': n, 'bc_type': bc_type, 'bc_value': bc_value}
            for v, n in zip(Vv, Nv)]

# ---------------------------------------------------------------------
# Batch generation (one per thickness, always upper)
# ---------------------------------------------------------------------

def _fmt(x: float) -> str:
    return f"{x:g}"

def generate_one(radius: float,
                 thickness: float,
                 theta_res: int,
                 phi_res: int,
                 out_dir: Path,
                 inner_shift: float = 0.0,
                 inner_shift_axis: str = "z") -> None:
    """Generate one zipped *upper* hemisphere sample and write .pkl + outer-seed .npy."""

    # Build hemispheres (upper only)
    white_mesh = make_hemisphere(radius, theta_res, phi_res)
    pial_mesh  = make_hemisphere(radius + thickness, theta_res, phi_res)

    # --- NEW: optionally translate the inner hemisphere as a rigid body ---
    # Shift along chosen axis by inner_shift mm (morphology preserved).
    axis_to_idx = {'x': 0, 'y': 1, 'z': 2}
    shift_vec = np.zeros(3, dtype=float)

    if inner_shift != 0.0:
        ax_idx = axis_to_idx[inner_shift_axis]
        shift_vec[ax_idx] = inner_shift
        # Apply rigid translation to all inner (white) vertices
        white_mesh.points += shift_vec

    # Extract geometry (note: white_mesh may now be shifted, pial_mesh remains concentric)
    wpoints, wfaces = _pv_faces_to_tris(white_mesh)
    ppoints, pfaces = _pv_faces_to_tris(pial_mesh)

    # Boundary loops (equators)
    white_loop = _find_boundary_loop(wpoints, wfaces)
    pial_loop  = _find_boundary_loop(ppoints, pfaces)

    # Side band bridging the equators (use the actual coordinates of each loop)
    side_points, side_faces = _robust_match_loops(wpoints[white_loop], ppoints[pial_loop])

    # ---- Orient all faces outward for the closed shell ----
    pfaces_or, wfaces_or, sfaces_or = orient_outward_with_pymeshlab(
        ppoints, pfaces, wpoints, wfaces, side_points, side_faces,
        use_weld=False, weld_tol=1e-7, rays=128, parity_sampling=True
    )

    # Assemble BC-tagged triangles using oriented faces
    tris: List[Dict] = []
    tris += _faces_to_tri_dicts(ppoints, pfaces_or, 'dirichlet', 1.0)   # pial outer
    tris += _faces_to_tri_dicts(wpoints, wfaces_or, 'dirichlet', 0.0)   # white inner (shifted)
    tris += _faces_to_tri_dicts(side_points, sfaces_or, 'neumann', 0.0) # equatorial band

    # Seeds:
    # - inner_seed moves with the inner hemisphere.
    # - outer_seed stays where it was (you can change this if your solver expects something else).
    inner_seed = np.array([0.0, 0.0, radius + 0.0], dtype=float) + shift_vec
    outer_seed = np.array([0.0, 0.0, -(radius + thickness)], dtype=float)

    # Naming: "{R}mm_{t}" and "{R}mm_{t}_os"
    base = f"{_fmt(radius)}mm_{_fmt(thickness)}"
    pkl_path = out_dir / f"{base}.pkl"
    os_path  = out_dir / f"{base}_os.npy"  # outer seed

    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'triangles': tris,
            'white': {'points': wpoints, 'faces': wfaces_or},
            'pial':  {'points': ppoints, 'faces': pfaces_or},
            'side':  {'points': side_points, 'faces': sfaces_or},
            'outer_seed': outer_seed,
            'inner_seed': inner_seed,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(os_path, outer_seed)
    print(
        f"  saved {pkl_path.name}  "
        f"(V: w={wpoints.shape[0]}, p={ppoints.shape[0]}, s={side_points.shape[0]}, "
        f"shift={inner_shift:g}mm along {inner_shift_axis})"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate zipped upper hemispheres at multiple thicknesses",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--inner-radius', type=float, default=15.0, help='Inner radius R [mm]')
    p.add_argument('--thicknesses', type=float, nargs='+', default=[0.25, 0.5, 1.0, 2.0, 3.0, 4.0],
                   help='List of radial gaps (outer-inner) [mm]')
    p.add_argument('--theta-res', type=int, default=30, help='Azimuthal resolution (longitude divisions)')
    p.add_argument('--phi-res', type=int, default=15, help='Polar resolution (pole→equator divisions)')
    p.add_argument(
        '--inner-shift',
        type=float,
        default=0.5,
        help='Translate inner hemisphere by this distance [mm] away from outer shell (0 = concentric)'
    )
    p.add_argument(
        '--inner-shift-axis',
        type=str,
        choices=['x', 'y', 'z'],
        default='z',
        help='Axis along which to shift the inner hemisphere'
    )
    p.add_argument('--out-dir', type=Path,
                   default=Path(r'C:/Users/uqasnell/Documents/GitHub/laplace_thickness/validation/data/analytical/shifted_hemis'),
                   help='Output directory')
    return p.parse_args()

def main() -> None:
    args = parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inner radius: {args.inner_radius} mm | theta_res: {args.theta_res} | phi_res: {args.phi_res}")
    print(f"Thicknesses: {', '.join(_fmt(t) for t in args.thicknesses)} mm")
    print(f"Hemisphere: upper (z>=0)")
    print(f"Inner shift: {args.inner_shift:g} mm along {args.inner_shift_axis}-axis")
    print(f"Output dir: {out_dir.resolve()}\n")

    for t in args.thicknesses:
        print(f"=== thickness = {_fmt(t)} mm ===")
        generate_one(
            radius=args.inner_radius,
            thickness=t,
            theta_res=args.theta_res,
            phi_res=args.phi_res,
            out_dir=out_dir,
            inner_shift=args.inner_shift,
            inner_shift_axis=args.inner_shift_axis,
        )

    print("\nDone.")

if __name__ == '__main__':
    main()
