#!/usr/bin/env python3
"""
Perturb cortical slab phantoms with Gaussian surface noise and resave.

Inputs (default):
  C:/Users/uqasnell/Documents/GitHub/laplace_thickness/validation/data/phantoms
    e.g. 0.5x15_1.pkl   (zipped surface with BC-tagged triangles, and possibly meshes)
         0.5x15_1.npy   (OUTER seed; optional)

Outputs:
  C:/Users/uqasnell/Documents/GitHub/laplace_thickness/validation/data/perturbed
    For each sigma in --sigmas:
      e.g. 0.5x15_1_s0.1.pkl       (perturbed surfaces)
           0.5x15_1_s0.1_os.npy   (outer seed snapped to perturbed pial)

Behaviour:
  • By default, the boundary rim vertices are *frozen* (not perturbed) so the rim stays fixed.
  • The side band is rebuilt from the perturbed white/pial rims to keep a watertight ribbon.
  • The OUTER seed is always written:
      - If an outer seed exists in the input (.pkl or companion .npy), it is
        snapped to the nearest vertex on the **perturbed pial** and saved as *_os.npy*.
      - If none exists, a seed is derived from the perturbed pial centroid and saved as *_os.npy*. 

  • By default, the boundary rim vertices are *frozen* (not perturbed) so the rim stays fixed.
  • The side band is rebuilt from the perturbed white/pial rims to keep a watertight ribbon.
  • The OUTER seed is always written:
      - If an outer seed exists in the input (.pkl or companion .npy), it is
        snapped to the nearest vertex on the **perturbed pial** and saved.
      - If none exists, a seed is derived from the perturbed pial centroid and saved.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import shutil
from typing import Dict, Tuple, List

import numpy as np
from scipy.spatial import cKDTree

# --------------------------- small utils --------------------------- #

def _fmt(x: float) -> str:
    return f"{x:g}"

# --------------------------- geometry helpers --------------------------- #

def _faces_to_tri_dicts(points: np.ndarray, faces: np.ndarray, bc_type: str, bc_value: float,
                        atol: float = 1e-12) -> List[dict]:
    V = points[faces].astype(np.float64, copy=False)  # (m,3,3)
    finite_mask = np.isfinite(V).all(axis=(1, 2))
    e1 = V[:, 1] - V[:, 0]
    e2 = V[:, 2] - V[:, 0]
    cross = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    nondeg_mask = area >= atol
    valid = finite_mask & nondeg_mask
    Vv = V[valid]
    Cv = cross[valid]
    norms = np.linalg.norm(Cv, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    Nv = (Cv / norms).astype(np.float64, copy=False)
    return [{'vertices': v, 'normal': n, 'bc_type': bc_type, 'bc_value': bc_value}
            for v, n in zip(Vv, Nv)]


def _find_boundary_loop(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    e01 = np.sort(faces[:, [0, 1]], axis=1)
    e12 = np.sort(faces[:, [1, 2]], axis=1)
    e20 = np.sort(faces[:, [2, 0]], axis=1)
    all_edges = np.vstack((e01, e12, e20))
    uniq, counts = np.unique(all_edges, axis=0, return_counts=True)
    boundary_edges = uniq[counts == 1]
    if boundary_edges.size == 0:
        raise ValueError("No boundary edges found; mesh may be closed.")
    from collections import defaultdict
    adj: Dict[int, List[int]] = defaultdict(list)
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


def _robust_match_loops(loop_a: np.ndarray, loop_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(loop_a) <= len(loop_b):
        small, large = loop_a, loop_b
    else:
        small, large = loop_b, loop_a
    n_small, n_large = len(small), len(large)
    t_small, t_large = cKDTree(small), cKDTree(large)
    _, large_to_small = t_small.query(large)
    _, small_to_large = t_large.query(small)
    matches = np.column_stack((np.arange(n_large, dtype=int), large_to_small))
    used_small = set(large_to_small.tolist())
    missing_small = np.setdiff1d(np.arange(n_small, dtype=int), np.fromiter(used_small, int))
    if missing_small.size:
        extra = np.column_stack((small_to_large[missing_small], missing_small))
        matches = np.vstack((matches, extra))
    matches = matches[np.argsort(matches[:, 0])]
    L, S = matches[:, 0], matches[:, 1]
    L_next, S_next = np.roll(L, -1), np.roll(S, -1)
    side_points = np.vstack((large, small))
    s_offset = n_large
    faces_upper = np.column_stack((L, L_next, S + s_offset))
    faces_lower = np.column_stack((L_next, S_next + s_offset, S + s_offset))
    side_faces = np.vstack((faces_upper, faces_lower)).astype(int, copy=False)
    return side_points, side_faces


def _vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = points
    f = faces
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)  # 2*area * unit_normal
    norms = np.zeros_like(v)
    for i in range(3):
        np.add.at(norms, f[:, i], n)
    lens = np.linalg.norm(norms, axis=1, keepdims=True)
    lens[lens == 0.0] = 1.0
    return norms / lens


def _weld_from_triangles(tris: List[dict], tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    V = np.array([t['vertices'] for t in tris], dtype=np.float64)  # (m,3,3)
    flat = V.reshape(-1, 3)
    q = np.round(flat / tol).astype(np.int64)
    uq, inv = np.unique(q, axis=0, return_inverse=True)
    points = (uq.astype(np.float64) * tol)
    faces = inv.reshape(-1, 3).astype(np.int32)
    return points, faces


# --------------------------- IO helpers --------------------------- #

def _load_phantom_pkl(pkl_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (wpoints, wfaces, ppoints, pfaces, spoints, sfaces, outer_seed_or_none).
    If the pickle contains only triangle dicts, reconstruct meshes per BC type.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    outer_seed = data.get('outer_seed') if isinstance(data, dict) else None
    if outer_seed is not None:
        outer_seed = np.asarray(outer_seed, dtype=float)

    if isinstance(data, dict) and all(k in data for k in ('white', 'pial', 'side')):
        wpoints = data['white']['points'].astype(np.float64)
        wfaces  = data['white']['faces'].astype(np.int32)
        ppoints = data['pial']['points'].astype(np.float64)
        pfaces  = data['pial']['faces'].astype(np.int32)
        spoints = data['side']['points'].astype(np.float64)
        sfaces  = data['side']['faces'].astype(np.int32)
        return wpoints, wfaces, ppoints, pfaces, spoints, sfaces, outer_seed

    # Fallback: reconstruct from triangles by BC type
    tris: List[dict] = data['triangles']
    pial_tris  = [t for t in tris if t.get('bc_type') == 'dirichlet' and abs(float(t.get('bc_value', 0.0)) - 1.0) < 1e-9]
    white_tris = [t for t in tris if t.get('bc_type') == 'dirichlet' and abs(float(t.get('bc_value', 0.0)) - 0.0) < 1e-9]
    side_tris  = [t for t in tris if t.get('bc_type') == 'neumann']

    if not (pial_tris and white_tris and side_tris):
        raise ValueError(f"{pkl_path.name}: cannot split triangles by BC types; unexpected format.")

    ppoints, pfaces = _weld_from_triangles(pial_tris)
    wpoints, wfaces = _weld_from_triangles(white_tris)
    spoints, sfaces = _weld_from_triangles(side_tris)
    return wpoints, wfaces, ppoints, pfaces, spoints, sfaces, outer_seed


def _save_phantom_pkl(out_path: Path,
                      wpoints: np.ndarray, wfaces: np.ndarray,
                      ppoints: np.ndarray, pfaces: np.ndarray,
                      spoints: np.ndarray, sfaces: np.ndarray,
                      *, sigma: float,
                      outer_seed: np.ndarray | None) -> None:
    tris: List[dict] = []
    tris += _faces_to_tri_dicts(ppoints, pfaces, 'dirichlet', 1.0)
    tris += _faces_to_tri_dicts(wpoints, wfaces, 'dirichlet', 0.0)
    tris += _faces_to_tri_dicts(spoints, sfaces, 'neumann', 0.0)
    with open(out_path, 'wb') as f:
        payload = {
            'triangles': tris,
            'white': {'points': wpoints, 'faces': wfaces},
            'pial':  {'points': ppoints, 'faces': pfaces},
            'side':  {'points': spoints, 'faces': sfaces},
            'sigma': float(sigma),
        }
        if outer_seed is not None:
            payload['outer_seed'] = np.asarray(outer_seed, dtype=float)
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


# --------------------------- perturbation --------------------------- #

def _perturb_surface(points: np.ndarray, faces: np.ndarray, sigma: float, rng: np.random.Generator,
                     freeze_boundary: bool = True) -> np.ndarray:
    """Displace vertices along area-weighted vertex normals by N(0, sigma).
    If freeze_boundary, boundary-loop vertices are left unchanged.
    """
    pts = points.copy()
    normals = _vertex_normals(pts, faces)
    noise = rng.normal(loc=0.0, scale=sigma, size=(pts.shape[0], 1))
    if freeze_boundary:
        try:
            loop = _find_boundary_loop(pts, faces)
            mask = np.ones((pts.shape[0], 1), dtype=bool)
            mask[loop] = False
            noise = noise * mask  # zero at boundary
        except Exception:
            pass  # if no boundary found, perturb all
    pts += normals * noise
    return pts


# --------------------------- main driver --------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Add Gaussian perturbations to slab phantoms and resave',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--in-dir', type=Path,
                   default=Path(r'C:/Users/uqasnell/Documents/GitHub/laplace_thickness/validation/data/phantoms_families'),
                   help='Input directory containing *.pkl (and optional matching *.npy for outer seed)')
    p.add_argument('--out-dir', type=Path,
                   default=Path(r'C:/Users/uqasnell/Documents/GitHub/laplace_thickness/validation/data/perturbed_families'),
                   help='Output directory for perturbed phantoms')
    p.add_argument('--sigmas', type=float, nargs='+', default=[0.1, 0.2, 0.5],
                   help='One or more std devs of normal displacement [mm]')
    p.add_argument('--seed', type=int, default=42, help='RNG seed for reproducibility')
    p.add_argument('--no-freeze-boundary', action='store_true', help='Also perturb boundary rim vertices')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pkls = sorted([p for p in args.in_dir.glob('*.pkl')])
    if not pkls:
        print(f'No .pkl files found in {args.in_dir}')
        return

    print(f"Found {len(pkls)} phantoms in {args.in_dir}")
    print(f"Sigmas: {', '.join(_fmt(s) for s in args.sigmas)} mm | Freeze boundary: {not args.no_freeze_boundary}")
    print(f"Output: {args.out_dir}\n")

    for pkl_path in pkls:
        base = pkl_path.stem  # e.g. '0.5x15_1'
        npy_in = pkl_path.with_suffix('.npy')

        try:
            wpts, wfcs, ppts, pfcs, spts, sfcs, outer_seed_in = _load_phantom_pkl(pkl_path)
            # If no seed in pickle, try companion .npy
            if outer_seed_in is None and npy_in.exists():
                try:
                    outer_seed_in = np.load(npy_in)
                except Exception:
                    outer_seed_in = None

            for sigma in args.sigmas:
                sub_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))

                # Perturb inner (white) and outer (pial) surfaces; rebuild side band
                wpts_pert = _perturb_surface(wpts, wfcs, sigma=float(sigma), rng=sub_rng,
                                             freeze_boundary=not args.no_freeze_boundary)
                ppts_pert = _perturb_surface(ppts, pfcs, sigma=float(sigma), rng=sub_rng,
                                             freeze_boundary=not args.no_freeze_boundary)

                wloop = _find_boundary_loop(wpts_pert, wfcs)
                ploop = _find_boundary_loop(ppts_pert, pfcs)
                side_pts, side_fcs = _robust_match_loops(wpts_pert[wloop], ppts_pert[ploop])

                # Determine output outer seed: snap input (or derived) to perturbed pial
                if outer_seed_in is None:
                    # derive from centroid of perturbed pial
                    seed_ref = ppts_pert.mean(axis=0)
                else:
                    seed_ref = np.asarray(outer_seed_in, dtype=float)
                kdt = cKDTree(ppts_pert)
                _, idx = kdt.query(seed_ref)
                outer_seed_out = ppts_pert[idx]

                out_base = f"{base}_s{_fmt(sigma)}"
                out_pkl = args.out_dir / f"{out_base}.pkl"
                _save_phantom_pkl(out_pkl, wpts_pert, wfcs, ppts_pert, pfcs, side_pts, side_fcs,
                                  sigma=float(sigma), outer_seed=outer_seed_out)

                # Save outer seed with sigma in filename
                np.save(args.out_dir / f"{out_base}_os.npy", outer_seed_out)

                print(f"  Saved {out_pkl.name}")

        except Exception as e:
            print(f"  [SKIP] {base}: {e}")

    print("\nDone.")


if __name__ == '__main__':
    main()
