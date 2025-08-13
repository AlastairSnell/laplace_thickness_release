import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle
import pyvista as pv
from matplotlib.widgets import CheckButtons
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_seed(path: str) -> np.ndarray | None:
    try:
        return np.load(path)
    except FileNotFoundError:
        return None

def _pv_faces_to_tris(mesh: pv.PolyData) -> Tuple[np.ndarray, np.ndarray]:
    """Return (points, faces) where faces is (m,3) int array."""
    points = mesh.points.astype(np.float64, copy=False)
    faces = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32, copy=False)
    return points, faces

def _build_poly(points: np.ndarray, faces: np.ndarray, **kwargs) -> Poly3DCollection:
    """Efficient Poly3DCollection from numpy arrays."""
    # Poly3DCollection accepts a (m, 3, 3) array as a sequence of triangles.
    return Poly3DCollection(points[faces], **kwargs)

# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

def find_boundary_loop(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Identify and return the single boundary loop as vertex indices.
    Assumes exactly one continuous open boundary (each boundary vertex has degree 2).
    """
    # All (undirected) edges in the mesh (vectorized)
    e01 = np.sort(faces[:, [0, 1]], axis=1)
    e12 = np.sort(faces[:, [1, 2]], axis=1)
    e20 = np.sort(faces[:, [2, 0]], axis=1)
    all_edges = np.vstack((e01, e12, e20))

    # Boundary edges appear only once
    uniq, counts = np.unique(all_edges, axis=0, return_counts=True)
    boundary_edges = uniq[counts == 1]
    if boundary_edges.size == 0:
        raise ValueError("No boundary edges found.")

    # Build adjacency on boundary
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    # Sanity: every boundary vertex should have degree 2 in a single loop
    start = next((v for v, nbrs in adj.items() if len(nbrs) == 2), None)
    if start is None:
        raise ValueError("Boundary is not a single 2-regular loop.")

    # Walk the loop
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
    """
    Bidirectional nearest-neighbor matching for two boundary loops (possibly different lengths).
    Returns:
      side_points (N,3) and side_faces (M,3) forming a bridging surface.
    """
    # Pick small/large
    if len(loop_small_coords) <= len(loop_large_coords):
        small, large = loop_small_coords, loop_large_coords
    else:
        small, large = loop_large_coords, loop_small_coords

    n_small, n_large = len(small), len(large)
    t_small, t_large = cKDTree(small), cKDTree(large)

    _, large_to_small = t_small.query(large)   # (n_large,)
    _, small_to_large = t_large.query(small)   # (n_small,)

    # Ensure every large has a small, and every small is used at least once
    matches = np.column_stack((np.arange(n_large, dtype=int), large_to_small))
    used_small = set(large_to_small.tolist())
    missing_small = np.setdiff1d(np.arange(n_small, dtype=int), np.fromiter(used_small, int))

    if missing_small.size:
        extra = np.column_stack((small_to_large[missing_small], missing_small))
        matches = np.vstack((matches, extra))

    # Sort by large index for consistent quads
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
    """
    Vectorized conversion: triangle list-of-dicts with vertices, unit normal, and BCs.
    Skips degenerate, duplicate-vertex, or non-finite triangles.
    """
    V = points[faces].astype(np.float64, copy=False)     # (m,3,3)

    # Duplicate-vertex checks (vectorized exact-ish with tolerance)
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

    # Unit normals
    norms = np.linalg.norm(Cv, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # safe divide (shouldn't occur after mask)
    Nv = (Cv / norms).astype(np.float64, copy=False)

    # Pack dicts
    tris = [{'vertices': v, 'normal': n, 'bc_type': bc_type, 'bc_value': bc_value}
            for v, n in zip(Vv, Nv)]

    skipped_zero_area = int((~nondeg_mask).sum())
    skipped_duplicate = int(dup_mask.sum())
    skipped_nonfinite = int((~finite_mask).sum())
    print(f"faces_to_tri_dicts: skipped {skipped_zero_area} zero-area, "
          f"{skipped_duplicate} duplicate-vertex, {skipped_nonfinite} non-finite triangles.")
    return tris

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    # 1) Optional seed points
    outer_seed = _load_seed("outer_seed_point.npy")
    inner_seed = _load_seed("inner_seed_point.npy")
    if outer_seed is not None or inner_seed is not None:
        print("Loaded seed points:",
              f"\n  outer_seed: {outer_seed if outer_seed is not None else 'None'}",
              f"\n  inner_seed: {inner_seed if inner_seed is not None else 'None'}", sep="")
    else:
        print("No seed points found. Continuing without them.")

    # 2) Read meshes
    white_mesh = pv.read("simulated_white.vtk")
    pial_mesh  = pv.read("simulated_pial.vtk")

    wpoints, wfaces = _pv_faces_to_tris(white_mesh)
    ppoints, pfaces = _pv_faces_to_tris(pial_mesh)

    print(f"White mesh: {wpoints.shape[0]} vertices, {wfaces.shape[0]} faces.")
    print(f"Pial mesh:  {ppoints.shape[0]} vertices, {pfaces.shape[0]} faces.")

    # 3) Boundary loops
    white_loop = find_boundary_loop(wpoints, wfaces)
    pial_loop  = find_boundary_loop(ppoints, pfaces)
    print(f"White boundary loop length: {len(white_loop)}")
    print(f"Pial boundary loop length:  {len(pial_loop)}")

    # 4) Loop coordinates
    white_coords = wpoints[white_loop]
    pial_coords  = ppoints[pial_loop]

    # 5) Robust loop matching → side surface
    side_points, side_faces = robust_match_loops(white_coords, pial_coords)

    # 6) Plot (toggleable)
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection='3d')

    white_poly = _build_poly(wpoints, wfaces, facecolor='silver', edgecolor='k', alpha=1, label='White')
    pial_poly  = _build_poly(ppoints, pfaces, facecolor='pink',   edgecolor='k', alpha=1, label='Pial')
    side_poly  = _build_poly(side_points, side_faces, facecolor='lime', edgecolor='k', alpha=1, label='Side')

    for poly in (white_poly, pial_poly, side_poly):
        ax.add_collection3d(poly)

    # ---------- KEEP THIS ----------
    all_triangles = []
    all_triangles += faces_to_tri_dicts(ppoints, pfaces, 'dirichlet', 1.0)
    all_triangles += faces_to_tri_dicts(wpoints, wfaces, 'dirichlet', 0.0)
    all_triangles += faces_to_tri_dicts(side_points, side_faces, 'neumann', 0.0)

    with open('zipped_surface.pkl', 'wb') as f:
        pickle.dump({'triangles': all_triangles}, f, protocol=pickle.HIGHEST_PROTOCOL)

    # UI: CheckButtons + dynamic legend
    chk_ax = fig.add_axes([0.02, 0.70, 0.15, 0.15])
    chk_ax.set_title("Toggle surfaces", fontsize=10)
    labels = ['White', 'Pial', 'Side']
    label_to_poly = {'White': white_poly, 'Pial': pial_poly, 'Side': side_poly}
    checks = CheckButtons(chk_ax, labels, [True, True, True])

    def _update_legend() -> None:
        visible = [(l, p) for l, p in label_to_poly.items() if p.get_visible()]
        if visible:
            ax.legend([p for _, p in visible], [l for l, _ in visible], loc='upper right')
        else:
            leg = ax.get_legend()
            if leg:
                leg.remove()
        fig.canvas.draw_idle()

    def _toggle(label: str) -> None:
        poly = label_to_poly[label]
        poly.set_visible(not poly.get_visible())
        _update_legend()

    checks.on_clicked(lambda _: _toggle(_))  # noqa: E731
    _update_legend()

    # Seed points
    if outer_seed is not None:
        ax.scatter(*outer_seed, color='red', s=50, label='outer_seed')
    if inner_seed is not None:
        ax.scatter(*inner_seed, color='blue', s=50, label='inner_seed')

    # Axes limits + equal aspect
    all_coords = np.vstack((wpoints, ppoints, side_points))
    mins = all_coords.min(axis=0)
    maxs = all_coords.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(maxs - mins)  # equal aspect ratio
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("White, Pial, and Side Surfaces")

    plt.show()

if __name__ == "__main__":
    main()
