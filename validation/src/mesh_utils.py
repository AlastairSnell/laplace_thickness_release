import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

__all__ = [
    "load_freesurfer_surf", "save_vtk",
    "poisson_sample", "gaussian_noise"
]

def load_freesurfer_surf(path):
    from nibabel.freesurfer.io import read_geometry
    v, f = read_geometry(str(path))
    return v.astype(np.float64), f.astype(np.int32)

def save_vtk(filename, verts, faces):
    import pyvista as pv
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32), faces])
    pv.PolyData(verts, faces_pv.ravel()).save(filename)

# ------------------------------------------------------------------
# Sampling
# ------------------------------------------------------------------
def pick_even_start_points(
        triangles,
        outer_seed_point,
        pct=50,
        target_spacing=None,
        max_points=None
):
    # ------------------------------------------------------------------
    # 1. Gather *top* vertices and build an index map  (unique → row id)
    # ------------------------------------------------------------------
    top_verts = {}
    faces_top = []
    for tri in triangles:
        if tri['bc_type'] == 'dirichlet' and np.isclose(tri['bc_value'], 1.0):
            idxs = []
            for v in tri['vertices']:
                key = tuple(v)
                idxs.append(top_verts.setdefault(key, len(top_verts)))
            faces_top.append(idxs)

    if len(top_verts) == 0:
        raise ValueError("No Dirichlet-1 triangles found in input surface.")

    V = np.asarray(list(top_verts.keys()), dtype=np.float64)  # (N,3)
    F = np.asarray(faces_top,             dtype=np.int32)     # (M,3)

    # ------------------------------------------------------------------
    # 2. Build sparse edge graph (undirected) with edge-length weights
    # ------------------------------------------------------------------
    rows, cols, dists = [], [], []
    for a, b, c in F:
        for i, j in ((a, b), (b, c), (c, a)):
            rows.append(i); cols.append(j)
            d = np.linalg.norm(V[i] - V[j])
            dists.append(d)
            # add symmetric entry
            rows.append(j); cols.append(i); dists.append(d)

    G = coo_matrix((dists, (rows, cols)), shape=(len(V), len(V)))

    # ------------------------------------------------------------------
    # 3. Locate vertex closest to outer_seed_point, run Dijkstra
    # ------------------------------------------------------------------
    seed_idx = np.argmin(np.linalg.norm(V - outer_seed_point, axis=1))
    geo_d   = dijkstra(G, directed=False, indices=seed_idx)
    geo_d   = np.asarray(geo_d)           # (N,)

    finite = np.isfinite(geo_d)
    geo_max = geo_d[finite].max()
    thresh  = (pct / 100.0) * geo_max

    admissible = np.where(geo_d <= thresh)[0]
    V_sub      = V[admissible]

    # ------------------------------------------------------------------
    # 4. Poisson-like farthest-point sampling for even spread
    # ------------------------------------------------------------------
    if V_sub.shape[0] == 0:
        raise ValueError("No vertices satisfy distance threshold.")

    # Heuristic spacing if none supplied
    if target_spacing is None:
        bb_diag = np.linalg.norm(V_sub.max(0) - V_sub.min(0))
        target_spacing = 0.05 * bb_diag     # 5 % of bbox diagonal

    chosen = [int(np.random.choice(admissible))]   # start from random admissible
    chosen_coords = [V[chosen[0]]]

    while True:
        # Compute each admissible vertex's distance to nearest chosen point
        dist_to_chosen = np.min(
            np.linalg.norm(V_sub[:, None, :] - np.array(chosen_coords)[None, :, :],
                           axis=2),
            axis=1
        )
        # pick the candidate with the largest min-distance
        next_idx = np.argmax(dist_to_chosen)
        if dist_to_chosen[next_idx] < target_spacing:
            break  # all remaining points too close → stop

        chosen.append(admissible[next_idx])
        chosen_coords.append(V_sub[next_idx])

        if max_points is not None and len(chosen) >= max_points:
            break

    return np.asarray(chosen_coords, dtype=np.float64)

# ------------------------------------------------------------------
# Noise
# ------------------------------------------------------------------
def gaussian_noise(verts, sigma):
    """Return a new verts array with N(0, σ² I₃) displacement."""
    return verts + np.random.normal(scale=sigma, size=verts.shape)
