import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import pickle, os
from concurrent.futures import ThreadPoolExecutor, as_completed

from scipy.spatial import KDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from path_trace_copy import (
    plot_bumpy_top,
    plot_bumpy_bottom,
    set_axes_equal,
    preprocess_triangles,
    pick_even_start_points,
)

from path_trace_copy2 import (
    assemble_and_solve,
    BEMConfig,
    path_trace_simple_bem,
)

# ---------------- CONFIG ----------------
N_CPU          = os.cpu_count() or 1

# Match Script A mechanics
ALPHA_INITIAL  = 0.05
MAX_ITER       = int(6.0 / ALPHA_INITIAL)
FIRST_STEP     = 0.05

PCT            = 50          # percentage of pial faces to keep
TARGET_SPACING = 0.1
# ----------------------------------------


def load_surfaces(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)['triangles']


def run_single_path(path_idx, triangles, start_pt, q, cfg):
    """
    Trace one ‘down’ path using the BEM-based path tracer.

    Returns:
        (idx, path_down, len_down)
    """
    path_down, len_down = path_trace_simple_bem(
        start_pt,
        triangles,
        q,
        cfg,
        'down',
        MAX_ITER,
        ALPHA_INITIAL,
        FIRST_STEP,
        debug=True  # match Script A; set to False if too noisy
    )
    return path_idx, path_down, len_down


def geodesic_idw(
        V, A_sparse,
        sample_xyz, sample_vals,
        *, power=2.0, verbose=True):
    """
    Interpolate scalar values on a mesh using inverse-distance weighting
    with *geodesic* distance.  Every sample influences every vertex
    (no max_dist cut-off).
    """
    V = np.ascontiguousarray(V, dtype=np.float64)

    # Map samples → nearest vertices
    kd = KDTree(V)
    src_idx = kd.query(sample_xyz)[1].astype(np.int32)

    if verbose:
        print(
            f"Computing full geodesic distance matrix "
            f"(samples={len(src_idx)}, vertices={len(V)}) …",
            flush=True
        )

    # Distances: shape (M, N)
    D = dijkstra(A_sparse, directed=False, indices=src_idx)

    # IDW weights: 1 / d^power  (avoid 0^-power at the source)
    W = 1.0 / np.maximum(D, 1e-6)**power  # (M, N)

    # Weighted sum across samples
    acc_num = (W * sample_vals[:, None]).sum(axis=0)
    acc_den = W.sum(axis=0)

    out = acc_num / np.maximum(acc_den, 1e-12)

    if verbose:
        print("…done.", flush=True)

    return out


def main():
    print("Loading surfaces...")
    outer_seed = np.load(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\big_folds\outer_seed_000.npy")
    triangles = load_surfaces(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\big_folds\zipped_patch_000.pkl")
    preprocess_triangles(triangles)

    # ---------- Build V (unique vertices) and F (indices) -------------
    all_verts = np.vstack([tri['vertices'] for tri in triangles])
    V, inverse = np.unique(all_verts, axis=0, return_inverse=True)
    F = inverse.reshape(-1, 3).astype(np.int32)

    # ---------- Sparse edge graph for surface Dijkstra ---------------
    i = F[:, [0, 1, 2]].ravel()
    j = F[:, [1, 2, 0]].ravel()
    w = np.linalg.norm(V[i] - V[j], axis=1)  # Euclidean edge length
    nV = len(V)
    A_sparse = coo_matrix((w, (i, j)), shape=(nV, nV))
    A_sparse = A_sparse.maximum(A_sparse.T).tocsr()

    # ---------- BEM assembly + solve (Script A mechanics) ------------
    print("Assembling BEM system...")
    cfg = BEMConfig(
        quad_order=3,
        TAU_NEAR=0.2,
        TOL_NEAR=1e-10,
        MAX_SUBDIV=3
    )
    q, A, b = assemble_and_solve(triangles, cfg)

    # ---------- Sample start points ----------------------------------
    start_coords = pick_even_start_points(
        triangles,
        outer_seed_point=outer_seed,
        pct=PCT,
        target_spacing=TARGET_SPACING,
        max_points=None
    )
    NUM_PATHS = len(start_coords)
    print(f"Tracing {NUM_PATHS} paths (one per start point).")

    sample_thick  = np.empty(NUM_PATHS, dtype=float)
    sample_coords = np.asarray(start_coords)

    # ---------- Path tracing (BEM-based) -----------------------------
    if NUM_PATHS > 0:
        with ThreadPoolExecutor(max_workers=N_CPU) as pool:
            futures = [
                pool.submit(
                    run_single_path,
                    i,
                    triangles,
                    start_pt=sample_coords[i],
                    q=q,
                    cfg=cfg,
                )
                for i in range(NUM_PATHS)
            ]

            for fut in as_completed(futures):
                idx, _, len_down = fut.result()
                sample_thick[idx] = len_down

        # ---------- Collect pial vertices -----------------------------
        pial_xyz = []
        for tri in triangles:
            if tri['bc_value'] >= 0.99:  # pial marker
                pial_xyz.extend(tri['vertices'])
        pial_xyz = np.unique(np.asarray(pial_xyz), axis=0)  # de-duplicate
        kd = KDTree(pial_xyz)

        # ---------- Filter clearly-bad path lengths -------------------
        LOWER_BOUND = 1.0

        K_NEIGH = 3
        kd = KDTree(sample_coords)
        clean_thick = sample_thick.copy()
        low_mask = clean_thick < LOWER_BOUND
        good_mask = ~low_mask  # anything not too thin

        for idx in np.where(low_mask)[0]:
            pt = sample_coords[idx]
            _, all_idx = kd.query(pt, k=len(sample_coords))

            neigh_idx = [
                j for j in all_idx
                if clean_thick[j] >= LOWER_BOUND
            ][:K_NEIGH]

            if len(neigh_idx) < K_NEIGH:
                print(
                    f" ⚠️  idx {idx} kept at {clean_thick[idx]:.2f} mm "
                    "(not enough good neighbours)"
                )
                continue

            new_val = np.median(clean_thick[neigh_idx])

            if new_val < LOWER_BOUND:  # still suspicious
                print(
                    f" ⚠️  idx {idx} upgrade still too thin "
                    f"({new_val:.2f} mm) – leaving original value"
                )
                continue

            clean_thick[idx] = new_val
            good_mask[idx] = True

        # ---------- Geodesic IDW interpolation ------------------------
        pial_thick_by_vert = geodesic_idw(
            V,
            A_sparse,
            sample_coords,  # (N,3) start points
            clean_thick,    # (N,)   cleaned path lengths
            power=2.0,
        )

        # ---------- Seed-based pial clipping --------------------------
        kd_all = KDTree(V)

        # 1. locate the seed vertex
        seed_vert_idx = kd_all.query(outer_seed)[1]

        # 2. seed-to-vertex geodesics
        print("Computing seed-to-vertex geodesics …", flush=True)
        seed_dist = dijkstra(
            A_sparse,
            directed=False,
            indices=[seed_vert_idx]
        )[0]  # (nV,)

        # 3. gather pial faces with their mean distance
        pial_face_info = []  # (mean_dist, tri, idxs)

        for tri in triangles:
            if tri['bc_value'] >= 0.99:  # pial marker
                idxs = kd_all.query(tri['vertices'])[1]  # 3 vertex indices
                mean_d = seed_dist[idxs].mean()
                pial_face_info.append((mean_d, tri, idxs))

        # 4. keep the closest half (or whatever PCT sets)
        pial_face_info.sort(key=lambda t: t[0])
        keep_n = (len(pial_face_info) * PCT) // 100
        pial_face_info = pial_face_info[:keep_n]

        trimmed_triangles = [t[1] for t in pial_face_info]
        trimmed_face_vals = np.array([
            pial_thick_by_vert[idxs].mean()
            for _, _, idxs in pial_face_info
        ])

        print(f"Kept {keep_n} pial faces (nearest {PCT} %).")

        # ---------- Plotting: heatmap on top surface ------------------
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Suppose your axis is called `ax`
        ax.set_axis_off()          # hides box, ticks, labels, grid (Matplotlib ≥3.3)


        poly_top = plot_bumpy_top(
            ax,
            trimmed_triangles,           # clipped list
            face_vals=trimmed_face_vals,
            cmap='hot'
        )
        poly_bottom = plot_bumpy_bottom(ax, triangles)  # full bottom

        # Toggle top/bottom
        rax = plt.axes([0.02, 0.4, 0.12, 0.12])
        labels = ['Top surface', 'Bottom surface']
        visibility = [poly_top.get_visible(), poly_bottom.get_visible()]
        check = CheckButtons(rax, labels, visibility)

        def toggle_surfaces(label):
            if label.startswith('Top'):
                poly_top.set_visible(not poly_top.get_visible())
            else:
                poly_bottom.set_visible(not poly_bottom.get_visible())
            plt.draw()

        check.on_clicked(toggle_surfaces)
        set_axes_equal(ax)

    plt.show()


if __name__ == "__main__":
    main()
