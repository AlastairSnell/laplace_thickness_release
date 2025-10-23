
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import CheckButtons
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics as stats
import os
from path_trace_copy import (
    plot_bumpy_top,
    plot_bumpy_bottom,
    set_axes_equal,
    preprocess_triangles,
    pick_even_start_points,
    _print_length_stats,
    summarise_paths,
    compare_down_up,
    path_trace_simple)

from path_trace_copy2 import (
    assemble_and_solve, 
    BEMConfig,
    path_trace_simple_bem
)
N_CPU = os.cpu_count() or 1
FORCED_SUBDIV_THRESHOLD = 0.1
PLOT_PATH = True
ALPHA_INITIAL = 0.05
MAX_ITER = int(4.0 / ALPHA_INITIAL)
TOLERANCE = 1e-3
COMPUTE_UP_PATH = True
FIRST_STEP = 0.05

def load_surfaces(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)['triangles']
    
def run_single_path(path_idx, triangles, start_pt, q, cfg, *, do_up=COMPUTE_UP_PATH):
    """
    Trace one ‘down’ path (always) and, optionally, the reverse ‘up’ path.

    Returns:
        (idx, path_down, len_down, path_up, len_up)
        • path_up / len_up are sensible placeholders when do_up is False.
    """
    # --- always trace the downward leg
    path_down, len_down = path_trace_simple_bem(
        start_pt, triangles, q, cfg, 'down',
        MAX_ITER, ALPHA_INITIAL, FIRST_STEP, debug=True
    )

    # --- optionally trace back upward
    if do_up:
        bottom_pt          = path_down[-1]
        path_up, len_up = path_trace_simple_bem(
            bottom_pt, triangles, q, cfg, 'up',
            MAX_ITER, ALPHA_INITIAL, FIRST_STEP, debug=True
        )
    else:
        # placeholders keep downstream code unchanged
        path_up, len_up = [], 0.0

    return path_idx, path_down, len_down, path_up, len_up

def main():
    print("Loading surfaces.")
    #outer_seed = np.load(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\analytical\hemispheres\5mm_1_os.npy")
    #triangles = load_surfaces(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\analytical\hemispheres\5mm_1.pkl")
    outer_seed = np.load(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\analytical\disks\10mm_4_os.npy")
    triangles = load_surfaces(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\analytical\disks\10mm_4.pkl")
    preprocess_triangles(triangles)

    print("Assembling system.")
    #A, b = assemble_system(triangles, parallel=True, max_workers=N_CPU)
    cfg = BEMConfig(quad_order=3, TAU_NEAR=0.2, TOL_NEAR=1e-10, MAX_SUBDIV=3)
    q, A, b = assemble_and_solve(triangles, cfg)
    #pts = np.array([[0.0, 0.0, 3], [0.0, 5.0, 3],[0.0, 10.0, 3],[0.0, 12.0, 3],[0.0, 14.0, 3],[0.0, 14.9, 3],[0.0, 15.0, 3],[0.0, 15.1, 3],[0.0, 16.0, 3]])
    #phi = evaluate_potential(pts, triangles, q, cfg)
    #grad = evaluate_gradient(pts, triangles, q, cfg)
    #print(f"phi = {phi}, grad = {grad}")

    #store_solution(triangles, solve_system(A, b))

    all_paths_down, all_paths_up = [], []
    all_lengths_down, all_lengths_up = [], []


    start_pts_pool = pick_even_start_points(
        triangles,
        outer_seed_point=outer_seed,
        pct=90,
        target_spacing=5,
        max_points=None
    )

    NUM_PATHS = len(start_pts_pool)
    print(f"Tracing {NUM_PATHS} paths (one per start point).")

    if NUM_PATHS > 0:
        with ThreadPoolExecutor(max_workers=N_CPU) as pool:
            futures = [
                pool.submit(
                    run_single_path,
                    i,                   # path index
                    triangles,
                    start_pt=start_pts_pool[i],
                    q=q,
                    cfg=cfg,
                    do_up=COMPUTE_UP_PATH,
                )
                for i in range(NUM_PATHS)
            ]

            for fut in as_completed(futures):
                idx, p_down, len_down, p_up, len_up = fut.result()

                all_paths_down.append(p_down)
                all_lengths_down.append(len_down)
                all_paths_up.append(p_up)
                all_lengths_up.append(len_up)

                print(f"Path {idx+1}: down={len_down:.3f}, up={len_up:.3f}")

    if NUM_PATHS > 0:
        print("\nSummary of all path lengths:")
        print("Down lengths:", ", ".join(f"{l:.3f}" for l in all_lengths_down))
        print("Up lengths:  ", ", ".join(f"{l:.3f}" for l in all_lengths_up))

        _print_length_stats("Down-path", all_lengths_down)
        _print_length_stats("Up-path",   all_lengths_up)

        summarise_paths(
        all_paths_down,
        all_paths_up,
        debug=False,
        save_txt="all_traced_points.txt")

        compare_down_up(all_lengths_down, all_lengths_up,
                debug=False)

    if PLOT_PATH and NUM_PATHS > 0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        poly_top    = plot_bumpy_top(ax, triangles)
        poly_bottom = plot_bumpy_bottom(ax, triangles)

        ax.scatter(
            outer_seed[0], outer_seed[1], outer_seed[2],
            color='red', marker='*', s=120, label='Outer seed'
        )

        rax = plt.axes([0.02, 0.4, 0.12, 0.12])
        labels = ['Top surface', 'Bottom surface']
        visibility = [poly_top.get_visible(), poly_bottom.get_visible()]
        check = CheckButtons(rax, labels, visibility)

        def toggle_surfaces(label):
            if label.startswith('Top'):
                poly_top.set_visible(not poly_top.get_visible())
            else:                       # 'Bottom surface'
                poly_bottom.set_visible(not poly_bottom.get_visible())
            plt.draw()

        check.on_clicked(toggle_surfaces)

    for i, path_down in enumerate(all_paths_down):
        arr_down = np.array(path_down)
        ax.plot(arr_down[:,0], arr_down[:,1], arr_down[:,2],
                marker='o', alpha=1.0, 
                label=f'Down {i+1}')

    if COMPUTE_UP_PATH:
        for i, path_up in enumerate(all_paths_up):
            arr_up = np.array(path_up)
            ax.plot(arr_up[:,0], arr_up[:,1], arr_up[:,2],
                    marker='x', alpha=1.0, label=f'Up {i+1}')

    set_axes_equal(ax)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
