import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle

from path_trace_copy import (
    pick_top_triangle_point_in_bounds,
    trace_single_path,
    plot_bumpy_top,
    plot_bumpy_bottom,
    set_axes_equal,
    preprocess_triangles,
    assemble_system,
    solve_system,
    store_solution,
    find_closest_triangle_by_plane,
    compute_Iij_Jij
)

ADAPT_TOL = 1e-5
ADAPT_MAX_REFINE = 8
FORCED_SUBDIV_THRESHOLD = 0.1
SELF_NSUB = 2
PLOT_PATH = True
NUM_PATHS = 20
MAX_ITER = 200
ALPHA_INITIAL = 0.05
TOLERANCE = 1e-5
MIN_ALPHA = 1e-5

def load_surfaces(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)['triangles']

def main():
    print("Loading surfaces...")
    triangles = load_surfaces('C:/Users/uqasnell/Documents/GitHub/laplace_thickness/validation/data/phantoms/0.2x15_1.pkl')
    preprocess_triangles(triangles)
    
    A, b = assemble_system(triangles)
    print(f"System size: {A.shape}")
    store_solution(triangles, solve_system(A, b))
    
    all_paths_down = []
    all_paths_up = []
    all_lengths_down = []
    all_lengths_up = []

    if NUM_PATHS > 0:
        for path_i in range(NUM_PATHS):
            start_pt = pick_top_triangle_point_in_bounds(triangles)
            tri_closest_top = find_closest_triangle_by_plane(start_pt, triangles)
            face_normal_down = tri_closest_top['normal'] if tri_closest_top else None

            path_points_down, length_down = trace_single_path(
                start_pt,
                triangles,
                direction='down',
                max_iter=MAX_ITER,
                alpha_initial=ALPHA_INITIAL,
                tolerance=TOLERANCE,
                min_alpha=MIN_ALPHA,
                first_normal=face_normal_down
            )
            all_paths_down.append(path_points_down)
            all_lengths_down.append(length_down)

            bottom_pt = path_points_down[-1]
            tri_closest_bottom = find_closest_triangle_by_plane(bottom_pt, triangles)
            face_normal_up = tri_closest_bottom['normal'] if tri_closest_bottom else None

            path_points_up, length_up = trace_single_path(
                bottom_pt,
                triangles,
                direction='up',
                max_iter=MAX_ITER,
                alpha_initial=ALPHA_INITIAL,
                tolerance=TOLERANCE,
                min_alpha=MIN_ALPHA,
                first_normal=face_normal_up
            )
            all_paths_up.append(path_points_up)
            all_lengths_up.append(length_up)

            print(f"Path {path_i+1}: down={length_down:.3f}, up={length_up:.3f}")

    if NUM_PATHS > 0:
        print("\nSummary of all path lengths:")
        print("Down lengths:", ", ".join(f"{l:.3f}" for l in all_lengths_down))
        print("Up lengths:  ", ", ".join(f"{l:.3f}" for l in all_lengths_up))

    if PLOT_PATH and NUM_PATHS > 0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ...
        plot_bumpy_top(ax, triangles)

        plot_bumpy_bottom(ax, triangles)

    for i, path_down in enumerate(all_paths_down):
        arr_down = np.array(path_down)
        ax.plot(arr_down[:,0], arr_down[:,1], arr_down[:,2],
                marker='o', label=f'Down {i+1}')

    for i, path_up in enumerate(all_paths_up):
        arr_up = np.array(path_up)
        ax.plot(arr_up[:,0], arr_up[:,1], arr_up[:,2],
                marker='x', label=f'Up {i+1}')

    set_axes_equal(ax)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
