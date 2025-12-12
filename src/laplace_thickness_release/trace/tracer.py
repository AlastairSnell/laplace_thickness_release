from __future__ import annotations

import numpy as np

from ..bem.core import evaluate_gradient
from .normal import triangles_to_numeric_full, compute_meyer_normal_jit
from .intersections import find_exit_intersection

def path_trace_simple_bem(
    start_pt,
    triangles,
    q,
    cfg,
    direction='down',
    max_iter=200,
    alpha_initial=0.05,
    first_step=0.05,
    *,
    debug=False,
    angle_max_deg=30.0,
):
    """
    Steepest-descent (or ascent) path traced in the BEM-computed field.
    - No line search / Armijo.
    - Angle guard only (caps direction change per step).
    - Uses evaluate_gradient(...) from the BEM module for interior gradients.
    """

    def dbg(msg):
        if debug:
            print(msg, flush=True)

    def step_dir_from_grad(g):
        d = (-g if direction == 'down' else g)
        n = np.linalg.norm(d)
        return d / max(n, 1e-16)

    # --- state ---
    path_points  = [start_pt.copy()]
    total_length = 0.0
    x_current    = start_pt.copy()
    prev_step_dir = None

    # --- seed: push safely into the interior and align with -∇φ (for 'down') ---
    # 1) get a provisional surface normal (your routine, or infer per-cap)
    # --- seed step (step 0) along average normal (NO pre-flip)

    verts_arr, norms_arr, areas_arr, cents_arr = \
        triangles_to_numeric_full(triangles)

    avg_normal = compute_meyer_normal_jit(
        x_current, verts_arr, norms_arr, areas_arr, cents_arr,
        radius=-1.0, eps=1e-12, debug=False)

    if avg_normal is not None and np.linalg.norm(avg_normal) > 0:
        seed_dir    = avg_normal / np.linalg.norm(avg_normal)
        x_current = x_current + first_step * seed_dir
        total_length += first_step
        path_points.append(x_current.copy())
        prev_step_dir = seed_dir.copy()
        dbg(f"[SEED] moved {first_step:.5f}; seed_dir={seed_dir}")
    else:
        prev_step_dir = None
        dbg("[SEED] no normal; skipping seed move")

    # --- main loop ---
    for it in range(1, max_iter + 1):
        # 1) Gradient at current point
        g = evaluate_gradient(np.asarray([x_current]), triangles, q, cfg)[0]

        # 2) Check gradient validity
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) < 1e-12:
            dbg(f"[{it:03}] gradient too small or invalid → stop")
            break

        # 3) Candidate direction from gradient (unit)
        cand_dir = step_dir_from_grad(g)

        # 4) Angle guard: decide which direction to use
        used_dir = cand_dir
        if prev_step_dir is not None:
            # cosine of angle between candidate and previous USED direction
            a = float(np.clip(np.dot(cand_dir, prev_step_dir), -1.0, 1.0))
            ang = float(np.degrees(np.arccos(a)))

            if ang > angle_max_deg:
                # If flip is too large, ignore gradient direction
                # and reuse the *previous USED* direction
                used_dir = prev_step_dir
                dbg(
                    f"[{it:03}] ∠flip {ang:.2f}° > {angle_max_deg}° "
                    f"→ reuse prev_step_dir"
                )
            else:
                dbg(
                    f"[{it:03}] step good"
                )

        # 5) Propose step with FIXED alpha
        old_pt = x_current.copy()
        new_pt = old_pt + alpha_initial * used_dir  # alpha is constant

        # 6) Exit detection: intersect with boundary (e.g., bottom cap)
        X_int = find_exit_intersection(old_pt, new_pt, triangles)
        if X_int is not None:
            seg = np.linalg.norm(X_int - old_pt)
            if seg > 1e-8:
                total_length += seg
                path_points.append(X_int.copy())
                dbg(f"[{it:03}] EXIT (len={seg:.5f})")
            else:
                dbg(f"[{it:03}] EXIT at start (ε) — stopping")
            break

        # 7) Accept full step
        seg = np.linalg.norm(new_pt - old_pt)
        total_length += seg
        path_points.append(new_pt.copy())
        x_current = new_pt

        # Store the direction we ACTUALLY used this step
        prev_step_dir = used_dir.copy()

    dbg(f"[DONE] steps={len(path_points)-1}, total_len={total_length:.5f}")
    return path_points, total_length
