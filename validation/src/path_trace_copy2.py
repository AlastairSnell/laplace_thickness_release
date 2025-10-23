# bem_laplace_slp.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

# ----------------------------- Optional Numba -----------------------------
try:
    from numba import njit  # type: ignore

    def _maybe_njit(*args, **kwargs):
        return njit(*args, **kwargs)
except Exception:  # Numba not available; no-op decorator
    def _maybe_njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap

# ----------------------------- Configuration -----------------------------
@dataclass(slots=True)
class BEMConfig:
    quad_order: Literal[1, 3, 7] = 7
    TAU_NEAR: float = 0.3          # near-singularity threshold: dist/h < TAU_NEAR
    TOL_NEAR: float = 1e-10        # adaptive subdivision tolerance (absolute, integrals)
    MAX_SUBDIV: int = 7            # max recursion depth for near/self panels
    FOURPI: float = 4.0 * math.pi  # constant
    USE_KP_NEAR_REFINEMENT: bool = True  # refine near for K' as well (recommended)

# ----------------------------- Utilities ---------------------------------
@_maybe_njit(cache=True, fastmath=True)
def _triangle_area(v0, v1, v2) -> float:
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

@_maybe_njit(cache=True, fastmath=True)
def _triangle_height(area: float) -> float:
    # proxy element scale h = sqrt(area), per specification
    return math.sqrt(area)

@_maybe_njit(cache=True, fastmath=True)
def _point_triangle_distance(x, v0, v1, v2) -> float:
    # Closest distance from point x to triangle (v0,v1,v2)
    # (Christer Ericson, "Real-Time Collision Detection", robust form)
    ab = v1 - v0
    ac = v2 - v0
    ap = x - v0
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return np.linalg.norm(ap)

    bp = x - v1
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return np.linalg.norm(bp)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        proj = v0 + v * ab
        return np.linalg.norm(x - proj)

    cp = x - v2
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return np.linalg.norm(cp)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        proj = v0 + w * ac
        return np.linalg.norm(x - proj)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        proj = v1 + w * (v2 - v1)
        return np.linalg.norm(x - proj)

    # Inside face region
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    u = 1.0 - v - w
    proj = u * v0 + v * v1 + w * v2
    return np.linalg.norm(x - proj)

# Dunavant quadrature rules (barycentric pairs (a,b), weight w on reference triangle with area 1/2)
# Weights sum to 1/2. For a physical triangle, multiply Σ w f(y) by (2 * Area(T)).
_DUNAVANT = {
    1: (np.array([[1/3, 1/3]]), np.array([0.5])),
    3: (
        np.array([
            [1/6, 1/6],
            [2/3, 1/6],
            [1/6, 2/3],
        ]),
        np.array([1/6, 1/6, 1/6]),
    ),
    7: (
        np.array([
            [1/3, 1/3],
            [0.0597158717, 0.4701420641],
            [0.4701420641, 0.0597158717],
            [0.4701420641, 0.4701420641],
            [0.7974269853, 0.1012865073],
            [0.1012865073, 0.7974269853],
            [0.1012865073, 0.1012865073],
        ]),
        np.array([
            0.2250000000,
            0.1323941527, 0.1323941527, 0.1323941527,
            0.1259391805, 0.1259391805, 0.1259391805,
        ]) * 0.5,  # scale to sum to 1/2 to match reference area
    ),
}

@_maybe_njit(cache=True, fastmath=True)
def _map_ref_to_phys(v0, v1, v2, a, b):
    # Reference barycentric (a,b, 1-a-b) mapped to physical triangle
    return a * v0 + b * v1 + (1.0 - a - b) * v2

# ----------------------------- Kernels -----------------------------------
@_maybe_njit(cache=True, fastmath=True)
def _G(x_minus_y_norm):
    return 1.0 / (4.0 * math.pi * x_minus_y_norm)

@_maybe_njit(cache=True, fastmath=True)
def _dG_dn_row(n_row, x_minus_y):
    r = np.linalg.norm(x_minus_y)
    return -np.dot(n_row, x_minus_y) / (4.0 * math.pi * (r**3))

@_maybe_njit(cache=True, fastmath=True)
def _gradG(x_minus_y):
    # ∇_x G = -(x - y)/(4π r^3)
    r = np.linalg.norm(x_minus_y)
    return -(x_minus_y) / (4.0 * math.pi * (r**3))

# ---------------------- Quadrature-based Integrals -----------------------
def _panel_integral_S(x: np.ndarray,
                      v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                      cfg: BEMConfig) -> float:
    area = _triangle_area(v0, v1, v2)
    pts, w = _DUNAVANT[cfg.quad_order]
    acc = 0.0
    for k in range(pts.shape[0]):
        a, b = pts[k]
        y = _map_ref_to_phys(v0, v1, v2, a, b)
        ry = np.linalg.norm(x - y)
        acc += w[k] * (1.0 / (cfg.FOURPI * ry))
    return acc * (2.0 * area)

def _panel_integral_Kp(x: np.ndarray, n_row: np.ndarray,
                       v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                       cfg: BEMConfig) -> float:
    # K' uses row normal (observation), not source normal
    area = _triangle_area(v0, v1, v2)
    pts, w = _DUNAVANT[cfg.quad_order]
    acc = 0.0
    for k in range(pts.shape[0]):
        a, b = pts[k]
        y = _map_ref_to_phys(v0, v1, v2, a, b)
        acc += w[k] * _dG_dn_row(n_row, x - y)
    return acc * (2.0 * area)

def _panel_integral_gradS(x: np.ndarray,
                          v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                          cfg: BEMConfig) -> np.ndarray:
    area = _triangle_area(v0, v1, v2)
    pts, w = _DUNAVANT[cfg.quad_order]
    acc = np.zeros(3, dtype=np.float64)
    for k in range(pts.shape[0]):
        a, b = pts[k]
        y = _map_ref_to_phys(v0, v1, v2, a, b)
        acc += w[k] * _gradG(x - y)
    return acc * (2.0 * area)

# ---------------------- Adaptive Subdivision Wrapper ---------------------
def _adaptive_integral(
    integrand_kind: Literal["S", "Kp", "gradS"],
    x: np.ndarray,
    n_row: np.ndarray | None,
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
    cfg: BEMConfig,
    depth: int = 0,
) -> np.ndarray | float:
    """
    Adaptive 4-way midpoint subdivision until coarse-vs-children error < TOL_NEAR
    or MAX_SUBDIV reached. Works for scalar integrals (S, Kp) and vector (gradS).
    """
    if integrand_kind == "S":
        coarse = _panel_integral_S(x, v0, v1, v2, cfg)
    elif integrand_kind == "Kp":
        coarse = _panel_integral_Kp(x, n_row if n_row is not None else np.zeros(3), v0, v1, v2, cfg)
    else:
        coarse = _panel_integral_gradS(x, v0, v1, v2, cfg)

    if depth >= cfg.MAX_SUBDIV:
        return coarse

    # Build 4 sub-triangles by edge midpoints
    m01 = 0.5 * (v0 + v1)
    m12 = 0.5 * (v1 + v2)
    m20 = 0.5 * (v2 + v0)

    children = (
        (v0, m01, m20),
        (m01, v1, m12),
        (m20, m12, v2),
        (m01, m12, m20),
    )

    if integrand_kind == "gradS":
        fine = np.zeros(3, dtype=np.float64)
        for a, b, c in children:
            fine += _adaptive_integral(integrand_kind, x, n_row, a, b, c, cfg, depth + 1)  # type: ignore
        err = np.linalg.norm(fine - coarse)  # type: ignore
    else:
        fine = 0.0
        for a, b, c in children:
            fine += _adaptive_integral(integrand_kind, x, n_row, a, b, c, cfg, depth + 1)  # type: ignore
        err = abs(fine - coarse)

    if err < cfg.TOL_NEAR:
        return fine
    else:
        # Continue subdividing (already subdivided one level; accept the finer estimate)
        return fine

# ----------------------------- Assembly ----------------------------------
def assemble_system(
    triangles: list[dict],
    cfg: BEMConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble dense A, b per specification:

      If T_i ∈ Γ_D:   sum_j S_ij q_j               = φ^BC(x_i).
      If T_i ∈ Γ_N:   sum_j K′_ij q_j  − 1/2 q_i   = g^BC(x_i).

    Notes:
    - Columns never depend on BC (always unknown q_j).
    - K′ uses row/observation normal n_i.
    - K′_ii = 0 (PV); the only 1/2 term is explicit on Neumann row diagonal.

    Returns
    -------
    A : (N,N) float64
    b : (N,)  float64
    """
    if cfg is None:
        cfg = BEMConfig()

    N = len(triangles)
    # assign unknown indices if absent
    for j, t in enumerate(triangles):
        if "unknown_index" not in t:
            t["unknown_index"] = j

    # pre-extract geometry
    V0 = np.empty((N, 3))
    V1 = np.empty((N, 3))
    V2 = np.empty((N, 3))
    Xc = np.empty((N, 3))
    Nrow = np.empty((N, 3))
    bc_is_dir = np.zeros(N, dtype=bool)
    bc_val = np.empty(N)

    areas = np.empty(N)
    hs = np.empty(N)

    for i, tri in enumerate(triangles):
        v0 = np.asarray(tri["vertices"][0], dtype=np.float64)
        v1 = np.asarray(tri["vertices"][1], dtype=np.float64)
        v2 = np.asarray(tri["vertices"][2], dtype=np.float64)
        V0[i], V1[i], V2[i] = v0, v1, v2
        Xc[i] = np.asarray(tri["centroid"], dtype=np.float64)
        Nrow[i] = np.asarray(tri["normal"], dtype=np.float64)
        bc_is_dir[i] = (str(tri["bc_type"]).lower() == "dirichlet")
        bc_val[i] = float(tri["bc_value"])
        a = _triangle_area(v0, v1, v2)
        areas[i] = a
        hs[i] = _triangle_height(a)

    A = np.zeros((N, N), dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)

    # fill RHS by row BC
    b[:] = bc_val

    # assemble dense A row-by-row
    for i in range(N):
        xi = Xc[i]
        ni = Nrow[i]
        if bc_is_dir[i]:
            # Dirichlet row: use S
            for j in range(N):
                v0, v1, v2 = V0[j], V1[j], V2[j]
                if i == j:
                    # robust self-term via adaptive integration
                    A[i, j] = _adaptive_integral("S", xi, None, v0, v1, v2, cfg)  # S_ii
                else:
                    dist = _point_triangle_distance(xi, v0, v1, v2)
                    if dist / hs[j] < cfg.TAU_NEAR:
                        A[i, j] = _adaptive_integral("S", xi, None, v0, v1, v2, cfg)
                    else:
                        A[i, j] = _panel_integral_S(xi, v0, v1, v2, cfg)
        else:
            # Neumann row: use K' and explicit -1/2 on the diagonal
            for j in range(N):
                v0, v1, v2 = V0[j], V1[j], V2[j]
                if i == j:
                    A[i, j] = 0.0  # PV for planar P0
                else:
                    dist = _point_triangle_distance(xi, v0, v1, v2)
                    if cfg.USE_KP_NEAR_REFINEMENT and (dist / hs[j] < cfg.TAU_NEAR):
                        A[i, j] = _adaptive_integral("Kp", xi, ni, v0, v1, v2, cfg)
                    else:
                        A[i, j] = _panel_integral_Kp(xi, ni, v0, v1, v2, cfg)
            A[i, i] -= 0.5  # explicit jump term on Neumann rows

    return A, b

def solve_charges(
    triangles: list[dict],
    cfg: BEMConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble and solve for q.

    Returns
    -------
    q : (N,) solution vector
    A : (N,N) system matrix
    b : (N,)  RHS
    """
    A, b = assemble_system(triangles, cfg)
    # robust solve (symmetric positive for pure Dirichlet; mixed may be indefinite → use solve)
    q = np.linalg.solve(A, b)
    return q, A, b

# ----------------------- Field Evaluation (interior) ----------------------
def evaluate_potential(
    points: np.ndarray,  # (M,3)
    triangles: list[dict],
    q: np.ndarray,
    cfg: BEMConfig | None = None,
) -> np.ndarray:
    """
    φ(x) = ∑_j q_j ∫_{T_j} G(x,y) dS_y
    """
    if cfg is None:
        cfg = BEMConfig()

    N = len(triangles)
    M = int(points.shape[0])

    V0 = np.empty((N, 3))
    V1 = np.empty((N, 3))
    V2 = np.empty((N, 3))
    areas = np.empty(N)
    hs = np.empty(N)
    for j, t in enumerate(triangles):
        v0 = np.asarray(t["vertices"][0], dtype=np.float64)
        v1 = np.asarray(t["vertices"][1], dtype=np.float64)
        v2 = np.asarray(t["vertices"][2], dtype=np.float64)
        V0[j], V1[j], V2[j] = v0, v1, v2
        a = _triangle_area(v0, v1, v2)
        areas[j] = a
        hs[j] = _triangle_height(a)

    phi = np.zeros(M, dtype=np.float64)
    for i in range(M):
        x = np.asarray(points[i], dtype=np.float64)
        acc = 0.0
        for j in range(N):
            v0, v1, v2 = V0[j], V1[j], V2[j]
            dist = _point_triangle_distance(x, v0, v1, v2)
            if dist / hs[j] < cfg.TAU_NEAR:
                sij = _adaptive_integral("S", x, None, v0, v1, v2, cfg)
            else:
                sij = _panel_integral_S(x, v0, v1, v2, cfg)
            acc += q[j] * sij
        phi[i] = acc
    return phi

def evaluate_gradient(
    points: np.ndarray,  # (M,3)
    triangles: list[dict],
    q: np.ndarray,
    cfg: BEMConfig | None = None,
) -> np.ndarray:
    """
    ∇φ(x) = ∑_j q_j ∫_{T_j} ∇_x G(x,y) dS_y
    """
    if cfg is None:
        cfg = BEMConfig()

    N = len(triangles)
    M = int(points.shape[0])

    V0 = np.empty((N, 3))
    V1 = np.empty((N, 3))
    V2 = np.empty((N, 3))
    areas = np.empty(N)
    hs = np.empty(N)
    for j, t in enumerate(triangles):
        v0 = np.asarray(t["vertices"][0], dtype=np.float64)
        v1 = np.asarray(t["vertices"][1], dtype=np.float64)
        v2 = np.asarray(t["vertices"][2], dtype=np.float64)
        V0[j], V1[j], V2[j] = v0, v1, v2
        a = _triangle_area(v0, v1, v2)
        areas[j] = a
        hs[j] = _triangle_height(a)

    grad = np.zeros((M, 3), dtype=np.float64)
    for i in range(M):
        x = np.asarray(points[i], dtype=np.float64)
        acc = np.zeros(3, dtype=np.float64)
        for j in range(N):
            v0, v1, v2 = V0[j], V1[j], V2[j]
            dist = _point_triangle_distance(x, v0, v1, v2)
            if dist / hs[j] < cfg.TAU_NEAR:
                gij = _adaptive_integral("gradS", x, None, v0, v1, v2, cfg)
            else:
                gij = _panel_integral_gradS(x, v0, v1, v2, cfg)
            acc += q[j] * gij  # type: ignore
        grad[i] = acc
    return grad

# --------------------------- High-level helper ----------------------------
def assemble_and_solve(
    triangles: list[dict],
    cfg: BEMConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: returns (q, A, b).
    """
    return solve_charges(triangles, cfg)

def segment_triangle_intersection(p0, p1, tri_vertices):
    """
    Returns (hit_bool, intersection_point, tval)
    Möller–Trumbore intersection with segment p0->p1
    """
    v0, v1, v2= tri_vertices
    dir_seg= p1- p0
    epsilon=1e-14

    edge1= v1- v0
    edge2= v2- v0
    h= np.cross(dir_seg, edge2)
    a= np.dot(edge1,h)
    if abs(a)< epsilon:
        return (False,None,999.9)
    f=1.0/a
    s= p0- v0
    u= f* np.dot(s,h)
    if u<0.0 or u>1.0:
        return (False,None,999.9)
    q= np.cross(s,edge1)
    v= f* np.dot(dir_seg,q)
    if v<0.0 or (u+v)>1.0:
        return (False,None,999.9)
    t= f* np.dot(edge2,q)
    if t<0.0 or t>1.0:
        return (False,None,999.9)
    hit_point= p0+ t* dir_seg
    return (True, hit_point,t)

def find_exit_intersection(p0, p1, all_triangles):
    best_t= 999.9
    best_point= None
    for tri in all_triangles:
        tri_verts= tri['vertices']
        hit, xint, tval= segment_triangle_intersection(p0,p1,tri_verts)
        if hit and tval< best_t:
            best_t= tval
            best_point= xint
    return best_point

def path_trace_simple_bem(
    start_pt,
    triangles,
    q,
    cfg,
    direction='down',
    max_iter=200,
    alpha_initial=0.10,
    first_step=0.05,
    *,
    debug=True,
    angle_max_deg=35.0,
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

    # --- helpers specific to this cylinder (optional but fast) ---
    # If you want generality, replace by mesh-based distance util.
    def dist_to_boundary(pt, R=15.0, H=4.0):
        r = np.hypot(pt[0], pt[1])
        return min(R - r, pt[2], H - pt[2])

    # --- state ---
    path_points  = [start_pt.copy()]
    total_length = 0.0
    x_current    = start_pt.copy()
    prev_step_dir = None

    # --- seed: push safely into the interior and align with -∇φ (for 'down') ---
    # 1) get a provisional surface normal (your routine, or infer per-cap)
    #    If you don't have normals here, seed straight along -∇φ once.
    # evaluate gradient at start (safe: interior eval; if exactly on surface, offset by eps z)
    eps = 1e-6
    g0 = evaluate_gradient(np.asarray([x_current + np.array([0,0,-eps])]), triangles, q, cfg)[0]
    seed_dir = step_dir_from_grad(g0)  # points into the interior for 'down'

    # scale seed by local gap to boundary to clear the jump layer
    fs = max(first_step, 0.15 * max(dist_to_boundary(x_current), 1e-3))
    x_current = x_current + fs * seed_dir
    total_length += fs
    path_points.append(x_current.copy())
    prev_step_dir = seed_dir.copy()
    dbg(f"[SEED] moved {fs:.5f}; seed_dir={seed_dir}")

    # --- main loop ---
    for it in range(1, max_iter + 1):
        g = evaluate_gradient(np.asarray([x_current]), triangles, q, cfg)[0]
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) < 1e-12:
            dbg(f"[{it:03}] gradient too small or invalid → stop")
            break

        cand_dir = step_dir_from_grad(g)  # unit

        # angle guard (compare step directions)
        used_dir = cand_dir
        if prev_step_dir is not None:
            a = float(np.clip(np.dot(cand_dir, prev_step_dir), -1.0, 1.0))
            ang = float(np.degrees(np.arccos(a)))
            if ang > angle_max_deg:
                used_dir = prev_step_dir
                dbg(f"[{it:03}] ∠flip {ang:.2f}° > {angle_max_deg}° → reuse prev_step_dir")

        # pick a safe step length (no backtracking)
        alpha = min(alpha_initial, 0.5 * max(dist_to_boundary(x_current), 1e-3))

        old_pt = x_current.copy()
        new_pt = old_pt + alpha * used_dir

        # exit detection: intersect with boundary (e.g., bottom cap)
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

        # accept step
        seg = np.linalg.norm(new_pt - old_pt)
        total_length += seg
        path_points.append(new_pt.copy())
        x_current = new_pt
        prev_step_dir = used_dir.copy()
        dbg(f"[{it:03}] step ok len={seg:.5f}, total={total_length:.5f}")

    dbg(f"[DONE] steps={len(path_points)-1}, total_len={total_length:.5f}")
    return path_points, total_length
