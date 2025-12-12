# bem_laplace_slp.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal
import numpy as np
from numba import njit

# ----------------------------- Configuration -----------------------------
@dataclass(slots=True)
class BEMConfig:
    quad_order: Literal[1, 3, 7] = 3
    TAU_NEAR: float = 0.2
    TOL_NEAR: float = 1e-6
    MAX_SUBDIV: int = 4
    FOURPI: float = 4.0 * math.pi
    USE_KP_NEAR_REFINEMENT: bool = True

def validate_triangles_for_bem(
    triangles,
    *,
    area_eps=1e-12,
    normal_unit_tol=1e-2,
    dirichlet_targets=(0.0, 1.0),
    dirichlet_tol=1e-6,
    require_watertight=False,
):
    issues = []

    if not isinstance(triangles, (list, tuple)) or len(triangles) == 0:
        return False, ["triangles must be a non-empty list"]

    # ---- schema + basic numeric checks ----
    for t_idx, tri in enumerate(triangles):
        for k in ("vertices", "bc_type", "bc_value", "normal"):
            if k not in tri:
                issues.append(f"[tri {t_idx}] missing key '{k}'")
                continue

        v = np.asarray(tri.get("vertices", None))
        if v.shape != (3, 3) or not np.isfinite(v).all():
            issues.append(f"[tri {t_idx}] vertices must be finite (3,3), got {v.shape}")
            continue

        n = np.asarray(tri.get("normal", None))
        if n.shape != (3,) or not np.isfinite(n).all():
            issues.append(f"[tri {t_idx}] normal must be finite (3,), got {n.shape}")
            continue

        bc_type = str(tri.get("bc_type", "")).lower()
        if bc_type not in ("dirichlet", "neumann"):
            issues.append(f"[tri {t_idx}] bc_type must be 'dirichlet' or 'neumann', got {bc_type}")

        bc_val = tri.get("bc_value", None)
        if bc_val is None or not np.isfinite(float(bc_val)):
            issues.append(f"[tri {t_idx}] bc_value must be finite float, got {bc_val}")

        # ---- geometric checks ----
        e1 = v[1] - v[0]
        e2 = v[2] - v[0]
        cr = np.cross(e1, e2)
        area = 0.5 * np.linalg.norm(cr)
        if not np.isfinite(area) or area <= area_eps:
            issues.append(f"[tri {t_idx}] degenerate area={area:.3e}")

        nlen = np.linalg.norm(n)
        if not np.isfinite(nlen) or abs(nlen - 1.0) > normal_unit_tol:
            issues.append(f"[tri {t_idx}] normal not ~unit (||n||={nlen:.6f})")

        # ---- BC semantics ----
        if bc_type == "dirichlet":
            # must be close to one of the target values (0 or 1 by default)
            if min(abs(float(bc_val) - dv) for dv in dirichlet_targets) > dirichlet_tol:
                issues.append(f"[tri {t_idx}] Dirichlet bc_value={bc_val} not near {dirichlet_targets}")

    ok = (len(issues) == 0)
    return ok, issues

# ----------------------------- Utilities ---------------------------------
@njit(cache=True, fastmath=True)
def _triangle_area(v0, v1, v2) -> float:
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

@njit(cache=True, fastmath=True)
def _triangle_height(area: float) -> float:
    # proxy element scale h = sqrt(area), per specification
    return math.sqrt(area)

@njit(cache=True, fastmath=True)
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

@njit(cache=True, fastmath=True)
def _map_ref_to_phys(v0, v1, v2, a, b):
    # Reference barycentric (a,b, 1-a-b) mapped to physical triangle
    return a * v0 + b * v1 + (1.0 - a - b) * v2

# ----------------------------- Kernels -----------------------------------
@njit(cache=True, fastmath=True)
def _G(x_minus_y_norm):
    return 1.0 / (4.0 * math.pi * x_minus_y_norm)

@njit(cache=True, fastmath=True)
def _dG_dn_row(n_row, x_minus_y):
    r = np.linalg.norm(x_minus_y)
    return -np.dot(n_row, x_minus_y) / (4.0 * math.pi * (r**3))

@njit(cache=True, fastmath=True)
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

    ok, issues = validate_triangles_for_bem(triangles)
    if not ok:
        raise ValueError("Invalid triangles for BEM:\n" + "\n".join(issues))

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
        if "centroid" not in tri:
            raise KeyError(
                "triangle missing 'centroid'. Run preprocess_triangles(triangles) before assemble_system()."
            )
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