import numpy as np
from numba import njit, prange

BARY_3PT = np.array([
    [1/2, 1/2, 0.0],
    [1/2, 0.0, 1/2],
    [0.0, 1/2, 1/2],
], dtype=np.float64)

BARY_7PT = np.array([
    [1/3,          1/3,          1/3        ],
    [0.059715871789770, 0.470142064105115, 0.470142064105115],
    [0.470142064105115, 0.059715871789770, 0.470142064105115],
    [0.470142064105115, 0.470142064105115, 0.059715871789770],
    [0.797426985353087, 0.101286507323456, 0.101286507323456],
    [0.101286507323456, 0.797426985353087, 0.101286507323456],
    [0.101286507323456, 0.101286507323456, 0.797426985353087]
], dtype=np.float64)

def triangles_to_numeric_full(triangles):
    """
    Convert list-of-dict ‘triangles’ → contiguous numeric arrays.

    Returns
    -------
    verts      : (n, 3, 3) float64   – triangle vertices
    norms      : (n, 3)     float64   – unit normals
    areas      : (n,)       float64   – triangle areas
    flux       : (n,)       float64   – Neumann data  q_j
    phi        : (n,)       float64   – Dirichlet data φ_j
    centroids  : (n, 3)     float64   – pre-computed centroids
    """
    n = len(triangles)

    verts     = np.empty((n, 3, 3), dtype=np.float64)
    norms     = np.empty((n, 3),     dtype=np.float64)
    areas     = np.empty(n,          dtype=np.float64)
    flux      = np.empty(n,          dtype=np.float64)
    phi       = np.empty(n,          dtype=np.float64)
    centroids = np.empty((n, 3),     dtype=np.float64)

    for i, tri in enumerate(triangles):
        verts[i]     = tri['vertices']
        norms[i]     = tri['normal']
        areas[i]     = tri['area']
        flux[i]      = tri['flux']
        phi[i]       = tri['potential']
        centroids[i] = tri['centroid']

    return verts, norms, areas, centroids, flux, phi

def closest_centroid_idx(pt, cents_arr):
    """
    Return the index of the triangle whose centroid is nearest to `pt`.

    Parameters
    ----------
    pt        : (3,) array_like
        Query point (x, y, z).
    cents_arr : (N, 3) float64
        Centroids of all N triangles.

    Returns
    -------
    idx : int
        Index of the closest centroid, or -1 if cents_arr is empty.
    """
    cents_arr = np.asarray(cents_arr, dtype=float)
    if cents_arr.size == 0:
        return -1

    pt        = np.asarray(pt, dtype=float)
    # Squared Euclidean distances, then argmin
    d2        = np.sum((cents_arr - pt)**2, axis=1)
    return int(np.argmin(d2))

def compute_meyer_normal(pt,
                             verts_arr,
                             norms_arr,
                             areas_arr,
                             cents_arr,
                             radius=-1.0,
                             eps=1e-12,
                             debug=False):
    """
    Meyer-style area-weighted pseudo-normal at arbitrary point `pt`.

    radius < 0  → base face + edge-sharing neighbours (“one-ring”)
    radius >= 0 → faces whose centroids lie within `radius` of `pt`
    """
    n_faces = cents_arr.shape[0]

    # ----------------------------------------------------------------------
    # 1) Find the face whose centroid is closest to `pt`
    # ----------------------------------------------------------------------
    base_idx = closest_centroid_idx(pt, cents_arr)
    if base_idx == -1:
        return norms_arr[0]          # degenerate fallback

    # ----------------------------------------------------------------------
    # 2) Build neighbour mask
    # ----------------------------------------------------------------------
    neigh_mask = np.zeros(n_faces, np.uint8)
    neigh_mask[base_idx] = 1

    if radius < 0.0:                 # literal one-ring
        v0, v1, v2 = verts_arr[base_idx]
        for j in range(n_faces):
            if j == base_idx:
                continue
            shared = 0
            for vv in verts_arr[j]:
                if (np.all(vv == v0) or np.all(vv == v1) or np.all(vv == v2)):
                    shared += 1
            if shared >= 2:          # shares an edge
                neigh_mask[j] = 1
    else:                            # centroid-radius set
        for j in range(n_faces):
            if np.linalg.norm(cents_arr[j] - pt) < radius:
                neigh_mask[j] = 1

    # ----------------------------------------------------------------------
    # 3) Area-weighted face-normal sum  (Meyer 2003, Eq. (11))
    # ----------------------------------------------------------------------
    n_sum = np.zeros(3, np.float64)

    for j in range(n_faces):
        if neigh_mask[j] == 0:
            continue

        # --- Weight choice -------------------------------------------------
        # Meyer pseudo-normal:           w = area_f
        # Angle-weighted pseudo-normal:  w = angle at pt in face f
        w = areas_arr[j]

        # Uncomment below for angle weighting (pt must coincide with a vertex)
        # if False:
        #     verts = verts_arr[j]
        #     # find which vertex indexes match pt
        #     vidx = -1
        #     for k in range(3):
        #         if np.all(np.abs(verts[k] - pt) < eps):
        #             vidx = k
        #             break
        #     if vidx >= 0:
        #         a = verts[(vidx + 1) % 3] - verts[vidx]
        #         b = verts[(vidx + 2) % 3] - verts[vidx]
        #         cos_ang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)
        #         w = np.arccos(np.clip(cos_ang, -1.0, 1.0))

        if debug:
            print(f"[dbg] face {j:4d}: area={areas_arr[j]:.4e}, w={w:.4e}, n={norms_arr[j]}")

        n_sum += norms_arr[j] * w

    # ----------------------------------------------------------------------
    # 4) Normalise and orient consistently with base face
    # ----------------------------------------------------------------------
    n_len = np.linalg.norm(n_sum)
    if n_len < eps:
        return norms_arr[base_idx]   # fallback if something went wrong

    n_sum /= n_len
    if np.dot(n_sum, norms_arr[base_idx]) < 0.0:
        n_sum *= -1.0

    return n_sum

@njit
def _point_triangle_distance(pt, vertices):
    """
    Mirror of original point_triangle_distance():
    • exact plane distance if projection inside
    • else min distance to the three edges
    """
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]

    # --- plane normal ---
    cross = np.cross(v1 - v0, v2 - v0)
    denom = np.linalg.norm(cross)
    if denom < 1e-14:           # degenerate triangle
        d0 = np.linalg.norm(pt - v0)
        d1 = np.linalg.norm(pt - v1)
        d2 = np.linalg.norm(pt - v2)
        return min(d0, d1, d2)

    normal = cross / denom
    dist_plane = abs(np.dot(pt - v0, normal))

    # --- projection of pt onto plane ---
    proj = pt - np.dot(pt - v0, normal) * normal

    # barycentric‐inside test
    v0p = proj - v0
    v1p = v1 - v0
    v2p = v2 - v0

    dot11 = np.dot(v1p, v1p)
    dot12 = np.dot(v1p, v2p)
    dot1p = np.dot(v1p, v0p)
    dot22 = np.dot(v2p, v2p)
    dot2p = np.dot(v2p, v0p)

    inv_den = 1.0 / (dot11 * dot22 - dot12 * dot12)
    alpha   = (dot22 * dot1p - dot12 * dot2p) * inv_den
    beta    = (dot11 * dot2p - dot12 * dot1p) * inv_den

    if alpha >= 0.0 and beta >= 0.0 and (alpha + beta) <= 1.0:
        # projection lies inside the triangle
        return dist_plane
    else:
        # distance to each edge
        d0 = _point_segment_distance(pt, v0, v1)
        d1 = _point_segment_distance(pt, v1, v2)
        d2 = _point_segment_distance(pt, v2, v0)
        return d0 if d0 < d1 and d0 < d2 else (d1 if d1 < d2 else d2)

@njit
def _point_segment_distance(pt, a, b):
    """
    Distance from point `pt` to segment [a,b].
    """
    ab   = b - a
    ab2  = np.dot(ab, ab)
    if ab2 < 1e-14:
        return np.linalg.norm(pt - a)

    t = np.dot(pt - a, ab) / ab2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    closest = a + t * ab
    return np.linalg.norm(pt - closest)

@njit
def characteristic_size(vertices):
    v0, v1, v2 = vertices
    e01 = np.linalg.norm(v1 - v0)
    e12 = np.linalg.norm(v2 - v1)
    e02 = np.linalg.norm(v2 - v0)
    return (e01 + e12 + e02) / 3.0

@njit
def _subdivide_triangle(verts):
    """
    Split into 4 children (Loop subdivision style).
    Returns (4,3,3) array.
    """
    m01 = 0.5 * (verts[0] + verts[1])
    m12 = 0.5 * (verts[1] + verts[2])
    m20 = 0.5 * (verts[2] + verts[0])
    out = np.empty((4, 3, 3), dtype=np.float64)
    # child 0
    out[0, 0], out[0, 1], out[0, 2] = verts[0], m01, m20
    # child 1
    out[1, 0], out[1, 1], out[1, 2] = m01, verts[1], m12
    # child 2
    out[2, 0], out[2, 1], out[2, 2] = m20, m12, verts[2]
    # centre child
    out[3, 0], out[3, 1], out[3, 2] = m01, m12, m20
    return out

@njit
def _tri_integrate_vec(func_id, x, verts, n_y, coarse):
    bset = BARY_3PT if coarse else BARY_7PT
    A = tri_area(verts)
    acc = np.zeros(3)
    for k in range(bset.shape[0]):
        bc = bset[k]
        y = bc[0]*verts[0] + bc[1]*verts[1] + bc[2]*verts[2]
        acc += _integrand_vec(func_id, x, y, n_y)
    return acc * (A / bset.shape[0])

@njit
def grad_x_G(x, y):
    """
    grad_x G(x, y) = -(x - y) / (4 pi |x - y|^3)
    Returns a 3D vector
    """
    r_vec = x - y
    r = np.linalg.norm(r_vec)
    if r < 1e-14:
        return np.zeros(3)
    return -r_vec / (4.0 * np.pi * r**3)

@njit
def grad_x_dGdn(x, y, n_y):
    """
    grad_x[dG/dn_y](x, y) = derivative wrt x of dot((x-y), n_y)/(4 pi |x-y|^3).
    Uses quotient rule to handle the 1/|x-y|^3 factor.
    """
    r_vec = x - y
    r = np.linalg.norm(r_vec)
    if r < 1e-14:
        return np.zeros(3)

    A = np.dot(r_vec, n_y)
    B = r**3
    gradA = n_y
    gradB = 3.0 * r * r_vec
    out = (gradA * B - A * gradB) / (B**2)
    out /= (4.0 * np.pi)
    return out

@njit
def _integrand_vec(func_id, x, y, n_y):
    if func_id == 2:
        return grad_x_G(x, y)
    else:
        return grad_x_dGdn(x, y, n_y)

@njit
def tri_area(vertices):
    e1 = vertices[1] - vertices[0]
    e2 = vertices[2] - vertices[0]
    return 0.5 * np.linalg.norm(np.cross(e1, e2))

@njit
def _adaptive_vec(func_id, x, verts, n_y,
                  tol=1e-3, max_ref=3):
    if max_ref > 0:
        d = _point_triangle_distance(x, verts)
        h = characteristic_size(verts)
        if d < 0.1*h:
            childs = _subdivide_triangle(verts)
            acc = np.zeros(3)
            for c in range(4):
                acc += _adaptive_vec(func_id, x, childs[c], n_y,
                                     tol, max_ref-1)
            return acc

    if max_ref == 0:
        return _tri_integrate_vec(func_id, x, verts, n_y, coarse=False)

    coarse = _tri_integrate_vec(func_id, x, verts, n_y, coarse=True)
    fine   = _tri_integrate_vec(func_id, x, verts, n_y, coarse=False)
    if np.linalg.norm(fine - coarse) < tol:
        return fine

    childs = _subdivide_triangle(verts)
    acc = np.zeros(3)
    for c in range(4):
        acc += _adaptive_vec(func_id, x, childs[c], n_y,
                             tol, max_ref-1)
    return acc

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

def segment_triangle_intersection(p0, p1, tri_vertices):
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

@njit(parallel=True)
def evaluate_gradient_numba(x, verts, normals, flux, phi,
                            tol=1e-3, max_ref=3):
    """
    Mirrors evaluate_gradient() in full nopython mode.
    Returns 3-vector gradient.
    """
    n_tris = verts.shape[0]
    grad = np.zeros(3)
    for j in prange(n_tris):
        I_vec = _adaptive_vec(2, x, verts[j], normals[j], tol, max_ref)
        J_vec = _adaptive_vec(3, x, verts[j], normals[j], tol, max_ref) 
        grad += flux[j]*I_vec - phi[j]*J_vec
    return grad

def path_trace_simple(
    start_pt,
    triangles,
    direction='down',
    max_iter=200,
    alpha_initial=0.1,
    first_step=0.05,
    *,
    debug=False,):
    # 1) ------------------------------------------------ dbg helper
    def dbg(msg):
        if debug:
            print(msg)

    # 2) -------------------------------------------- forced full‑α step
    def forced_step(old_pt, g_prev, α_full):
        step_dir  = (-g_prev if direction == 'down' else g_prev)
        step_dir /= np.linalg.norm(step_dir)
        return old_pt + α_full * step_dir

    # 3) ------------------------------- mesh → contiguous numeric arrays
    verts_arr, norms_arr, areas_arr, cents_arr, q_arr, phi_arr = \
        triangles_to_numeric_full(triangles)
    faces = np.arange(verts_arr.size // 3).reshape(-1, 3)

    # 4) --------------------------- bookkeeping / initial state
    path_points  = [start_pt.copy()]
    total_length = 0.0
    x_current    = start_pt.copy()
    alpha        = alpha_initial
    prev_grad    = None

    # 5) ----------------------------- tiny seed step off the surface
    avg_normal = compute_meyer_normal(
        x_current, verts_arr, norms_arr, areas_arr, cents_arr,
        radius=-1.0, eps=1e-12, debug=False)

    if direction == 'up':
        avg_normal = -avg_normal

    if avg_normal is not None:
        seed_vec = first_step * avg_normal
        x_current = x_current - seed_vec
        total_length += np.linalg.norm(seed_vec)
        path_points.append(x_current.copy())
        dbg(f"[SEED] moved {np.linalg.norm(seed_vec):.5f} along avg normal")

    # 6) ------------------------------ main iteration loop
    GRAD_NORM_MAX = 1.0
    for it in range(1, max_iter + 1):
        # compute ∇φ (needed for steps 8, 10, 12)
        grad_val  = evaluate_gradient_numba(
            x_current, verts_arr, norms_arr, q_arr, phi_arr)
        grad_norm = np.linalg.norm(grad_val)

        # 8) -------- sanity‑check angle between successive gradients
        angle_flip = False
        if prev_grad is not None and grad_norm > 0.0:
            cosθ = np.clip(
                np.dot(grad_val, prev_grad) /
                (grad_norm * np.linalg.norm(prev_grad)), -1.0, 1.0)
            if np.degrees(np.arccos(cosθ)) > 35.0:
                angle_flip = True
                dbg(f"[{it:03}] ∠flip too high ({np.degrees(np.arccos(cosθ))}) → use prev ∇φ")

        # 10.1) ----- flag huge gradient magnitude drift
        huge_grad = abs(grad_norm - 1.0) > GRAD_NORM_MAX

        # 10.3) ----- decide whether to reuse previous gradient
        use_prev = (huge_grad or angle_flip) and (prev_grad is not None)
        grad_use = prev_grad if use_prev else grad_val

        old_pt = x_current.copy()

        # 11) -------- forced step branch when trusting prev ∇φ
        if use_prev:
            new_pt = forced_step(old_pt, prev_grad, alpha_initial)
            X_int  = find_exit_intersection(old_pt, new_pt, triangles)

            if X_int is not None:                          # exited mesh
                seg = np.linalg.norm(X_int - old_pt)
                total_length += seg
                path_points.append(X_int.copy())
                dbg(f"[{it:03}] EXIT after forced step (len={seg:.5f})")
                break
            else:                                          # still inside
                seg = np.linalg.norm(new_pt - old_pt)
                total_length += seg
                path_points.append(new_pt.copy())
                x_current = new_pt
                continue                                   # next iter

        # 12) -------- ordinary step along current trusted gradient
        step_dir  = (-grad_use if direction == 'down' else grad_use)
        step_dir /= np.linalg.norm(step_dir)
        new_pt    = old_pt + alpha * step_dir

        # 15) -------- intersection test for this step
        X_int = find_exit_intersection(old_pt, new_pt, triangles)
        if X_int is not None:                              # crossed out
            seg = np.linalg.norm(X_int - old_pt)
            total_length += seg
            path_points.append(X_int.copy())
            dbg(f"[{it:03}] EXIT (len={seg:.5f})")
            break
        else:                                              # accepted step
            seg = np.linalg.norm(new_pt - old_pt)
            total_length += seg
            path_points.append(new_pt.copy())
            x_current  = new_pt
            prev_grad  = grad_use.copy()
            # α stays constant; no back‑tracking logic in this simple version

    # 16) ------------ loop ends naturally if max_iter reached
    dbg(f"[DONE] steps={len(path_points)-1}, total_len={total_length:.5f}")


    # 19) ------------ return polyline & length
    return path_points, total_length
