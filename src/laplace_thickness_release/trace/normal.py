import numpy as np

def triangles_to_numeric_full(triangles):
    """
    Convert list-of-dict ‘triangles’ → contiguous numeric arrays
    (geometry only; no BC/solution data).

    Returns
    -------
    verts      : (n, 3, 3) float64 – triangle vertices
    norms      : (n, 3)    float64 – unit normals
    areas      : (n,)      float64 – triangle areas
    centroids  : (n, 3)    float64 – pre-computed centroids
    """
    n = len(triangles)

    verts     = np.empty((n, 3, 3), dtype=np.float64)
    norms     = np.empty((n, 3),     dtype=np.float64)
    areas     = np.empty(n,          dtype=np.float64)
    centroids = np.empty((n, 3),     dtype=np.float64)

    for i, tri in enumerate(triangles):
        verts[i]     = tri['vertices']
        norms[i]     = tri['normal']
        areas[i]     = tri['area']
        centroids[i] = tri['centroid']

    return verts, norms, areas, centroids

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

def compute_meyer_normal_jit(pt,
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

    return -n_sum