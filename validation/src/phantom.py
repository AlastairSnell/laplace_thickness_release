#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv
from pathlib import Path
import nibabel as nib
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from collections import defaultdict, Counter

# ---------- config ----------
SUBJ_MID      = Path(r'\\wsl.localhost\Ubuntu\home\uqasnell\freesurfer_subjects\good_output\surf\lh.graymid')
KEEP_PC       = 90.0   # keep inner % geodesically (percentile)
GEO_RADIUS_MM = 7      # initial patch radius (geodesic)
STEP_MM       = 2    # one outward/inward offset (mm)
SMOOTH_LAMBDA = 0.6    # Taubin smoothing
SMOOTH_MU     = -0.6
MAX_ATTEMPTS  = 2000   # safety cap on resampling attempts


# ---------- IO ----------
def read_fs_surf(path: Path):
    v, f = nib.freesurfer.read_geometry(path)
    return v.astype(np.float64), f.astype(np.int32)


# ---------- mesh/graph utils ----------
def unique_edges(faces: np.ndarray) -> np.ndarray:
    e01, e12, e20 = faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]
    edges = np.vstack((e01, e12, e20))
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def csr_graph(points: np.ndarray, faces: np.ndarray) -> csr_matrix:
    edges = unique_edges(faces)
    w = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    i = np.concatenate([edges[:, 0], edges[:, 1]])
    j = np.concatenate([edges[:, 1], edges[:, 0]])
    d = np.concatenate([w, w])
    return csr_matrix((d, (i, j)), shape=(len(points), len(points)))


def geodesic_ball(points: np.ndarray, faces: np.ndarray, seed_idx: int, radius_mm: float) -> np.ndarray:
    G = csr_graph(points, faces)
    dist = dijkstra(G, directed=False, indices=seed_idx)
    mask = np.zeros(len(points), dtype=bool)
    mask[np.isfinite(dist) & (dist < radius_mm)] = True
    return mask


def largest_component_faces(faces: np.ndarray) -> np.ndarray:
    """Keep only the largest vertex-connected component of 'faces'."""
    if faces.size == 0:
        return faces

    used = np.unique(faces.ravel())
    mapping = -np.ones(used.max() + 1, dtype=np.int64)
    mapping[used] = np.arange(used.size, dtype=np.int64)
    faces_local = mapping[faces]

    edges = unique_edges(faces_local)
    n = used.size
    A = csr_matrix(
        (np.ones(edges.shape[0] * 2),
         (np.r_[edges[:, 0], edges[:, 1]],
          np.r_[edges[:, 1], edges[:, 0]])),
        shape=(n, n)
    )

    visited = np.zeros(n, dtype=bool)
    comp_id = -np.ones(n, dtype=int)
    cid = 0

    for v in range(n):
        if visited[v]:
            continue
        stack = [v]
        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            comp_id[u] = cid
            nbrs = A.indices[A.indptr[u]:A.indptr[u + 1]]
            for w in nbrs:
                if not visited[w]:
                    stack.append(w)
        cid += 1

    sizes = np.bincount(comp_id[comp_id >= 0])
    keep_comp = sizes.argmax()
    keep_mask = np.all(comp_id[faces_local] == keep_comp, axis=1)
    return faces[keep_mask]


def reindex_patch(verts: np.ndarray, faces: np.ndarray):
    used = np.unique(faces.ravel())
    mapping = -np.ones(verts.shape[0], dtype=np.int64)
    mapping[used] = np.arange(used.size, dtype=np.int64)
    return verts[used], mapping[faces]


def compute_face_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1e-12
    return n / ln


def compute_vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    fn = compute_face_normals(points, faces)
    vnorms = np.zeros_like(points)
    counts = np.zeros((points.shape[0], 1), dtype=np.int32)
    for k in range(3):
        np.add.at(vnorms, faces[:, k], fn)
        np.add.at(counts, faces[:, k], 1)
    counts[counts == 0] = 1
    vnorms /= counts
    ln = np.linalg.norm(vnorms, axis=1, keepdims=True)
    ln[ln == 0] = 1e-12
    return vnorms / ln


def taubin_smooth_once(points: np.ndarray, faces: np.ndarray, lam=0.5, mu=-0.53, fixed=None) -> np.ndarray:
    edges = unique_edges(faces)
    n = len(points)
    A = csr_matrix(
        (np.ones(edges.shape[0] * 2),
         (np.r_[edges[:, 0], edges[:, 1]],
          np.r_[edges[:, 1], edges[:, 0]])),
        shape=(n, n)
    )
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0

    P = points.copy()
    fixed_mask = np.zeros(n, dtype=bool) if fixed is None else fixed

    avg = A.dot(P) / deg[:, None]
    P[~fixed_mask] += lam * (avg[~fixed_mask] - P[~fixed_mask])
    avg = A.dot(P) / deg[:, None]
    P[~fixed_mask] += mu * (avg[~fixed_mask] - P[~fixed_mask])
    return P


def move_vertices_in_out_once(points: np.ndarray, faces: np.ndarray, seed_point: np.ndarray, step_mm: float):
    seed_idx = int(np.argmin(np.linalg.norm(points - seed_point, axis=1)))
    vnorms = compute_vertex_normals(points, faces)

    outer = points + step_mm * vnorms
    inner = points - step_mm * vnorms
    fixed = np.zeros(len(points), dtype=bool)
    fixed[seed_idx] = True

    outer = taubin_smooth_once(outer, faces, SMOOTH_LAMBDA, SMOOTH_MU, fixed)
    inner = taubin_smooth_once(inner, faces, SMOOTH_LAMBDA, SMOOTH_MU, fixed)
    return outer, inner, seed_idx, outer[seed_idx], inner[seed_idx]


def geodesic_keep_inner_percent(points: np.ndarray, faces: np.ndarray, seed_idx: int, keep_pc: float):
    G = csr_graph(points, faces)
    dist = dijkstra(G, directed=False, indices=seed_idx)
    finite = np.isfinite(dist)
    thr = np.quantile(dist[finite], keep_pc / 100.0)
    keep_mask = finite & (dist <= thr)

    new_idx = -np.ones(len(points), dtype=np.int64)
    new_idx[keep_mask] = np.arange(keep_mask.sum(), dtype=np.int64)
    faces_mapped = new_idx[faces]
    ok = np.all(faces_mapped >= 0, axis=1)
    return points[keep_mask], faces_mapped[ok].astype(np.int32)


# ---------- self-intersection checks ----------
def _tri_aabb(P, tri):
    pts = P[tri]
    return pts.min(axis=0), pts.max(axis=0)


def _seg_tri_intersect(p0, p1, v0, v1, v2, eps=1e-12):
    # Segment–triangle via Möller–Trumbore
    dirv = p1 - p0
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(dirv, e2)
    a_ = np.dot(e1, h)
    if abs(a_) < eps:
        return False
    f = 1.0 / a_
    s = p0 - v0
    u = f * np.dot(s, h)
    if u < -eps or u > 1 + eps:
        return False
    qv = np.cross(s, e1)
    v = f * np.dot(dirv, qv)
    if v < -eps or u + v > 1 + eps:
        return False
    t = f * np.dot(e2, qv)
    return (t >= -eps) and (t <= 1 + eps)


def _tri_tri_intersect(P, Q, eps=1e-12):
    # Quick AABB reject
    pa, pb = P.min(0), P.max(0)
    qa, qb = Q.min(0), Q.max(0)
    if (pa > qb + eps).any() or (qa > pb + eps).any():
        return False

    # Edge–triangle tests both ways
    for i in range(3):
        if _seg_tri_intersect(P[i], P[(i + 1) % 3], Q[0], Q[1], Q[2], eps):
            return True
    for i in range(3):
        if _seg_tri_intersect(Q[i], Q[(i + 1) % 3], P[0], P[1], P[2], eps):
            return True
    return False


def _fallback_find_self_intersections(points, faces, voxel_size=None, eps=1e-10):
    if len(faces) == 0:
        return False

    if voxel_size is None:
        e = points[faces[:, [0, 1, 1, 2, 2, 0]]].reshape(-1, 2, 3)
        edge_lens = np.linalg.norm(e[:, 0] - e[:, 1], axis=1)
        voxel_size = np.median(edge_lens) * 1.5 if len(edge_lens) else 1.0

    mins, maxs = [], []
    for f in faces:
        mn, mx = _tri_aabb(points, f)
        mins.append(mn)
        maxs.append(mx)
    mins, maxs = np.vstack(mins), np.vstack(maxs)
    origin = mins.min(axis=0)

    def voxkeys(mn, mx):
        lo = np.floor((mn - origin) / voxel_size).astype(int)
        hi = np.floor((mx - origin) / voxel_size).astype(int)
        for x in range(lo[0], hi[0] + 1):
            for y in range(lo[1], hi[1] + 1):
                for z in range(lo[2], hi[2] + 1):
                    yield (x, y, z)

    grid = defaultdict(list)
    for idx in range(len(faces)):
        for key in voxkeys(mins[idx], maxs[idx]):
            grid[key].append(idx)

    face_verts = [set(faces[i]) for i in range(len(faces))]
    seen = set()

    for idxs in grid.values():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            ia = idxs[a]
            for b in range(a + 1, len(idxs)):
                ib = idxs[b]
                key = (min(ia, ib), max(ia, ib))
                if key in seen:
                    continue
                seen.add(key)
                if face_verts[ia] & face_verts[ib]:
                    continue
                P = points[faces[ia]]
                Q = points[faces[ib]]
                if _tri_tri_intersect(P, Q, eps=eps):
                    return True
    return False


def has_self_intersections(V: np.ndarray, F: np.ndarray) -> bool:
    Vc = np.asarray(V, dtype=np.float64)
    Fc = np.asarray(F, dtype=np.int32)
    return _fallback_find_self_intersections(Vc, Fc)


# ---------- viz ----------
def plot_two_shells(outer_pts, inner_pts, faces_outer, faces_inner):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection([outer_pts[f] for f in faces_outer], facecolor="#d62728", alpha=0.45)
    )
    ax.add_collection3d(
        Poly3DCollection([inner_pts[f] for f in faces_inner], facecolor="#1f77b4", alpha=0.9)
    )

    allc = np.vstack([outer_pts, inner_pts])
    ax.set_xlim(allc[:, 0].min(), allc[:, 0].max())
    ax.set_ylim(allc[:, 1].min(), allc[:, 1].max())
    ax.set_zlim(allc[:, 2].min(), allc[:, 2].max())
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


# ---------- build one candidate ----------
def build_candidate(verts, faces):
    # 1) random seed on lh.graymid
    seed_idx0 = np.random.randint(len(verts))
    seed_point0 = verts[seed_idx0]

    # 2) geodesic patch on the mid-surface
    mask = geodesic_ball(verts, faces, seed_idx0, GEO_RADIUS_MM)
    patch_faces = faces[np.all(mask[faces], axis=1)]
    patch_faces = largest_component_faces(patch_faces)
    if len(patch_faces) == 0:
        raise RuntimeError("Empty patch after geodesic cut; try again.")

    patch_verts, patch_faces = reindex_patch(verts, patch_faces)

    # 3) (optional) subdivide the patch
    faces_pv = np.hstack([np.full((len(patch_faces), 1), 3, dtype=np.int32), patch_faces]).ravel()
    mesh = pv.PolyData(patch_verts, faces_pv)
    mesh = mesh.subdivide(1, subfilter="loop")
    points = mesh.points
    faces_arr = mesh.faces.reshape(-1, 4)[:, 1:]

    # 4) offset ±STEP_MM with light Taubin; seed pinned
    outer_pts, inner_pts, seed_idx_subdiv, outer_seed, inner_seed = \
        move_vertices_in_out_once(points, faces_arr, seed_point0, STEP_MM)

    # 5) keep inner % geodesically
    outer_pts_c, faces_outer_c = geodesic_keep_inner_percent(outer_pts, faces_arr, seed_idx_subdiv, KEEP_PC)
    inner_pts_c, faces_inner_c = geodesic_keep_inner_percent(inner_pts, faces_arr, seed_idx_subdiv, KEEP_PC)

    return {
        "seed_point0": seed_point0,
        "outer_pts_c": outer_pts_c,
        "faces_outer_c": faces_outer_c,
        "inner_pts_c": inner_pts_c,
        "faces_inner_c": faces_inner_c,
        "outer_seed": outer_seed,
        "inner_seed": inner_seed,
        "patch": mesh,
    }


# ---------- main ----------
def main():
    verts, faces = read_fs_surf(SUBJ_MID)
    attempt = 0

    while True:
        attempt += 1
        if attempt > MAX_ATTEMPTS:
            raise RuntimeError(f"Exceeded MAX_ATTEMPTS={MAX_ATTEMPTS} without finding a clean sample.")

        try:
            cand = build_candidate(verts, faces)
        except Exception as e:
            print(f"[attempt {attempt}] Build failed ({e}); resampling…")
            continue

        has_outer_int = has_self_intersections(cand["outer_pts_c"], cand["faces_outer_c"])
        has_inner_int = has_self_intersections(cand["inner_pts_c"], cand["faces_inner_c"])
        if has_outer_int or has_inner_int:
            which = []
            if has_outer_int:
                which.append("outer")
            if has_inner_int:
                which.append("inner")
            print(f"[attempt {attempt}] Self-intersection detected ({' & '.join(which)}). Resampling…")
            continue

        print(f"[attempt {attempt}] Clean shells — saving.")

        # Quick preview
        plot_two_shells(cand["outer_pts_c"], cand["inner_pts_c"], cand["faces_outer_c"], cand["faces_inner_c"])

        # Save mid-surface patch
        cand["patch"].save("tessellated_patch.vtk")
        np.save("seed_point.npy", cand["seed_point0"])
        np.save("outer_seed_point.npy", cand["outer_seed"])
        np.save("inner_seed_point.npy", cand["inner_seed"])

        # Save shells
        inner_faces_pv = np.hstack(
            [np.full((len(cand["faces_inner_c"]), 1), 3, dtype=np.int32), cand["faces_inner_c"]]
        ).ravel()
        pv.PolyData(cand["inner_pts_c"], inner_faces_pv).save("simulated_white.vtk")

        outer_faces_pv = np.hstack(
            [np.full((len(cand["faces_outer_c"]), 1), 3, dtype=np.int32), cand["faces_outer_c"]]
        ).ravel()
        pv.PolyData(cand["outer_pts_c"], outer_faces_pv).save("simulated_pial.vtk")
        break


if __name__ == "__main__":
    main()
