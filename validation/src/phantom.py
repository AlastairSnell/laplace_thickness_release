import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv
from pathlib import Path
import nibabel as nib
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# ---------- config ----------
SUBJ_MID = Path(r'\\wsl.localhost\Ubuntu\home\uqasnell\freesurfer_subjects\good_output\surf\lh.graymid')
KEEP_PC = 95.0         # keep inner % geodesically
GEO_RADIUS_MM = 15.0   # initial patch radius
STEP_MM = 0.5          # one outward/inward offset
SMOOTH_LAMBDA = 0.5    # Taubin smoothing (single pass)
SMOOTH_MU = -0.53


def read_fs_surf(path: Path):
    v, f = nib.freesurfer.read_geometry(path)
    return v.astype(np.float64), f.astype(np.int32)


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
    n = len(points)
    return csr_matrix((d, (i, j)), shape=(n, n))


def geodesic_ball(points: np.ndarray, faces: np.ndarray, seed_idx: int, radius_mm: float) -> np.ndarray:
    G = csr_graph(points, faces)
    dist = dijkstra(G, directed=False, indices=seed_idx)
    mask = np.zeros(len(points), dtype=bool)
    mask[np.isfinite(dist) & (dist < radius_mm)] = True
    return mask


def largest_component_faces(faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return faces
    edges = unique_edges(faces)
    n = faces.max() + 1
    A = csr_matrix((np.ones(edges.shape[0] * 2),
                    (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                   shape=(n, n))
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
            stack.extend([w for w in nbrs if not visited[w]])
        cid += 1
    sizes = np.bincount(comp_id[comp_id >= 0])
    keep_comp = sizes.argmax()
    return faces[np.all(comp_id[faces] == keep_comp, axis=1)]


def reindex_patch(verts: np.ndarray, faces: np.ndarray):
    used = np.unique(faces.ravel())
    mapping = -np.ones(verts.shape[0], dtype=np.int64)
    mapping[used] = np.arange(used.size, dtype=np.int64)
    return verts[used], mapping[faces]


def compute_face_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1e-12
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
    ln = np.linalg.norm(vnorms, axis=1, keepdims=True); ln[ln == 0] = 1e-12
    return vnorms / ln


def taubin_smooth_once(points: np.ndarray, faces: np.ndarray, lam=0.5, mu=-0.53, fixed=None) -> np.ndarray:
    edges = unique_edges(faces)
    n = len(points)
    A = csr_matrix((np.ones(edges.shape[0] * 2),
                    (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                   shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel(); deg[deg == 0] = 1.0
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
    fixed = np.zeros(len(points), dtype=bool); fixed[seed_idx] = True
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
    return points[keep_mask], faces_mapped[ok]


def plot_two_shells(outer_pts, inner_pts, faces_outer, faces_inner):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection([outer_pts[f] for f in faces_outer],
                                         facecolor="#d62728", alpha=0.45))
    ax.add_collection3d(Poly3DCollection([inner_pts[f] for f in faces_inner],
                                         facecolor="#1f77b4", alpha=0.9))
    allc = np.vstack([outer_pts, inner_pts])
    ax.set_xlim(allc[:, 0].min(), allc[:, 0].max())
    ax.set_ylim(allc[:, 1].min(), allc[:, 1].max())
    ax.set_zlim(allc[:, 2].min(), allc[:, 2].max())
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.show()


def main():
    verts, faces = read_fs_surf(SUBJ_MID)
    seed_idx0 = np.random.randint(len(verts))
    seed_point0 = verts[seed_idx0]

    mask = geodesic_ball(verts, faces, seed_idx0, GEO_RADIUS_MM)
    patch_faces = faces[np.all(mask[faces], axis=1)]
    patch_faces = largest_component_faces(patch_faces)
    patch_verts, patch_faces = reindex_patch(verts, patch_faces)

    faces_pv = np.hstack([np.full((len(patch_faces), 1), 3, dtype=np.int32), patch_faces]).ravel()
    mesh = pv.PolyData(patch_verts, faces_pv)
    #mesh = mesh.subdivide(1, subfilter="loop")
    mesh.save("tessellated_patch.vtk")
    np.save("seed_point.npy", seed_point0)

    points = mesh.points
    faces_arr = mesh.faces.reshape(-1, 4)[:, 1:]

    outer_pts, inner_pts, seed_idx_subdiv, outer_seed, inner_seed = \
        move_vertices_in_out_once(points, faces_arr, seed_point0, STEP_MM)

    outer_pts_c, faces_outer_c = geodesic_keep_inner_percent(outer_pts, faces_arr, seed_idx_subdiv, KEEP_PC)
    inner_pts_c, faces_inner_c = geodesic_keep_inner_percent(inner_pts, faces_arr, seed_idx_subdiv, KEEP_PC)

    plot_two_shells(outer_pts_c, inner_pts_c, faces_outer_c, faces_inner_c)

    np.save("outer_seed_point.npy", outer_seed)
    np.save("inner_seed_point.npy", inner_seed)

    inner_faces_pv = np.hstack([np.full((len(faces_inner_c), 1), 3, dtype=np.int32), faces_inner_c]).ravel()
    pv.PolyData(inner_pts_c, inner_faces_pv).save("simulated_white.vtk")

    outer_faces_pv = np.hstack([np.full((len(faces_outer_c), 1), 3, dtype=np.int32), faces_outer_c]).ravel()
    pv.PolyData(outer_pts_c, outer_faces_pv).save("simulated_pial.vtk")


if __name__ == "__main__":
    main()
