import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import trimesh
import pyvista as pv

__all__ = [
    "load_freesurfer_surf", "save_vtk",
    "poisson_sample", "gaussian_noise"
]

def load_freesurfer_surf(path):
    from nibabel.freesurfer.io import read_geometry
    v, f = read_geometry(str(path))
    return v.astype(np.float64), f.astype(np.int32)

def save_vtk(filename, verts, faces):
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32), faces])
    pv.PolyData(verts, faces_pv.ravel()).save(filename)

# Sampling
def poisson_sample(verts, faces, n_points, min_spacing=1.0):
    mesh = trimesh.Trimesh(verts, faces, process=False)
    seeds = mesh.sample_surface_even(n_points, radius=min_spacing)
    if len(seeds) < n_points:
        # pad with random barycentric samples
        extra = mesh.sample(n_points - len(seeds))
        seeds = np.vstack([seeds, extra])
    return seeds

# Noise
def gaussian_noise(verts, sigma):
    """Return a new verts array with N(0, σ² I₃) displacement."""
    return verts + np.random.normal(scale=sigma, size=verts.shape)
