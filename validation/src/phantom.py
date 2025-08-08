import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv
import networkx as nx

# --- 1. Load the graymid surface ---
graymid_path = "/data/home/uqasnell/tutorial_data_20190918_1558/buckner_data/tutorial_subjs/good_output/surf/lh.graymid"
vertices, faces = nib.freesurfer.read_geometry(graymid_path)

# --- 2. Pick a random vertex (seed) ---
seed_idx = np.random.randint(vertices.shape[0])
seed_point = vertices[seed_idx]

# --- 3. Compute geodesic distances over the surface ---
# Build a graph: each vertex is a node, and edges exist between vertices that share a face.
G = nx.Graph()
# Add edges from all faces; for each face, add each pair of vertices with weight equal to their Euclidean distance.
for f in faces:
    a, b, c = f
    for (i, j) in [(a, b), (b, c), (c, a)]:
        # Compute the Euclidean distance between vertices i and j
        d = np.linalg.norm(vertices[i] - vertices[j])
        # Add edge (or update if needed)
        if G.has_edge(i, j):
            # In case of duplicates, keep the smaller weight
            if d < G[i][j]['weight']:
                G[i][j]['weight'] = d
        else:
            G.add_edge(i, j, weight=d)

# Use Dijkstra's algorithm to compute geodesic distances from the seed vertex
geo_dists = nx.single_source_dijkstra_path_length(G, seed_idx)

# Create a boolean mask: True for vertices with geodesic distance below the threshold.
threshold = 4.0
mask = np.zeros(vertices.shape[0], dtype=bool)
for v, dist in geo_dists.items():
    if dist < threshold:
        mask[v] = True

# --- 4. Filter faces: keep only faces where all vertices are within the patch ---
patch_faces = np.array([f for f in faces if mask[f[0]] and mask[f[1]] and mask[f[2]]])

# --- 5. Remove non-contiguous parts; keep only the largest connected component ---
def get_largest_component(faces):
    # Build adjacency list for vertices in patch
    graph = {}
    unique_vertices = set(faces.flatten())
    for v in unique_vertices:
        graph[v] = set()
    for f in faces:
        a, b, c = f
        graph[a].update([b, c])
        graph[b].update([a, c])
        graph[c].update([a, b])
    visited = set()
    components = []
    for v in graph:
        if v not in visited:
            stack = [v]
            comp = set()
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                comp.add(curr)
                stack.extend(list(graph[curr] - visited))
            components.append(comp)
    return max(components, key=len)

largest_component = get_largest_component(patch_faces)
final_patch_faces = np.array([f for f in patch_faces if all(v in largest_component for v in f)])

# --- 6. Re-index the patch vertices ---
final_vertex_indices = np.unique(final_patch_faces)
final_coords = vertices[final_vertex_indices]
mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(final_vertex_indices)}
final_faces = np.array([[mapping[v] for v in face] for face in final_patch_faces])

# --- 7. Create a PyVista mesh for the patch ---
faces_pv = np.column_stack((np.full(len(final_faces), 3, dtype=np.int32), final_faces)).ravel()
patch_mesh = pv.PolyData(final_coords, faces_pv)

# --- 8. Re-tessellate (refine) the patch using Loop subdivision ---
# This will add new vertices and triangles for a smoother, more refined mesh.
# The 'nsub' parameter controls the number of subdivision iterations.
subdivided_patch = patch_mesh.subdivide(nsub=1, subfilter='loop')

# --- 9. Plot the re-tessellated patch using matplotlib ---
# We extract the triangles from the subdivided mesh.
sub_coords = subdivided_patch.points
# PyVista stores faces as a flat array: [3, v0, v1, v2, 3, v3, v4, v5, ...]
sub_faces = subdivided_patch.faces.reshape((-1, 4))[:, 1:]  # remove the leading count per triangle

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

triangles = [sub_coords[face] for face in sub_faces]
patch_collection = Poly3DCollection(triangles, facecolor='cyan', edgecolor='k', alpha=0.7)
ax.add_collection3d(patch_collection)

# Highlight the original seed if it is in the refined patch
# (This may not be exactly present after subdivision, but we can mark the nearest point.)
dists_sub = np.linalg.norm(sub_coords - seed_point, axis=1)
seed_sub_idx = np.argmin(dists_sub)
ax.scatter(sub_coords[seed_sub_idx,0], sub_coords[seed_sub_idx,1], sub_coords[seed_sub_idx,2],
           color='red', s=50)

ax.set_xlim(sub_coords[:,0].min(), sub_coords[:,0].max())
ax.set_ylim(sub_coords[:,1].min(), sub_coords[:,1].max())
ax.set_zlim(sub_coords[:,2].min(), sub_coords[:,2].max())
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("Re-tessellated (Subdivided) Patch from Graymid Surface")
# Save the tessellated patch mesh to a file (VTK format in this example)
subdivided_patch.save("tessellated_patch.vtk")
np.save("seed_point.npy", seed_point)
plt.show()
