import argparse
from pathlib import Path
import numpy as np
import pyvista as pv


def _hexagonal_grid_points(radius: float, edge_len: float) -> np.ndarray:
    """Return roughly hex‑packed interior points for a flat disc.

    A simple hexagonal (triangular) lattice is laid over the bounding box of the
    disc.  Points whose centres fall inside the circle are kept.  The lattice
    spacing is chosen so that resulting triangles have edges **~ edge_len**.
    """
    # Hex (triangular) lattice vectors
    dx = edge_len
    dy = edge_len * np.sqrt(3) / 2.0

    # Determine lattice bounds that cover the circle
    nx = int(np.ceil(radius / dx))
    ny = int(np.ceil(radius / dy))

    pts = []
    for j in range(-ny, ny + 1):
        row_shift = (j & 1) * 0.5 * dx  # every second row is shifted half‑step
        y = j * dy
        for i in range(-nx, nx + 1):
            x = i * dx + row_shift
            if x * x + y * y < (radius - 1e-6) ** 2:  # keep only points *strictly* inside
                pts.append((x, y))
    return np.asarray(pts, dtype=float)


def make_disc(
    radius: float,
    n_boundary: int,
    z: float = 0.0,
    edge_len: float | None = None,
) -> tuple[pv.PolyData, np.ndarray]:
    """Create a uniformly tessellated 2‑D disc (flat patch).

    * `n_boundary` equally spaced vertices define the outer loop (retained in the
      final mesh so downstream scripts can still locate the rim).
    * Interior vertices are generated on a hexagonal lattice and the whole set
      is triangulated with a constrained 2‑D Delaunay (`pyvista`).
    * The mesh has **no central hub vertex**, avoiding the spoke/fan pattern.
    """
    if n_boundary < 3:
        raise ValueError("n_boundary must be ≥ 3 for a valid disc")

    # --------------------------- boundary loop --------------------------- #
    theta = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
    xy_boundary = np.stack((radius * np.cos(theta), radius * np.sin(theta)), axis=1)

    # --------------------------- interior mesh --------------------------- #
    if edge_len is None:
        # Target interior edge length ≈ boundary segment length
        edge_len = 2.0 * np.pi * radius / n_boundary
    xy_interior = _hexagonal_grid_points(radius, edge_len)

    # --------------------------- point cloud ---------------------------- #
    z_col_boundary = np.full((n_boundary, 1), z, dtype=float)
    boundary_pts = np.hstack((xy_boundary, z_col_boundary))

    z_col_interior = np.full((xy_interior.shape[0], 1), z, dtype=float)
    interior_pts = np.hstack((xy_interior, z_col_interior))

    points = np.vstack((boundary_pts, interior_pts))

    # --------------------------- triangulation -------------------------- #
    cloud = pv.PolyData(points)
    mesh = cloud.delaunay_2d()  # constrained by convex hull of point set (≈ disc)

    # --------------------------- seed (centre) -------------------------- #
    centre = np.array([0.0, 0.0, z], dtype=float)

    return mesh, centre


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate two vertically offset cortical‑like surface patches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--radius", type=float, default=15.0, help="Radius of both discs [mm]")
    parser.add_argument("--thickness", type=float, default=0.25, help="Vertical separation between discs [mm]")
    parser.add_argument("--resolution", type=int, default=60, help="Number of vertices around the boundary loop (controls perimeter density)")
    parser.add_argument("--edge-len", type=float, default=None, help="Approx. target interior edge length [mm] (defaults to boundary segment length)")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Directory to save VTK + seed files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    r = args.radius
    dz = args.thickness
    n_boundary = args.resolution
    edge_len = args.edge_len

    # Create inner (white) at z = 0 and outer (pial) at z = dz
    inner_mesh, inner_seed = make_disc(r, n_boundary, z=0.0, edge_len=edge_len)
    outer_mesh, outer_seed = make_disc(r, n_boundary, z=dz, edge_len=edge_len)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    inner_mesh_path = out_dir / "simulated_white.vtk"
    outer_mesh_path = out_dir / "simulated_pial.vtk"

    inner_mesh.save(inner_mesh_path)
    outer_mesh.save(outer_mesh_path)

    np.save(out_dir / "inner_seed_point.npy", inner_seed)
    np.save(out_dir / "outer_seed_point.npy", outer_seed)

    print("\nCreated:")
    print(f"  {inner_mesh_path}  (vertices={inner_mesh.n_points}, faces={inner_mesh.n_faces})")
    print(f"  {outer_mesh_path}  (vertices={outer_mesh.n_points}, faces={outer_mesh.n_faces})")
    print("  inner_seed_point.npy  (centre = [0, 0, 0])")
    print(f"  outer_seed_point.npy  (centre = [0, 0, {dz}])\n")
    print("Run 'zipper.py' next to generate the bridging side surface.")


if __name__ == "__main__":
    main()
