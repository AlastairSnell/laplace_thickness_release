#!/usr/bin/env python3
"""
Generate two concentric hemispherical surface patches (open at the equator)
separated by a specified thickness (difference in radii), similar to the prior
flat-disc generator. Saves:

  simulated_white.vtk   # inner hemisphere (radius = R)
  simulated_pial.vtk    # outer hemisphere (radius = R + thickness)
  inner_seed_point.npy  # seed at the inner north pole  [0, 0,  R]
  outer_seed_point.npy  # seed at the outer north pole  [0, 0, R+thickness]

Use with your existing zipper/solver steps that expect VTK PolyData meshes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pyvista as pv


def make_hemisphere(
    radius: float,
    theta_res: int,
    phi_res: int,
    *,
    keep_upper: bool = True,
) -> pv.PolyData:
    """Create a triangulated hemisphere using PyVista's sphere + clipping.

    Args:
      radius: Sphere radius.
      theta_res: Azimuthal (longitude) resolution.
      phi_res: Polar (latitude) resolution (from pole to equator for hemisphere).
      keep_upper: If True, keep z >= 0 (north/up hemisphere). If False, keep z <= 0.

    Returns:
      PyVista PolyData containing only triangular faces.
    """
    # Full sphere first (PyVista ensures triangular faces by default)
    sph = pv.Sphere(radius=radius, theta_resolution=theta_res, phi_resolution=2*phi_res)

    # Clip at the equatorial plane z = 0 to keep a hemisphere
    # For upper hemisphere, keep points with z >= 0; for lower, invert selection.
    plane_normal = (0.0, 0.0, -1.0)  # plane with normal -z and origin at 0 keeps z >= 0 by default
    hemi = sph.clip(normal=plane_normal, origin=(0.0, 0.0, 0.0), invert=not keep_upper)

    # Clean to merge coincident points introduced by clipping
    hemi = hemi.clean()

    # Ensure triangle faces array (PyVista spheres are triangles already, but clean())
    if not hemi.is_all_triangles:
        hemi = hemi.triangulate()
    return hemi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate two concentric hemispheres at radii R and R+thickness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--inner-radius", type=float, default=5.0, help="Inner hemisphere radius R [mm]")
    p.add_argument("--thickness", type=float, default=2, help="Radial gap (outer - inner) [mm]")
    p.add_argument("--theta-res", type=int, default=30, help="Azimuthal resolution (longitude divisions)")
    p.add_argument("--phi-res", type=int, default=15, help="Polar resolution for hemisphere (pole→equator)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory to save VTK + seed files",
    )
    p.add_argument(
        "--lower",
        action="store_true",
        help="Generate the LOWER hemisphere (z<=0) instead of the default upper (z>=0)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    R = float(args["inner_radius"]) if isinstance(args, dict) else args.inner_radius
    t = float(args["thickness"]) if isinstance(args, dict) else args.thickness
    theta_res = args.theta_res
    phi_res = args.phi_res
    keep_upper = not args.lower

    outer_R = R + t

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build inner ("white") and outer ("pial") hemispheres
    white_mesh = make_hemisphere(R, theta_res, phi_res, keep_upper=keep_upper)
    pial_mesh  = make_hemisphere(outer_R, theta_res, phi_res, keep_upper=keep_upper)

    inner_mesh_path = out_dir / "simulated_white.vtk"
    outer_mesh_path = out_dir / "simulated_pial.vtk"

    white_mesh.save(inner_mesh_path)
    pial_mesh.save(outer_mesh_path)

    # Seed points at the north/south pole on each hemisphere
    pole_z = outer_R if keep_upper else -outer_R
    outer_seed = np.array([0.0, 0.0, -pole_z], dtype=float)

    pole_z_inner = R if keep_upper else -R
    inner_seed = np.array([0.0, 0.0, -pole_z_inner], dtype=float)

    np.save(out_dir / "inner_seed_point.npy", inner_seed)
    np.save(out_dir / "outer_seed_point.npy", outer_seed)

    # Report
    print("\nCreated concentric hemispheres:")
    print(f"  {inner_mesh_path}  (R = {R:g}, verts = {white_mesh.n_points}, faces = {white_mesh.n_faces})")
    print(f"  {outer_mesh_path}  (R = {outer_R:g}, verts = {pial_mesh.n_points}, faces = {pial_mesh.n_faces})")
    hemi_name = "upper (z>=0)" if keep_upper else "lower (z<=0)"
    print(f"  Hemisphere: {hemi_name}")
    print(f"  inner_seed_point.npy  (pole = [0, 0, {pole_z_inner:g}])")
    print(f"  outer_seed_point.npy  (pole = [0, 0, {pole_z:g}])\n")
    print("Run 'zipper.py' (adapted for spherical rims) if you want to cap the equator.")


if __name__ == "__main__":
    main()
