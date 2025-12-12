# laplace_thickness/viz/plotting.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_bumpy_top(ax, triangles, face_vals=None, cmap: str = "hot"):
    """
    Plot the Dirichlet top surface (bc_value ≳ 1.0) as a Poly3DCollection.

    If `face_vals` is provided, faces are coloured by those values and a colorbar
    is added to the figure.
    """
    faces_xyz = []
    colours = []

    for tri in triangles:
        if tri["bc_value"] >= 0.99:
            verts = tri.get("vertices_xyz", tri["vertices"])
            faces_xyz.append(verts)
            if face_vals is not None:
                # Keep consistent with legacy behaviour: index corresponds to kept faces only.
                colours.append(face_vals[len(colours)])

    poly = Poly3DCollection(faces_xyz, linewidths=0)
    ax.add_collection3d(poly)  # add FIRST (legacy order)
    poly.set_label("Top surface")

    if face_vals is not None:
        norm = plt.Normalize(np.nanmin(face_vals), np.nanmax(face_vals))
        poly.set_facecolor(plt.cm.get_cmap(cmap)(norm(colours)))

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(face_vals)

        # pass ax=ax so Matplotlib knows which axes to shrink
        ax.figure.colorbar(
            mappable,
            ax=ax,
            shrink=0.5,
            pad=0.02,
            label="Thickness (mm)",
        )
    else:
        poly.set_facecolor("pink")
        poly.set_alpha(0.5)
        poly.set_edgecolors("k")

    return poly


def plot_bumpy_bottom(ax, triangles):
    """
    Plot the Dirichlet bottom surface (bc_value ≈ 0.0) as a Poly3DCollection.
    """
    faces = []
    for tri in triangles:
        if tri["bc_type"] == "dirichlet" and abs(tri["bc_value"] - 0.0) < 1e-10:
            faces.append(tri["vertices"])

    poly_bottom = Poly3DCollection(faces, facecolors="lightblue", alpha=0.5)
    ax.add_collection3d(poly_bottom)
    return poly_bottom


def set_axes_equal(ax) -> None:
    """
    Sets equal scaling for a 3D plot so that the scale for x, y, and z axes are equal.
    This ensures that a cube appears as a cube rather than a rectangular prism.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot radius is half of the maximum range
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
