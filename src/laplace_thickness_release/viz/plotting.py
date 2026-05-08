# laplace_thickness/viz/plotting.py
from __future__ import annotations

import inspect
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_bumpy_top(ax, triangles, face_vals=None, cmap: str = "hot", clim: tuple[float, float] | None = None):
    """
    Plot the Dirichlet top surface (bc_value ≳ 1.0) as a Poly3DCollection.

    If `face_vals` is provided, faces are coloured by those values and a colorbar
    is added to the figure. Use `clim` to fix the color range.
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
        if clim is None:
            norm = plt.Normalize(np.nanmin(face_vals), np.nanmax(face_vals))
        else:
            norm = plt.Normalize(clim[0], clim[1])
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


def plot_bumpy_bottom(ax, triangles, face_vals=None, cmap: str = "hot", clim: tuple[float, float] | None = None):
    """
    Plot the Dirichlet bottom surface (bc_value ~ 0.0) as a Poly3DCollection.
    """
    faces = []
    colours = []
    for tri in triangles:
        if tri["bc_type"] == "dirichlet" and abs(tri["bc_value"] - 0.0) < 1e-10:
            faces.append(tri["vertices"])
            if face_vals is not None:
                colours.append(face_vals[len(colours)])

    poly_bottom = Poly3DCollection(faces, linewidths=0)
    ax.add_collection3d(poly_bottom)
    poly_bottom.set_label("Bottom surface")

    if face_vals is not None:
        if clim is None:
            norm = plt.Normalize(np.nanmin(face_vals), np.nanmax(face_vals))
        else:
            norm = plt.Normalize(clim[0], clim[1])
        poly_bottom.set_facecolor(plt.cm.get_cmap(cmap)(norm(colours)))

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(face_vals)

        ax.figure.colorbar(
            mappable,
            ax=ax,
            shrink=0.5,
            pad=0.02,
            label="Thickness (mm)",
        )
    else:
        poly_bottom.set_facecolor("lightblue")
        poly_bottom.set_alpha(0.5)

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


def _faces_to_pv(faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return np.empty((0,), dtype=np.int64)
    faces = np.asarray(faces, dtype=np.int64)
    return np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()


def plot_heatmap_pyvista(
    *,
    V_top: np.ndarray,
    F_top: np.ndarray,
    top_vertex_vals: np.ndarray,
    keep_mask_top: np.ndarray,
    V_bottom: np.ndarray | None = None,
    F_bottom: np.ndarray | None = None,
    keep_mask_bottom: np.ndarray | None = None,
    clim: tuple[float, float] | None = None,
    cmap: str = "hot",
    bottom_color: str = "lightblue",
    bottom_opacity: float = 0.35,
    smooth_shading: bool = True,
    screenshot_path: str | None = None,
    window_size: tuple[int, int] | None = None,
    image_scale: int = 1,
    camera_views: list[tuple[float, float, str]] | None = None,
    disable_shadows: bool = True,
) -> None:
    import pyvista as pv

    faces_top = F_top[keep_mask_top] if keep_mask_top is not None else F_top
    mesh_top = pv.PolyData(V_top, _faces_to_pv(faces_top))
    mesh_top.point_data["thickness"] = top_vertex_vals

    off_screen = screenshot_path is not None or bool(camera_views)
    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
    if disable_shadows:
        plotter.disable_shadows()
    mesh_args = dict(
        scalars="thickness",
        cmap=cmap,
        smooth_shading=smooth_shading,
        show_edges=False,
        show_scalar_bar=False,
    )
    if clim is not None:
        mesh_args["clim"] = clim
    plotter.add_mesh(mesh_top, **mesh_args)
    plotter.add_scalar_bar(
        title="",
        vertical=False,
    )

    if V_bottom is not None and F_bottom is not None and keep_mask_bottom is not None:
        faces_bottom = F_bottom[keep_mask_bottom] if keep_mask_bottom is not None else F_bottom
        if faces_bottom.size > 0:
            mesh_bottom = pv.PolyData(V_bottom, _faces_to_pv(faces_bottom))
            plotter.add_mesh(
                mesh_bottom,
                color=bottom_color,
                opacity=bottom_opacity,
                smooth_shading=smooth_shading,
                show_edges=False,
            )

    def _compute_camera_position() -> tuple[list[float], float]:
        points = V_top
        if V_bottom is not None:
            points = np.vstack([V_top, V_bottom])
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = (mins + maxs) * 0.5
        radius = float(np.max(maxs - mins)) * 1.8
        if radius <= 0:
            radius = 1.0
        return center.tolist(), radius

    def _camera_from_angles(center: list[float], radius: float, az_deg: float, el_deg: float):
        az = math.radians(az_deg)
        el = math.radians(el_deg)
        x = math.cos(el) * math.cos(az)
        y = math.cos(el) * math.sin(az)
        z = math.sin(el)
        pos = [center[0] + radius * x, center[1] + radius * y, center[2] + radius * z]
        viewup = [0.0, 0.0, 1.0]
        if abs(z) > 0.95:
            viewup = [0.0, 1.0, 0.0]
        return pos, center, viewup

    def _screenshot(plotter, path: str, scale_value: int) -> None:
        scale_value = max(1, int(scale_value))
        sig = inspect.signature(plotter.screenshot)
        kwargs = {"return_img": False}
        if "image_scale" in sig.parameters:
            kwargs["image_scale"] = scale_value
        elif "scale" in sig.parameters:
            kwargs["scale"] = scale_value
        elif window_size is not None and scale_value > 1:
            kwargs["window_size"] = (window_size[0] * scale_value, window_size[1] * scale_value)
        plotter.screenshot(path, **kwargs)

    if camera_views:
        plotter.show(auto_close=False)
        center, radius = _compute_camera_position()
        for az_deg, el_deg, path in camera_views:
            pos, focus, viewup = _camera_from_angles(center, radius, az_deg, el_deg)
            plotter.camera_position = (pos, focus, viewup)
            plotter.render()
            _screenshot(plotter, path, image_scale)
        plotter.close()
    elif screenshot_path is not None:
        plotter.show(auto_close=False)
        _screenshot(plotter, screenshot_path, image_scale)
        plotter.close()
    else:
        plotter.show()
