#!/usr/bin/env python3
import os, gc, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import CheckButtons, Button
import pyvista as pv

# --------- CONFIG ---------
REPO_ROOT = Path(__file__).resolve().parents[2]
PHANTOMS_DIR = REPO_ROOT / "validation/data/patient_example"
ALPHA = 0.85
EDGE_W = 0.15
# --------------------------

def load_pack(pkl_path: Path):
    mesh = pv.read(str(pkl_path))
    mesh = mesh.extract_surface().triangulate()

    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    points = np.asarray(mesh.points, dtype=float)

    cell_data = mesh.cell_data
    part_id = cell_data.get("part_id", None)
    bc_type = cell_data.get("bc_type", None)
    bc_value = cell_data.get("bc_value", None)

    top, bottom, side = [], [], []

    for ci, f in enumerate(faces):
        tri = points[f]

        if part_id is not None:
            pid = int(part_id[ci])
            if pid == 1:
                top.append(tri)
            elif pid == 2:
                bottom.append(tri)
            elif pid == 3:
                side.append(tri)
            continue

        if bc_type is None or bc_value is None:
            continue

        bt = bc_type[ci]
        bv = float(bc_value[ci])
        if isinstance(bt, (bytes, str)):
            bt_s = str(bt).lower()
            if bt_s == "dirichlet" and np.isclose(bv, 1.0):
                top.append(tri)
            elif bt_s == "dirichlet" and np.isclose(bv, 0.0):
                bottom.append(tri)
            elif bt_s == "neumann":
                side.append(tri)
        else:
            bt_i = int(bt)
            if bt_i == 0 and np.isclose(bv, 1.0):
                top.append(tri)
            elif bt_i == 0 and np.isclose(bv, 0.0):
                bottom.append(tri)
            elif bt_i == 1:
                side.append(tri)

    return top, bottom, side

def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    center = limits.mean(axis=1); span = (limits[:,1] - limits[:,0]).max()
    r = 0.5 * span if span > 0 else 1.0
    ax.set_xlim3d(center[0]-r, center[0]+r)
    ax.set_ylim3d(center[1]-r, center[1]+r)
    ax.set_zlim3d(center[2]-r, center[2]+r)

def all_points_from_poly_lists(*poly_lists):
    if not any(poly_lists):  # all empty
        return np.zeros((0,3))
    arrays = []
    for lst in poly_lists:
        if lst:
            arrays.append(np.reshape(np.array(lst), (-1,3)))
    return np.vstack(arrays) if arrays else np.zeros((0,3))

# --- NEW: normals helpers -----------------------------------------------------
def _tri_centroids_normals(polys):
    """Return (C,N) for triangle list: centroids and unit normals (right-hand rule)."""
    if not polys:
        return np.zeros((0,3)), np.zeros((0,3))
    P = np.asarray(polys)            # (T,3,3)
    v0, v1, v2 = P[:,0], P[:,1], P[:,2]
    n = np.cross(v1 - v0, v2 - v0)   # right-hand normal
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    n = n / ln
    c = (v0 + v1 + v2) / 3.0
    return c, n

def _suggest_arrow_len(*poly_lists):
    """Median edge length across provided triangles (scaled) for nice arrow size."""
    all_tris = [p for lst in poly_lists for p in (lst or [])]
    if not all_tris:
        return 1.0
    P = np.asarray(all_tris)  # (T,3,3)
    e = np.concatenate([
        np.linalg.norm(P[:,0]-P[:,1], axis=1),
        np.linalg.norm(P[:,1]-P[:,2], axis=1),
        np.linalg.norm(P[:,2]-P[:,0], axis=1),
    ])
    med = np.median(e)
    return float(0.6 * med if np.isfinite(med) and med > 0 else 1.0)
# -----------------------------------------------------------------------------

class PhantomViewer:
    def __init__(self, folder: Path):
        self.folder = folder
        self.files = sorted(p for p in folder.glob("*.vtk"))
        if not self.files:
            print(f"No .vtk files found in {folder}")
            sys.exit(1)
        self.i = 0

        self.fig = plt.figure(figsize=(9, 7))
        self.ax  = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1,1,1])

        # UI: checkboxes for surfaces
        rax = self.fig.add_axes([0.02, 0.50, 0.16, 0.19])  # taller to fit "Normals"
        self.check = CheckButtons(
            rax,
            ['Pial','White','Side wall','Normals'],  # NEW: Normals
            [True, True, True, False]
        )
        self.check.on_clicked(self._on_toggle)

        # UI: buttons
        bax_prev = self.fig.add_axes([0.02, 0.35, 0.07, 0.05])
        bax_next = self.fig.add_axes([0.11, 0.35, 0.07, 0.05])
        self.btn_prev = Button(bax_prev, 'Prev')
        self.btn_next = Button(bax_next, 'Next')
        self.btn_prev.on_clicked(lambda event: self.show(self.i - 1))
        self.btn_next.on_clicked(lambda event: self.show(self.i + 1))

        # State handles
        self.poly_top = self.poly_bot = self.poly_side = None
        self.seed_scatter = self.title_text = None

        # NEW: normals state/handles
        self.quiv_top = self.quiv_bot = self.quiv_side = None
        self.show_normals = False      # checkbox state
        self._arrow_len = 1.0

        # Keys
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Initial draw
        self.show(self.i)

    def _clear_plot(self):
        self.ax.cla()
        self.ax.set_box_aspect([1,1,1])
        self.ax.grid(False)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.pane.fill = False

    def _add_poly(self, polys, facecolor, label):
        if not polys:
            return None
        coll = Poly3DCollection(
            polys, alpha=ALPHA, facecolor=facecolor,
            edgecolor='k' if EDGE_W>0 else 'none', linewidths=EDGE_W
        )
        coll.set_label(label)
        self.ax.add_collection3d(coll)
        return coll

    # --- NEW: helpers to (re)draw normals ------------------------------------
    def _add_normals(self, polys):
        """Return a quiver artist for given triangle list, or None if empty."""
        C, N = _tri_centroids_normals(polys)
        if not len(C):
            return None
        # length uses suggested scale; normalize=True uses unit vectors
        return self.ax.quiver(
            C[:,0], C[:,1], C[:,2],
            N[:,0], N[:,1], N[:,2],
            length=self._arrow_len, normalize=True, linewidths=0.6
        )

    def _toggle_quiver(self, q, vis):
        if q is not None:
            q.set_visible(vis)
    # -------------------------------------------------------------------------

    def show(self, idx: int):
        self.i = idx % len(self.files)
        pkl_path = self.files[self.i]

        top, bottom, side = load_pack(pkl_path)

        self._clear_plot()

        self.poly_top  = self._add_poly(top,    facecolor=(1.0, 0.6, 0.2),   label='Pial')
        self.poly_bot  = self._add_poly(bottom, facecolor=(0.2, 0.8, 1.0),   label='White')
        self.poly_side = self._add_poly(side,   facecolor=(0.75, 0.75, 0.75),label='Side wall')

        # NEW: compute arrow length based on current mesh(es) and (re)draw quivers
        self._arrow_len = _suggest_arrow_len(top, bottom, side)
        self.quiv_top  = self._add_normals(top)
        self.quiv_bot  = self._add_normals(bottom)
        self.quiv_side = self._add_normals(side)
        # Respect current checkbox state for Normals
        self._toggle_quiver(self.quiv_top,  self.show_normals and (self.poly_top  is None or self.poly_top.get_visible()))
        self._toggle_quiver(self.quiv_bot,  self.show_normals and (self.poly_bot  is None or self.poly_bot.get_visible()))
        self._toggle_quiver(self.quiv_side, self.show_normals and (self.poly_side is None or self.poly_side.get_visible()))

        self.seed_scatter = None

        # Fit axes
        pts = all_points_from_poly_lists(top, bottom, side)
        if pts.size:
            mins = pts.min(axis=0); maxs = pts.max(axis=0)
            pad = 0.05 * np.linalg.norm(maxs - mins)
            self.ax.set_xlim(mins[0]-pad, maxs[0]+pad)
            self.ax.set_ylim(mins[1]-pad, maxs[1]+pad)
            self.ax.set_zlim(mins[2]-pad, maxs[2]+pad)
            set_axes_equal(self.ax)

        # Title
        n_top, n_bot, n_side = len(top), len(bottom), len(side)
        tbits = []
        tbits.append(f"tris: top={n_top}, bottom={n_bot}, side={n_side}")
        self.ax.set_title(f"{pkl_path.name}  [{self.i+1}/{len(self.files)}]\n" + "  ".join(tbits))

        self.ax.legend(loc='upper right', fontsize=9)
        self.fig.canvas.draw_idle()
        gc.collect()

    def _on_toggle(self, label):
        if 'Top' in label and self.poly_top is not None:
            vis = not self.poly_top.get_visible()
            self.poly_top.set_visible(vis)
            # NEW: keep normals in sync with surface visibility (if normals enabled)
            self._toggle_quiver(self.quiv_top, self.show_normals and vis)
        elif 'Bottom' in label and self.poly_bot is not None:
            vis = not self.poly_bot.get_visible()
            self.poly_bot.set_visible(vis)
            self._toggle_quiver(self.quiv_bot, self.show_normals and vis)
        elif 'Side' in label and self.poly_side is not None:
            vis = not self.poly_side.get_visible()
            self.poly_side.set_visible(vis)
            self._toggle_quiver(self.quiv_side, self.show_normals and vis)
        elif 'Normals' in label:
            # NEW: master toggle
            self.show_normals = not self.show_normals
            self._toggle_quiver(self.quiv_top,  self.show_normals and (self.poly_top  is None or self.poly_top.get_visible()))
            self._toggle_quiver(self.quiv_bot,  self.show_normals and (self.poly_bot  is None or self.poly_bot.get_visible()))
            self._toggle_quiver(self.quiv_side, self.show_normals and (self.poly_side is None or self.poly_side.get_visible()))
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in ('right', ' '):
            self.show(self.i + 1)
        elif event.key in ('left',):
            self.show(self.i - 1)
        elif event.key in ('t','T'):
            if self.poly_top is not None:
                self.poly_top.set_visible(not self.poly_top.get_visible())
                self._toggle_quiver(self.quiv_top, self.show_normals and self.poly_top.get_visible())  # NEW
                self.fig.canvas.draw_idle()
        elif event.key in ('b','B'):
            if self.poly_bot is not None:
                self.poly_bot.set_visible(not self.poly_bot.get_visible())
                self._toggle_quiver(self.quiv_bot, self.show_normals and self.poly_bot.get_visible())  # NEW
                self.fig.canvas.draw_idle()
        elif event.key in ('s','S'):
            if self.poly_side is not None:
                self.poly_side.set_visible(not self.poly_side.get_visible())
                self._toggle_quiver(self.quiv_side, self.show_normals and self.poly_side.get_visible())  # NEW
                self.fig.canvas.draw_idle()
        elif event.key in ('n','N'):  # NEW: keyboard toggle for normals
            self.show_normals = not self.show_normals
            self._toggle_quiver(self.quiv_top,  self.show_normals and (self.poly_top  is None or self.poly_top.get_visible()))
            self._toggle_quiver(self.quiv_bot,  self.show_normals and (self.poly_bot  is None or self.poly_bot.get_visible()))
            self._toggle_quiver(self.quiv_side, self.show_normals and (self.poly_side is None or self.poly_side.get_visible()))
            self.fig.canvas.draw_idle()
        elif event.key in ('q','Q'):
            plt.close(self.fig)

def main():
    viewer = PhantomViewer(PHANTOMS_DIR)
    print("Controls:\n"
          "  Right arrow / Space  : next phantom\n"
          "  Left arrow           : previous phantom\n"
          "  t / b / s            : toggle Top / Bottom / Side\n"
          "  n                    : toggle Normals\n"
          "  Checkboxes           : same toggles (incl. Normals)\n"
          "  q                    : quit\n")
    plt.show()

if __name__ == "__main__":
    main()
