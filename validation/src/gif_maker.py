#!/usr/bin/env python3
from pathlib import Path
import pickle, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image  # pip install pillow

# --------- INPUTS (edit these) ----------
PKL = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\folds\zipped_patch_017.pkl")
OUT = PKL.with_suffix(".gif")
FPS = 24
N_FRAMES = 180       # 180 frames @24 fps ≈ 7.5 s
ELEV = 20            # camera elevation
AZIM0, AZIM1 = 0, 360
ALPHA = 0.85
EDGE_W = 0.15
# ----------------------------------------

def load_pack(p):
    with open(p, "rb") as f:
        pack = pickle.load(f)
    if isinstance(pack, dict) and "triangles" in pack:
        return pack["triangles"], pack.get("meta", {})
    return pack, {}

def split_bc(tris):
    top, bottom, side = [], [], []
    for t in tris:
        bc_type  = t.get('bc_type', '')
        bc_value = t.get('bc_value', None)
        verts = np.asarray(t['vertices'], float)
        if bc_type == 'dirichlet' and np.isclose(bc_value, 1.0):   top.append(verts)
        elif bc_type == 'dirichlet' and np.isclose(bc_value, 0.0): bottom.append(verts)
        elif bc_type == 'neumann':                                 side.append(verts)
    return top, bottom, side

def set_axes_equal(ax):
    import numpy as np
    lim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], float)
    c, span = lim.mean(axis=1), (lim[:,1]-lim[:,0]).max()
    r = 0.5*span if span>0 else 1.0
    ax.set_xlim3d(c[0]-r, c[0]+r); ax.set_ylim3d(c[1]-r, c[1]+r); ax.set_zlim3d(c[2]-r, c[2]+r)

def add_poly(ax, polys, color, label):
    if not polys: return None
    coll = Poly3DCollection(polys, alpha=ALPHA, facecolor=color,
                            edgecolor='k' if EDGE_W>0 else 'none', linewidths=EDGE_W)
    coll.set_label(label); ax.add_collection3d(coll); return coll

def main():
    tris, meta = load_pack(PKL)
    top, bottom, side = split_bc(tris)

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d'); ax.set_box_aspect([1,1,1])
    add_poly(ax, top,    (1.0, 0.6, 0.2),   "Top (pial)")
    add_poly(ax, bottom, (0.2, 0.8, 1.0),   "Bottom (white)")
    add_poly(ax, side,   (0.75,0.75,0.75),  "Side wall")

    # Fit axes
    pts = np.vstack([np.reshape(np.array(lst), (-1,3)) for lst in (top, bottom, side) if lst] or [np.zeros((0,3))])
    if pts.size:
        mins, maxs = pts.min(axis=0), pts.max(axis=0)
        pad = 0.05 * np.linalg.norm(maxs - mins)
        ax.set_xlim(mins[0]-pad, maxs[0]+pad)
        ax.set_ylim(mins[1]-pad, maxs[1]+pad)
        ax.set_zlim(mins[2]-pad, maxs[2]+pad)
        set_axes_equal(ax)

    ax.set_axis_off()  # cleaner GIF; remove if you want axes
    fig.tight_layout()

    frames = []
    H, W = fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0]
    for i in range(N_FRAMES):
        az = AZIM0 + (AZIM1 - AZIM0) * (i / (N_FRAMES - 1))
        ax.view_init(elev=ELEV, azim=az)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = buf.reshape(H, W, 3)
        frames.append(Image.fromarray(img))

    frames[0].save(OUT, save_all=True, append_images=frames[1:],
                   duration=int(1000/FPS), loop=0, disposal=2)
    print(f"Saved: {OUT}")
    plt.close(fig)

if __name__ == "__main__":
    main()
