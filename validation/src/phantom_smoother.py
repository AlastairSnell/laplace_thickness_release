#!/usr/bin/env python3
# phantom_smoother_iso_taubin.py
"""
Batch-improve tessellation quality for cortical ribbon phantoms by:
  1) PyMeshLab isotropic remeshing (uniformize edges, preserve boundary)
  2) PyVista Taubin smoothing (gentle, morphology-friendly)
Preserves per-face BCs via nearest original-face mapping.
Writes outputs to phantoms3 with suffix _iso_taubin.pkl
"""

import pickle
from pathlib import Path
import numpy as np

# ---- deps ----
# pip install pymeshlab pyvista scipy
import pymeshlab as pml
import pyvista as pv
from scipy.spatial import cKDTree

# ----------------- config -----------------
IN_DIR  = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\phantoms2")
OUT_DIR = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\data\phantoms3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# PyMeshLab isotropic remeshing params (light)
TARGET_EDGE_MM   = 0.5    # ≈ desired edge length; try ~median original edge
ISO_ITERATIONS   = 4      # small number = gentle
ISO_ADAPTIVITY   = 0.0    # 0 = uniform edges
ISO_PRESERVE_BND = True
ISO_CREASE_ANGLE = 120.0  # high = less feature preservation pressure

# Taubin smoothing (gentle)
TAUBIN_ITERS     = 6
TAUBIN_PASSBAND  = 0.10   # 0.05–0.15 gentle band

# Fidelity guard
HAUSDORFF_P95_MAX_MM = 0.25   # allow small drift; tighten if you like

# Backoff ladder (if drift too high we retry lighter ops)
# Each step is a dict of overrides applied before retry.
BACKOFF_STEPS = [
    {"TAUBIN_ITERS": 4, "TAUBIN_PASSBAND": 0.12},          # gentler smoothing
    {"ISO_ITERATIONS": 3},                                  # gentler iso
    {"TARGET_EDGE_MM": 0.6},                                # slightly coarser target
    {"TAUBIN_ITERS": 2, "TAUBIN_PASSBAND": 0.15},          # very gentle smoothing
]


# ----------------- helpers -----------------
def tri_dicts_to_numpy(tris):
    """Deduplicate vertices; return V (N,3) and F (M,3)."""
    vmap, verts, faces = {}, [], []
    for t in tris:
        Vt = np.asarray(t['vertices'], dtype=np.float64)
        idx = []
        for v in Vt:
            key = tuple(np.round(v, 7))
            if key not in vmap:
                vmap[key] = len(verts)
                verts.append(v)
            idx.append(vmap[key])
        faces.append(idx)
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int32)
    return V, F

def numpy_to_pv(V, F):
    cells = np.hstack([np.c_[np.full(len(F), 3), F]]).ravel()
    return pv.PolyData(V, cells).triangulate().clean()

def pv_to_numpy(mesh):
    F = mesh.faces.reshape((-1, 4))[:, 1:].astype(np.int32)
    V = mesh.points.astype(np.float64, copy=False)
    return V, F

def face_normals(V, F):
    v0, v1, v2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    N = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    s = np.linalg.norm(N, axis=1, keepdims=True); s[s==0] = 1.0
    return (N / s).astype(np.float32)

def hausdorff_p95(A: np.ndarray, B: np.ndarray) -> float:
    ta = cKDTree(A); tb = cKDTree(B)
    da, _ = ta.query(B, k=1)
    db, _ = tb.query(A, k=1)
    return float(np.percentile(np.r_[da, db], 95))

def assign_bc_from_original(tris_src, V_new, F_new):
    cents_src = np.array([np.asarray(t['vertices'], float).mean(0) for t in tris_src])
    bcs_src   = np.array([(t['bc_type'], float(t['bc_value'])) for t in tris_src], dtype=object)
    kdt = cKDTree(cents_src)

    face_cents = V_new[F_new].mean(axis=1)
    _, idx = kdt.query(face_cents, k=1)
    bc_per_face = bcs_src[idx]

    N_new = face_normals(V_new, F_new)
    tris_out = []
    for (i, j, k), n, (bc_type, bc_val) in zip(F_new, N_new, bc_per_face):
        tris_out.append({
            "vertices": V_new[[i, j, k]].astype(np.float32),
            "normal": n,
            "bc_type": str(bc_type),
            "bc_value": float(bc_val),
        })
    return tris_out

def isotropic_remesh(V, F, target_edge, iterations, adaptivity, preserve_bnd, crease_angle):
    ms = pml.MeshSet()
    m = pml.Mesh(vertex_matrix=V, face_matrix=F)
    ms.add_mesh(m, "phantom")
    ms.apply_filter(
        "meshing_isotropic_explicit_remeshing",
        targetlen=target_edge,
        iterations=int(iterations),
        adaptivity=float(adaptivity),
        creaseangle=float(crease_angle),
        selectedonly=False,
        preserveboundary=bool(preserve_bnd),
    )
    m2 = ms.current_mesh()
    V2 = m2.vertex_matrix()
    F2 = m2.face_matrix().astype(np.int32)
    return V2, F2

def taubin_smooth_pv(V, F, n_iter, pass_band):
    mesh = numpy_to_pv(V, F)
    mesh = mesh.smooth_taubin(
        n_iter=int(n_iter),
        pass_band=float(pass_band),
        boundary_smoothing=False,
        feature_edge_smoothing=False,
        non_manifold_smoothing=False,
        normalize_coordinates=True,
    ).clean()
    return pv_to_numpy(mesh)


# ----------------- main work -----------------
def improve_one(pkl_path: Path):
    pack = pickle.loads(pkl_path.read_bytes())
    tris_src = pack["triangles"]
    V0, F0 = tri_dicts_to_numpy(tris_src)

    # — baseline params —
    params = dict(
        TARGET_EDGE_MM=TARGET_EDGE_MM,
        ISO_ITERATIONS=ISO_ITERATIONS,
        ISO_ADAPTIVITY=ISO_ADAPTIVITY,
        ISO_PRESERVE_BND=ISO_PRESERVE_BND,
        ISO_CREASE_ANGLE=ISO_CREASE_ANGLE,
        TAUBIN_ITERS=TAUBIN_ITERS,
        TAUBIN_PASSBAND=TAUBIN_PASSBAND,
    )

    # try baseline + backoff ladder
    attempts = [dict()] + BACKOFF_STEPS
    for step in attempts:
        local = params.copy()
        local.update(step)

        try:
            # 1) isotropic remesh (light)
            V1, F1 = isotropic_remesh(
                V0, F0,
                target_edge=local["TARGET_EDGE_MM"],
                iterations=local["ISO_ITERATIONS"],
                adaptivity=local["ISO_ADAPTIVITY"],
                preserve_bnd=local["ISO_PRESERVE_BND"],
                crease_angle=local["ISO_CREASE_ANGLE"],
            )

            # 2) gentle Taubin smoothing
            V2, F2 = taubin_smooth_pv(
                V1, F1,
                n_iter=local["TAUBIN_ITERS"],
                pass_band=local["TAUBIN_PASSBAND"],
            )

            # 3) drift check vs ORIGINAL (V0, not V1)
            p95 = hausdorff_p95(V0, V2)
            if p95 <= HAUSDORFF_P95_MAX_MM:
                tris_out = assign_bc_from_original(tris_src, V2, F2)
                out_pack = {"triangles": tris_out, "meta": dict(pack.get("meta", {}))}
                out_path = OUT_DIR / pkl_path.name.replace(".pkl", "_iso_taubin.pkl")
                out_path.write_bytes(pickle.dumps(out_pack, protocol=pickle.HIGHEST_PROTOCOL))
                print(f"[OK] {pkl_path.name} → {out_path.name} | faces {len(F0)}→{len(F2)} | p95={p95:.4f} mm")
                return
            else:
                print(f"[TRY] {pkl_path.name}: drift {p95:.4f} mm > {HAUSDORFF_P95_MAX_MM} mm; backoff…")

        except Exception as e:
            print(f"[ERR] {pkl_path.name}: {e}; backoff…")

    print(f"[SKIP] {pkl_path.name}: all backoff levels exceeded drift limit.")

def main():
    pkls = sorted(IN_DIR.glob("*.pkl"))
    if not pkls:
        print(f"No .pkl files found in {IN_DIR}")
        return
    for p in pkls:
        improve_one(p)

if __name__ == "__main__":
    main()
