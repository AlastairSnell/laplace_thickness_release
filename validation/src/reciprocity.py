# src/reciprocity.py
"""
Two‑way path reciprocity experiment.
"""
import argparse, json, numpy as np, pickle, pathlib as _P
from mesh_utils import load_freesurfer_surf, poisson_sample
from metrics    import euclidean_disp, asymmetry, hausdorff, path_length
from tracing_utils import path_trace_simple

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--patch_dir", required=True, help="folder of *.pkl patch files")
parser.add_argument("--seeds", type=int, default=500)
parser.add_argument("--out",   required=True)
args = parser.parse_args()

np.random.seed(42)

patch_dir = _P.Path(args.patch_dir)
out_dir   = _P.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
def load_patch(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)          # assume dict{'triangles': …}
    return data["triangles"]

# ----------------------------------------------------------------------
summary_rows = []

for pkl in patch_dir.glob("*.pkl"):
    tri = load_patch(pkl)

    # TODO: pre‑solve Laplace and store Φ, ∇Φ if your tracer expects it
    # preprocess_triangles(tri)  # ← your function

    # -------------------------------- seeds
    outer_seed_point = tri[0]['vertices'][0]   # dummy; replace if needed
    seeds = poisson_sample(
        verts=np.vstack([t['vertices'] for t in tri]),
        faces=np.array([[0, 1, 2]]),          # bogus – not used by poisson
        n_points=args.seeds, min_spacing=1.0
    )

    # -------------------------------- run each seed
    disp_list, asym_list, haus_list = [], [], []

    for s in seeds:
        fwd_path, len_fwd = path_trace_simple(s, tri, direction='down')
        rev_path, len_rev = path_trace_simple(fwd_path[-1], tri, direction='up')

        disp_list.append(euclidean_disp(fwd_path[0], rev_path[-1]))
        asym_list.append(asymmetry(len_fwd, len_rev))
        haus_list.append(hausdorff(np.vstack(fwd_path), np.vstack(rev_path)))

    # -------------------------------- save CSV‑like
    np.savez(out_dir / f"{pkl.stem}.npz",
             disp=disp_list, asym=asym_list, haus=haus_list)

    summary_rows.append({
        "patch": pkl.stem,
        "pass_rate": (np.array(disp_list) < 0.05).mean()
    })

# ----------------------------------------------------------------------
with open(out_dir / "summary.json", "w") as f:
    json.dump(summary_rows, f, indent=2)
print(f"Done. Results → {out_dir}")
