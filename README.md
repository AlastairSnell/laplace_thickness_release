# Laplace Thickness

Command-line tooling for cortical thickness on surface patches using a BEM Laplace solve and gradient-flow tracing.

Main entrypoint:

```bash
python -m laplace_thickness_release.main.main
```

All commands below assume the current working directory is the repository root (the folder containing `pyproject.toml`).

---

## Installation

Core package:

```bash
python -m pip install -e .
```

Validation/data tooling dependencies (required for figure and table reproduction):

```bash
python -m pip install -e ".[validation]"
```

---

## Reproducing Paper Figures and Tables

After installing with `[validation]` extras, run:

```bash
python reproduce_figures.py
```

Figures from reproduce_figures.py are written to `validation/figures/`:

| File | Description |
|---|---|
| `fig3_heatmap_view01–04.png` | Figure 3A–D — thickness heatmap, four camera angles |
| `fig4_bland_altman.png` | Figure 4 — Bland-Altman validation (disks, hemispheres, phantoms) |
| `fig5A_shape_mean.png` | Figure 5A — % valid paths vs mean shape quality |
| `fig5B_shape_cv.png` | Figure 5B — % valid paths vs shape quality CV |

Tables are written to `validation/tables/`:

| File | Description |
|---|---|
| `table1_disks.csv` | Table 1 — two-way reciprocity, cylindrical patches |
| `table2_hemispheres.csv` | Table 2 — two-way reciprocity, hemispherical annulus patches |
| `table3_synthetic.csv` | Table 3 — two-way reciprocity, synthetic patches (6 thickness groups) |
| `table4_real.csv` | Table 4 — two-way reciprocity, real cortical patches |

Pre-computed path CSVs in `validation/results/` are used by default. To recompute them from the raw meshes before plotting (takes 10–30 min), pass `--recompute`:

```bash
python reproduce_figures.py --recompute
```

---

## Input Requirement for `main.py`

`main.py` expects a prebuilt patch mesh (`.vtk` or `.vtp`) with required cell arrays:
- `bc_type`
- `bc_value`
- `normal`

Raw FreeSurfer `pial` and `white` surfaces are not consumed directly by `main.py`; convert them first with `validation/scripts/patch_maker.py`.

---

## Heatmap Example

Run heatmap generation on an included example patch:

```bash
python -m laplace_thickness_release.main.main \
  --mesh validation\data\patient_example\zipped_patch.vtk \
  --heatmap \
  --parallel
```

---

## Patch Generation from FreeSurfer

Create one zipped patch (`.vtk`) from FreeSurfer surfaces.

Region mode (`aparc` labels):

```bash
python validation/scripts/patch_maker.py \
  --fs-surf-dir path/to/subject/surf \
  --aparc-dir path/to/subject/label \
  --region-name supramarginal \
  --hemi lh \
  --out-dir validation/data/new_folds
```

Surface-RAS mode:

```bash
python validation/scripts/patch_maker.py \
  --fs-surf-dir path/to/subject/surf \
  --surface-ras "[12.3,-45.6,78.9]" \
  --out-dir validation/data/new_folds
```

Notes:
- Do not combine `--region-name` and `--surface-ras`.
- `--hemi` is required when using `--region-name`.
- `--surface-ras` must be the "Surface RAS" coordinate from Freeview.

---

## License

This repository is released under the MIT License (`LICENSE`).
Use, modification, redistribution, and commercial use are permitted.

