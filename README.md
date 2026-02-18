## Laplace Thickness

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

Validation/data tooling dependencies:

```bash
python -m pip install -e ".[validation]"
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
  --mesh validation/data/folds_selected/zipped_patch_000.vtk \
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

## Two-Way Validation on `validation/data`

Run two-way testing over selected subfolders under `validation/data`:

```bash
python validation/scripts/twoway_testing.py \
  --data-root validation/data \
  --subfolders analytical/hemispheres analytical/shifted_hemis phantoms_final folds_selected perturbed \
  --pattern "*.vtk" \
  --outdir validation/tests/twoway_results_run01
```

Summarize generated per-surface CSVs:

```bash
python validation/scripts/twoway_analysis.py \
  --results-dir validation/tests/twoway_results_run01 \
  --pattern "*_paths.csv"
```

This writes:
- `two_way_summary.csv`
- `two_way_summary_table.png`

Operational note:
- `twoway_testing.py` appends to the master CSV in `--outdir`. Use a fresh output directory per run to avoid mixing runs.

---

## License

This repository is released under the MIT License (`LICENSE`).
Use, modification, redistribution, and commercial use are permitted.

