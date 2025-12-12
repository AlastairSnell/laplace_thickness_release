## Laplace Thickness

Command-line tool for computing cortical thickness on surface patches using a BEM Laplace solve + gradient-flow tracing.

Main entrypoint:

```bash
python -m laplace_thickness_release.main.main
````

All commands assume you are in the repo root with pyproject.toml

---

## Installation

Install the package in editable mode:

```bash
python -m pip install -e .
```
---

## Examples from the paper

The `validation/data/paper_examples/` folder contains the patches used in the manuscript figures.

### Technical example: single test patch

This is the basic “technical” example in the paper: default streamline tracing on a single cortical patch:

```bash
python -m laplace_thickness_release.main.main \
  --mesh validation/data/paper_examples/zipped_patch_000.vtk
---

## Streamline tracing

Trace pial to white surface and plot the the streamlines:

```bash
python -m laplace_thickness_release.main.main \
  --mesh validation/data/folds_selected/zipped_patch_004.vtk
```

Parallelised version (recommended):

```bash
python -m laplace_thickness_release.main.main \
  --mesh validation/data/folds_selected/zipped_patch_004.vtk \
  --parallel
```

Default number of workers is set to the number of available CPUs, but this can be changed:

```bash
python -m laplace_thickness_release.main.main \
  --mesh validation/data/folds_selected/zipped_patch_004.vtk \
  --parallel --workers 8
```

Spatial density of paths in tracing mode is controlled by:

```bash
--trace-spacing <float>
```

---

## Heatmap mode

Thickness heatmap on the pial surface:

```bash
python -m laplace_thickness_release.main.main \
  --mesh validation/data/folds_selected/zipped_patch_004.vtk \
  --heatmap \
  --parallel
```

Key points:

* Only down-tracing is used in heatmap mode.

* A central inner patch of the pial surface is kept; default:

  ```bash
  --heatmap-pct 50.0
  ```

* Sampling density for the heatmap is controlled by:

  ```bash
  --heatmap-spacing 1
  ```
where the default is maximal density.

If `--heatmap-pct` is set above `50`, results are less reliable.
