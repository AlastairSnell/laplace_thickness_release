#!/usr/bin/env python3
"""Reproduce all paper figures and tables.

Run from the repository root:
    python reproduce_figures.py

Figures are written to validation/figures/:
    fig3_heatmap_view01.png  -- Figure 3A  (heatmap, camera angle 1)
    fig3_heatmap_view02.png  -- Figure 3B  (heatmap, camera angle 2)
    fig3_heatmap_view03.png  -- Figure 3C  (heatmap, camera angle 3)
    fig3_heatmap_view04.png  -- Figure 3D  (heatmap, camera angle 4)
    fig4_bland_altman.png    -- Figure 4   (Bland-Altman; disks, hemispheres, phantoms)
    fig5A_shape_mean.png     -- Figure 5A  (% valid paths vs mean shape quality)
    fig5B_shape_cv.png       -- Figure 5B  (% valid paths vs shape quality CV)

Tables are written to validation/tables/:
    table1_disks.csv         -- Table 1   (cylindrical patches)
    table2_hemispheres.csv   -- Table 2   (hemispherical annulus patches)
    table3_synthetic.csv     -- Table 3   (synthetic/phantom patches, 6 thickness groups)
    table4_real.csv          -- Table 4   (real cortical patches)

Prerequisites
-------------
Install the package and its validation extras before running:
    pip install -e ".[validation]"

Figure 3 uses PyVista for off-screen rendering.  Four views are saved (one per
azimuth in FIG3_AZIMUTHS at the elevation in FIG3_ELEVATIONS).  Adjust the
angles and/or FIG3_HEATMAP_SPACING to match the final paper figures.

Figures 4 and 5 read from pre-computed two-way path CSVs already committed to
the repository.  To recompute those CSVs from scratch before generating plots,
pass --recompute to this script (this can take tens of minutes).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
FIGURES = ROOT / "validation" / "figures"
TABLES = ROOT / "validation" / "tables"
SCRIPTS = ROOT / "validation" / "scripts"
DATA = ROOT / "validation" / "data"
RESULTS = ROOT / "validation" / "results"

# ── Camera settings for Figure 3 ─────────────────────────────────────────────
# Four azimuths at one elevation → four output files: _view01 … _view04.png.
# Adjust to match the exact orientations used in the paper.
FIG3_AZIMUTHS = "0,90,180,270"   # degrees; front, right, back, left
FIG3_ELEVATIONS = "30"            # degrees above horizontal
FIG3_HEATMAP_SPACING = "0.1"     # mm; geodesic spacing between heatmap sample points


def _run(*args: str) -> None:
    cmd = [sys.executable, *args]
    label = " ".join(str(a) for a in cmd)
    print(f"\n>>> {label}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"\n[FAIL] Exit code {result.returncode}: {label}")


def _recompute_analytical() -> None:
    """Re-run two-way testing on disks, hemispheres, and phantoms."""
    datasets = [
        ("analytical/disks", RESULTS / "disks_results"),
        ("analytical/hemispheres", RESULTS / "hemispheres_results"),
        ("phantoms_final", RESULTS / "phantom_results"),
    ]
    for subfolder, outdir in datasets:
        outdir.mkdir(parents=True, exist_ok=True)
        _run(
            str(SCRIPTS / "twoway_testing.py"),
            "--data-root", str(DATA),
            "--subfolders", subfolder,
            "--pattern", "*.vtk",
            "--outdir", str(outdir),
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--recompute",
        action="store_true",
        default=False,
        help="Re-run two-way testing before generating Figure 4 (slow; ~10–30 min).",
    )
    args = p.parse_args(argv)

    FIGURES.mkdir(parents=True, exist_ok=True)

    # ── Figure 3: thickness heatmap ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Figure 3: Thickness heatmap")
    print("=" * 60)
    _run(
        "-m", "laplace_thickness_release.main.main",
        "--mesh", str(DATA / "patient_example" / "zipped_patch.vtk"),
        "--heatmap",
        "--parallel",
        "--start-surface", "white",          # trace white → pial (one-way)
        "--heatmap-spacing", FIG3_HEATMAP_SPACING,
        "--heatmap-screenshot", str(FIGURES / "fig3_heatmap.png"),
        "--heatmap-azimuths", FIG3_AZIMUTHS,
        "--heatmap-elevations", FIG3_ELEVATIONS,
    )

    # ── Figure 4: Bland-Altman ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Figure 4: Bland-Altman validation")
    print("=" * 60)
    if args.recompute:
        print("Recomputing two-way path CSVs ...")
        _recompute_analytical()

    _run(
        str(SCRIPTS / "bland_altman.py"),
        "--disks-dir", str(RESULTS / "disks_results"),
        "--hemispheres-dir", str(RESULTS / "hemispheres_results"),
        "--phantoms-dir", str(RESULTS / "phantom_results"),
        "--out", str(FIGURES / "fig4_bland_altman.png"),
    )

    # ── Figure 5: robustness to mesh perturbation ────────────────────────────
    print("\n" + "=" * 60)
    print("Figure 5: Robustness to mesh noise")
    print("=" * 60)
    _run(
        str(SCRIPTS / "perturbation_analysis.py"),
        "--results-dir", str(RESULTS / "perturbations_results"),
        "--out-a", str(FIGURES / "fig5A_shape_mean.png"),
        "--out-b", str(FIGURES / "fig5B_shape_cv.png"),
    )

    # ── Tables 1–4: two-way path reciprocity ────────────────────────────────
    print("\n" + "=" * 60)
    print("Tables 1–4: Two-way path reciprocity")
    print("=" * 60)
    TABLES.mkdir(parents=True, exist_ok=True)
    _run(
        str(SCRIPTS / "generate_tables.py"),
        "--out-dir", str(TABLES),
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Done. Figures written to {FIGURES}:")
    for f in sorted(FIGURES.iterdir()):
        print(f"  {f.name}")
    print(f"\nTables written to {TABLES}:")
    for f in sorted(TABLES.iterdir()):
        print(f"  {f.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
