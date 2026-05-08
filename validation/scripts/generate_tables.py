#!/usr/bin/env python3
"""Generate Tables 1–4: two-way path reciprocity results as CSV files.

Outputs written to validation/tables/:
    table1_disks.csv        -- Table 1  (cylindrical patches)
    table2_hemispheres.csv  -- Table 2  (hemispherical annulus patches)
    table3_synthetic.csv    -- Table 3  (synthetic/phantom patches, 6 thickness groups)
    table4_real.csv         -- Table 4  (real cortical patches)
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent.resolve()
RESULTS = ROOT / "validation" / "results"
TABLES = ROOT / "validation" / "tables"

GEO_THR = 0.05   # mm
ASYM1_THR = 0.01  # 1 %
ASYM5_THR = 0.05  # 5 %

STANDARD_FIELDS = [
    "Data",
    "Thickness (mm)",
    "Paths (n)",
    "Geo < 0.05 mm (%)",
    "Asym < 1% (%)",
    "Asym < 5% (%)",
    "p95 asym (%)",
]

REAL_FIELDS = [
    "Data",
    "Paths (n)",
    "Geo < 0.05 mm (%)",
    "Asym < 1% (%)",
    "Asym < 5% (%)",
    "p95 asym (%)",
    "shape_mean",
    "shape_cv",
]


def _safe_float(v: str) -> float:
    v = v.strip().lower()
    if v in {"", "nan", "inf", "+inf", "-inf"}:
        return float("nan")
    return float(v)


def _load_paths(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (len_down, len_up, geodesic_disp), dropping non-finite rows."""
    ld, lu, geo = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = _safe_float(row.get("len_down", "nan"))
            u = _safe_float(row.get("len_up", "nan"))
            g = _safe_float(row.get("geodesic_disp", "nan"))
            if np.isfinite(d) and np.isfinite(u) and np.isfinite(g):
                ld.append(d)
                lu.append(u)
                geo.append(g)
    return np.array(ld), np.array(lu), np.array(geo)


def _load_paths_with_shape(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Like _load_paths but also extracts shape_mean and shape_cv (constant per patch)."""
    ld, lu, geo = [], [], []
    shape_means: list[float] = []
    shape_cvs: list[float] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = _safe_float(row.get("len_down", "nan"))
            u = _safe_float(row.get("len_up", "nan"))
            g = _safe_float(row.get("geodesic_disp", "nan"))
            if np.isfinite(d) and np.isfinite(u) and np.isfinite(g):
                ld.append(d)
                lu.append(u)
                geo.append(g)
            sm = _safe_float(row.get("shape_mean", "nan"))
            sc = _safe_float(row.get("shape_cv", "nan"))
            if np.isfinite(sm):
                shape_means.append(sm)
            if np.isfinite(sc):
                shape_cvs.append(sc)
    shape_mean = float(np.mean(shape_means)) if shape_means else float("nan")
    shape_cv = float(np.mean(shape_cvs)) if shape_cvs else float("nan")
    return np.array(ld), np.array(lu), np.array(geo), shape_mean, shape_cv


def _metrics(
    ld: np.ndarray, lu: np.ndarray, geo: np.ndarray
) -> tuple[int, float, float, float, float]:
    """Compute (n, geo_pct, asym1_pct, asym5_pct, p95_asym_pct).

    Asymmetry = |len_up - len_down| / len_down, consistent with twoway_analysis.py.
    """
    n = len(ld)
    if n == 0:
        nan = float("nan")
        return 0, nan, nan, nan, nan
    asym = np.abs(lu - ld) / np.clip(ld, 1e-9, None)
    geo_pct = float(100.0 * np.mean(geo < GEO_THR))
    asym1_pct = float(100.0 * np.mean(asym < ASYM1_THR))
    asym5_pct = float(100.0 * np.mean(asym < ASYM5_THR))
    p95 = float(np.percentile(asym * 100.0, 95))
    return n, geo_pct, asym1_pct, asym5_pct, p95


def _fmt(val: float, decimals: int = 1) -> str:
    return f"{val:.{decimals}f}" if np.isfinite(val) else ""


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"  [OK] {path.name}")


def _standard_row(
    data: str, thickness: float, n: int,
    geo: float, a1: float, a5: float, p95: float,
) -> dict:
    return {
        "Data": data,
        "Thickness (mm)": f"{thickness:g}",
        "Paths (n)": n,
        "Geo < 0.05 mm (%)": _fmt(geo),
        "Asym < 1% (%)": _fmt(a1),
        "Asym < 5% (%)": _fmt(a5),
        "p95 asym (%)": _fmt(p95, 2),
    }


def table1_disks(results_dir: Path, out_dir: Path) -> None:
    """Table 1: cylindrical (disk) patches, one row per thickness level."""
    rows = []
    for csv_path in sorted(results_dir.glob("*_paths.csv")):
        m = re.match(r"15mm_(\d+(?:\.\d+)?)_paths", csv_path.stem)
        if not m:
            continue
        thickness = float(m.group(1))
        ld, lu, geo = _load_paths(csv_path)
        n, geo_pct, a1, a5, p95 = _metrics(ld, lu, geo)
        rows.append(_standard_row("Disk", thickness, n, geo_pct, a1, a5, p95))
    rows.sort(key=lambda r: float(r["Thickness (mm)"]))
    _write_csv(out_dir / "table1_disks.csv", STANDARD_FIELDS, rows)


def table2_hemispheres(results_dir: Path, out_dir: Path) -> None:
    """Table 2: hemispherical annulus patches, one row per thickness level."""
    rows = []
    for csv_path in sorted(results_dir.glob("*_paths.csv")):
        m = re.match(r"15mm_(\d+(?:\.\d+)?)_paths", csv_path.stem)
        if not m:
            continue
        thickness = float(m.group(1))
        ld, lu, geo = _load_paths(csv_path)
        n, geo_pct, a1, a5, p95 = _metrics(ld, lu, geo)
        rows.append(_standard_row("Hemisphere", thickness, n, geo_pct, a1, a5, p95))
    rows.sort(key=lambda r: float(r["Thickness (mm)"]))
    _write_csv(out_dir / "table2_hemispheres.csv", STANDARD_FIELDS, rows)


def table3_synthetic(results_dir: Path, out_dir: Path) -> None:
    """Table 3: synthetic patches pooled into 6 groups by ground-truth thickness."""
    by_thickness: dict[float, tuple[list, list, list]] = {}
    for csv_path in sorted(results_dir.glob("*_paths.csv")):
        m = re.match(r"(\d+(?:\.\d+)?)x10_", csv_path.name)
        if not m:
            continue
        t = float(m.group(1))
        ld, lu, geo = _load_paths(csv_path)
        if t not in by_thickness:
            by_thickness[t] = ([], [], [])
        by_thickness[t][0].extend(ld.tolist())
        by_thickness[t][1].extend(lu.tolist())
        by_thickness[t][2].extend(geo.tolist())

    rows = []
    for group_idx, (t, (ld_list, lu_list, geo_list)) in enumerate(
        sorted(by_thickness.items()), start=1
    ):
        ld = np.array(ld_list)
        lu = np.array(lu_list)
        geo = np.array(geo_list)
        n, geo_pct, a1, a5, p95 = _metrics(ld, lu, geo)
        rows.append(_standard_row(f"Group {group_idx}", t, n, geo_pct, a1, a5, p95))
    _write_csv(out_dir / "table3_synthetic.csv", STANDARD_FIELDS, rows)


def table4_real(results_dir: Path, perturb_dir: Path, out_dir: Path) -> None:
    """Table 4: real cortical patches.

    Shape quality (shape_mean, shape_cv) is read from the unperturbed
    (jitter_0p000) perturbation results, which carry those columns.
    """
    rows = []
    for patch_idx, csv_path in enumerate(
        sorted(results_dir.glob("zipped_patch_*_paths.csv")), start=1
    ):
        patch_stem = csv_path.stem.replace("_paths", "")
        ld, lu, geo = _load_paths(csv_path)
        n, geo_pct, a1, a5, p95 = _metrics(ld, lu, geo)

        shape_mean, shape_cv = float("nan"), float("nan")
        jitter_csv = perturb_dir / f"{patch_stem}_jitter_0p000_paths.csv"
        if jitter_csv.exists():
            _, _, _, shape_mean, shape_cv = _load_paths_with_shape(jitter_csv)
        else:
            print(f"  [WARN] Shape quality CSV not found: {jitter_csv.name}")

        rows.append({
            "Data": f"Patch {patch_idx}",
            "Paths (n)": n,
            "Geo < 0.05 mm (%)": _fmt(geo_pct),
            "Asym < 1% (%)": _fmt(a1),
            "Asym < 5% (%)": _fmt(a5),
            "p95 asym (%)": _fmt(p95, 2),
            "shape_mean": _fmt(shape_mean, 4),
            "shape_cv": _fmt(shape_cv, 4),
        })
    _write_csv(out_dir / "table4_real.csv", REAL_FIELDS, rows)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", type=Path, default=TABLES, help="Output directory for table CSVs.")
    p.add_argument("--disks-dir", type=Path, default=RESULTS / "disks_results")
    p.add_argument("--hemispheres-dir", type=Path, default=RESULTS / "hemispheres_results")
    p.add_argument("--synthetic-dir", type=Path, default=RESULTS / "phantom_results")
    p.add_argument("--folds-dir", type=Path, default=RESULTS / "folds_results")
    p.add_argument(
        "--perturb-dir", type=Path, default=RESULTS / "perturbations_results",
        help="Perturbation results dir (used to fetch shape quality for Table 4).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    print("Generating Table 1: Disks ...")
    table1_disks(args.disks_dir, args.out_dir)

    print("Generating Table 2: Hemispheres ...")
    table2_hemispheres(args.hemispheres_dir, args.out_dir)

    print("Generating Table 3: Synthetic (phantom) ...")
    table3_synthetic(args.synthetic_dir, args.out_dir)

    print("Generating Table 4: Real cortical ...")
    table4_real(args.folds_dir, args.perturb_dir, args.out_dir)

    print(f"\n[OK] Tables written to {args.out_dir}:")
    for f in sorted(args.out_dir.iterdir()):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
