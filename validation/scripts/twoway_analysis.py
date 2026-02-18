#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PathRow:
    file: str
    truth_mm: float
    path_idx: int
    len_down: float
    len_up: float
    len_asym_abs: float
    len_ratio_up_over_down: float
    geodesic_disp: float


@dataclass
class Thresholds:
    geo_thr_mm: float
    asym_thr: float


def _safe_float(value: str) -> float:
    v = value.strip().lower()
    if v in {"", "nan"}:
        return float("nan")
    if v in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if v in {"-inf", "-infinity"}:
        return float("-inf")
    return float(v)


def _parse_truth_from_filename(name: str) -> float | None:
    patterns = [
        r"^(?P<truth>\d+(?:\.\d+)?)x10_",
        r"^15mm_(?P<truth>\d+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return float(m.group("truth"))
    return None


def _read_paths_csv(path: Path, truth_mm: float) -> list[PathRow]:
    rows: list[PathRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "path_idx",
            "len_down",
            "len_up",
            "len_asym_abs",
            "len_ratio_up_over_down",
            "geodesic_disp",
        }
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path.name}: missing required columns: {sorted(missing)}")

        for rec in reader:
            row = PathRow(
                file=path.name,
                truth_mm=float(truth_mm),
                path_idx=int(rec["path_idx"]),
                len_down=_safe_float(rec["len_down"]),
                len_up=_safe_float(rec["len_up"]),
                len_asym_abs=_safe_float(rec["len_asym_abs"]),
                len_ratio_up_over_down=_safe_float(rec["len_ratio_up_over_down"]),
                geodesic_disp=_safe_float(rec["geodesic_disp"]),
            )
            if not np.isfinite(row.len_down):
                continue
            if not np.isfinite(row.len_up):
                continue
            if not np.isfinite(row.geodesic_disp):
                continue
            rows.append(row)

    return rows


def _pct(mask: np.ndarray) -> float:
    if mask.size == 0:
        return float("nan")
    return float(100.0 * np.mean(mask))


def _p95(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, 95))


def _summarise(rows: list[PathRow], th: Thresholds) -> dict[str, float]:
    len_down = np.asarray([r.len_down for r in rows], dtype=float)
    len_up = np.asarray([r.len_up for r in rows], dtype=float)
    geo = np.asarray([r.geodesic_disp for r in rows], dtype=float)

    denom = np.clip(len_down, 1e-9, None)
    asym_ratio = np.abs(len_up - len_down) / denom

    ok_geo = geo < th.geo_thr_mm
    ok_asym = asym_ratio < th.asym_thr
    ok_all = ok_geo & ok_asym

    return {
        "n_paths": float(len(rows)),
        "pct_geo_lt_thr": _pct(ok_geo),
        "pct_asym_lt_thr": _pct(ok_asym),
        "pct_all": _pct(ok_all),
        "p95_geo_mm": _p95(geo),
        "p95_asym_pct": float(100.0 * _p95(asym_ratio)),
        "mean_len_down": float(np.mean(len_down)) if len_down.size else float("nan"),
        "mean_len_up": float(np.mean(len_up)) if len_up.size else float("nan"),
    }


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save_truth_table_png(
    out_path: Path,
    truth_levels: list[float],
    truth_summaries: dict[float, dict[str, float]],
    th: Thresholds,
    dpi: int,
) -> None:
    table_rows = []
    for t in truth_levels:
        s = truth_summaries[t]
        table_rows.append(
            [
                f"{t:g}",
                f"{s['pct_geo_lt_thr']:.1f}",
                f"{s['pct_asym_lt_thr']:.1f}",
                f"{s['pct_all']:.1f}",
                f"{int(round(s['n_paths']))}",
            ]
        )

    headers = [
        "Truth (mm)",
        f"geo<{th.geo_thr_mm:.2f}mm (%)",
        f"asym<{100*th.asym_thr:.1f}% (%)",
        "ALL pass (%)",
        "n paths",
    ]

    fig_h = 0.8 + 0.45 * (len(table_rows) + 1)
    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)

    for (r, _c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Summarise two-way path CSVs produced by twoway_testing.py "
            "(no Hausdorff metric required)."
        )
    )
    p.add_argument("--results-dir", type=Path, required=True, help="Directory containing *_paths.csv files.")
    p.add_argument("--pattern", type=str, default="*_paths.csv", help="Glob pattern for per-surface path CSV files.")
    p.add_argument("--out-csv", type=Path, default=None, help="Output summary CSV path. Default: <results-dir>/two_way_summary.csv")
    p.add_argument("--out-png", type=Path, default=None, help="Output PNG table path. Default: <results-dir>/two_way_summary_table.png")
    p.add_argument("--geo-thr-mm", type=float, default=0.05)
    p.add_argument("--asym-thr", type=float, default=0.01, help="Asymmetry ratio threshold (0.01 = 1%%).")
    p.add_argument("--dpi", type=int, default=300)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if not args.results_dir.exists():
        parser.error(f"Results directory does not exist: {args.results_dir}")

    out_csv = args.out_csv or (args.results_dir / "two_way_summary.csv")
    out_png = args.out_png or (args.results_dir / "two_way_summary_table.png")
    th = Thresholds(geo_thr_mm=float(args.geo_thr_mm), asym_thr=float(args.asym_thr))

    files = sorted(args.results_dir.glob(args.pattern))
    if not files:
        print(f"[WARN] No files matched pattern '{args.pattern}' in {args.results_dir}")
        return 0

    per_file_rows: list[dict[str, object]] = []
    pooled_rows: list[PathRow] = []

    for path in files:
        truth = _parse_truth_from_filename(path.name)
        if truth is None:
            print(f"[WARN] Skipping {path.name}: could not parse truth thickness from filename")
            continue

        try:
            rows = _read_paths_csv(path, truth)
        except Exception as exc:
            print(f"[WARN] Skipping {path.name}: {exc}")
            continue

        if not rows:
            print(f"[WARN] Skipping {path.name}: no valid rows")
            continue

        pooled_rows.extend(rows)
        s = _summarise(rows, th)
        per_file_rows.append(
            {
                "level": "file",
                "file": path.name,
                "truth_mm": truth,
                "n_paths": int(round(s["n_paths"])),
                "pct_geo_lt_thr": s["pct_geo_lt_thr"],
                "pct_asym_lt_thr": s["pct_asym_lt_thr"],
                "pct_all": s["pct_all"],
                "p95_geo_mm": s["p95_geo_mm"],
                "p95_asym_pct": s["p95_asym_pct"],
                "mean_len_down": s["mean_len_down"],
                "mean_len_up": s["mean_len_up"],
            }
        )

        print(
            f"{path.name:28s} n={int(round(s['n_paths'])):3d} | "
            f"geo<{th.geo_thr_mm:.2f}mm: {s['pct_geo_lt_thr']:5.1f}% | "
            f"asym<{100*th.asym_thr:.1f}%: {s['pct_asym_lt_thr']:5.1f}% | "
            f"ALL: {s['pct_all']:5.1f}%"
        )

    if not pooled_rows:
        print("[WARN] No valid rows found.")
        return 0

    truth_levels = sorted({r.truth_mm for r in pooled_rows})
    truth_summaries: dict[float, dict[str, float]] = {}
    for t in truth_levels:
        subset = [r for r in pooled_rows if np.isclose(r.truth_mm, t)]
        s = _summarise(subset, th)
        truth_summaries[t] = s
        per_file_rows.append(
            {
                "level": "truth",
                "file": f"TRUTH_{t:g}",
                "truth_mm": t,
                "n_paths": int(round(s["n_paths"])),
                "pct_geo_lt_thr": s["pct_geo_lt_thr"],
                "pct_asym_lt_thr": s["pct_asym_lt_thr"],
                "pct_all": s["pct_all"],
                "p95_geo_mm": s["p95_geo_mm"],
                "p95_asym_pct": s["p95_asym_pct"],
                "mean_len_down": s["mean_len_down"],
                "mean_len_up": s["mean_len_up"],
            }
        )

    overall = _summarise(pooled_rows, th)
    per_file_rows.append(
        {
            "level": "overall",
            "file": "ALL",
            "truth_mm": "",
            "n_paths": int(round(overall["n_paths"])),
            "pct_geo_lt_thr": overall["pct_geo_lt_thr"],
            "pct_asym_lt_thr": overall["pct_asym_lt_thr"],
            "pct_all": overall["pct_all"],
            "p95_geo_mm": overall["p95_geo_mm"],
            "p95_asym_pct": overall["p95_asym_pct"],
            "mean_len_down": overall["mean_len_down"],
            "mean_len_up": overall["mean_len_up"],
        }
    )

    _write_summary_csv(out_csv, per_file_rows)
    _save_truth_table_png(out_png, truth_levels, truth_summaries, th, int(args.dpi))

    print(f"\n[OK] Summary CSV: {out_csv}")
    print(f"[OK] Summary PNG: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
