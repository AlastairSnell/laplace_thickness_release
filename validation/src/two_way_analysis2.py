#!/usr/bin/env python3
import re
import sys
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG DEFAULTS ----------------
DEFAULT_RESULTS_DIR = r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\phantom_results"
DEFAULT_GLOB = "*_paths.csv"
GEO_THR_MM = 0.05
ASYM_THR_1 = 0.01  # 1% asymmetry gate
ASYM_THR_5 = 0.05  # 5% asymmetry summary gate
EPS_DEN = 1e-9
# -------------------------------------------------

REQ_COLS = ["len_down", "len_up", "geodesic_disp"]


def p95(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(np.percentile(s.to_numpy(dtype=float), 95))


def parse_truth_from_name(name: str) -> float:
    # Try to parse thickness value from file name
    m = re.match(r"^([0-9.]+)x\d+_\d+_paths\.csv$", name)
    if m:
        return float(m.group(1))

    m = re.match(r"^\d+mm_([0-9.]+)_paths\.csv$", name)
    if m:
        return float(m.group(1))

    m = re.match(r"^truth([0-9.]+).*_paths\.csv$", name)
    if m:
        return float(m.group(1))

    m = re.search(r"([0-9]*\.?[0-9]+)", name)
    if m:
        return float(m.group(1))

    return float("nan")


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read {path.name}: {e}")
        return None


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in REQ_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df.dropna(subset=REQ_COLS)


def add_metrics(df: pd.DataFrame, truth: float) -> pd.DataFrame:
    denom = np.clip(df["len_down"].to_numpy(dtype=float), EPS_DEN, None)
    asym = (
        np.abs(
            df["len_up"].to_numpy(dtype=float) - df["len_down"].to_numpy(dtype=float)
        )
        / denom
    )

    df = df.copy()
    df["len_asym_ratio"] = asym

    # Gates
    df["ok_geo"] = df["geodesic_disp"].to_numpy(dtype=float) < GEO_THR_MM
    df["ok_asym1"] = df["len_asym_ratio"] < ASYM_THR_1  # <1%
    df["ok_asym5"] = df["len_asym_ratio"] < ASYM_THR_5  # <5%
    df["ok_all"] = df["ok_geo"] & df["ok_asym1"]
    df["truth_mm"] = float(truth)
    return df


def discover_files(results_dir: Path, pattern: str) -> list[tuple[Path, float]]:
    all_paths = sorted(results_dir.glob(pattern))
    paths: list[Path] = []

    for p in all_paths:
        # Skip master/aggregate file entirely
        if p.name.startswith("_MASTER"):
            print(f"[INFO] Skipping aggregate file {p.name}")
            continue
        paths.append(p)

    if not paths:
        print(
            f"[FATAL] No files matched pattern '{pattern}' in {results_dir.resolve()}"
        )
        sys.exit(1)

    return [(p, parse_truth_from_name(p.name)) for p in paths]


def per_file_summary(df: pd.DataFrame, fname: str) -> dict:
    n = len(df)
    pct_geo = 100 * df["ok_geo"].mean()
    pct_asym1 = 100 * df["ok_asym1"].mean()
    pct_asym5 = 100 * df["ok_asym5"].mean()
    pct_all = 100 * df["ok_all"].mean()
    p95_asymP = 100 * p95(df["len_asym_ratio"])

    print(
        f"{fname:30s} n={n:4d} | "
        f"geo<{GEO_THR_MM:.2f}mm: {pct_geo:5.1f}% | "
        f"asym<1%: {pct_asym1:5.1f}% | "
        f"asym<5%: {pct_asym5:5.1f}% | "
        f"ALL gates: {pct_all:5.1f}% | "
        f"p95 asym: {p95_asymP:5.2f}%"
    )

    return {
        "level": "file",
        "file": fname,
        "truth_mm": float(df["truth_mm"].iloc[0]),
        "n_paths": n,
        "pct_geo_lt_0.05mm": pct_geo,
        "pct_asym_lt_1pct": pct_asym1,
        "pct_asym_lt_5pct": pct_asym5,
        "pct_all_gates": pct_all,
        "p95_asym_pct": p95_asymP,
    }


def overall_block(all_df: pd.DataFrame) -> dict:
    n_all = len(all_df)
    pooled = {
        "level": "overall",
        "file": "ALL",
        "truth_mm": np.nan,
        "n_paths": n_all,
        "pct_geo_lt_0.05mm": 100 * all_df["ok_geo"].mean(),
        "pct_asym_lt_1pct": 100 * all_df["ok_asym1"].mean(),
        "pct_asym_lt_5pct": 100 * all_df["ok_asym5"].mean(),
        "pct_all_gates": 100 * all_df["ok_all"].mean(),
        "p95_asym_pct": 100 * p95(all_df["len_asym_ratio"]),
    }

    print("\n=== Overall pooled results ===")
    print(f"Total paths: {n_all}")
    print(
        f"Geodesic displacement <{GEO_THR_MM:.2f} mm: "
        f"{pooled['pct_geo_lt_0.05mm']:.1f}%"
    )
    print(
        f"Path-length asymmetry <1% : "
        f"{pooled['pct_asym_lt_1pct']:.1f}%"
    )
    print(
        f"Path-length asymmetry <5% : "
        f"{pooled['pct_asym_lt_5pct']:.1f}%"
    )
    print(
        f"ALL gates pass : "
        f"{pooled['pct_all_gates']:.1f}%"
    )
    print(f"p95 asymmetry: {pooled['p95_asym_pct']:.2f}%")

    return pooled


def per_truth_summary(df: pd.DataFrame, truth: float) -> dict:
    """
    Summarise all paths for a given truth thickness.
    One row per unique truth_mm.
    """
    n = len(df)
    pct_geo = 100 * df["ok_geo"].mean()
    pct_asym1 = 100 * df["ok_asym1"].mean()
    pct_asym5 = 100 * df["ok_asym5"].mean()
    pct_all = 100 * df["ok_all"].mean()
    p95_asymP = 100 * p95(df["len_asym_ratio"])

    return {
        "level": "file",  # treat like a "data row" in the table
        "file": f"truth_{truth}",  # label; not too important
        "truth_mm": float(truth),
        "n_paths": n,
        "pct_geo_lt_0.05mm": pct_geo,
        "pct_asym_lt_1pct": pct_asym1,
        "pct_asym_lt_5pct": pct_asym5,
        "pct_all_gates": pct_all,
        "p95_asym_pct": p95_asymP,
    }


def save_table_image(
    summary_df: pd.DataFrame,
    out_dir: Path,
    label_prefix: str,
    show_thickness: bool = True,
) -> Path:
    """
    Render the summary table as a high-DPI PNG in out_dir.
    Uses human-friendly labels (e.g. 'Patch 1', 'Patch 2', 'All data').
    If show_thickness is True, includes a 'Thickness (mm)' column.
    """
    df = summary_df.copy()

    # Build display labels: Patch 1, Patch 2, ..., All data
    labels: list[str] = []
    idx = 1
    for _, row in df.iterrows():
        if row["level"] == "file":
            labels.append(f"{label_prefix} {idx}")
            idx += 1
        else:
            labels.append("All data")

    df["Data"] = labels

    # Columns to show in the figure
    display_cols = ["Data"]
    if show_thickness:
        display_cols.append("truth_mm")

    display_cols += [
        "n_paths",
        "pct_geo_lt_0.05mm",
        "pct_asym_lt_1pct",
        "pct_asym_lt_5pct",
        "p95_asym_pct",
    ]

    rename_map = {
        "Data": "Data",
        "n_paths": "Paths (n)",
        "pct_geo_lt_0.05mm": f"Geo < {GEO_THR_MM:.2f} mm (%)",
        "pct_asym_lt_1pct": "Asym < 1% (%)",
        "pct_asym_lt_5pct": "Asym < 5% (%)",
        "p95_asym_pct": "p95 asym (%)",
    }

    if show_thickness:
        rename_map["truth_mm"] = "Thickness (mm)"

    display_df = df.loc[:, display_cols].rename(columns=rename_map)

    # Build cell text with light formatting
    cell_text: list[list[str]] = []
    for _, row in display_df.iterrows():
        formatted_row: list[str] = []
        for col in display_df.columns:
            val = row[col]
            if isinstance(val, (int, float, np.integer, np.floating)):
                if pd.isna(val):
                    formatted_row.append("—")
                else:
                    # Integers as integers, others to 2dp
                    if float(val).is_integer():
                        formatted_row.append(f"{int(val)}")
                    else:
                        formatted_row.append(f"{float(val):.2f}")
            else:
                formatted_row.append(str(val))
        cell_text.append(formatted_row)

    col_labels = list(display_df.columns)

    # Figure sizing heuristic: scale with row/column count
    n_rows = len(display_df)
    n_cols = len(col_labels)
    fig_width = max(9.0, 1.8 * n_cols)
    fig_height = max(2.5, 0.45 * (n_rows + 1))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    # Styling: header + zebra striping + subtle borders
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:  # Header row
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(weight="bold")
            cell.set_edgecolor("#555555")
            cell.set_linewidth(0.6)
        else:
            # Zebra striping for readability
            if row_idx % 2 == 0:
                cell.set_facecolor("#ffffff")
            else:
                cell.set_facecolor("#f9f9f9")
            cell.set_edgecolor("#cccccc")
            cell.set_linewidth(0.4)

    plt.tight_layout(pad=0.5)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"path_validation_summary_{timestamp}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved summary table image to {out_path.resolve()}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d",
        "--dir",
        default=DEFAULT_RESULTS_DIR,
        help="Results directory",
    )
    ap.add_argument(
        "-g",
        "--glob",
        default=DEFAULT_GLOB,
        help="Glob pattern for input files",
    )
    ap.add_argument(
        "-o",
        "--out",
        default=None,
        help="Optional output CSV path",
    )
    ap.add_argument(
        "--label",
        default="Group",
        help="Base label for data rows in the figure (e.g. 'Patch', 'Chunk')",
    )
    ap.add_argument(
        "--show-thickness",
        action="store_true",
        help="Include a 'Thickness (mm)' column in the table image",
    )

    args = ap.parse_args()

    results_dir = Path(args.dir)
    items = discover_files(results_dir, args.glob)

    print(f"[INFO] RESULTS_DIR: {results_dir.resolve()}")
    print(f"[INFO] Found {len(items)} file(s).")

    all_rows: list[pd.DataFrame] = []

    for path, truth in items:
        df = safe_read_csv(path)
        if df is None:
            continue

        df = clean_df(df)
        if df.empty:
            print(f"{path.name:30s} n= 0 | (no valid rows)")
            continue

        df = add_metrics(df, truth)

        # Keep the per-file printout for debugging/logging:
        per_file_summary(df, path.name)
        all_rows.append(df)

    if not all_rows:
        print("[FATAL] No valid rows found.")
        sys.exit(1)

    all_df = pd.concat(all_rows, ignore_index=True)

    # --- NEW: one summary row per unique truth thickness ---
    summaries_by_truth: list[dict] = []
    for truth, df_t in sorted(all_df.groupby("truth_mm"), key=lambda x: x[0]):
        summaries_by_truth.append(per_truth_summary(df_t, truth))

    # Add pooled "ALL" row at the end
    summaries_by_truth.append(overall_block(all_df))

    # Collect summaries into a DataFrame
    summary_df = pd.DataFrame(summaries_by_truth)

    # Optional CSV output
    if args.out:
        out_path = Path(args.out)
        summary_df.to_csv(out_path, index=False)
        print(f"\n[INFO] Saved summary CSV to {out_path.resolve()}")

    # Always save a publication-style table image in the results directory
    save_table_image(
        summary_df,
        results_dir,
        label_prefix=args.label,
        show_thickness=args.show_thickness,
    )


if __name__ == "__main__":
    main()
