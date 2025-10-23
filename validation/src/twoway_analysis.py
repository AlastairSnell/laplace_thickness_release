#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# -------- CONFIG --------
RESULTS_DIR = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\phantom_final2_results")

# List of (csv filename, ground truth thickness in mm)


FILES = [
    (f"{t}x10_{i}_paths.csv", float(t))
    for t in ["0.25", "0.5", "1", "2", "3", "4"]
    for i in range(1, 11)
]

'''
FILES = [
    (f"5mm_{i}_paths.csv", float(i))
    for i in ["0.25", "0.5", "1", "2", "3", "4"]
]
'''
'''
FILES = [
    (f"gyrus15_{i}_paths.csv", float(0))
     for i in range(1,10)
]
'''


OUT_CSV = RESULTS_DIR / "two_way_phantoms.csv"
# Thresholds
GEO_THR_MM  = 0.05
HAUS_THR_MM = 0.05
ASYM_THR    = 0.01
EPS_DEN     = 1e-9
# =========================
def save_per_truth_table_png(all_df: pd.DataFrame,
                             truth_levels: list[float],
                             out_path: Path,
                             dpi: int = 300) -> None:
    """Render the per-truth summaries as a PNG table (without p95 cols)."""
    rows = []
    for t in truth_levels:
        sub = all_df[np.isclose(all_df["truth_mm"], t)]
        if sub.empty:
            continue
        rows.append({
            "Truth (mm)": f"{t:g}",
            f"geo<{GEO_THR_MM:.2f}mm (%)": f"{100*sub['ok_geo'].mean():.1f}",
            f"haus<{HAUS_THR_MM:.2f}mm (%)": f"{100*sub['ok_hd'].mean():.1f}",
            "asym<1% (%)": f"{100*sub['ok_asym'].mean():.1f}",
            "ALL pass (%)": f"{100*sub['ok_all'].mean():.1f}",
            "n paths": f"{len(sub)}",
        })

    if not rows:
        print("[WARN] No per-truth rows to render.")
        return

    tbl_df = pd.DataFrame(rows)

    # Figure size scales with number of rows
    nrows, ncols = tbl_df.shape
    fig_w = 8
    fig_h = 0.6 + 0.4 * (nrows + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=tbl_df.values,
        colLabels=tbl_df.columns.tolist(),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    # Bold header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved per-truth table PNG to: {out_path}")



def safe_read_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV if it exists; return None if missing or unreadable."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"[WARN] Missing file, skipping: {path.name}")
        return None
    except Exception as e:
        print(f"[WARN] Failed to read {path.name}: {e}")
        return None
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf with NaN, drop rows missing required columns."""
    req = ["len_down", "len_up", "geodesic_disp", "hausdorff"]
    df = df.replace([np.inf, -np.inf], np.nan)
    # if any of the required columns are missing, create them so dropna works deterministically
    for c in req:
        if c not in df.columns:
            df[c] = np.nan
    df = df.dropna(subset=req)
    return df


def add_derived_and_gates(df: pd.DataFrame, truth: float) -> pd.DataFrame:
    """Add derived metrics and pass/fail gates."""
    # Asymmetry ratio with safe denominator
    denom = np.clip(df["len_down"].to_numpy(dtype=float), EPS_DEN, None)
    df["len_asym_ratio"] = np.abs(df["len_up"].to_numpy(dtype=float) - df["len_down"].to_numpy(dtype=float)) / denom

    # Pass/fail gates
    df["ok_geo"]  = df["geodesic_disp"].to_numpy(dtype=float) < GEO_THR_MM
    df["ok_hd"]   = df["hausdorff"].to_numpy(dtype=float)     < HAUS_THR_MM
    df["ok_asym"] = df["len_asym_ratio"]                      < ASYM_THR
    df["ok_all"]  = df["ok_geo"] & df["ok_hd"] & df["ok_asym"]

    # Truth for grouping
    df["truth_mm"] = float(truth)
    return df


def p95(series: pd.Series) -> float:
    """95th percentile ignoring NaNs; returns NaN if empty."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(np.percentile(s.to_numpy(dtype=float), 95))


def analyse_files(files: list[tuple[str, float]]) -> None:
    all_summaries: list[dict] = []
    all_rows = []

    # ---- Per-file loop ----
    for fname, truth in files:
        path = RESULTS_DIR / fname
        print(f"[INFO] Reading: {path}")
        df = safe_read_csv(path)
        if df is None:
            continue

        df = clean_df(df)
        if df.empty:
            print(f"{fname:25s} n=  0 | (no valid rows after cleanup)")
            continue

        df = add_derived_and_gates(df, truth)
        all_rows.append(df)

        # Per-file stats
        n = len(df)
        pct_geo   = 100 * df["ok_geo"].mean()
        pct_hd    = 100 * df["ok_hd"].mean()
        pct_asym  = 100 * df["ok_asym"].mean()
        pct_all   = 100 * df["ok_all"].mean()
        p95_haus  = p95(df["hausdorff"])
        p95_asymP = 100 * p95(df["len_asym_ratio"])

        print(
            f"{fname:25s} n={n:3d} | "
            f"geo<{GEO_THR_MM:.2f}mm: {pct_geo:5.1f}% | "
            f"haus<{HAUS_THR_MM:.2f}mm: {pct_hd:5.1f}% | "
            f"asym<1%: {pct_asym:5.1f}% | "
            f"ALL: {pct_all:5.1f}% | "
            f"p95 HD: {p95_haus:5.3f} mm | p95 asym: {p95_asymP:5.2f}%"
        )

        all_summaries.append({
            "level": "file",
            "file": fname,
            "truth_mm": truth,
            "n_paths": n,
            "pct_geo_lt_0.05mm": pct_geo,
            "pct_haus_lt_0.05mm": pct_hd,
            "pct_asym_lt_1pct": pct_asym,
            "pct_all_three": pct_all,
            "p95_haus_mm": p95_haus,
            "p95_asym_pct": p95_asymP,
        })

    if not all_rows:
        print("No data loaded.")
        return

    all_df = pd.concat(all_rows, ignore_index=True)
    n_all = len(all_df)

    # ---- Per-truth summaries ----
    print("\n=== Per-truth summaries ===")
    truth_levels = [t for t in dict.fromkeys(truth for _, truth in files)]  # preserve order
    # ---- Save per-truth table as PNG ----
    png_path = RESULTS_DIR / "per_truth_summaries.png"
    save_per_truth_table_png(all_df, truth_levels, png_path)

    for t in truth_levels:
        sub = all_df[np.isclose(all_df["truth_mm"], t)]
        if sub.empty:
            continue
        n = len(sub)
        pct_geo   = 100 * sub["ok_geo"].mean()
        pct_hd    = 100 * sub["ok_hd"].mean()
        pct_asym  = 100 * sub["ok_asym"].mean()
        pct_all   = 100 * sub["ok_all"].mean()
        p95_haus  = p95(sub["hausdorff"])
        p95_asymP = 100 * p95(sub["len_asym_ratio"])

        print(
            f"truth={t:>4} mm  n={n:3d} | "
            f"geo<{GEO_THR_MM:.2f}mm: {pct_geo:5.1f}% | "
            f"haus<{HAUS_THR_MM:.2f}mm: {pct_hd:5.1f}% | "
            f"asym<1%: {pct_asym:5.1f}% | "
            f"ALL: {pct_all:5.1f}% | "
            f"p95 HD: {p95_haus:5.3f} mm | p95 asym: {p95_asymP:5.2f}%"
        )

        all_summaries.append({
            "level": "truth",
            "file": f"TRUTH_{t}",
            "truth_mm": t,
            "n_paths": n,
            "pct_geo_lt_0.05mm": pct_geo,
            "pct_haus_lt_0.05mm": pct_hd,
            "pct_asym_lt_1pct": pct_asym,
            "pct_all_three": pct_all,
            "p95_haus_mm": p95_haus,
            "p95_asym_pct": p95_asymP,
        })

    # ---- Overall pooled ----
    pooled = {
        "level": "overall",
        "file": "ALL",
        "truth_mm": np.nan,
        "n_paths": n_all,
        "pct_geo_lt_0.05mm": 100 * all_df["ok_geo"].mean(),
        "pct_haus_lt_0.05mm": 100 * all_df["ok_hd"].mean(),
        "pct_asym_lt_1pct": 100 * all_df["ok_asym"].mean(),
        "pct_all_three": 100 * all_df["ok_all"].mean(),
        "p95_haus_mm": p95(all_df["hausdorff"]),
        "p95_asym_pct": 100 * p95(all_df["len_asym_ratio"]),
    }
    print("\n=== Overall pooled results ===")
    print(f"Total paths: {n_all}")
    print(f"Geodesic displacement <{GEO_THR_MM:.2f} mm: {pooled['pct_geo_lt_0.05mm']:.1f}%")
    print(f"Hausdorff distance   <{HAUS_THR_MM:.2f} mm: {pooled['pct_haus_lt_0.05mm']:.1f}%")
    print(f"Path-length asymmetry   <1%       : {pooled['pct_asym_lt_1pct']:.1f}%")
    print(f"ALL three gates pass              : {pooled['pct_all_three']:.1f}%")
    print(f"p95 Hausdorff: {pooled['p95_haus_mm']:.3f} mm")
    print(f"p95 asymmetry: {pooled['p95_asym_pct']:.2f}%")

    all_summaries.append(pooled)

    # ---- Save ----
    pd.DataFrame(all_summaries).to_csv(OUT_CSV, index=False)
    print(f"\nSaved summary CSV to {OUT_CSV}")


if __name__ == "__main__":
    analyse_files(FILES)