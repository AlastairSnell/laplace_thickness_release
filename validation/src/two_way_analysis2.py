#!/usr/bin/env python3
import re, sys, argparse, datetime
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------- CONFIG DEFAULTS ----------------
DEFAULT_RESULTS_DIR = r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\phantom_final2_results"
DEFAULT_GLOB        = "*_paths.csv"     # change if needed
GEO_THR_MM          = 0.05
HAUS_THR_MM         = 0.05   # << tightened from 0.07
ASYM_THR            = 0.01
EPS_DEN             = 1e-9
# -------------------------------------------------

REQ_COLS = ["len_down", "len_up", "geodesic_disp", "hausdorff"]

def p95(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return float("nan")
    return float(np.percentile(s.to_numpy(dtype=float), 95))

def parse_truth_from_name(name: str) -> float:
    # Try to parse thickness value from file name
    m = re.match(r"^([0-9.]+)x\d+_\d+_paths\.csv$", name)
    if m: return float(m.group(1))
    m = re.match(r"^\d+mm_([0-9.]+)_paths\.csv$", name)
    if m: return float(m.group(1))
    m = re.match(r"^truth([0-9.]+).*_paths\.csv$", name)
    if m: return float(m.group(1))
    m = re.search(r"([0-9]*\.?[0-9]+)", name)
    if m: return float(m.group(1))
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
        if c not in df.columns: df[c] = np.nan
    return df.dropna(subset=REQ_COLS)

def add_metrics(df: pd.DataFrame, truth: float) -> pd.DataFrame:
    denom = np.clip(df["len_down"].to_numpy(dtype=float), EPS_DEN, None)
    asym = np.abs(df["len_up"].to_numpy(dtype=float) - df["len_down"].to_numpy(dtype=float)) / denom
    df = df.copy()
    df["len_asym_ratio"] = asym
    df["ok_geo"]  = df["geodesic_disp"].to_numpy(dtype=float) < GEO_THR_MM
    df["ok_hd"]   = df["hausdorff"].to_numpy(dtype=float)     < HAUS_THR_MM
    df["ok_asym"] = df["len_asym_ratio"]                      < ASYM_THR
    df["ok_all"]  = df["ok_geo"] & df["ok_hd"] & df["ok_asym"]
    df["truth_mm"] = float(truth)
    return df

def discover_files(results_dir: Path, pattern: str) -> list[tuple[Path, float]]:
    paths = sorted(results_dir.glob(pattern))
    if not paths:
        print(f"[FATAL] No files matched pattern '{pattern}' in {results_dir.resolve()}")
        sys.exit(1)
    return [(p, parse_truth_from_name(p.name)) for p in paths]

def per_file_summary(df: pd.DataFrame, fname: str) -> dict:
    n = len(df)
    pct_geo, pct_hd = 100*df["ok_geo"].mean(), 100*df["ok_hd"].mean()
    pct_asym, pct_all = 100*df["ok_asym"].mean(), 100*df["ok_all"].mean()
    p95_haus, p95_asymP = p95(df["hausdorff"]), 100*p95(df["len_asym_ratio"])
    print(f"{fname:30s} n={n:4d} | "
          f"geo<{GEO_THR_MM:.2f}mm: {pct_geo:5.1f}% | "
          f"haus<{HAUS_THR_MM:.2f}mm: {pct_hd:5.1f}% | "
          f"asym<1%: {pct_asym:5.1f}% | ALL: {pct_all:5.1f}% | "
          f"p95 HD: {p95_haus:6.3f} mm | p95 asym: {p95_asymP:5.2f}%")
    return {
        "level": "file", "file": fname, "truth_mm": float(df["truth_mm"].iloc[0]),
        "n_paths": n, "pct_geo_lt_0.05mm": pct_geo, "pct_haus_lt_0.05mm": pct_hd,
        "pct_asym_lt_1pct": pct_asym, "pct_all_three": pct_all,
        "p95_haus_mm": p95_haus, "p95_asym_pct": p95_asymP,
    }

def overall_block(all_df: pd.DataFrame) -> dict:
    n_all = len(all_df)
    pooled = {
        "level": "overall", "file": "ALL", "truth_mm": np.nan, "n_paths": n_all,
        "pct_geo_lt_0.05mm": 100*all_df["ok_geo"].mean(),
        "pct_haus_lt_0.05mm": 100*all_df["ok_hd"].mean(),
        "pct_asym_lt_1pct": 100*all_df["ok_asym"].mean(),
        "pct_all_three": 100*all_df["ok_all"].mean(),
        "p95_haus_mm": p95(all_df["hausdorff"]),
        "p95_asym_pct": 100*p95(all_df["len_asym_ratio"]),
    }
    print("\n=== Overall pooled results ===")
    print(f"Total paths: {n_all}")
    print(f"Geodesic displacement <{GEO_THR_MM:.2f} mm: {pooled['pct_geo_lt_0.05mm']:.1f}%")
    print(f"Hausdorff distance   <{HAUS_THR_MM:.2f} mm: {pooled['pct_haus_lt_0.05mm']:.1f}%")
    print(f"Path-length asymmetry   <1%       : {pooled['pct_asym_lt_1pct']:.1f}%")
    print(f"ALL three gates pass              : {pooled['pct_all_three']:.1f}%")
    print(f"p95 Hausdorff: {pooled['p95_haus_mm']:.3f} mm")
    print(f"p95 asymmetry: {pooled['p95_asym_pct']:.2f}%")
    return pooled

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", default=DEFAULT_RESULTS_DIR, help="Results directory")
    ap.add_argument("-g", "--glob", default=DEFAULT_GLOB, help="Glob pattern for input files")
    ap.add_argument("-o", "--out", default=None, help="Optional output CSV path")
    args = ap.parse_args()

    results_dir = Path(args.dir)
    items = discover_files(results_dir, args.glob)
    print(f"[INFO] RESULTS_DIR: {results_dir.resolve()}")
    print(f"[INFO] Found {len(items)} file(s).")

    summaries, all_rows = [], []
    for path, truth in items:
        df = safe_read_csv(path)
        if df is None: continue
        df = clean_df(df)
        if df.empty:
            print(f"{path.name:30s} n=   0 | (no valid rows)")
            continue
        df = add_metrics(df, truth)
        summaries.append(per_file_summary(df, path.name))
        all_rows.append(df)

    if not all_rows:
        print("[FATAL] No valid rows found.")
        sys.exit(1)

    all_df = pd.concat(all_rows, ignore_index=True)
    summaries.append(overall_block(all_df))

    if args.out:
        out_path = Path(args.out)
        pd.DataFrame(summaries).to_csv(out_path, index=False)
        print(f"\n[INFO] Saved summary CSV to {out_path.resolve()}")

if __name__ == "__main__":
    main()
