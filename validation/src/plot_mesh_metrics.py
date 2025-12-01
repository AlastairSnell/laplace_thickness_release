#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make 4 scatter plots:
  y-axis:  % of valid (non-degenerate) paths per (surface,deformation)
  x-axis:  shape_mean, shape_cv, gauss_mean, gauss_cv (one figure each)

Degenerate path definition:
  - length asymmetry > 1%   (relative, using mean length)
  - OR geodesic distance > 0.05 mm

Inputs:
  _MASTER_paths.csv produced by your perturbation runs

Outputs:
  PNGs in ...\validation\figures\
"""

from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------- CONFIG -----------
MASTER_CSV = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\perturbations_results\_MASTER_paths.csv")
FIG_DIR    = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\figures")

# thresholds
ASYM_PCT_THRESHOLD = 1.0      # percent
GEO_DISP_THRESHOLD = 0.05     # mm

# which metrics to plot on x-axis
X_METRICS = ["shape_mean", "shape_cv", "gauss_mean", "gauss_cv"]
# ------------------------------


def load_master(master_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(master_csv)
    # Ensure required columns exist
    required = {
        "folder","surface","deformation","sigma_factor",
        "len_down","len_up","geodesic_disp",
        "shape_mean","shape_cv","gauss_mean","gauss_cv",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MASTER CSV missing columns: {sorted(missing)}")
    return df


def compute_valid_mask(df: pd.DataFrame) -> pd.Series:
    """Return boolean Series: True = valid (non-degenerate), False = degenerate."""
    # finite rows only
    finite = np.isfinite(df["len_down"]) & np.isfinite(df["len_up"]) & np.isfinite(df["geodesic_disp"])
    # relative length asymmetry using mean length (symmetric % diff)
    mean_len = (df["len_down"] + df["len_up"]) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        asym_pct = np.abs(df["len_up"] - df["len_down"]) / mean_len * 100.0

    # degenerate if asymmetry > 1% OR geodesic_disp > 0.05 mm
    degenerate = (asym_pct > ASYM_PCT_THRESHOLD) | (df["geodesic_disp"] > GEO_DISP_THRESHOLD)

    # Only paths with finite metrics are considered; others treated as invalid/ignored
    valid = finite & (~degenerate)
    # if not finite, mark as False so they don't count as valid
    valid = valid & finite
    return valid


def aggregate_by_surface_def(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (folder, surface, deformation, sigma_factor) and compute:
      - n_paths_used
      - pct_valid
      - representative x-metrics (take first finite value in the group)
    """
    key = ["folder", "surface", "deformation", "sigma_factor"]
    df = df.copy()

    # Compute per-row validity
    df["_valid"] = compute_valid_mask(df)
    df["_finite"] = np.isfinite(df["len_down"]) & np.isfinite(df["len_up"]) & np.isfinite(df["geodesic_disp"])

    def _first_finite(s: pd.Series):
        s = s[np.isfinite(s)]
        return s.iloc[0] if len(s) else np.nan

    agg = df.groupby(key).apply(
        lambda g: pd.Series({
            "n_paths_used": int(g["_finite"].sum()),
            "n_valid":      int((g["_finite"] & g["_valid"]).sum()),
            **{m: _first_finite(g[m]) for m in X_METRICS},
        })
    ).reset_index()

    # % valid among usable (finite) paths
    with np.errstate(divide="ignore", invalid="ignore"):
        agg["pct_valid"] = np.where(agg["n_paths_used"] > 0,
                                    100.0 * agg["n_valid"] / agg["n_paths_used"],
                                    np.nan)
    # Drop groups with no usable paths
    agg = agg[np.isfinite(agg["pct_valid"])]
    return agg


def make_scatter(df: pd.DataFrame, xcol: str, outpath: Path):
    plt.figure(figsize=(6, 5))
    plt.scatter(df[xcol], df["pct_valid"], alpha=0.7)
    plt.xlabel(xcol)
    plt.ylabel("% valid (non-degenerate)")
    plt.title(f"% valid vs {xcol}")
    plt.grid(True, alpha=0.3)

    # (Optional) annotate by deformation for debugging
    # for _, r in df.iterrows():
    #     plt.annotate(str(r["deformation"]), (r[xcol], r["pct_valid"]), fontsize=7, alpha=0.6)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_master(MASTER_CSV)
    agg = aggregate_by_surface_def(df)

    # Save the aggregated table alongside the plots for reproducibility/debugging
    agg_csv = FIG_DIR / "validity_vs_mesh_metrics_agg.csv"
    agg.to_csv(agg_csv, index=False)

    for x in X_METRICS:
        out = FIG_DIR / f"valid_vs_{x}.png"
        make_scatter(agg, x, out)
        print(f"Saved {out}")

    print(f"\nAggregated data saved to: {agg_csv}")


if __name__ == "__main__":
    main()
