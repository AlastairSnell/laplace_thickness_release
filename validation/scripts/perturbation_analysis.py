#!/usr/bin/env python3
"""Generate Figure 5: % valid paths vs mesh shape quality.

Reads all *_paths.csv files from validation/results/perturbations_results/,
applies the same validity criterion as plot_mesh_metrics.py (asymmetry <= 1%
AND geodesic displacement <= 0.05 mm), aggregates per (surface, deformation),
and produces two scatter plots:

  Figure 5A: mean shape quality  vs % valid paths
  Figure 5B: shape quality CV    vs % valid paths

Points are coloured by sigma (jitter) level.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ASYM_PCT_THRESHOLD = 1.0   # percent
GEO_DISP_THRESHOLD = 0.05  # mm


def _load_csvs(results_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in sorted(results_dir.glob("*_paths.csv")):
        m = re.match(r"(zipped_patch_\d+)_(jitter_\w+)_paths", p.stem)
        if not m:
            continue
        df = pd.read_csv(p)
        df["surface"] = m.group(1)
        df["deformation"] = m.group(2)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _valid_mask(df: pd.DataFrame) -> pd.Series:
    finite = (
        np.isfinite(df["len_down"])
        & np.isfinite(df["len_up"])
        & np.isfinite(df["geodesic_disp"])
    )
    mean_len = (df["len_down"] + df["len_up"]) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        asym_pct = np.abs(df["len_up"] - df["len_down"]) / mean_len * 100.0
    degenerate = (asym_pct > ASYM_PCT_THRESHOLD) | (df["geodesic_disp"] > GEO_DISP_THRESHOLD)
    return finite & ~degenerate


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    finite = (
        np.isfinite(df["len_down"])
        & np.isfinite(df["len_up"])
        & np.isfinite(df["geodesic_disp"])
    )
    df["_finite"] = finite
    df["_valid"] = _valid_mask(df)

    def _first_finite(s: pd.Series) -> float:
        vals = s[np.isfinite(s)]
        return float(vals.iloc[0]) if len(vals) else np.nan

    agg = (
        df.groupby(["surface", "deformation"])
        .apply(
            lambda g: pd.Series({
                "n_finite": int(g["_finite"].sum()),
                "n_valid": int((g["_finite"] & g["_valid"]).sum()),
                "shape_mean": _first_finite(g["shape_mean"]),
                "shape_cv": _first_finite(g["shape_cv"]),
                "sigma_factor": float(g["sigma_factor"].iloc[0]),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        agg["pct_valid"] = np.where(
            agg["n_finite"] > 0,
            100.0 * agg["n_valid"] / agg["n_finite"],
            np.nan,
        )
    return agg[np.isfinite(agg["pct_valid"])].reset_index(drop=True)


def _scatter_panel(ax: plt.Axes, agg: pd.DataFrame, xcol: str, xlabel: str) -> None:
    sigmas = sorted(agg["sigma_factor"].unique())
    cmap = plt.get_cmap("viridis", len(sigmas))
    for i, sig in enumerate(sigmas):
        sub = agg[np.isclose(agg["sigma_factor"], sig)]
        ax.scatter(
            sub[xcol], sub["pct_valid"],
            color=cmap(i), alpha=0.85, s=45, edgecolors="none",
            label=f"σ = {sig:g}",
        )
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("% valid (non-degenerate) paths", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, title="Jitter (σ)", title_fontsize=9, framealpha=0.8)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate Figure 5: robustness of path tracing to mesh perturbation."
    )
    p.add_argument(
        "--results-dir", type=Path,
        default=Path("validation/results/perturbations_results"),
        help="Directory containing perturbed *_paths.csv files.",
    )
    p.add_argument(
        "--out-a", type=Path,
        default=Path("validation/figures/fig5A_shape_mean.png"),
        help="Output PNG for Figure 5A (shape_mean).",
    )
    p.add_argument(
        "--out-b", type=Path,
        default=Path("validation/figures/fig5B_shape_cv.png"),
        help="Output PNG for Figure 5B (shape_cv).",
    )
    p.add_argument("--dpi", type=int, default=300)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    if not args.results_dir.exists():
        print(f"[ERR] Results directory not found: {args.results_dir}")
        return 1

    print(f"Loading perturbed CSVs from {args.results_dir} ...")
    df = _load_csvs(args.results_dir)
    if df.empty:
        print(f"[ERR] No matching *_paths.csv files found in {args.results_dir}")
        return 1

    print(f"  {len(df)} total path rows from {df['surface'].nunique()} surfaces "
          f"x {df['deformation'].nunique()} jitter levels")

    agg = _aggregate(df)
    print(f"  {len(agg)} aggregated (surface, deformation) groups")

    panels = [
        (args.out_a, "shape_mean", "Mean shape quality"),
        (args.out_b, "shape_cv", "Shape quality coefficient of variation"),
    ]

    for out_path, xcol, xlabel in panels:
        fig, ax = plt.subplots(figsize=(6, 5))
        _scatter_panel(ax, agg, xcol, xlabel)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
