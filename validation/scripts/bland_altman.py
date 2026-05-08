#!/usr/bin/env python3
"""Generate Figure 4: Bland-Altman validation plots.

Three panels: disks (standard BA), hemispheres (standard BA), phantoms (relative BA).

Standard BA:
  x = (measured + truth) / 2
  y = measured - truth

Relative BA (phantoms):
  x = (measured + truth) / 2
  y = (measured - truth) / measured
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_TRUTH_PATTERNS = [
    r"^(?P<truth>\d+(?:\.\d+)?)x10_",
    r"^15mm_(?P<truth>\d+(?:\.\d+)?)",
]


def _parse_truth(name: str) -> float | None:
    for pat in _TRUTH_PATTERNS:
        m = re.search(pat, name)
        if m:
            return float(m.group("truth"))
    return None


def _load_dataset(results_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (truth_values, measured_values) from all *_paths.csv in results_dir."""
    truth_vals: list[float] = []
    measured_vals: list[float] = []
    for csv_path in sorted(results_dir.glob("*_paths.csv")):
        truth = _parse_truth(csv_path.name)
        if truth is None:
            print(f"  [WARN] Could not parse truth from filename, skipping: {csv_path.name}")
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    ld = float(row["len_down"])
                except (ValueError, KeyError):
                    continue
                if np.isfinite(ld):
                    measured_vals.append(ld)
                    truth_vals.append(truth)
    return np.array(truth_vals), np.array(measured_vals)


def _ba_panel(
    ax: plt.Axes,
    truth: np.ndarray,
    measured: np.ndarray,
    relative: bool,
    title: str,
) -> None:
    levels = sorted(set(truth))
    cmap = plt.get_cmap("tab10", max(len(levels), 1))

    x = (measured + truth) / 2.0
    if relative:
        with np.errstate(divide="ignore", invalid="ignore"):
            y = (measured - truth) / measured
        ylabel = "(measured − truth) / measured"
    else:
        y = measured - truth
        ylabel = "Difference (mm)\n(measured − truth)"

    for i, lv in enumerate(levels):
        mask = np.isclose(truth, lv)
        ax.scatter(
            x[mask], y[mask],
            alpha=0.4, s=8, color=cmap(i), label=f"{lv:g} mm",
            rasterized=True,
        )

    mean_d = float(np.mean(y))
    std_d = float(np.std(y, ddof=1))
    loa_lo = mean_d - 1.96 * std_d
    loa_hi = mean_d + 1.96 * std_d

    ax.axhline(mean_d, color="black", lw=1.2, ls="--", zorder=3)
    ax.axhline(loa_lo, color="#666666", lw=0.9, ls=":")
    ax.axhline(loa_hi, color="#666666", lw=0.9, ls=":")
    ax.axhline(0.0, color="black", lw=0.4, alpha=0.4)

    if not relative:
        ax.set_ylim(-0.04, 0.04)
    ax.set_xlabel("Mean of methods (mm)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, markerscale=2.5, title="Truth", title_fontsize=7,
              loc="upper left", framealpha=0.8)

    stats_text = f"Bias: {mean_d:+.4f}\nLoA: [{loa_lo:+.4f}, {loa_hi:+.4f}]"
    ax.text(
        0.98, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=7, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate Figure 4: Bland-Altman plots.")
    p.add_argument(
        "--disks-dir", type=Path,
        default=Path("validation/results/disks_results"),
        help="Directory containing disks *_paths.csv files.",
    )
    p.add_argument(
        "--hemispheres-dir", type=Path,
        default=Path("validation/results/hemispheres_results"),
        help="Directory containing hemispheres *_paths.csv files.",
    )
    p.add_argument(
        "--phantoms-dir", type=Path,
        default=Path("validation/results/phantom_results"),
        help="Directory containing phantom *_paths.csv files.",
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("validation/figures/fig4_bland_altman.png"),
        help="Output PNG path.",
    )
    p.add_argument("--dpi", type=int, default=300)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    datasets = [
        (args.disks_dir, "Disks (analytical)", False),
        (args.hemispheres_dir, "Hemispheres (analytical)", False),
        (args.phantoms_dir, "Phantoms (synthetic)", True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (d, title, relative) in zip(axes, datasets):
        if not d.exists():
            print(f"[WARN] Directory not found, skipping: {d}")
            ax.set_visible(False)
            continue
        print(f"Loading {d.name} ...")
        truth, measured = _load_dataset(d)
        if truth.size == 0:
            print(f"[WARN] No valid data in {d}")
            ax.set_visible(False)
            continue
        print(f"  {truth.size} path measurements across {len(set(truth))} truth levels")
        _ba_panel(ax, truth, measured, relative=relative, title=title)

    fig.suptitle("Bland-Altman Analysis — Two-way Path Tracing", fontsize=11)
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Figure 4 saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
