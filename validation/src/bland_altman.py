#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- CONFIG ----------
RESULTS_DIR = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\other_final2_results")

# List of (csv filename, ground truth thickness in mm)

FILES = [
    ("5mm_4_paths.csv", 4.0),
    ("5mm_3_paths.csv", 3.0),
    ("5mm_2_paths.csv", 2.0),
    ("5mm_1_paths.csv", 1.0),
    ("5mm_0.5_paths.csv", 0.5),
    ("5mm_0.25_paths.csv", 0.25)
]

import re
'''
FILES = [
    (f"{t}x10_{i}_paths.csv", float(t))
    for t in ["0.25", "0.5", "1", "2", "3", "4"]
    for i in range(1, 11)
]
'''
SAVE_PNG = True
PNG_NAME = "bland_altman_hemi_final2.png"
# ----------------------------

def bland_altman_stats(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = diff.size
    bias = np.nan if n == 0 else float(np.mean(diff))
    sd   = np.nan if n <= 1 else float(np.std(diff, ddof=1))
    loa_lo = bias - 1.96*sd if np.isfinite(sd) else np.nan
    loa_hi = bias + 1.96*sd if np.isfinite(sd) else np.nan
    return n, bias, sd, loa_lo, loa_hi

def main():
    # Load & pool
    frames = []
    for fname, truth in FILES:
        p = RESULTS_DIR / fname
        df = pd.read_csv(p)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["len_down"])
        df = df.assign(
            file=fname,
            truth=float(truth),
            that=df["len_down"].astype(float)
        )
        frames.append(df[["file", "that", "truth"]])

    if not frames:
        raise SystemExit("No files provided.")
    data = pd.concat(frames, ignore_index=True)

    # Per-file stats (optional but handy)
    print("\nPer-file Bland–Altman stats (len_down − truth):")
    for file, grp in data.groupby("file"):
        n, bias, sd, loa_lo, loa_hi = bland_altman_stats(grp["that"] - grp["truth"])
        print(f"  {file:>20}  n={n:3d}  bias={bias:.4f}  sd={sd:.4f}  LoA=[{loa_lo:.4f}, {loa_hi:.4f}]")

    # Pooled BA inputs
    mean_vals = (data["that"] + data["truth"]) / 2.0
    diff_vals = data["that"] - data["truth"]

    # Overall stats
    n_all, bias, sd, loa_lo, loa_hi = bland_altman_stats(diff_vals)
    print(f"\n Bland–Altman (len_down − truth): n={n_all}")
    print(f"  Bias = {bias:.4f} mm")
    print(f"  SD   = {sd:.4f} mm")
    print(f"  LoA  = [{loa_lo:.4f}, {loa_hi:.4f}] mm")

    # Plot (single pooled BA)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(mean_vals, diff_vals, s=18, alpha=0.7)
    ax.axhline(bias,   linestyle="--", label=f"Bias = {bias:.3f}")
    if np.isfinite(loa_lo): ax.axhline(loa_lo, linestyle=":", label=f"LoA = [{loa_lo:.3f}, {loa_hi:.3f}]")
    if np.isfinite(loa_hi): ax.axhline(loa_hi, linestyle=":")
    ax.set_title("Bland–Altman (measured vs true thickness)")
    ax.set_xlabel("Mean of (len_down, true thickness) [mm]")
    ax.set_ylabel("len_down − true thickness")
    ax.set_ylim(-0.05, 0.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if SAVE_PNG:
        out = RESULTS_DIR / PNG_NAME
        fig.savefig(out, dpi=200)
        print(f"\nSaved: {out}")

    plt.show()

if __name__ == "__main__":
    main()
