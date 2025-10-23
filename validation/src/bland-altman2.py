#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- CONFIG ----------
RESULTS_DIR = Path(r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\phantom_final2_results")

FILES = [
    (f"{t}x10_{i}_paths.csv", float(t))
    for t in ["0.25", "0.5", "1", "2", "3", "4"]  # keep as-is; skip if missing on disk
    for i in range(1, 11)
]
SAVE_PNG = True
PNG_NAME = "bland_altman_phantoms_final2.png"
# ----------------------------

REQ_COL = "len_down"  # what we compare to truth

def bland_altman_stats(diff: np.ndarray):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = diff.size
    if n == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    bias = float(np.mean(diff))
    sd   = float(np.std(diff, ddof=1)) if n > 1 else np.nan
    loa_lo = bias - 1.96*sd if np.isfinite(sd) else np.nan
    loa_hi = bias + 1.96*sd if np.isfinite(sd) else np.nan
    return n, bias, sd, loa_lo, loa_hi

def safe_read_csv(p: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(p)
    except FileNotFoundError:
        print(f"[WARN] Missing file, skip: {p.name}")
    except Exception as e:
        print(f"[WARN] Failed to read {p.name}: {e}")
    return None

def main():
    frames = []
    for fname, truth in FILES:
        p = (RESULTS_DIR / fname)
        df = safe_read_csv(p)
        if df is None:
            continue
        # Ensure required col exists and is numeric
        if REQ_COL not in df.columns:
            print(f"[WARN] {p.name} lacks '{REQ_COL}', skipping.")
            continue
        s = pd.to_numeric(df[REQ_COL], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            print(f"[WARN] {p.name} has no valid '{REQ_COL}', skipping.")
            continue
        frames.append(pd.DataFrame({"file": fname, "that": s.values.astype(float), "truth": float(truth)}))

    if not frames:
        print("[FATAL] No valid data found. Check RESULTS_DIR or filenames.")
        sys.exit(1)

    data = pd.concat(frames, ignore_index=True)

    # Per-file BA stats
    print("\nPer-file Bland–Altman stats (len_down − truth):")
    for file, grp in data.groupby("file", sort=True):
        n, bias, sd, loa_lo, loa_hi = bland_altman_stats(grp["that"].values - grp["truth"].values)
        print(f"  {file:>24}  n={n:4d}  bias={bias: .4f}  sd={sd: .4f}  LoA=[{loa_lo: .4f}, {loa_hi: .4f}]")

    # Pooled BA
    mean_vals = (data["that"].values + data["truth"].values) / 2.0
    diff_vals = data["that"].values - data["truth"].values
    n_all, bias, sd, loa_lo, loa_hi = bland_altman_stats(diff_vals)

    print(f"\nBland–Altman (len_down − truth), pooled across files:")
    print(f"  n    = {n_all}")
    print(f"  Bias = {bias:.4f} mm")
    print(f"  SD   = {sd:.4f} mm")
    print(f"  LoA  = [{loa_lo:.4f}, {loa_hi:.4f}] mm")

    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(mean_vals, diff_vals, s=18, alpha=0.7)
    if np.isfinite(bias):   ax.axhline(bias,   linestyle="--", label=f"Bias = {bias:.3f} mm")
    if np.isfinite(loa_lo): ax.axhline(loa_lo, linestyle=":",  label=f"LoA = [{loa_lo:.3f}, {loa_hi:.3f}] mm")
    if np.isfinite(loa_hi): ax.axhline(loa_hi, linestyle=":")
    ax.set_title("Bland–Altman (measured vs true thickness)")
    ax.set_xlabel("Mean of (len_down, truth) [mm]")
    ax.set_ylabel("len_down − truth [mm]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    if SAVE_PNG:
        out = (RESULTS_DIR / PNG_NAME).resolve()
        fig.savefig(out, dpi=200)
        print(f"\nSaved: {out}")

    plt.show()

if __name__ == "__main__":
    main()
