from __future__ import annotations

import numpy as np
import statistics as stats
from math import sqrt

def compare_down_up(all_lengths_down, all_lengths_up, *, label="Down↔Up", debug=False):
    n_pairs = min(len(all_lengths_down), len(all_lengths_up))
    if n_pairs == 0:
        print("[WARN] No paired paths to compare.")
        return

    d = np.asarray(all_lengths_down[:n_pairs], dtype=float)
    u = np.asarray(all_lengths_up[:n_pairs], dtype=float)

    delta = u - d
    abs_delta = np.abs(delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d != 0.0, u / d, np.nan)

    mean_delta = float(delta.mean())
    sd_delta = float(delta.std(ddof=1)) if n_pairs > 1 else float("nan")
    t_val = (mean_delta / (sd_delta / sqrt(n_pairs))) if n_pairs > 1 and sd_delta > 0 else float("nan")
    r_val = float(np.corrcoef(d, u)[0, 1]) if n_pairs > 1 else float("nan")

    lines = [
        f"# === {label} comparison (n={n_pairs}) ===",
        f"Mean Δ (up−down):   {mean_delta:.3f}",
        f"SD   Δ:             {sd_delta:.3f}",
        f"Mean |Δ|:           {abs_delta.mean():.3f}",
        f"Median |Δ|:         {np.median(abs_delta):.3f}",
        f"Mean ratio up/down: {np.nanmean(ratio):.3f}",
        f"Median ratio:       {np.nanmedian(ratio):.3f}",
        f"Pearson r:          {r_val:.3f}",
        f"Paired t value:     {t_val:.3f} (df={n_pairs-1})",
    ]

    if debug:
        lines.append("\n# per-pair rows = idx | down | up | Δ | |Δ| | ratio")
        for i in range(n_pairs):
            lines.append(
                f"{i+1:3d} {d[i]:8.3f} {u[i]:8.3f} {delta[i]:7.3f} {abs_delta[i]:7.3f} {ratio[i]:7.3f}"
            )

    print("\n" + "\n".join(lines))

def summarise_paths(all_paths_down, all_paths_up, *, debug=False, save_txt=None):
    def fmt(pt):
        return f"({pt[0]:.5f}, {pt[1]:.5f}, {pt[2]:.5f})"

    lines = ["# === DOWN paths ==="]
    for i, path in enumerate(all_paths_down, 1):
        lines.append(f"Path {i:<3d}: " + ", ".join(fmt(p) for p in path))

    lines.append("# === UP paths ===")
    for i, path in enumerate(all_paths_up, 1):
        lines.append(f"Path {i:<3d}: " + ", ".join(fmt(p) for p in path))

    txt = "\n".join(lines)
    if debug:
        print("\n" + txt)
    if save_txt is not None:
        with open(save_txt, "w") as fh:
            fh.write(txt + "\n")

def _print_length_stats(name: str, lengths: list[float]) -> None:
    if not lengths:
        return
    n = len(lengths)
    mean_val = stats.mean(lengths)
    med_val = stats.median(lengths)
    sd_val = stats.stdev(lengths) if n > 1 else 0.0
    print(f"\n{name} statistics (n={n}):")
    print(f"  mean   : {mean_val:.3f}")
    print(f"  median : {med_val:.3f}")
    print(f"  stdev  : {sd_val:.3f}")
    print(f"  min    : {min(lengths):.3f}")
    print(f"  max    : {max(lengths):.3f}")
