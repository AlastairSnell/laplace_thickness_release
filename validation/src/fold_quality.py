#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------ build data table ------------------------ #

def build_table() -> pd.DataFrame:
    # Base path/asym table
    base_rows = [
        {"Data": "Patch 1",  "Paths (n)": 80,  "Geo<0.05mm (%)": 100.0, "Asym<1% (%)": 100.00, "Asym<5% (%)": 100.0, "p95 asym (%)": 0.37},
        {"Data": "Patch 2",  "Paths (n)": 77,  "Geo<0.05mm (%)": 100.0, "Asym<1% (%)":  96.10, "Asym<5% (%)": 100.0, "p95 asym (%)": 0.58},
        {"Data": "Patch 3",  "Paths (n)": 81,  "Geo<0.05mm (%)": 100.0, "Asym<1% (%)": 100.00, "Asym<5% (%)": 100.0, "p95 asym (%)": 0.21},
        {"Data": "Patch 4",  "Paths (n)": 97,  "Geo<0.05mm (%)": 100.0, "Asym<1% (%)": 100.00, "Asym<5% (%)": 100.0, "p95 asym (%)": 0.15},
        {"Data": "Patch 5",  "Paths (n)": 96,  "Geo<0.05mm (%)": 100.0, "Asym<1% (%)": 100.00, "Asym<5% (%)": 100.0, "p95 asym (%)": 0.12},
        {"Data": "Patch 6",  "Paths (n)": 82,  "Geo<0.05mm (%)": 100.0, "Asym<1% (%)": 100.00, "Asym<5% (%)": 100.0, "p95 asym (%)": 0.08},
        {"Data": "Patch 7",  "Paths (n)": 84,  "Geo<0.05mm (%)": 100.0, "Asym<1% (%)":  97.62, "Asym<5% (%)": 100.0, "p95 asym (%)": 0.61},
        {"Data": "Patch 8",  "Paths (n)": 59,  "Geo<0.05mm (%)": 100.0, "Asym<1% (%)": 100.00, "Asym<5% (%)": 100.0, "p95 asym (%)": 0.41},
        {"Data": "Patch 9",  "Paths (n)": 83,  "Geo<0.05mm (%)": 100.0, "Asym<1% (%)": 100.00, "Asym<5% (%)": 100.0, "p95 asym (%)": 0.12},
        {"Data": "Patch 10", "Paths (n)": 59,  "Geo<0.05mm (%)": 100.0, "Asym<1% (%)": 100.00, "Asym<5% (%)": 100.0, "p95 asym (%)": 0.55},
        {"Data": "Patch 11", "Paths (n)": 81,  "Geo<0.05mm (%)": 100.0, "Asym<1% (%)": 100.00, "Asym<5% (%)": 100.0, "p95 asym (%)": 0.42},
        {"Data": "Patch 12", "Paths (n)": 74,  "Geo<0.05mm (%)": 100.0, "Asym<1% (%)": 100.00, "Asym<5% (%)": 100.0, "p95 asym (%)": 0.21},
        {"Data": "All data", "Paths (n)": 953, "Geo<0.05mm (%)": 100.0, "Asym<1% (%)":  99.48, "Asym<5% (%)": 100.0, "p95 asym (%)": 0.33},
    ]
    base = pd.DataFrame(base_rows)

    # Shape-quality data
    shape_rows = [
        {"surface": "zipped_patch_000", "n_cells": 1724, "shape_mean": 0.8656073923477019, "shape_cv": 0.16074155080965555},
        {"surface": "zipped_patch_001", "n_cells": 1518, "shape_mean": 0.8674966944480201, "shape_cv": 0.14880603153700617},
        {"surface": "zipped_patch_002", "n_cells": 1941, "shape_mean": 0.895401877774185,  "shape_cv": 0.12512134629193825},
        {"surface": "zipped_patch_003", "n_cells": 1970, "shape_mean": 0.9364092583829985, "shape_cv": 0.07806260959105957},
        {"surface": "zipped_patch_004", "n_cells": 1975, "shape_mean": 0.9073361924044486, "shape_cv": 0.11288169430967972},
        {"surface": "zipped_patch_005", "n_cells": 1982, "shape_mean": 0.9156016031573999, "shape_cv": 0.10680685513612144},
        {"surface": "zipped_patch_006", "n_cells": 1834, "shape_mean": 0.8967242879420053, "shape_cv": 0.11571567002653362},
        {"surface": "zipped_patch_007", "n_cells": 1352, "shape_mean": 0.863669323007863,  "shape_cv": 0.1957882264449878},
        {"surface": "zipped_patch_008", "n_cells": 1726, "shape_mean": 0.9083281379316023, "shape_cv": 0.10077329176969323},
        {"surface": "zipped_patch_009", "n_cells": 1546, "shape_mean": 0.8550351429281632, "shape_cv": 0.19619071609573213},
        {"surface": "zipped_patch_010", "n_cells": 1805, "shape_mean": 0.8832755083096201, "shape_cv": 0.13240859417125336},
        {"surface": "zipped_patch_011", "n_cells": 1966, "shape_mean": 0.8782952148914793, "shape_cv": 0.15353456341944224},
    ]
    for i, r in enumerate(shape_rows):
        r["Data"] = f"Patch {i+1}"

    shape_df = pd.DataFrame(shape_rows)

    # Merge shape stats onto base
    merged = base.merge(
        shape_df[["Data", "n_cells", "shape_mean", "shape_cv"]],
        on="Data",
        how="left",
    )

    # Global shape_mean / shape_cv (weighted by n_cells)
    valid = shape_df.dropna(subset=["shape_mean", "shape_cv", "n_cells"])
    N = float(valid["n_cells"].sum())
    mu_global = float((valid["shape_mean"] * valid["n_cells"]).sum() / N)
    sigma_i_sq = (valid["shape_cv"] * valid["shape_mean"]) ** 2
    sigma2_global = float(
        ((sigma_i_sq + valid["shape_mean"] ** 2) * valid["n_cells"]).sum() / N
        - mu_global ** 2
    )
    sigma_global = float(np.sqrt(max(sigma2_global, 0.0)))
    cv_global = float(sigma_global / mu_global) if mu_global != 0 else np.nan

    mask_all = merged["Data"].eq("All data")
    merged.loc[mask_all, "shape_mean"] = mu_global
    merged.loc[mask_all, "shape_cv"] = cv_global

    # Final column order
    merged = merged[
        [
            "Data",
            "shape_mean",
            "shape_cv",
            "Paths (n)",
            "Geo<0.05mm (%)",
            "Asym<1% (%)",
            "Asym<5% (%)",
            "p95 asym (%)",
        ]
    ]

    # Round to 3 d.p. max (leave Paths as int)
    for col in merged.columns:
        if col in ("Data", "Paths (n)"):
            continue
        merged[col] = merged[col].round(3)

    return merged


# ------------------------ styling & save ------------------------ #

def save_table_png(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Make a copy with string formatting
    disp = df.copy()
    for col in disp.columns:
        if col == "Data":
            continue
        if col == "Paths (n)":
            disp[col] = disp[col].astype(int).astype(str)
        else:
            disp[col] = disp[col].map(
                lambda x: "" if pd.isna(x) else f"{x:.3f}".rstrip("0").rstrip(".")
            )

    n_rows, n_cols = disp.shape

    fig_width = 12
    fig_height = 0.5 + 0.5 * (n_rows + 1)  # header + rows
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=disp.values,
        colLabels=disp.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.4)

    # Styling: header grey, bold; alternating row shading; light grid
    header_color = "#f1f1f1"
    stripe_color = "#fafafa"
    edge_color = "#d0d0d0"

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)

        # Header row
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold")
        else:
            # Alternating row background
            if row % 2 == 1:
                cell.set_facecolor("#ffffff")
            else:
                cell.set_facecolor(stripe_color)

        # Left-align first column, centre others
        if col == 0:
            cell._loc = "w"

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    df = build_table()

    out_path = Path(
        r"C:\Users\uqasnell\Documents\GitHub\laplace_thickness\validation\folds_results\folds_validation_table.png"
    )
    save_table_png(df, out_path)
    print("Saved PNG to:", out_path)


if __name__ == "__main__":
    main()
