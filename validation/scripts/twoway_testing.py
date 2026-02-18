#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from laplace_thickness_release.bem.core import BEMConfig, assemble_and_solve
from laplace_thickness_release.main.main import load_triangles_from_vtk_polydata
from laplace_thickness_release.mesh.preprocess import preprocess_triangles
from laplace_thickness_release.trace.start_points import pick_even_start_points
from laplace_thickness_release.trace.tracer import path_trace_simple_bem


def _csv_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.16g}"


@dataclass
class RunConfig:
    data_root: str
    outdir: str
    subfolders: list[str]
    pattern: str
    workers: int
    target_n_paths: int
    init_pct: int
    max_pct: int
    pct_step: int
    target_spacing: float | None
    alpha_initial: float
    first_step: float
    max_iter: int
    angle_max_deg: float
    debug_trace: bool
    quad_order: int
    tau_near: float
    tol_near: float
    max_subdiv: int


def top_surface_vertices_faces(triangles: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    vert_index: dict[tuple[float, float, float], int] = {}
    verts: list[np.ndarray] = []
    faces: list[list[int]] = []

    for tri in triangles:
        if str(tri.get("bc_type", "")).lower() != "dirichlet":
            continue
        if not np.isclose(float(tri.get("bc_value", 0.0)), 1.0):
            continue

        idxs: list[int] = []
        for v in tri["vertices"]:
            key = tuple(np.asarray(v, dtype=np.float64))
            if key not in vert_index:
                vert_index[key] = len(verts)
                verts.append(np.asarray(key, dtype=np.float64))
            idxs.append(vert_index[key])
        faces.append(idxs)

    if not verts:
        raise ValueError("No pial surface triangles (Dirichlet bc_value=1.0) found.")

    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def build_geodesic_graph(vertices: np.ndarray, faces: np.ndarray) -> csr_matrix:
    edges: set[tuple[int, int]] = set()
    for a, b, c in faces:
        edges.add((min(a, b), max(a, b)))
        edges.add((min(b, c), max(b, c)))
        edges.add((min(c, a), max(c, a)))

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for i, j in edges:
        w = float(np.linalg.norm(vertices[i] - vertices[j]))
        rows.extend((i, j))
        cols.extend((j, i))
        vals.extend((w, w))

    n = int(vertices.shape[0])
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def geodesic_distance_on_top(
    vertices: np.ndarray,
    graph: csr_matrix,
    src_pt: np.ndarray,
    dst_pt: np.ndarray,
    tree: cKDTree | None = None,
) -> float:
    tree = tree or cKDTree(vertices)
    i_src = int(tree.query(src_pt, k=1)[1])
    i_dst = int(tree.query(dst_pt, k=1)[1])
    dist = dijkstra(graph, directed=False, indices=i_src, limit=np.inf)
    return float(dist[i_dst])


def select_vertex_start_points(
    triangles: list[dict[str, Any]],
    *,
    target_n: int,
    init_pct: int,
    max_pct: int,
    pct_step: int,
    target_spacing: float | None,
) -> np.ndarray:
    pct = int(init_pct)

    while True:
        pts = pick_even_start_points(
            triangles,
            pct=float(pct),
            target_spacing=target_spacing,
            max_points=None,
            bc_value=1.0,
            seed_mode="vertex",
        )
        pts = np.asarray(pts, dtype=np.float64)

        if pts.shape[0] >= target_n or pct >= max_pct:
            break
        pct = min(int(max_pct), pct + int(pct_step))

    if pts.shape[0] == 0:
        return pts

    if pts.shape[0] <= target_n:
        return pts

    pick = np.linspace(0, pts.shape[0] - 1, num=target_n, dtype=int)
    return pts[pick]


def trace_one(
    idx: int,
    start_pt: np.ndarray,
    triangles: list[dict[str, Any]],
    q: np.ndarray,
    cfg: BEMConfig,
    *,
    alpha_initial: float,
    first_step: float,
    max_iter: int,
    angle_max_deg: float,
    debug: bool,
) -> tuple[int, list[np.ndarray], float, dict[str, Any], list[np.ndarray], float, dict[str, Any]]:
    path_down, len_down, meta_down = path_trace_simple_bem(
        start_pt,
        triangles,
        q,
        cfg,
        direction="down",
        max_iter=max_iter,
        alpha_initial=alpha_initial,
        first_step=first_step,
        debug=debug,
        angle_max_deg=angle_max_deg,
    )

    path_up: list[np.ndarray] = []
    len_up = float("nan")
    meta_up: dict[str, Any] = {}

    if path_down:
        path_up, len_up, meta_up = path_trace_simple_bem(
            path_down[-1],
            triangles,
            q,
            cfg,
            direction="up",
            max_iter=max_iter,
            alpha_initial=alpha_initial,
            first_step=first_step,
            debug=debug,
            angle_max_deg=angle_max_deg,
        )

    return idx, path_down, float(len_down), meta_down, path_up, float(len_up), meta_up


def process_surface(mesh_path: Path, cfg: argparse.Namespace) -> list[dict[str, float | int]]:
    triangles = load_triangles_from_vtk_polydata(mesh_path)
    preprocess_triangles(triangles)

    bem_cfg = BEMConfig(
        quad_order=int(cfg.quad_order),
        TAU_NEAR=float(cfg.tau_near),
        TOL_NEAR=float(cfg.tol_near),
        MAX_SUBDIV=int(cfg.max_subdiv),
    )
    q, _, _ = assemble_and_solve(triangles, bem_cfg)

    top_v, top_f = top_surface_vertices_faces(triangles)
    top_g = build_geodesic_graph(top_v, top_f)
    top_tree = cKDTree(top_v)

    starts = select_vertex_start_points(
        triangles,
        target_n=int(cfg.target_n_paths),
        init_pct=int(cfg.init_pct),
        max_pct=int(cfg.max_pct),
        pct_step=int(cfg.pct_step),
        target_spacing=cfg.target_spacing,
    )

    n_paths = int(starts.shape[0])
    if n_paths == 0:
        print(f"[WARN] No vertex start points found for {mesh_path.name}")
        return []

    rows: list[dict[str, float | int] | None] = [None] * n_paths
    with ThreadPoolExecutor(max_workers=int(cfg.workers)) as pool:
        futures = [
            pool.submit(
                trace_one,
                i,
                np.asarray(starts[i], dtype=np.float64),
                triangles,
                q,
                bem_cfg,
                alpha_initial=float(cfg.alpha_initial),
                first_step=float(cfg.first_step),
                max_iter=int(cfg.max_iter),
                angle_max_deg=float(cfg.angle_max_deg),
                debug=bool(cfg.debug_trace),
            )
            for i in range(n_paths)
        ]

        for fut in as_completed(futures):
            (
                idx,
                path_down,
                len_down,
                _meta_down,
                path_up,
                len_up,
                _meta_up,
            ) = fut.result()

            if len(path_down) == 0:
                rows[idx] = {
                    "path_idx": idx,
                    "len_down": float("nan"),
                    "len_up": float("nan"),
                    "len_asym_abs": float("nan"),
                    "len_ratio_up_over_down": float("nan"),
                    "geodesic_disp": float("nan"),
                }
                continue

            start_down = np.asarray(path_down[0], dtype=np.float64)
            if path_up:
                end_up = np.asarray(path_up[-1], dtype=np.float64)
            else:
                end_up = np.asarray(path_down[-1], dtype=np.float64)

            geodesic_disp = geodesic_distance_on_top(top_v, top_g, start_down, end_up, top_tree)
            len_asym_abs = float(abs(len_up - len_down))
            len_ratio = float(len_up / len_down) if len_down > 0 else float("inf")

            rows[idx] = {
                "path_idx": int(idx),
                "len_down": float(len_down),
                "len_up": float(len_up),
                "len_asym_abs": len_asym_abs,
                "len_ratio_up_over_down": len_ratio,
                "geodesic_disp": float(geodesic_disp),
            }

    return [r for r in rows if r is not None]


def _write_rows_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames = [
        "path_idx",
        "len_down",
        "len_up",
        "len_asym_abs",
        "len_ratio_up_over_down",
        "geodesic_disp",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _ensure_master_header(path: Path) -> None:
    if path.exists():
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([
            "folder",
            "surface",
            "path_idx",
            "len_down",
            "len_up",
            "len_asym_abs",
            "len_ratio_up_over_down",
            "geodesic_disp",
        ])


def _append_master(path: Path, folder: str, surface: str, rows: list[dict[str, float | int]]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        for row in rows:
            w.writerow([
                folder,
                surface,
                row["path_idx"],
                _csv_float(float(row["len_down"])),
                _csv_float(float(row["len_up"])),
                _csv_float(float(row["len_asym_abs"])),
                _csv_float(float(row["len_ratio_up_over_down"])),
                _csv_float(float(row["geodesic_disp"])),
            ])


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run two-way validation tracing on VTK surfaces using pial-vertex seeds. "
            "Outputs per-surface *_paths.csv files with no Hausdorff metric."
        )
    )
    p.add_argument("--data-root", type=Path, default=Path("validation/data"), help="Root directory containing validation surface folders.")
    p.add_argument(
        "--subfolders",
        nargs="+",
        default=["analytical/hemispheres"],
        help="One or more subfolders under --data-root to scan.",
    )
    p.add_argument("--pattern", type=str, default="15mm_*.vtk", help="Glob pattern for meshes inside each subfolder.")
    p.add_argument("--outdir", type=Path, default=Path("validation/tests/twoway_results"), help="Directory for output CSV files.")
    p.add_argument("--workers", type=int, default=(os.cpu_count() or 1), help="Thread worker count for per-path tracing.")

    p.add_argument("--target-n-paths", type=int, default=100)
    p.add_argument("--init-pct", type=int, default=50)
    p.add_argument("--max-pct", type=int, default=50)
    p.add_argument("--pct-step", type=int, default=10)
    p.add_argument("--target-spacing", type=float, default=None)

    p.add_argument("--alpha-initial", type=float, default=0.05)
    p.add_argument("--first-step", type=float, default=0.05)
    p.add_argument("--max-iter", type=int, default=400)
    p.add_argument("--angle-max-deg", type=float, default=30.0)
    p.add_argument("--debug-trace", action="store_true")

    p.add_argument("--quad-order", type=int, choices=[1, 3, 7], default=3)
    p.add_argument("--tau-near", type=float, default=0.2)
    p.add_argument("--tol-near", type=float, default=1e-6)
    p.add_argument("--max-subdiv", type=int, default=4)

    p.add_argument("--master-name", type=str, default="_MASTER_paths.csv", help="Filename for combined master CSV in --outdir.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.target_n_paths <= 0:
        parser.error("--target-n-paths must be > 0.")
    if args.workers <= 0:
        parser.error("--workers must be > 0.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    master_csv = args.outdir / args.master_name
    _ensure_master_header(master_csv)

    run_cfg = RunConfig(
        data_root=str(args.data_root),
        outdir=str(args.outdir),
        subfolders=[str(s) for s in args.subfolders],
        pattern=str(args.pattern),
        workers=int(args.workers),
        target_n_paths=int(args.target_n_paths),
        init_pct=int(args.init_pct),
        max_pct=int(args.max_pct),
        pct_step=int(args.pct_step),
        target_spacing=(None if args.target_spacing is None else float(args.target_spacing)),
        alpha_initial=float(args.alpha_initial),
        first_step=float(args.first_step),
        max_iter=int(args.max_iter),
        angle_max_deg=float(args.angle_max_deg),
        debug_trace=bool(args.debug_trace),
        quad_order=int(args.quad_order),
        tau_near=float(args.tau_near),
        tol_near=float(args.tol_near),
        max_subdiv=int(args.max_subdiv),
    )

    metadata_path = args.outdir / "_twoway_testing_run.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "config": asdict(run_cfg),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    for sub in args.subfolders:
        folder = args.data_root / sub
        if not folder.exists():
            print(f"[WARN] Missing folder: {folder}")
            continue

        meshes = sorted(folder.glob(args.pattern))
        if not meshes:
            print(f"[INFO] No meshes matched '{args.pattern}' in {folder}")
            continue

        print(f"\n>>> Folder: {folder} ({len(meshes)} meshes)")
        for mesh_path in meshes:
            print(f"[INFO] Processing {mesh_path.name}")
            rows = process_surface(mesh_path, args)
            if not rows:
                continue

            surface_csv = args.outdir / f"{mesh_path.stem}_paths.csv"
            _write_rows_csv(surface_csv, rows)
            _append_master(master_csv, str(sub), mesh_path.stem, rows)
            print(f"[OK] Wrote {len(rows)} paths -> {surface_csv}")

    print(f"\n[OK] Master CSV: {master_csv}")
    print(f"[OK] Run metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

