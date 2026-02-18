from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons

from laplace_thickness_release.bem.core import BEMConfig, assemble_and_solve
from laplace_thickness_release.mesh.preprocess import preprocess_triangles
from laplace_thickness_release.trace.normal import build_surface_vertex_normals
from laplace_thickness_release.trace.start_points import pick_even_start_points
from laplace_thickness_release.trace.tracer import path_trace_simple_bem
from laplace_thickness_release.viz.plotting import (
    plot_bumpy_bottom,
    plot_bumpy_top,
    set_axes_equal,
)

LOG = logging.getLogger(__name__)
DEFAULT_WORKERS = os.cpu_count() or 1

# Policy defaults (non-expert)
DEFAULT_START_PCT_TRACE = 50.0
DEFAULT_START_SPACING_TRACE = 1.0
DEFAULT_START_SPACING_HEATMAP = 0.5

# Tracing defaults
DEFAULT_ALPHA = 0.05
DEFAULT_MAX_ITER = 150
DEFAULT_FIRST_STEP = 0.05

# Heatmap defaults (hidden unless --expert)
DEFAULT_HEATMAP_PCT = 50.0
DEFAULT_HEATMAP_LOWER_BOUND = 1.0
DEFAULT_HEATMAP_POWER = 2.0
DEFAULT_HEATMAP_K_NEIGH = 3

def load_triangles_from_vtk_polydata(
    mesh_path: Path,
    *,
    bc_type_array: str = "bc_type",
    bc_value_array: str = "bc_value",
    normal_array: str = "normal",
    require_all_triangles: bool = True,
) -> list[dict[str, Any]]:
    """
    Load PolyData from .vtk or .vtp and convert to the legacy triangles list-of-dicts.

    Requirements in the VTK file:
      - PolyData surface
      - CellData arrays:
          * bc_type (string OR numeric)
          * bc_value (float)
          * normal  (float[3])  [default name: 'normal']
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    from vtkmodules.vtkIOLegacy import vtkPolyDataReader
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

    suffix = mesh_path.suffix.lower()
    if suffix == ".vtp":
        reader = vtkXMLPolyDataReader()
    elif suffix == ".vtk":
        reader = vtkPolyDataReader()
    else:
        raise ValueError(f"Unsupported mesh extension '{suffix}'. Use .vtk or .vtp (PolyData).")

    reader.SetFileName(str(mesh_path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None:
        raise RuntimeError(f"VTK reader produced no output for {mesh_path}")

    points = poly.GetPoints()
    if points is None:
        raise RuntimeError("Mesh has no points.")

    cell_data = poly.GetCellData()
    if cell_data is None:
        raise RuntimeError("Mesh has no CellData.")

    # --- required arrays ---
    bc_type = cell_data.GetAbstractArray(bc_type_array)
    if bc_type is None:
        bc_type = cell_data.GetArray(bc_type_array)
    if bc_type is None:
        raise RuntimeError(f"Missing required CellData array '{bc_type_array}'")

    bc_val = cell_data.GetArray(bc_value_array)
    if bc_val is None:
        raise RuntimeError(f"Missing required CellData array '{bc_value_array}'")

    normals = cell_data.GetArray(normal_array)
    if normals is None:
        raise RuntimeError(f"Missing required CellData array '{normal_array}'")

    n_cells = poly.GetNumberOfCells()
    if n_cells <= 0:
        raise RuntimeError("No cells in mesh.")

    triangles: list[dict[str, Any]] = []
    non_tri_count = 0

    for cid in range(n_cells):
        cell = poly.GetCell(cid)
        if cell is None:
            continue
        if cell.GetNumberOfPoints() != 3:
            non_tri_count += 1
            if require_all_triangles:
                raise RuntimeError(
                    f"Mesh contains non-triangle cell at cid={cid} with "
                    f"{cell.GetNumberOfPoints()} points. Set require_all_triangles=False to skip."
                )
            continue

        pid0 = cell.GetPointId(0)
        pid1 = cell.GetPointId(1)
        pid2 = cell.GetPointId(2)

        # Preserve vertex order exactly as stored in the file
        v0 = np.asarray(points.GetPoint(pid0), dtype=np.float64)
        v1 = np.asarray(points.GetPoint(pid1), dtype=np.float64)
        v2 = np.asarray(points.GetPoint(pid2), dtype=np.float64)

        # bc_type: string or numeric
        try:
            bt = bc_type.GetValue(cid)  # vtkStringArray supports this
            bt_str = str(bt).strip().lower()
        except Exception:
            bt_num = float(bc_type.GetTuple1(cid))
            bt_str = "dirichlet" if int(bt_num) == 0 else "neumann"

        bv = float(bc_val.GetTuple1(cid))

        # normal: take as-is (no normalization, no flipping)
        n = np.asarray(normals.GetTuple3(cid), dtype=np.float64)

        triangles.append(
            {
                "vertices": np.stack([v0, v1, v2], axis=0),
                "bc_type": bt_str,
                "bc_value": bv,
                "normal": n,
            }
        )

    if not triangles:
        raise RuntimeError("No triangles extracted from mesh (unexpected).")

    return triangles

def run_single_path(
    idx: int,
    *,
    triangles: list[dict[str, Any]],
    start_pt: np.ndarray,
    q: np.ndarray,
    cfg: BEMConfig,
    direction_down: str,
    max_iter: int,
    alpha_initial: float,
    first_step: float,
    seed_dir: np.ndarray | None,
    seed_face_idx: int | None,
    debug: bool,
):
    path_down, len_down, meta = path_trace_simple_bem(
        start_pt,
        triangles,
        q,
        cfg,
        direction=direction_down,
        max_iter=max_iter,
        alpha_initial=alpha_initial,
        first_step=first_step,
        seed_dir=seed_dir,
        seed_face_idx=seed_face_idx,
        debug=debug,
    )
    return idx, path_down, float(len_down), meta


def run_two_way_path(
    idx: int,
    *,
    triangles: list[dict[str, Any]],
    start_pt: np.ndarray,
    q: np.ndarray,
    cfg: BEMConfig,
    direction_down: str,
    direction_up: str,
    max_iter: int,
    alpha_initial: float,
    first_step: float,
    seed_dir: np.ndarray | None,
    seed_face_idx: int | None,
    tri_index_map: dict[int, int] | None,
    centroid_seeds: bool,
    debug: bool,
):
    path_down, len_down, meta_down = path_trace_simple_bem(
        start_pt,
        triangles,
        q,
        cfg,
        direction=direction_down,
        max_iter=max_iter,
        alpha_initial=alpha_initial,
        first_step=first_step,
        seed_dir=seed_dir,
        seed_face_idx=seed_face_idx,
        debug=debug,
    )

    path_up = None
    len_up = float("nan")
    meta_up: dict[str, Any] = {}

    exit_idx = meta_down.get("exit_tri_idx", None)
    if exit_idx is not None and path_down:
        back_seed_face_idx = None
        if centroid_seeds and tri_index_map is not None:
            back_seed_face_idx = tri_index_map.get(int(exit_idx))

        path_up, len_up, meta_up = path_trace_simple_bem(
            path_down[-1],
            triangles,
            q,
            cfg,
            direction=direction_up,
            max_iter=max_iter,
            alpha_initial=alpha_initial,
            first_step=first_step,
            seed_dir=None,
            seed_face_idx=back_seed_face_idx,
            debug=debug,
        )

    return idx, path_down, float(len_down), meta_down, path_up, float(len_up), meta_up


def _has_expert_flag(argv: Sequence[str]) -> bool:
    return "--expert" in argv


def _parse_window_size(value: str) -> tuple[int, int]:
    cleaned = value.strip().lower().replace(" ", "")
    if "x" in cleaned:
        parts = cleaned.split("x")
    elif "," in cleaned:
        parts = cleaned.split(",")
    else:
        raise argparse.ArgumentTypeError("Window size must be WxH (e.g. 2400x2400).")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Window size must be WxH (e.g. 2400x2400).")
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Window size values must be integers.") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Window size values must be positive integers.")
    return width, height


def _parse_csv_floats(value: str) -> list[float]:
    cleaned = value.strip().replace(" ", "")
    if not cleaned:
        raise argparse.ArgumentTypeError("Value list cannot be empty.")
    parts = [p for p in cleaned.split(",") if p]
    if not parts:
        raise argparse.ArgumentTypeError("Value list cannot be empty.")
    out: list[float] = []
    for part in parts:
        try:
            out.append(float(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid float: '{part}'.") from exc
    return out


def build_argparser(*, expert: bool) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trace_bem_main",
        description=(
            "Trace gradient-flow paths in a BEM-computed Laplace field (VTK input).\n"
            "Use --expert to reveal advanced tuning options."
        ),
    )

    p.add_argument("--expert", action="store_true", help="Show/enable expert options in --help.")
    p.add_argument("--mesh", type=Path, required=True, help="Surface mesh (.vtk or .vtp PolyData).")

    p.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable 3D plot.")
    p.add_argument("--heatmap", action=argparse.BooleanOptionalAction, default=False, help="Overlay thickness heatmap.")
    p.add_argument(
        "--heatmap-screenshot",
        type=Path,
        default=None,
        help="Save PyVista heatmap to an image file (implies off-screen rendering).",
    )
    p.add_argument(
        "--heatmap-window-size",
        type=_parse_window_size,
        default=(2400, 2400),
        help="PyVista window size for heatmap screenshot (WxH). Only used with --heatmap-screenshot.",
    )
    p.add_argument(
        "--heatmap-image-scale",
        type=int,
        default=3,
        help="Supersampling scale for PyVista heatmap screenshots.",
    )
    p.add_argument(
        "--heatmap-shadows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable PyVista shadows for heatmap rendering.",
    )
    p.add_argument(
        "--heatmap-azimuths",
        type=_parse_csv_floats,
        default=None,
        help="Comma-separated azimuth angles (degrees) for multi-view screenshots.",
    )
    p.add_argument(
        "--heatmap-elevations",
        type=_parse_csv_floats,
        default=None,
        help="Comma-separated elevation angles (degrees) for multi-view screenshots.",
    )
    p.add_argument(
        "--heatmap-clim",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Fix heatmap color range as two floats: MIN MAX (mm).",
    )
    p.add_argument("--start-surface", choices=["pial", "white"], default="white", help="Seed surface (default: white).")
    p.add_argument(
        "--centroid-seeds",
        action="store_true",
        help=(
            "Use centroid-based start points and centroid-neighborhood normals (legacy). "
            "Not compatible with --heatmap."
        ),
    )
    p.add_argument(
        "--two-way",
        action="store_true",
        help="Trace from the start surface to the opposite surface, then back again.",
    )
    p.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=False, help="Parallel path tracing.")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Worker count when --parallel.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument(
        "--allow-degenerate",
        action="store_true",
        help="Keep degenerate paths in plots/heatmaps (no filtering).",
    )

    # Start-point sampling knobs
    p.add_argument(
        "--trace-spacing",
        type=float,
        default=DEFAULT_START_SPACING_TRACE,
        help="Target geodesic spacing between start points for line/path tracing (larger=faster).",
    )
    p.add_argument(
        "--heatmap-spacing",
        type=float,
        default=DEFAULT_START_SPACING_HEATMAP,
        help="Target geodesic spacing between start points for heatmap sampling (smaller=denser/slower).",
    )

    if expert:
        g = p.add_argument_group("Expert options")

        # Tracing mechanics
        g.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Fixed step size for tracing.")
        g.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="Maximum tracing iterations.")
        g.add_argument("--first-step", type=float, default=DEFAULT_FIRST_STEP, help="Seed step length in tracer.")
        g.add_argument("--debug", action="store_true", help="Verbose tracing debug output.")

        # Start-point sampling safety valve
        g.add_argument("--start-max-points", type=int, default=None, help="Max number of start points (None=unlimited).")

        # BEM (only override defaults in expert mode)
        dcfg = BEMConfig()
        g.add_argument("--quad-order", type=int, choices=[1, 3, 7], default=dcfg.quad_order)
        g.add_argument("--tau-near", type=float, default=dcfg.TAU_NEAR)
        g.add_argument("--tol-near", type=float, default=dcfg.TOL_NEAR)
        g.add_argument("--max-subdiv", type=int, default=dcfg.MAX_SUBDIV)

        # VTK array names
        g.add_argument("--bc-type-array", type=str, default="bc_type", help="CellData array name for bc_type.")
        g.add_argument("--bc-value-array", type=str, default="bc_value", help="CellData array name for bc_value.")

        # Heatmap tuning
        g.add_argument("--heatmap-pct", type=float, default=DEFAULT_HEATMAP_PCT)
        g.add_argument("--heatmap-lower-bound", type=float, default=DEFAULT_HEATMAP_LOWER_BOUND)
        g.add_argument("--heatmap-power", type=float, default=DEFAULT_HEATMAP_POWER)
        g.add_argument("--heatmap-k-neigh", type=int, default=DEFAULT_HEATMAP_K_NEIGH)

    return p


def _degenerate_reasons(
    length: float,
    meta: dict[str, Any],
    tri_by_idx: dict[int, dict[str, Any]],
    start_bc_value: float,
) -> list[str]:
    reasons: list[str] = []
    if length < 0.5:
        reasons.append("short<0.5mm")
    if bool(meta.get("terminated_by_max_iter", False)):
        reasons.append("max_iter")

    exit_idx = meta.get("exit_tri_idx", None)
    if exit_idx is not None and exit_idx in tri_by_idx:
        tri = tri_by_idx[exit_idx]
        if str(tri.get("bc_type", "")).lower() == "dirichlet":
            bc_val = float(tri.get("bc_value", float("nan")))
            if np.isclose(bc_val, float(start_bc_value), atol=1e-6):
                reasons.append("exit_same_surface")

    dir_steps = int(meta.get("dir_steps", 0))
    used_prev = int(meta.get("used_prev_dir_steps", 0))
    if dir_steps > 0 and (used_prev / dir_steps) > 0.5:
        reasons.append("prev_dir>50%")

    return reasons


def _log_path_length(idx: int, length: float, reasons: list[str]) -> None:
    if reasons:
        LOG.info("Path %d: len=%.3f (degenerate: %s)", idx + 1, length, ",".join(reasons))
    else:
        LOG.info("Path %d: len=%.3f", idx + 1, length)


def _log_path_length_two_way(
    idx: int,
    length_down: float,
    length_up: float,
    label_down: str,
    label_up: str,
    reasons: list[str],
) -> None:
    def _fmt(x: float) -> str:
        return "n/a" if not np.isfinite(x) else f"{x:.3f}"

    base = f"Path {idx + 1}: {label_down}: {_fmt(length_down)}, {label_up}: {_fmt(length_up)}"
    if reasons:
        LOG.info("%s (degenerate: %s)", base, ",".join(reasons))
    else:
        LOG.info("%s", base)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        import sys
        argv = sys.argv[1:]

    expert = _has_expert_flag(argv)
    parser = build_argparser(expert=expert)

    try:
        args = parser.parse_args(argv)
    except SystemExit:
        if not expert:
            expert_only_tokens = {
                "--alpha",
                "--max-iter",
                "--first-step",
                "--debug",
                "--start-max-points",
                "--quad-order",
                "--tau-near",
                "--tol-near",
                "--max-subdiv",
                "--bc-type-array",
                "--bc-value-array",
                "--heatmap-pct",
                "--heatmap-lower-bound",
                "--heatmap-power",
                "--heatmap-k-neigh",
            }
            if any(tok in argv for tok in expert_only_tokens):
                print("\nNote: some advanced options are only available with --expert.")
        raise

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    if not bool(args.parallel):
        warnings.warn(
            "Parallel tracing is disabled (--no-parallel). Enabling --parallel can substantially reduce runtime.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        LOG.info("Parallel tracing enabled with %d worker(s).", int(args.workers))

    heatmap_mode = bool(args.heatmap)
    start_surface = str(getattr(args, "start_surface", "white")).lower()
    heatmap_surface = start_surface
    start_bc_value = 1.0 if start_surface == "pial" else 0.0
    direction_down = "down" if start_surface == "pial" else "up"
    centroid_seeds = bool(getattr(args, "centroid_seeds", False))
    if centroid_seeds and heatmap_mode:
        parser.error("--centroid-seeds cannot be used with --heatmap (heatmap uses vertex-based grading).")
    two_way = bool(getattr(args, "two_way", False))
    if two_way and heatmap_mode:
        parser.error("--two-way cannot be used with --heatmap (heatmap uses vertex-based grading).")

    # Expert-only VTK array names
    bc_type_array = getattr(args, "bc_type_array", "bc_type")
    bc_value_array = getattr(args, "bc_value_array", "bc_value")

    # Tracer parameters
    alpha = float(getattr(args, "alpha", DEFAULT_ALPHA))
    max_iter = int(getattr(args, "max_iter", DEFAULT_MAX_ITER))
    first_step = float(getattr(args, "first_step", DEFAULT_FIRST_STEP))
    debug = bool(getattr(args, "debug", False))
    allow_degenerate = bool(getattr(args, "allow_degenerate", False))

    heatmap_pct = float(getattr(args, "heatmap_pct", DEFAULT_HEATMAP_PCT))
    if heatmap_mode and (heatmap_pct <= 0.0 or heatmap_pct > 100.0):
        parser.error("--heatmap-pct must be in (0, 100].")

    # In heatmap mode, sampling coverage follows --heatmap-pct.
    start_pct = heatmap_pct if heatmap_mode else DEFAULT_START_PCT_TRACE
    start_spacing = float(args.heatmap_spacing) if heatmap_mode else float(args.trace_spacing)

    # Expert-only safety valve
    start_max_points = getattr(args, "start_max_points", None)

    # BEM config:
    # - non-expert: EXACT library defaults
    # - expert: allow overrides
    cfg = BEMConfig()
    if expert and hasattr(args, "quad_order"):
        cfg = BEMConfig(
            quad_order=int(args.quad_order),
            TAU_NEAR=float(args.tau_near),
            TOL_NEAR=float(args.tol_near),
            MAX_SUBDIV=int(args.max_subdiv),
        )

    LOG.info("Loading mesh -> triangles from VTK.")
    triangles = load_triangles_from_vtk_polydata(
        args.mesh,
        bc_type_array=bc_type_array,
        bc_value_array=bc_value_array,
        normal_array="normal"
    )
    preprocess_triangles(triangles)

    if centroid_seeds:
        LOG.info("Using centroid-based seed points (legacy).")
    else:
        LOG.info("Building surface vertex normals for seed step.")
        surface_vertices, surface_normals = build_surface_vertex_normals(
            triangles,
            bc_value=float(start_bc_value),
        )

    LOG.info("Assembling + solving BEM system.")
    q, _, _ = assemble_and_solve(triangles, cfg)

    seed_mode = "centroid" if centroid_seeds else "vertex"
    LOG.info(
        "Selecting start points (surface=%s, pct=%.1f, spacing=%.3f, mode=%s).",
        start_surface,
        float(start_pct),
        float(start_spacing),
        seed_mode,
    )
    seed_face_idx = None
    if centroid_seeds:
        start_pts_pool, seed_face_idx = pick_even_start_points(
            triangles,
            pct=float(start_pct),
            target_spacing=float(start_spacing),
            max_points=start_max_points,
            bc_value=float(start_bc_value),
            seed_mode=seed_mode,
            return_indices=True,
        )
        seed_face_idx = np.asarray(seed_face_idx, dtype=int)
    else:
        start_pts_pool = pick_even_start_points(
            triangles,
            pct=float(start_pct),
            target_spacing=float(start_spacing),
            max_points=start_max_points,
            bc_value=float(start_bc_value),
            seed_mode=seed_mode,
        )
    start_pts_pool = np.asarray(start_pts_pool, dtype=float)
    num_paths = int(start_pts_pool.shape[0])

    seed_dirs = None
    if (not centroid_seeds) and num_paths > 0:
        coord_to_idx = {tuple(v): i for i, v in enumerate(surface_vertices)}
        start_idx = np.full(num_paths, -1, dtype=int)
        missing = []
        for i, pt in enumerate(start_pts_pool):
            idx = coord_to_idx.get(tuple(np.asarray(pt, dtype=np.float64)))
            if idx is None:
                missing.append(i)
            else:
                start_idx[i] = idx

        if missing:
            from scipy.spatial import KDTree

            kd = KDTree(surface_vertices)
            missing_idx = np.asarray(missing, dtype=int)
            _, nn_idx = kd.query(start_pts_pool[missing_idx])
            start_idx[missing_idx] = np.asarray(nn_idx, dtype=int)
            LOG.warning(
                "Mapped %d/%d start points to nearest vertex (exact match not found).",
                len(missing),
                num_paths,
            )

        seed_dirs = -1.0 * surface_normals[start_idx]

    LOG.info("Tracing %d paths.", num_paths)

    tri_by_idx = {int(tri.get("unknown_index", i)): tri for i, tri in enumerate(triangles)}
    tri_index_map = {int(tri.get("unknown_index", i)): i for i, tri in enumerate(triangles)}
    degenerate_flags = [False] * num_paths
    degenerate_reasons = [[] for _ in range(num_paths)]

    all_paths_down: list[list[np.ndarray] | None] = [None] * num_paths
    all_lengths_down = np.zeros(num_paths, dtype=float)
    all_paths_up: list[list[np.ndarray] | None] | None = None
    all_lengths_up: np.ndarray | None = None

    trimmed_top_tris: list[dict[str, Any]] | None = None
    trimmed_top_vals: np.ndarray | None = None
    trimmed_bottom_tris: list[dict[str, Any]] | None = None
    heatmap_result = None
    lengths_for_log: np.ndarray | None = None

    if num_paths > 0:
        # Warm-up (down only)
        _ = run_single_path(
            0,
            triangles=triangles,
            start_pt=start_pts_pool[0],
            q=q,
            cfg=cfg,
            direction_down=direction_down,
            max_iter=max_iter,
            alpha_initial=alpha,
            first_step=first_step,
            seed_dir=None if seed_dirs is None else seed_dirs[0],
            seed_face_idx=None if seed_face_idx is None else int(seed_face_idx[0]),
            debug=False,
        )

        if heatmap_mode:
            sample_thick = np.zeros(num_paths, dtype=float)

            if bool(args.parallel):
                with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
                    futures = [
                        pool.submit(
                            run_single_path,
                            i,
                            triangles=triangles,
                            start_pt=start_pts_pool[i],
                            q=q,
                            cfg=cfg,
                            direction_down=direction_down,
                            max_iter=max_iter,
                            alpha_initial=alpha,
                            first_step=first_step,
                            seed_dir=None if seed_dirs is None else seed_dirs[i],
                            seed_face_idx=None if seed_face_idx is None else int(seed_face_idx[i]),
                            debug=debug,
                        )
                        for i in range(num_paths)
                    ]
                    for fut in as_completed(futures):
                        idx, _p_down, len_down, meta = fut.result()
                        sample_thick[idx] = len_down
                        reasons = _degenerate_reasons(len_down, meta, tri_by_idx, start_bc_value)
                        if reasons:
                            degenerate_flags[idx] = True
                            degenerate_reasons[idx] = reasons
            else:
                for i in range(num_paths):
                    idx, _p_down, len_down, meta = run_single_path(
                        i,
                        triangles=triangles,
                        start_pt=start_pts_pool[i],
                        q=q,
                        cfg=cfg,
                        direction_down=direction_down,
                        max_iter=max_iter,
                        alpha_initial=alpha,
                        first_step=first_step,
                        seed_dir=None if seed_dirs is None else seed_dirs[i],
                        seed_face_idx=None if seed_face_idx is None else int(seed_face_idx[i]),
                        debug=debug,
                    )
                    sample_thick[idx] = len_down
                    reasons = _degenerate_reasons(len_down, meta, tri_by_idx, start_bc_value)
                    if reasons:
                        degenerate_flags[idx] = True
                        degenerate_reasons[idx] = reasons
            lengths_for_log = sample_thick

            heatmap_lower_bound = float(getattr(args, "heatmap_lower_bound", DEFAULT_HEATMAP_LOWER_BOUND))
            heatmap_power = float(getattr(args, "heatmap_power", DEFAULT_HEATMAP_POWER))
            heatmap_k_neigh = int(getattr(args, "heatmap_k_neigh", DEFAULT_HEATMAP_K_NEIGH))

            if allow_degenerate:
                valid_mask = np.ones(num_paths, dtype=bool)
            else:
                valid_mask = ~np.asarray(degenerate_flags, dtype=bool)
                if np.any(~valid_mask):
                    LOG.warning(
                        "Removed %d/%d degenerate paths from heatmap.",
                        int(np.sum(~valid_mask)),
                        int(num_paths),
                    )
                if not np.any(valid_mask):
                    warnings.warn(
                        "All paths were degenerate; skipping heatmap rendering.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    start_pts_pool = start_pts_pool[valid_mask]
                    sample_thick = sample_thick[valid_mask]

            if heatmap_pct > 50.0:
                warnings.warn(
                    f"--heatmap-pct={heatmap_pct:.1f} keeps faces near patch edges; "
                    "accuracy is typically less reliable close to patch boundaries.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            if np.any(valid_mask):
                from laplace_thickness_release.viz.heatmap import compute_thickness_heatmap

                heatmap_result = compute_thickness_heatmap(
                    triangles=triangles,
                    sample_coords=start_pts_pool,
                    sample_thickness=sample_thick,
                    pct=heatmap_pct,
                    lower_bound=heatmap_lower_bound,
                    power=heatmap_power,
                    k_neigh=heatmap_k_neigh,
                    heatmap_surface=heatmap_surface,
                    logger=LOG,
                )
                trimmed_top_tris = heatmap_result.trimmed_top_tris
                trimmed_top_vals = heatmap_result.trimmed_top_vals
                trimmed_bottom_tris = heatmap_result.trimmed_bottom_tris

        else:
            # Normal tracing mode
            if two_way:
                direction_up = "up" if direction_down == "down" else "down"
                start_bc_value_up = 1.0 if np.isclose(start_bc_value, 0.0) else 0.0
                all_paths_up = [None] * num_paths
                all_lengths_up = np.full(num_paths, np.nan, dtype=float)

                if bool(args.parallel):
                    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
                        futures = [
                            pool.submit(
                                run_two_way_path,
                                i,
                                triangles=triangles,
                                start_pt=start_pts_pool[i],
                                q=q,
                                cfg=cfg,
                                direction_down=direction_down,
                                direction_up=direction_up,
                                max_iter=max_iter,
                                alpha_initial=alpha,
                                first_step=first_step,
                                seed_dir=None if seed_dirs is None else seed_dirs[i],
                                seed_face_idx=None if seed_face_idx is None else int(seed_face_idx[i]),
                                tri_index_map=tri_index_map,
                                centroid_seeds=centroid_seeds,
                                debug=debug,
                            )
                            for i in range(num_paths)
                        ]
                        for fut in as_completed(futures):
                            (
                                idx,
                                p_down,
                                len_down,
                                meta_down,
                                p_up,
                                len_up,
                                meta_up,
                            ) = fut.result()
                            all_lengths_down[idx] = len_down
                            all_lengths_up[idx] = len_up

                            reasons_down = _degenerate_reasons(len_down, meta_down, tri_by_idx, start_bc_value)
                            reasons = [f"down:{r}" for r in reasons_down]
                            if p_up is None or not np.isfinite(len_up):
                                reasons_up = ["no_return_exit"]
                            else:
                                reasons_up = _degenerate_reasons(len_up, meta_up, tri_by_idx, start_bc_value_up)
                            if reasons_up:
                                reasons.extend([f"up:{r}" for r in reasons_up])
                            if reasons:
                                degenerate_flags[idx] = True
                                degenerate_reasons[idx] = reasons
                                if not allow_degenerate:
                                    continue
                            all_paths_down[idx] = p_down
                            all_paths_up[idx] = p_up
                else:
                    for i in range(num_paths):
                        (
                            idx,
                            p_down,
                            len_down,
                            meta_down,
                            p_up,
                            len_up,
                            meta_up,
                        ) = run_two_way_path(
                            i,
                            triangles=triangles,
                            start_pt=start_pts_pool[i],
                            q=q,
                            cfg=cfg,
                            direction_down=direction_down,
                            direction_up=direction_up,
                            max_iter=max_iter,
                            alpha_initial=alpha,
                            first_step=first_step,
                            seed_dir=None if seed_dirs is None else seed_dirs[i],
                            seed_face_idx=None if seed_face_idx is None else int(seed_face_idx[i]),
                            tri_index_map=tri_index_map,
                            centroid_seeds=centroid_seeds,
                            debug=debug,
                        )
                        all_lengths_down[idx] = len_down
                        all_lengths_up[idx] = len_up

                        reasons_down = _degenerate_reasons(len_down, meta_down, tri_by_idx, start_bc_value)
                        reasons = [f"down:{r}" for r in reasons_down]
                        if p_up is None or not np.isfinite(len_up):
                            reasons_up = ["no_return_exit"]
                        else:
                            reasons_up = _degenerate_reasons(len_up, meta_up, tri_by_idx, start_bc_value_up)
                        if reasons_up:
                            reasons.extend([f"up:{r}" for r in reasons_up])
                        if reasons:
                            degenerate_flags[idx] = True
                            degenerate_reasons[idx] = reasons
                            if not allow_degenerate:
                                continue
                        all_paths_down[idx] = p_down
                        all_paths_up[idx] = p_up
                lengths_for_log = None
            else:
                if bool(args.parallel):
                    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
                        futures = [
                            pool.submit(
                                run_single_path,
                                i,
                                triangles=triangles,
                                start_pt=start_pts_pool[i],
                                q=q,
                                cfg=cfg,
                                direction_down=direction_down,
                                max_iter=max_iter,
                                alpha_initial=alpha,
                                first_step=first_step,
                                seed_dir=None if seed_dirs is None else seed_dirs[i],
                                seed_face_idx=None if seed_face_idx is None else int(seed_face_idx[i]),
                                debug=debug,
                            )
                            for i in range(num_paths)
                        ]
                        for fut in as_completed(futures):
                            idx, p_down, len_down, meta = fut.result()
                            reasons = _degenerate_reasons(len_down, meta, tri_by_idx, start_bc_value)
                            all_lengths_down[idx] = len_down
                            if reasons:
                                degenerate_flags[idx] = True
                                degenerate_reasons[idx] = reasons
                                if not allow_degenerate:
                                    continue
                            all_paths_down[idx] = p_down
                else:
                    for i in range(num_paths):
                        idx, p_down, len_down, meta = run_single_path(
                            i,
                            triangles=triangles,
                            start_pt=start_pts_pool[i],
                            q=q,
                            cfg=cfg,
                            direction_down=direction_down,
                            max_iter=max_iter,
                            alpha_initial=alpha,
                            first_step=first_step,
                            seed_dir=None if seed_dirs is None else seed_dirs[i],
                            seed_face_idx=None if seed_face_idx is None else int(seed_face_idx[i]),
                            debug=debug,
                        )
                        reasons = _degenerate_reasons(len_down, meta, tri_by_idx, start_bc_value)
                        all_lengths_down[idx] = len_down
                        if reasons:
                            degenerate_flags[idx] = True
                            degenerate_reasons[idx] = reasons
                            if not allow_degenerate:
                                continue
                        all_paths_down[idx] = p_down
                lengths_for_log = all_lengths_down

        if not heatmap_mode and (not allow_degenerate) and any(degenerate_flags):
            LOG.warning(
                "Removed %d/%d degenerate paths from plot.",
                int(sum(degenerate_flags)),
                int(num_paths),
            )
        if lengths_for_log is not None:
            for i in range(num_paths):
                _log_path_length(i, float(lengths_for_log[i]), degenerate_reasons[i])
        elif two_way and all_lengths_up is not None:
            label_down = direction_down.title()
            label_up = ("up" if direction_down == "down" else "down").title()
            for i in range(num_paths):
                _log_path_length_two_way(
                    i,
                    float(all_lengths_down[i]),
                    float(all_lengths_up[i]),
                    label_down,
                    label_up,
                    degenerate_reasons[i],
                )

    # ---------------- PLOTTING ----------------
    if bool(args.plot) and num_paths > 0:
        if heatmap_mode and heatmap_result is not None:
            try:
                from laplace_thickness_release.viz.plotting import plot_heatmap_pyvista
                heatmap_screenshot = getattr(args, "heatmap_screenshot", None)
                heatmap_window_size = getattr(args, "heatmap_window_size", None)
                heatmap_image_scale = int(getattr(args, "heatmap_image_scale", 1))
                heatmap_shadows = bool(getattr(args, "heatmap_shadows", False))
                heatmap_azimuths = getattr(args, "heatmap_azimuths", None)
                heatmap_elevations = getattr(args, "heatmap_elevations", None)
                heatmap_clim = getattr(args, "heatmap_clim", None)
                if heatmap_clim is not None:
                    heatmap_clim = (float(heatmap_clim[0]), float(heatmap_clim[1]))
                    if not (heatmap_clim[0] < heatmap_clim[1]):
                        raise ValueError("--heatmap-clim requires MIN < MAX.")
                camera_views = None

                if heatmap_screenshot and (heatmap_azimuths or heatmap_elevations):
                    azimuths = heatmap_azimuths if heatmap_azimuths else [0.0]
                    elevations = heatmap_elevations if heatmap_elevations else [30.0]
                    base = Path(heatmap_screenshot)
                    camera_views = []
                    view_idx = 1
                    for el in elevations:
                        for az in azimuths:
                            if len(azimuths) * len(elevations) == 1:
                                out_path = base
                            else:
                                out_path = base.with_name(f"{base.stem}_view{view_idx:02d}{base.suffix}")
                            camera_views.append((float(az), float(el), str(out_path)))
                            view_idx += 1

                plot_heatmap_pyvista(
                    V_top=heatmap_result.V_top,
                    F_top=heatmap_result.F_top,
                    top_vertex_vals=heatmap_result.top_vertex_vals,
                    keep_mask_top=heatmap_result.keep_mask_top,
                    V_bottom=heatmap_result.V_bottom,
                    F_bottom=heatmap_result.F_bottom,
                    keep_mask_bottom=heatmap_result.keep_mask_bottom,
                    cmap="hot",
                    clim=heatmap_clim,
                    screenshot_path=str(heatmap_screenshot) if heatmap_screenshot else None,
                    window_size=heatmap_window_size if heatmap_screenshot else None,
                    image_scale=heatmap_image_scale if heatmap_screenshot else 1,
                    camera_views=camera_views,
                    disable_shadows=not heatmap_shadows,
                )
                return 0
            except Exception as exc:
                LOG.warning(
                    "PyVista heatmap rendering failed (%s). Falling back to Matplotlib.",
                    exc,
                )

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_axis_off()

        if heatmap_mode and trimmed_top_tris is not None and trimmed_top_vals is not None:
            heatmap_clim = getattr(args, "heatmap_clim", None)
            if heatmap_clim is not None:
                heatmap_clim = (float(heatmap_clim[0]), float(heatmap_clim[1]))
                if not (heatmap_clim[0] < heatmap_clim[1]):
                    raise ValueError("--heatmap-clim requires MIN < MAX.")
            if heatmap_surface == "pial":
                poly_top = plot_bumpy_top(
                    ax,
                    trimmed_top_tris,
                    face_vals=trimmed_top_vals,
                    cmap="hot",
                    clim=heatmap_clim,
                )
                poly_bot = plot_bumpy_bottom(ax, trimmed_bottom_tris or triangles)
            else:
                poly_bot = plot_bumpy_bottom(
                    ax,
                    trimmed_top_tris,
                    face_vals=trimmed_top_vals,
                    cmap="hot",
                    clim=heatmap_clim,
                )
                poly_top = plot_bumpy_top(ax, trimmed_bottom_tris or triangles)
        else:
            poly_top = plot_bumpy_top(ax, triangles)
            poly_bot = plot_bumpy_bottom(ax, triangles)

            paths_to_plot = [p for p in all_paths_down if p is not None]
            if two_way and all_paths_up is not None:
                paths_to_plot.extend([p for p in all_paths_up if p is not None])

            for path_down in paths_to_plot:
                arr = np.asarray(path_down, dtype=float)
                ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], marker="o", alpha=1.0)

        # Checkbox controls
        rax = plt.axes([0.02, 0.4, 0.12, 0.12])
        labels = ["Top surface", "Bottom surface"]
        visibility = [poly_top.get_visible(), poly_bot.get_visible()]
        check = CheckButtons(rax, labels, visibility)

        def toggle_surfaces(label: str):
            if label.startswith("Top"):
                poly_top.set_visible(not poly_top.get_visible())
            else:
                poly_bot.set_visible(not poly_bot.get_visible())
            plt.draw()

        check.on_clicked(toggle_surfaces)

        set_axes_equal(ax)
        ax.legend()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
