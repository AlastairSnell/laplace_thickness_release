from __future__ import annotations

import argparse
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons

from laplace_thickness_release.bem.core import BEMConfig, assemble_and_solve
from laplace_thickness_release.mesh.preprocess import preprocess_triangles
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
DEFAULT_START_PCT_TRACE = 0.1
DEFAULT_START_PCT_HEATMAP = 50.0
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

# ---------------- ProcessPool worker state ----------------
_WORKER_TRIANGLES = None
_WORKER_Q = None
_WORKER_CFG_KW = None
_WORKER_DO_UP = None
_WORKER_MAX_ITER = None
_WORKER_ALPHA = None
_WORKER_FIRST_STEP = None
_WORKER_DEBUG = None


def _init_worker(
    triangles,
    q,
    cfg_kwargs: dict[str, Any],
    do_up: bool,
    max_iter: int,
    alpha: float,
    first_step: float,
    debug: bool,
):
    """
    Runs once per worker process (Windows spawn-safe).
    Stores heavy objects (triangles, q) in process-local globals so we do not
    pickle them for every submitted job.
    """
    global _WORKER_TRIANGLES, _WORKER_Q, _WORKER_CFG_KW
    global _WORKER_DO_UP, _WORKER_MAX_ITER, _WORKER_ALPHA, _WORKER_FIRST_STEP, _WORKER_DEBUG

    _WORKER_TRIANGLES = triangles
    _WORKER_Q = q
    _WORKER_CFG_KW = cfg_kwargs

    _WORKER_DO_UP = do_up
    _WORKER_MAX_ITER = max_iter
    _WORKER_ALPHA = alpha
    _WORKER_FIRST_STEP = first_step
    _WORKER_DEBUG = debug


def _run_single_path_worker(idx: int, start_pt: np.ndarray):
    """
    ProcessPool target: uses process-local globals set by _init_worker.
    """
    cfg = BEMConfig(**_WORKER_CFG_KW)  # reconstruct in the worker (avoids pickling issues)

    return run_single_path(
        idx,
        triangles=_WORKER_TRIANGLES,
        start_pt=start_pt,
        q=_WORKER_Q,
        cfg=cfg,
        do_up=bool(_WORKER_DO_UP),
        max_iter=int(_WORKER_MAX_ITER),
        alpha_initial=float(_WORKER_ALPHA),
        first_step=float(_WORKER_FIRST_STEP),
        debug=bool(_WORKER_DEBUG),
    )
# --------------------------------------------------------------------------


def load_triangles_from_vtk_polydata(
    mesh_path: Path,
    *,
    bc_type_array: str = "bc_type",
    bc_value_array: str = "bc_value",
) -> list[dict[str, Any]]:
    """
    Load a triangular surface mesh from .vtp (XML PolyData) or legacy .vtk (PolyData),
    and convert to the legacy triangles list-of-dicts structure used by the BEM/tracing code.

    Requirements on the VTK file:
      - PolyData surface
      - CellData arrays:
          * bc_type (string or numeric; numeric assumed 0=dirichlet, 1=neumann)
          * bc_value (float)
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    from vtkmodules.vtkFiltersCore import vtkPolyDataNormals, vtkTriangleFilter
    from vtkmodules.vtkIOLegacy import vtkPolyDataReader
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

    suffix = mesh_path.suffix.lower()
    if suffix == ".vtp":
        reader = vtkXMLPolyDataReader()
    elif suffix == ".vtk":
        reader = vtkPolyDataReader()
    else:
        raise ValueError(
            f"Unsupported mesh extension '{suffix}'. Use .vtk or .vtp (PolyData)."
        )

    reader.SetFileName(str(mesh_path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None:
        raise RuntimeError(f"VTK reader produced no output for {mesh_path}")

    tri_f = vtkTriangleFilter()
    tri_f.SetInputData(poly)
    tri_f.Update()
    tri_poly = tri_f.GetOutput()

    n_f = vtkPolyDataNormals()
    n_f.SetInputData(tri_poly)
    n_f.ComputePointNormalsOff()
    n_f.ComputeCellNormalsOn()
    n_f.SplittingOff()
    n_f.ConsistencyOn()
    n_f.AutoOrientNormalsOn()
    n_f.Update()
    tri_poly = n_f.GetOutput()

    points = tri_poly.GetPoints()
    if points is None:
        raise RuntimeError("Mesh has no points.")

    cell_data = tri_poly.GetCellData()
    if cell_data is None:
        raise RuntimeError("Mesh has no CellData.")

    bc_type = cell_data.GetAbstractArray(bc_type_array)
    if bc_type is None:
        bc_type = cell_data.GetArray(bc_type_array)
    if bc_type is None:
        raise RuntimeError(f"Missing required CellData array '{bc_type_array}'")

    bc_val = cell_data.GetArray(bc_value_array)
    if bc_val is None:
        raise RuntimeError(f"Missing required CellData array '{bc_value_array}'")

    normals = cell_data.GetArray("Normals")
    if normals is None:
        raise RuntimeError(
            "Expected CellData 'Normals' after vtkPolyDataNormals, but not found."
        )

    n_cells = tri_poly.GetNumberOfCells()
    if n_cells <= 0:
        raise RuntimeError("No cells in mesh.")

    triangles: list[dict[str, Any]] = []

    for cid in range(n_cells):
        cell = tri_poly.GetCell(cid)
        if cell is None or cell.GetNumberOfPoints() != 3:
            continue

        pid0 = cell.GetPointId(0)
        pid1 = cell.GetPointId(1)
        pid2 = cell.GetPointId(2)

        v0 = np.asarray(points.GetPoint(pid0), dtype=np.float64)
        v1 = np.asarray(points.GetPoint(pid1), dtype=np.float64)
        v2 = np.asarray(points.GetPoint(pid2), dtype=np.float64)

        try:
            bt = bc_type.GetValue(cid)  # type: ignore[attr-defined]
        except Exception:
            bt_num = float(bc_type.GetTuple1(cid))  # type: ignore[call-arg]
            bt = "dirichlet" if int(bt_num) == 0 else "neumann"

        bv = float(bc_val.GetTuple1(cid))
        n = np.asarray(normals.GetTuple3(cid), dtype=np.float64)

        triangles.append(
            {
                "vertices": np.stack([v0, v1, v2], axis=0),
                "bc_type": str(bt).lower(),
                "bc_value": bv,
                "normal": n / max(float(np.linalg.norm(n)), 1e-16),
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
    cfg: Any,
    do_up: bool,
    max_iter: int,
    alpha_initial: float,
    first_step: float,
    debug: bool,
):
    path_down, len_down = path_trace_simple_bem(
        start_pt,
        triangles,
        q,
        cfg,
        direction="down",
        max_iter=max_iter,
        alpha_initial=alpha_initial,
        first_step=first_step,
        debug=debug,
    )

    if do_up:
        bottom_pt = np.asarray(path_down[-1], dtype=float)
        path_up, len_up = path_trace_simple_bem(
            bottom_pt,
            triangles,
            q,
            cfg,
            direction="up",
            max_iter=max_iter,
            alpha_initial=alpha_initial,
            first_step=first_step,
            debug=debug,
        )
    else:
        path_up, len_up = [], 0.0

    return idx, path_down, float(len_down), path_up, float(len_up)


def _has_expert_flag(argv: Sequence[str]) -> bool:
    return ("--expert" in argv)


def build_argparser(*, expert: bool) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trace_bem_main",
        description=(
            "Trace gradient-flow paths in a BEM-computed Laplace field (VTK input).\n"
            "Use --expert to reveal advanced tuning options."
        ),
    )

    p.add_argument(
        "--expert",
        action="store_true",
        help="Show/enable expert options in --help.",
    )
    p.add_argument("--mesh", type=Path, required=True, help="Surface mesh (.vtk or .vtp PolyData).")

    p.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable 3D plot.",
    )
    p.add_argument(
        "--heatmap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute and overlay a geodesic-IDW thickness heatmap on the top surface.",
    )
    p.add_argument(
        "--up",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable up-tracing (default: disabled).",
    )

    p.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable parallel execution (recommended for speed on most machines).",
    )

    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Thread pool size (when --parallel).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # -------- BASIC sampling knobs (clear separation) --------
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

        # BEM
        g.add_argument("--quad-order", type=int, choices=[1, 3, 7], default=3)
        g.add_argument("--tau-near", type=float, default=0.2)
        g.add_argument("--tol-near", type=float, default=1e-10)
        g.add_argument("--max-subdiv", type=int, default=3)

        # VTK array names
        g.add_argument("--bc-type-array", type=str, default="bc_type", help="CellData array name for bc_type.")
        g.add_argument("--bc-value-array", type=str, default="bc_value", help="CellData array name for bc_value.")

        # Heatmap tuning
        g.add_argument("--heatmap-pct", type=float, default=DEFAULT_HEATMAP_PCT)
        g.add_argument("--heatmap-lower-bound", type=float, default=DEFAULT_HEATMAP_LOWER_BOUND)
        g.add_argument("--heatmap-power", type=float, default=DEFAULT_HEATMAP_POWER)
        g.add_argument("--heatmap-k-neigh", type=int, default=DEFAULT_HEATMAP_K_NEIGH)

    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        import sys
        argv = sys.argv[1:]

    # Only --expert is valid now
    expert = _has_expert_flag(argv)
    parser = build_argparser(expert=expert)

    # If non-expert, fail gracefully when they try expert-only flags.
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

    # Configure logging early so LOG.* messages are visible
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    # Parallelism warnings/info
    if not bool(args.parallel):
        warnings.warn(
            "Parallel tracing is disabled (--no-parallel). For typical meshes, enabling --parallel "
            "can substantially reduce runtime.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        LOG.info("Parallel tracing enabled with %d worker(s).", int(args.workers))

    heatmap_mode = bool(args.heatmap)
    if heatmap_mode and bool(args.up):
        LOG.info("--heatmap enabled: ignoring --up (heatmap uses down-tracing only).")

    # Expert-only defaults / values
    bc_type_array = getattr(args, "bc_type_array", "bc_type")
    bc_value_array = getattr(args, "bc_value_array", "bc_value")

    alpha = getattr(args, "alpha", DEFAULT_ALPHA)
    max_iter = getattr(args, "max_iter", DEFAULT_MAX_ITER)
    first_step = getattr(args, "first_step", DEFAULT_FIRST_STEP)
    debug = bool(getattr(args, "debug", False))

    # Policy: start-pct is NOT user-configurable.
    start_pct = DEFAULT_START_PCT_HEATMAP if heatmap_mode else DEFAULT_START_PCT_TRACE

    # C) trace spacing is only ignored when heatmap is flagged:
    #    - normal mode: use --trace-spacing
    #    - heatmap mode: use --heatmap-spacing
    start_spacing = float(args.heatmap_spacing) if heatmap_mode else float(args.trace_spacing)

    # Expert-only safety valve
    start_max_points = getattr(args, "start_max_points", None)

    # BEM config: expert-only overrides
    cfg = BEMConfig(
        quad_order=int(getattr(args, "quad_order", 3)),
        TAU_NEAR=float(getattr(args, "tau_near", 0.2)),
        TOL_NEAR=float(getattr(args, "tol_near", 1e-10)),
        MAX_SUBDIV=int(getattr(args, "max_subdiv", 3)),
    )

    LOG.info("Loading mesh -> triangles from VTK.")
    triangles = load_triangles_from_vtk_polydata(
        args.mesh,
        bc_type_array=bc_type_array,
        bc_value_array=bc_value_array,
    )
    preprocess_triangles(triangles)

    LOG.info("Assembling + solving BEM system (SLP + K' with Neumann jump term).")
    q, _, _ = assemble_and_solve(triangles, cfg)

    LOG.info("Selecting start points (pct=%.1f, spacing=%.3f).", float(start_pct), float(start_spacing))
    start_pts_pool = pick_even_start_points(
        triangles,
        pct=float(start_pct),
        target_spacing=float(start_spacing),
        max_points=start_max_points,
    )
    start_pts_pool = np.asarray(start_pts_pool, dtype=float)
    num_paths = int(start_pts_pool.shape[0])

    LOG.info("Tracing %d paths.", num_paths)

    # Storage for normal tracing mode
    all_paths_down: list[list[np.ndarray] | None] = [None] * num_paths
    all_paths_up: list[list[np.ndarray] | None] = [None] * num_paths
    all_lengths_down = np.zeros(num_paths, dtype=float)
    all_lengths_up = np.zeros(num_paths, dtype=float)

    # Storage for heatmap mode
    trimmed_top_tris: list[dict[str, Any]] | None = None
    trimmed_top_vals: np.ndarray | None = None
    trimmed_bottom_tris: list[dict[str, Any]] | None = None

    if num_paths > 0:
        # Warm-up (down only; avoids first-call overhead surprises)
        _ = run_single_path(
            0,
            triangles=triangles,
            start_pt=start_pts_pool[0],
            q=q,
            cfg=cfg,
            do_up=False,
            max_iter=int(max_iter),
            alpha_initial=float(alpha),
            first_step=float(first_step),
            debug=False,
        )

        if heatmap_mode:
            # HEATMAP MODE:
            #  - only need down lengths (no lines plotted, no up-tracing)
            #  - start-spacing controlled by --heatmap-spacing
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
                            do_up=False,
                            max_iter=int(max_iter),
                            alpha_initial=float(alpha),
                            first_step=float(first_step),
                            debug=debug,
                        )
                        for i in range(num_paths)
                    ]
                    for fut in as_completed(futures):
                        idx, _p_down, len_down, _p_up, _len_up = fut.result()
                        sample_thick[idx] = len_down
            else:
                for i in range(num_paths):
                    idx, _p_down, len_down, _p_up, _len_up = run_single_path(
                        i,
                        triangles=triangles,
                        start_pt=start_pts_pool[i],
                        q=q,
                        cfg=cfg,
                        do_up=False,
                        max_iter=int(max_iter),
                        alpha_initial=float(alpha),
                        first_step=float(first_step),
                        debug=debug,
                    )
                    sample_thick[idx] = len_down

            # Heatmap tuning (expert-only overrides)
            heatmap_pct = float(getattr(args, "heatmap_pct", DEFAULT_HEATMAP_PCT))
            heatmap_lower_bound = float(getattr(args, "heatmap_lower_bound", DEFAULT_HEATMAP_LOWER_BOUND))
            heatmap_power = float(getattr(args, "heatmap_power", DEFAULT_HEATMAP_POWER))
            heatmap_k_neigh = int(getattr(args, "heatmap_k_neigh", DEFAULT_HEATMAP_K_NEIGH))

            if heatmap_pct > 50.0:
                warnings.warn(
                    f"--heatmap-pct={heatmap_pct:.1f} keeps faces near patch edges; "
                    "accuracy is typically less reliable close to patch boundaries.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            from laplace_thickness_release.viz.heatmap import compute_thickness_heatmap

            # IMPORTANT: compute_thickness_heatmap must now return BOTH:
            #   - trimmed_top_tris: inner {heatmap_pct}% pial faces
            #   - trimmed_top_vals: interpolated thickness values for those faces
            #   - trimmed_bottom_tris: matching inner {heatmap_pct}% white faces
            trimmed_top_tris, trimmed_top_vals, trimmed_bottom_tris = compute_thickness_heatmap(
                triangles=triangles,
                sample_coords=start_pts_pool,
                sample_thickness=sample_thick,
                pct=heatmap_pct,
                lower_bound=heatmap_lower_bound,
                power=heatmap_power,
                k_neigh=heatmap_k_neigh,
                logger=LOG,
            )

        else:
            # NORMAL TRACING MODE (no heatmap):
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
                            do_up=bool(args.up),
                            max_iter=int(max_iter),
                            alpha_initial=float(alpha),
                            first_step=float(first_step),
                            debug=debug,
                        )
                        for i in range(num_paths)
                    ]
                    for fut in as_completed(futures):
                        idx, p_down, len_down, p_up, len_up = fut.result()
                        all_paths_down[idx] = p_down
                        all_lengths_down[idx] = len_down
                        all_paths_up[idx] = p_up
                        all_lengths_up[idx] = len_up
                        LOG.info("Path %d: down=%.3f, up=%.3f", idx + 1, len_down, len_up)
            else:
                for i in range(num_paths):
                    idx, p_down, len_down, p_up, len_up = run_single_path(
                        i,
                        triangles=triangles,
                        start_pt=start_pts_pool[i],
                        q=q,
                        cfg=cfg,
                        do_up=bool(args.up),
                        max_iter=int(max_iter),
                        alpha_initial=float(alpha),
                        first_step=float(first_step),
                        debug=debug,
                    )
                    all_paths_down[idx] = p_down
                    all_lengths_down[idx] = len_down
                    all_paths_up[idx] = p_up
                    all_lengths_up[idx] = len_up
                    LOG.info("Path %d: down=%.3f, up=%.3f", idx + 1, len_down, len_up)

    # ---------------- PLOTTING ----------------
    if bool(args.plot) and num_paths > 0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        if heatmap_mode and trimmed_top_tris is not None and trimmed_top_vals is not None:
            # HEATMAP PLOT:
            #  - only the inner {heatmap_pct}% of the pial patch is shown
            #  - only the matching inner {heatmap_pct}% of the white patch is shown
            #  - no tracing lines

            ax.set_axis_off()

            # Top (pial) patch with heatmap
            poly_top = plot_bumpy_top(
                ax,
                trimmed_top_tris,
                face_vals=trimmed_top_vals,
                cmap="hot",
            )

            # Bottom (white) matching patch
            if trimmed_bottom_tris:
                poly_bot = plot_bumpy_bottom(ax, trimmed_bottom_tris)
            else:
                # Safety fallback: if trimming failed for any reason, show full bottom.
                poly_bot = plot_bumpy_bottom(ax, triangles)

        else:
            ax.set_axis_off()

            # STANDARD PLOT: full surfaces + path lines
            poly_top = plot_bumpy_top(ax, triangles)
            poly_bot = plot_bumpy_bottom(ax, triangles)

            paths_down = [p for p in all_paths_down if p is not None]
            paths_up = [p for p in all_paths_up if p is not None]

            for i, path_down in enumerate(paths_down):
                arr = np.asarray(path_down, dtype=float)
                ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], marker="o", alpha=1.0, label=f"Down {i + 1}")

            if bool(args.up):
                for i, path_up in enumerate(paths_up):
                    if not path_up:
                        continue
                    arr = np.asarray(path_up, dtype=float)
                    ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], marker="x", alpha=1.0, label=f"Up {i + 1}")

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
