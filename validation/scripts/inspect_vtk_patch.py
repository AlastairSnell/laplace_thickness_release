from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

def _vtk_array_info(arr):
    # Handles vtkDataArray and vtkAbstractArray
    if arr is None:
        return None
    name = arr.GetName()
    ncomp = arr.GetNumberOfComponents()
    ntup = arr.GetNumberOfTuples()
    cls = arr.GetClassName()
    dtype = None
    try:
        dtype = arr.GetDataTypeAsString()
    except Exception:
        pass
    return {"name": name, "class": cls, "dtype": dtype, "ncomp": ncomp, "ntuples": ntup}

def _print_arrays(title, fielddata):
    print(f"\n== {title} arrays ==")
    if fielddata is None:
        print("  (none)")
        return
    n = fielddata.GetNumberOfArrays()
    print(f"  count: {n}")
    for i in range(n):
        arr = fielddata.GetAbstractArray(i)
        info = _vtk_array_info(arr)
        print(f"  [{i}] {info}")

def _get_cell_tris(poly):
    """Return triangle cell point ids as (ntri,3) int array, skipping non-tris."""
    n_cells = poly.GetNumberOfCells()
    tris = []
    for cid in range(n_cells):
        cell = poly.GetCell(cid)
        if cell is None:
            continue
        if cell.GetNumberOfPoints() != 3:
            continue
        tris.append([cell.GetPointId(0), cell.GetPointId(1), cell.GetPointId(2)])
    if not tris:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(tris, dtype=np.int64)

def _points_to_numpy(points):
    n = points.GetNumberOfPoints()
    out = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        out[i] = points.GetPoint(i)
    return out

def _cell_normals_numpy(cell_normals, n_cells):
    # returns (n_cells,3) float, NaN for missing
    out = np.full((n_cells, 3), np.nan, dtype=np.float64)
    if cell_normals is None:
        return out
    nt = cell_normals.GetNumberOfTuples()
    # Some VTK arrays store only triangles or all cells; assume aligned with cell id when possible
    m = min(nt, n_cells)
    for i in range(m):
        out[i] = cell_normals.GetTuple3(i)
    return out

def _geom_normals_for_tris(V, F):
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    ng = np.cross(v1 - v0, v2 - v0)
    nrm = np.linalg.norm(ng, axis=1)
    ng = ng / np.maximum(nrm[:, None], 1e-16)
    return ng, nrm * 0.5  # unit normals, areas

def inspect(path_str: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)

    from vtkmodules.vtkIOLegacy import vtkPolyDataReader
    from vtkmodules.vtkFiltersCore import vtkTriangleFilter
    from vtkmodules.vtkFiltersCore import vtkPolyDataNormals

    print(f"FILE: {path}")
    print("---- Reading raw polydata (no filters) ----")
    r = vtkPolyDataReader()
    r.SetFileName(str(path))
    r.Update()
    poly_raw = r.GetOutput()
    if poly_raw is None:
        raise RuntimeError("Reader returned None output")

    print("Raw polydata:")
    print("  nPoints:", poly_raw.GetNumberOfPoints())
    print("  nCells :", poly_raw.GetNumberOfCells())

    _print_arrays("PointData", poly_raw.GetPointData())
    _print_arrays("CellData", poly_raw.GetCellData())

    # Build numpy V/F from raw
    P_raw = poly_raw.GetPoints()
    V_raw = _points_to_numpy(P_raw) if P_raw is not None else np.zeros((0, 3))
    F_raw = _get_cell_tris(poly_raw)
    print("\nRaw triangles extracted:")
    print("  nTris:", F_raw.shape[0])
    if F_raw.shape[0] > 0:
        ng_raw, area_raw = _geom_normals_for_tris(V_raw, F_raw)
        print("  area stats: min={:.6g} med={:.6g} max={:.6g}".format(
            float(np.min(area_raw)), float(np.median(area_raw)), float(np.max(area_raw))
        ))

    # Check for existing normals arrays on raw cell data
    cd_raw = poly_raw.GetCellData()
    normals_raw = None
    if cd_raw is not None:
        normals_raw = cd_raw.GetArray("Normals")
        if normals_raw is None:
            # try common names
            for nm in ("normal", "normals", "cell_normals", "face_normals"):
                normals_raw = cd_raw.GetArray(nm)
                if normals_raw is not None:
                    print(f"\nFound existing CellData normals array name='{nm}'")
                    break

    if normals_raw is not None and F_raw.shape[0] > 0:
        N_raw = _cell_normals_numpy(normals_raw, poly_raw.GetNumberOfCells())
        # compare normals only for the triangle cells we extracted
        # map triangle cell ids to row indices
        tri_cell_ids = []
        for cid in range(poly_raw.GetNumberOfCells()):
            cell = poly_raw.GetCell(cid)
            if cell is not None and cell.GetNumberOfPoints() == 3:
                tri_cell_ids.append(cid)
        tri_cell_ids = np.asarray(tri_cell_ids, dtype=np.int64)

        Ntri = N_raw[tri_cell_ids]
        Ntri = Ntri / np.maximum(np.linalg.norm(Ntri, axis=1, keepdims=True), 1e-16)

        dots = np.einsum("ij,ij->i", ng_raw, Ntri)
        bad = int(np.sum(dots < 0))
        print("\nRaw: geom normals vs stored CellData normals:")
        print(f"  dot<0 count: {bad}/{len(dots)}")
        print("  dot stats: min={:.6g} p01={:.6g} med={:.6g} p99={:.6g} max={:.6g}".format(
            float(np.min(dots)),
            float(np.quantile(dots, 0.01)),
            float(np.median(dots)),
            float(np.quantile(dots, 0.99)),
            float(np.max(dots)),
        ))
    else:
        print("\nRaw: no obvious CellData normals array found (or no tris).")

    print("\n---- Applying vtkTriangleFilter (like your loader) ----")
    tri_f = vtkTriangleFilter()
    tri_f.SetInputData(poly_raw)
    tri_f.Update()
    poly_tri = tri_f.GetOutput()
    print("After TriangleFilter:")
    print("  nPoints:", poly_tri.GetNumberOfPoints())
    print("  nCells :", poly_tri.GetNumberOfCells())
    F_tri = _get_cell_tris(poly_tri)
    print("  nTris  :", F_tri.shape[0])

    print("\n---- Applying vtkPolyDataNormals with AutoOrientNormalsOn (like your loader) ----")
    n_f = vtkPolyDataNormals()
    n_f.SetInputData(poly_tri)
    n_f.ComputePointNormalsOff()
    n_f.ComputeCellNormalsOn()
    n_f.SplittingOff()
    n_f.ConsistencyOn()
    n_f.AutoOrientNormalsOn()
    n_f.Update()
    poly_norm = n_f.GetOutput()

    print("After PolyDataNormals:")
    print("  nPoints:", poly_norm.GetNumberOfPoints())
    print("  nCells :", poly_norm.GetNumberOfCells())
    F_norm = _get_cell_tris(poly_norm)
    print("  nTris  :", F_norm.shape[0])

    # Compare raw-tri vs norm-tri geometry
    V_norm = _points_to_numpy(poly_norm.GetPoints())
    ng_norm, area_norm = _geom_normals_for_tris(V_norm, F_norm)
    print("  area stats: min={:.6g} med={:.6g} max={:.6g}".format(
        float(np.min(area_norm)), float(np.median(area_norm)), float(np.max(area_norm))
    ))

    # Pull the Normals array produced by vtkPolyDataNormals
    cd_norm = poly_norm.GetCellData()
    normals_prod = cd_norm.GetArray("Normals") if cd_norm is not None else None
    if normals_prod is None:
        print("\nNormals filter did NOT produce CellData 'Normals' array (unexpected).")
    else:
        # Compare produced normals to geom normals
        Np = _cell_normals_numpy(normals_prod, poly_norm.GetNumberOfCells())
        tri_cell_ids = np.arange(poly_norm.GetNumberOfCells(), dtype=np.int64)  # all should be tris now
        Ntri = Np[tri_cell_ids]
        Ntri = Ntri / np.maximum(np.linalg.norm(Ntri, axis=1, keepdims=True), 1e-16)

        dots = np.einsum("ij,ij->i", ng_norm, Ntri)
        bad = int(np.sum(dots < 0))
        print("\nAfter PolyDataNormals: geom normals vs produced 'Normals':")
        print(f"  dot<0 count: {bad}/{len(dots)}")
        print("  dot stats: min={:.6g} p01={:.6g} med={:.6g} p99={:.6g} max={:.6g}".format(
            float(np.min(dots)),
            float(np.quantile(dots, 0.01)),
            float(np.median(dots)),
            float(np.quantile(dots, 0.99)),
            float(np.max(dots)),
        ))

    print("\n---- Check whether TriangleFilter/Normals changed winding relative to raw ----")
    # This is approximate: we compare geometric normals from raw-tris and norm-tris only if counts match.
    if F_raw.shape[0] == F_norm.shape[0] and V_raw.shape[0] == V_norm.shape[0]:
        ng_raw2, _ = _geom_normals_for_tris(V_raw, F_raw)
        dots2 = np.einsum("ij,ij->i", ng_raw2, ng_norm)
        print("Compare geom normals (raw vs after filters):")
        print("  dot<0 count:", int(np.sum(dots2 < 0)), "/", len(dots2))
        print("  dot stats: min={:.6g} med={:.6g} max={:.6g}".format(
            float(np.min(dots2)), float(np.median(dots2)), float(np.max(dots2))
        ))
    else:
        print("  (counts differ; filters changed topology/triangulation, so direct 1:1 comparison not possible)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_vtk_patch.py <path-to-vtk>")
        raise SystemExit(2)
    inspect(sys.argv[1])
