# from tqdm import tqdm
import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import numpy_to_vtk as np2vtk
from vtkmodules.util.numpy_support import vtk_to_numpy as vtk2np
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid

from struphy.utils.arrays import xp


def make_ugrid_and_write_vtu(filename: str, writer, vtk_dir, gvec, s_range, u_range, v_range, periodic):
    """A helper function to orchestrate operations to run many test cases.

    This is not needed in practice.

    Parameters
    ----------
    filename : str
        Filename to write the ParaView file.
    writer : vtkWriter
        A `vtkWriter` class from `writer.paraview.vtk_writer`.
    vtk_dir : str
        Directory to store the output ParaView files.
    gvec : gvec_to_python.GVEC_functions.GVEC
        A wrapper class that maps logical coordinates (s,u,v) to Cartesian (x,y,z), among other things, such as computing MHD variables.
    s_range : numpy.ndarray
        Range of logical radial coordinates to transform into Cartesian vertices.
    u_range : numpy.ndarray
        Range of logical poloidal coordinates to transform into Cartesian vertices.
    v_range : numpy.ndarray
        Range of logical toroidal coordinates to transform into Cartesian vertices.
    periodic : boolean
        Whether the mesh is a periodic structure.
    """

    # Generate one set of data, then write them in ParaView files as using different graphics primitives.
    num_pts = s_range.shape[0] * u_range.shape[0] * v_range.shape[0]
    print("Number of points: {}".format(num_pts), flush=True)
    point_data = {}
    cell_data = {}
    vtk_points, suv_points, xyz_points, point_indices = gen_vtk_points(
        gvec, s_range, u_range, v_range, point_data, cell_data
    )
    print("vtk_points.GetNumberOfPoints()", vtk_points.GetNumberOfPoints(), flush=True)

    ugrid = setup_ugrid(vtk_points, num_pts)
    connect_cell(s_range, u_range, v_range, point_indices, ugrid, point_data, cell_data, periodic)
    set_data(ugrid, point_data, cell_data)
    writer.write(vtk_dir, filename, ugrid)
    # vtk_render(ugrid)


def gen_vtk_points(gvec, s_range, u_range, v_range, point_data, cell_data):
    """Generate vertices for `vtkUnstructuredGrid`.

    Parameters
    ----------
    gvec : gvec_to_python.GVEC_functions.GVEC
        A wrapper class that maps logical coordinates (s,u,v) to Cartesian (x,y,z), among other things, such as computing MHD variables.
    s_range : numpy.ndarray
        Range of logical radial coordinates to transform into Cartesian vertices.
    u_range : numpy.ndarray
        Range of logical poloidal coordinates to transform into Cartesian vertices.
    v_range : numpy.ndarray
        Range of logical toroidal coordinates to transform into Cartesian vertices.
    point_data : dict
        A dictionary of arrays to store data assoicated with each point/vertex.
    cell_data : dict
        A dictionary of arrays to store data assoicated with each cell in the mesh.

    Returns
    -------
    vtk_points : vtk.vtkPoints
        Vertices.
    suv_points : numpy.ndarray
        Associated (s,u,v) coordinate, indexed with the index of the (s,u,v) coordinate that generated that point.
    xyz_points : numpy.ndarray
        Associated Cartesian coordinate of each (s,u,v), indexed with the index of the (s,u,v) coordinate that generated that point.
    point_indices : numpy.ndarray
        Associated index of each `vtk_points`, indexed with the index of the (s,u,v) coordinate that generated that point.
    """

    pt_idx = 0
    vtk_points = vtk.vtkPoints()
    suv_points = xp.zeros((s_range.shape[0], u_range.shape[0], v_range.shape[0], 3))
    xyz_points = xp.zeros((s_range.shape[0], u_range.shape[0], v_range.shape[0], 3))
    point_indices = xp.zeros((s_range.shape[0], u_range.shape[0], v_range.shape[0]), dtype=xp.int_)

    # Add metadata to grid.
    num_pts = s_range.shape[0] * u_range.shape[0] * v_range.shape[0]
    point_data["s"] = xp.zeros(num_pts, dtype=xp.float_)
    point_data["u"] = xp.zeros(num_pts, dtype=xp.float_)
    point_data["v"] = xp.zeros(num_pts, dtype=xp.float_)
    point_data["x"] = xp.zeros(num_pts, dtype=xp.float_)
    point_data["y"] = xp.zeros(num_pts, dtype=xp.float_)
    point_data["z"] = xp.zeros(num_pts, dtype=xp.float_)
    point_data["theta"] = xp.zeros(num_pts, dtype=xp.float_)
    point_data["zeta"] = xp.zeros(num_pts, dtype=xp.float_)
    point_data["Point ID"] = xp.zeros(num_pts, dtype=xp.int_)
    point_data["pressure"] = xp.zeros(num_pts, dtype=xp.float_)
    point_data["phi"] = xp.zeros(num_pts, dtype=xp.float_)
    point_data["chi"] = xp.zeros(num_pts, dtype=xp.float_)
    point_data["iota"] = xp.zeros(num_pts, dtype=xp.float_)
    point_data["q"] = xp.zeros(num_pts, dtype=xp.float_)
    point_data["det"] = xp.zeros(num_pts, dtype=xp.float_)
    point_data["det/(2pi)^2"] = xp.zeros(num_pts, dtype=xp.float_)
    point_data["A"] = xp.zeros((num_pts, 3), dtype=xp.float_)
    point_data["A_vec"] = xp.zeros((num_pts, 3), dtype=xp.float_)
    point_data["A_1"] = xp.zeros((num_pts, 3), dtype=xp.float_)
    point_data["A_2"] = xp.zeros((num_pts, 3), dtype=xp.float_)
    point_data["B"] = xp.zeros((num_pts, 3), dtype=xp.float_)
    point_data["B_vec"] = xp.zeros((num_pts, 3), dtype=xp.float_)
    point_data["B_1"] = xp.zeros((num_pts, 3), dtype=xp.float_)
    point_data["B_2"] = xp.zeros((num_pts, 3), dtype=xp.float_)

    # pbar = tqdm(total=num_pts)
    for s_idx, s in enumerate(s_range):
        for u_idx, u in enumerate(u_range):
            for v_idx, v in enumerate(v_range):
                point = gvec.f(s, u, v)
                suv_points[s_idx, u_idx, v_idx, :] = xp.array([s, u, v])
                xyz_points[s_idx, u_idx, v_idx, :] = point
                point_indices[s_idx, u_idx, v_idx] = pt_idx
                vtk_points.InsertPoint(pt_idx, point)
                # vtk_points.InsertNextPoint(i, i, i)

                # Coordinates that correspond to each point.
                point_data["s"][pt_idx] = s
                point_data["u"][pt_idx] = u
                point_data["v"][pt_idx] = v
                point_data["x"][pt_idx] = point[0]
                point_data["y"][pt_idx] = point[1]
                point_data["z"][pt_idx] = point[2]
                point_data["Point ID"][pt_idx] = pt_idx
                point_data["pressure"][pt_idx] = gvec.P(s, u, v)
                point_data["phi"][pt_idx] = gvec.PHI(s, u, v)
                point_data["chi"][pt_idx] = gvec.CHI(s, u, v)
                point_data["iota"][pt_idx] = gvec.IOTA(s, u, v)
                point_data["det"][pt_idx] = gvec.df_det(s, u, v)
                point_data["A"][pt_idx] = gvec.A(s, u, v)
                point_data["A_vec"][pt_idx] = gvec.A_vec(s, u, v)
                point_data["A_1"][pt_idx] = gvec.A_1(s, u, v)
                point_data["A_2"][pt_idx] = gvec.A_2(s, u, v)
                point_data["B"][pt_idx] = gvec.B(s, u, v)  # TODO: if s > 1e-4: ...
                point_data["B_vec"][pt_idx] = gvec.B_vec(s, u, v)
                point_data["B_1"][pt_idx] = gvec.B_1(s, u, v)
                point_data["B_2"][pt_idx] = gvec.B_2(s, u, v)

                # pbar.update(1)
                pt_idx += 1

    # pbar.close()
    point_data["theta"] = 2 * xp.pi * point_data["u"]
    point_data["zeta"] = 2 * xp.pi * point_data["v"]
    point_data["q"] = 1 / point_data["iota"]
    point_data["det/(2pi)^2"] = point_data["det"] / (2 * xp.pi) ** 2

    return vtk_points, suv_points, xyz_points, point_indices


def setup_ugrid(pts, num_pts):
    """Associate vertices/points with a new `vtkUnstructuredGrid`.

    Parameters
    ----------
    pts : vtk.vtkPoints
        Cartesian coordinates of each vertex that is used to construct an unstructured grid.
    num_pts : int
        Number of vertices.

    Returns
    -------
    ugrid : vtk.vtkUnstructuredGrid
        An unstructured grid with vertices associated.
    """

    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(pts)
    ugrid.Allocate(num_pts)

    return ugrid


def set_data(ugrid, point_data, cell_data):
    """Associate point and cell data with an `vtkUnstructuredGrid`.

    Parameters
    ----------
    ugrid : vtk.vtkUnstructuredGrid
        An unstructured grid.
    point_data : dict
        A dictionary of arrays to store data assoicated with each point/vertex.
    cell_data : dict
        A dictionary of arrays to store data assoicated with each cell in the mesh.
    """

    # Getting the VTK data storage object.
    vtk_point_data = ugrid.GetPointData()
    vtk_cell_data = ugrid.GetCellData()

    # For each numpy data array, convert it into VTK formay, set its name, and add to grid.
    for i, (k, v) in enumerate(point_data.items()):
        vtk_array = np2vtk(v)
        vtk_array.SetName(k)
        vtk_point_data.AddArray(vtk_array)

    for i, (k, v) in enumerate(cell_data.items()):
        vtk_array = np2vtk(v)
        vtk_array.SetName(k)
        vtk_cell_data.AddArray(vtk_array)


def vtk_render(ugrid):  # pragma: no cover
    """Opens an interactive window that renders the current `vtkUnstructuredGrid`.

    Parameters
    ----------
    ugrid : vtk.vtkUnstructuredGrid
        An unstructured grid.
    """

    colors = vtk.vtkNamedColors()

    renderer = vtk.vtkRenderer()

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(renderer)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    ugridMapper = vtk.vtkDataSetMapper()
    ugridMapper.SetInputData(ugrid)

    ugridActor = vtk.vtkActor()
    ugridActor.SetMapper(ugridMapper)
    ugridActor.GetProperty().SetColor(colors.GetColor3d("Peacock"))
    ugridActor.GetProperty().EdgeVisibilityOn()
    ugridActor.GetProperty().SetOpacity(0.8)

    renderer.AddActor(ugridActor)
    renderer.SetBackground(colors.GetColor3d("Beige"))

    renderer.ResetCamera()
    renderer.GetActiveCamera().Elevation(60.0)
    renderer.GetActiveCamera().Azimuth(30.0)
    renderer.GetActiveCamera().Dolly(1.0)

    renWin.SetSize(640, 480)
    renWin.SetWindowName("UGrid")

    # Interact with the data.
    renWin.Render()

    iren.Start()


# ============================================================
# Connect vertices to form primitives
# e.g. points, lines, quads, cells.
# ============================================================


def connect_cell(s_range, u_range, v_range, point_indices, ugrid, point_data, cell_data, periodic):
    """Create (initialize) cells of a `vtkUnstructuredGrid` using connectivity of its vertices.

    Inserted cells are of type `vtk.VTK_HEXAHEDRON`. Connected cells form the volume of a torus.

    Parameters
    ----------
    s_range : numpy.ndarray
        Range of logical radial coordinates that was used to transform into Cartesian vertices.
    u_range : numpy.ndarray
        Range of logical poloidal coordinates that was used to transform into Cartesian vertices.
    v_range : numpy.ndarray
        Range of logical toroidal coordinates that was used to transform into Cartesian vertices.
    point_indices : numpy.ndarray
        Associated index of each `vtk_points`, indexed with the index of the (s,u,v) coordinate that generated that point.
    ugrid : vtk.vtkUnstructuredGrid
        An unstructured grid.
    point_data : dict
        (Unused) A dictionary of arrays to store data assoicated with each point/vertex.
    cell_data : dict
        (Unused) A dictionary of arrays to store data assoicated with each cell in the mesh.
    periodic : 3-tuple of bool
        Whether each direction is periodic.
        e.g. Connect a torus in poloidal and toroidal directions if periodic==[False,True,True].
    """

    cell_idx = 0
    cell_data["Cell ID"] = []

    len_s, len_u, len_v = s_range.shape[0], u_range.shape[0], v_range.shape[0]

    for s_idx, s in enumerate(s_range):
        for u_idx, u in enumerate(u_range):
            for v_idx, v in enumerate(v_range):
                if (
                    (periodic[0] or s_idx + 1 < len_s)
                    and (periodic[1] or u_idx + 1 < len_u)
                    and (periodic[2] or v_idx + 1 < len_v)
                ):
                    vertex1 = point_indices[s_idx, u_idx, v_idx]
                    vertex2 = point_indices[s_idx, (u_idx + 1) % len_u, v_idx]
                    vertex3 = point_indices[s_idx, (u_idx + 1) % len_u, (v_idx + 1) % len_v]
                    vertex4 = point_indices[s_idx, u_idx, (v_idx + 1) % len_v]
                    vertex5 = point_indices[(s_idx + 1), u_idx, v_idx]
                    vertex6 = point_indices[(s_idx + 1), (u_idx + 1) % len_u, v_idx]
                    vertex7 = point_indices[(s_idx + 1), (u_idx + 1) % len_u, (v_idx + 1) % len_v]
                    vertex8 = point_indices[(s_idx + 1), u_idx, (v_idx + 1) % len_v]

                    connected_idx = [vertex1, vertex2, vertex3, vertex4, vertex5, vertex6, vertex7, vertex8]
                    ugrid.InsertNextCell(vtk.VTK_HEXAHEDRON, len(connected_idx), connected_idx)
                    cell_data["Cell ID"].append(cell_idx)
                    cell_idx += 1

    cell_data["Cell ID"] = xp.array(cell_data["Cell ID"], dtype=xp.int_)
