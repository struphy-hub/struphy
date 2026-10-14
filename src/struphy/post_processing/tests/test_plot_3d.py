"""Loading and rendering the VTK files of a run in struphy.post_processing.plot_3d."""

import numpy as np
import pytest
import pyvista as pv

from struphy.post_processing.plot_3d import is_torus, load_field, plot_field_3d, vtk_files


def write_run(path_out, n=6):
    """A fake run with the VTK files of pproc(create_vtk=True, physical=True): a scalar ``phi`` and a vector field
    ``e`` (logical components in vtk/, Cartesian ones in vtk_phy/) on a cylinder, at two time steps."""
    r, theta, z = np.meshgrid(np.linspace(0.1, 1.0, n), np.linspace(0.0, 2 * np.pi, n), np.linspace(0.0, 1.0, n))
    grid = pv.StructuredGrid(r * np.cos(theta), r * np.sin(theta), z)
    x, y, zz = grid.points.T
    for physical in (False, True):
        folder = path_out / "post_processing" / "fields_data" / "em_fields" / ("vtk_phy" if physical else "vtk")
        folder.mkdir(parents=True)
        for step in (0, 1, 10):
            data = grid.copy()
            data.point_data["phi"] = step * np.exp(-((x - 0.5) ** 2) - y**2 - zz**2)
            for k, comp in enumerate((x, 2 * y, 0 * zz), start=1):
                data.point_data[f"e_{k}"] = step * comp if physical else np.ones_like(comp)
            data.save(folder / f"step_{step}.vts")


@pytest.mark.mpi_skip
def test_vtk_files_ordered_by_step(tmp_path):
    write_run(tmp_path)
    assert [f.name for f in vtk_files(tmp_path)] == ["step_0.vts", "step_1.vts", "step_10.vts"]
    with pytest.raises(FileNotFoundError):
        vtk_files(tmp_path, species="kinetic_ions")


@pytest.mark.mpi_skip
def test_load_scalar_and_vector_magnitude(tmp_path):
    write_run(tmp_path)
    grid, name = load_field(tmp_path, "phi", step=-1)
    assert name == "phi" and np.isclose(np.max(grid.point_data["phi"]), 10 * np.exp(0.0), rtol=0.5)

    grid, name = load_field(tmp_path, "e", step=1)
    x, y, _ = grid.points.T
    assert name == "|e|"
    assert np.allclose(grid.point_data["|e|"], np.hypot(x, 2 * y))

    with pytest.raises(KeyError):
        load_field(tmp_path, "b")


@pytest.mark.mpi_skip
@pytest.mark.parametrize("variable", ["phi", "e"])
def test_render_to_file(tmp_path, variable):
    write_run(tmp_path)
    filename = tmp_path / f"{variable}_3d.png"
    plot_field_3d(tmp_path, variable, filename=filename)
    assert filename.stat().st_size > 0


@pytest.mark.mpi_skip
def test_is_torus_without_simulation_data(tmp_path):
    # no simulation data: guessed from the grid, a cylinder reaching close to its axis is no torus
    write_run(tmp_path)
    grid, _ = load_field(tmp_path, "phi")
    assert not is_torus(tmp_path, grid)
    # the same grid moved away from the axis (radius + 3): a ring
    ring = grid.copy()
    x, y, z = ring.points.T
    radius = np.hypot(x, y)
    ring.points = np.column_stack([(radius + 3.0) * x / radius, (radius + 3.0) * y / radius, z])
    assert is_torus(tmp_path, ring)
