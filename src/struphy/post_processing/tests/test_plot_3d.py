"""Loading and rendering the VTK files of a run in struphy.post_processing.plot_3d."""

import numpy as np
import pytest
import pyvista as pv

from struphy.post_processing.plot_3d import (
    TORUS_DOMAINS,
    is_torus,
    load_field,
    plot_field_3d,
    repeat_around_axis,
    vtk_file,
    vtk_files,
)


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


@pytest.mark.mpi_skip
def test_torus_domains_exist():
    """The names in TORUS_DOMAINS are domain classes of struphy (renaming one would silently break is_torus)."""
    from struphy.geometry import base, domains

    for name in TORUS_DOMAINS:
        assert hasattr(domains, name) or hasattr(base, name), name


@pytest.mark.mpi_skip
def test_select_vtk_file_by_snapshot(tmp_path):
    write_run(tmp_path)
    assert vtk_file(tmp_path, snapshot=10).name == "step_10.vts"
    assert vtk_file(tmp_path, step=1).name == "step_1.vts"
    with pytest.raises(FileNotFoundError):
        vtk_file(tmp_path, snapshot=2)
    grid, _ = load_field(tmp_path, "phi", snapshot=1)
    assert np.isclose(np.max(grid.point_data["phi"]), 1.0, rtol=0.5)


@pytest.mark.mpi_skip
def test_repeat_sector_around_axis():
    # a sixth of a ring, between the angles -60 and 0 degrees
    r, phi, z = np.meshgrid(np.linspace(3.0, 4.0, 4), np.linspace(-np.pi / 3, 0.0, 5), np.linspace(-0.5, 0.5, 3))
    sector = pv.StructuredGrid(r * np.cos(phi), r * np.sin(phi), z)
    sector.point_data["f"] = np.cos(6 * np.arctan2(sector.points[:, 1], sector.points[:, 0]))
    sector.set_active_scalars("f")
    assert repeat_around_axis(sector, 1) is sector
    full = repeat_around_axis(sector, 6)
    assert full.active_scalars_name == "f"
    # the copies share their seams, which are merged
    assert 5 * sector.n_points < full.n_points <= 6 * sector.n_points
    x0, x1, y0, y1, _, _ = full.bounds
    assert np.allclose([x0, x1, y0, y1], [-4.0, 4.0, -4.0, 4.0], atol=0.1)
    # the field of the period of the sector is continued correctly
    angles = np.arctan2(full.points[:, 1], full.points[:, 0])
    assert np.allclose(full.point_data["f"], np.cos(6 * angles), atol=1e-12)


@pytest.mark.mpi_skip
def test_load_vector_component(tmp_path):
    write_run(tmp_path)
    # the FEEC form (logical components, all ones in the fake run)
    grid, name = load_field(tmp_path, "e", step=1, component=1)
    assert name == "e_2" and np.allclose(grid.point_data["e_2"], 1.0)
    # the Cartesian component 2 y (at step 1)
    grid, name = load_field(tmp_path, "e", step=1, component=1, physical=True)
    assert name == "e_2" and np.allclose(grid.point_data["e_2"], 2 * grid.points[:, 1])
    with pytest.raises(KeyError):
        load_field(tmp_path, "phi", component=0)
