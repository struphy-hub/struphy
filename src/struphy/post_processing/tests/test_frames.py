"""Frames of videos of FEEC fields (struphy.post_processing.frames) from short runs on a torus and other domains."""

import shutil

import numpy as np
import pytest
from matplotlib.image import imread
from maybempi import MPI

from struphy import BaseUnits, DerhamOptions, EnvironmentOptions, Output, Simulation, Time, domains, grids
from struphy.initial import perturbations
from struphy.post_processing.frames import (
    _check_clim,
    _finite,
    _grid_values,
    _limits,
    frame_steps,
    frames_to_video,
    saved_times,
    write_3d_frames,
    write_slice_frames,
    write_videos,
)

N_SAVED = 4  # saved time steps of the run (the initial one and three time steps)


def short_torus_run(out_folder, polar=False, domain=None) -> Output:
    """A Maxwell run of three time steps on a coarse torus (with ``polar``, an IGAPolarTorus with polar splines, which
    has an axis; or on ``domain``), with an m = 1, n = 1 electric field (2-form)."""
    from struphy.models import Maxwell

    if domain is None:
        domain = (
            domains.IGAPolarTorus(num_elements=(4, 9), degree=(2, 2), a=1.0, R0=3.0, tor_period=1)
            if polar
            else domains.HollowTorus(a1=0.2, a2=1.0, R0=3.0, tor_period=1)
        )

    model = Maxwell(base_units=BaseUnits())
    model.em_fields.e_field.save_data = True
    model.em_fields.b_field.save_data = True
    sim = Simulation(
        model=model,
        params_path=__file__,
        env=EnvironmentOptions(out_folders=out_folder, sim_folder="sim_frames"),
        time_opts=Time(dt=0.05, Tend=(N_SAVED - 1) * 0.05),
        domain=domain,
        grid=grids.TensorProductGrid(num_elements=(4, 9 if polar else 8, 4)),
        # polar splines: clamped in eta1 (free on the axis, Dirichlet at the boundary)
        derham_opts=DerhamOptions(degree=(2, 2, 1), bcs=(("free", "dirichlet"), None, None), polar_splines=True)
        if polar
        else DerhamOptions(degree=(2, 2, 1)),
    )
    model.em_fields.e_field.add_perturbation(
        perturbations.TorusModesSin(ms=(1,), ns=(1,), amps=(1e-3,), comp=0, given_in_basis="2")
    )
    sim.run()
    return Output(sim.env.path_out)


def check_frames(files, count):
    """``count`` frames, all of the same even size, listed in frames.txt."""
    assert len(files) == count
    shapes = {imread(f).shape for f in files}
    assert len(shapes) == 1
    height, width = shapes.pop()[:2]
    assert height % 2 == 0 and width % 2 == 0
    lines = (files[0].parent / "frames.txt").read_text().splitlines()
    assert len(lines) == count + 1


def test_frames(mpi_out_folder):
    run = short_torus_run(mpi_out_folder)
    rank = MPI.COMM_WORLD.Get_rank()
    folder = run.path_out / "frames"

    # at most max_frames evenly spaced saved steps, with the first and the last one
    assert frame_steps(run) == list(range(N_SAVED))
    assert frame_steps(run, max_frames=3) == [0, 2, 3]
    assert len(saved_times(run)) == N_SAVED
    with pytest.raises(ValueError):
        frame_steps(run, max_frames=0)

    # the radial component (normalized basis) in the poloidal plane, one scale over all frames
    files = write_slice_frames(run, "em_fields/e_field", folder / "e_r_rz", component=0, points_per_cell=2)
    if rank == 0:
        check_frames(files, N_SAVED)
    else:
        assert files == []

    # the magnitude of the field in the plane z = 0 at three of the saved steps, a scale per frame
    files = write_slice_frames(run, "em_fields/e_field", folder / "e_xy", plane="xy", max_frames=3, clim="frame")
    if rank == 0:
        check_frames(files, 3)

    # 3D, both views, the Cartesian component with fixed color limits
    for view in ("cut", "domain"):
        files = write_3d_frames(
            run,
            "em_fields/e_field",
            folder / f"e_3d_{view}",
            component=0,
            view=view,
            points_per_cell=2,
            window_size=(400, 400),
        )
        if rank == 0:
            check_frames(files, N_SAVED)
    files = write_3d_frames(
        run,
        "em_fields/e_field",
        folder / "e_x_3d",
        component=0,
        basis="physical",
        clim=(-1e-3, 1e-3),
        steps=[0, N_SAVED - 1],
        points_per_cell=1,
        window_size=(300, 300),
    )
    if rank == 0:
        check_frames(files, 2)

    if rank == 0 and shutil.which("ffmpeg") is not None:
        video = frames_to_video(folder / "e_r_rz", fps=2)
        assert video.exists() and video.stat().st_size > 0

    with pytest.raises(ValueError):
        write_slice_frames(run, "em_fields/e_field", folder / "bad", clim="global")
    with pytest.raises(ValueError):
        write_3d_frames(run, "em_fields/e_field", folder / "bad", view="side")
    if rank == 0:
        with pytest.raises(ValueError):
            write_slice_frames(run, "em_fields/e_field", folder / "bad", plane="ab")
    MPI.COMM_WORLD.Barrier()


@pytest.mark.mpi_skip
def test_off_grid_azimuth():
    from struphy.post_processing.plot_3d import off_grid_azimuth

    # points on 8 vertical half planes around the z-axis (azimuths k pi/4), at two radii
    angles = np.arange(8) * np.pi / 4
    points = np.array([[r * np.cos(a), r * np.sin(a), z] for a in angles for r in (1.0, 2.0) for z in (0.0, 1.0)])
    center = np.zeros(3)
    # on a plane of points: moved to the middle of the next gap
    assert np.isclose(off_grid_azimuth(points, center, np.pi / 2), np.pi / 2 + np.pi / 8)
    # within a gap: its middle, on the branch of the given azimuth
    assert np.isclose(off_grid_azimuth(points, center, 0.1), np.pi / 8)
    assert np.isclose(off_grid_azimuth(points, center, -0.1), -np.pi / 8)
    assert np.isclose(off_grid_azimuth(points, center, 2 * np.pi + 0.1), 2 * np.pi + np.pi / 8)
    # 6 planes (every pi/3): the middle of the gap puts the other face (+ pi/2) on a plane, a quarter does not
    sixths = np.arange(6) * np.pi / 3
    points6 = np.array([[np.cos(a), np.sin(a), 0.0] for a in sixths])
    assert np.isclose(off_grid_azimuth(points6, center, 0.0), np.pi / 12)
    # points on the axis only: unchanged
    assert off_grid_azimuth(np.zeros((3, 3)), center, 0.3) == 0.3


def test_frames_polar_axis(mpi_out_folder):
    """On a torus with a polar axis, the singular values of the push-forward on the axis are replaced by those of the
    next ring, so that the color scale stays finite."""
    run = short_torus_run(mpi_out_folder, polar=True)
    etas = {"eta1": np.linspace(0, 1, 5), "eta2": np.linspace(0, 1, 9), "eta3": np.linspace(0, 1, 5)}
    for component, basis in ((0, "norm"), (None, None), (0, "physical")):
        arrays, _ = _grid_values(run, "em_fields/e_field", N_SAVED - 1, etas, component, basis)
        assert all(np.all(np.isfinite(values)) for values in arrays.values())
    files = write_slice_frames(run, "em_fields/e_field", run.path_out / "frames" / "rz", component=0, max_frames=2)
    files += write_3d_frames(
        run,
        "em_fields/e_field",
        run.path_out / "frames" / "3d",
        max_frames=2,
        points_per_cell=1,
        window_size=(300, 300),
    )
    if MPI.COMM_WORLD.Get_rank() == 0:
        assert len(files) == 4
    MPI.COMM_WORLD.Barrier()


@pytest.mark.mpi_skip
def test_frame_helpers():
    # the axis (first index) takes the values of the next ring, other non-finite values become NaN
    values = _finite([[np.inf, 1.0], [2.0, np.nan], [3.0, 4.0]])
    assert values[0, 0] == 2.0 and values[0, 1] == 1.0 and np.isnan(values[1, 1])
    # color limits: symmetric about zero for a field changing sign, never empty
    assert _limits(None, None, -1.0, 2.0, None)[1] == (-2.0, 2.0)
    assert _limits("frame", np.array([0.5, 1.0, np.nan]), -9.0, 9.0, None)[1] == (0.0, 1.0)
    assert _limits(None, None, 0.0, 0.0, None)[1] == (-1.0, 1.0)
    assert _limits(np.array([-3, 1]), None, 0.0, 1.0, "viridis") == ("viridis", (-3.0, 1.0))
    _check_clim(np.array([0.0, 1.0]))
    for clim in ("global", (1.0, 0.0), np.array([1.0, 0.0]), (0.0, 1.0, 2.0)):
        with pytest.raises(ValueError):
            _check_clim(clim)


@pytest.mark.parametrize(
    "domain",
    [domains.Cuboid(r1=2.0, r2=1.0, r3=3.0), domains.Colella(Lx=2.0, Ly=1.0, alpha=0.1, Lz=3.0)],
    ids=["slab", "colella"],
)
def test_videos_on_other_domains(mpi_out_folder, domain):
    """All standard videos (the planes xy, xz, yz and both 3D views) of a slab and of the Colella mapping."""
    run = short_torus_run(mpi_out_folder, domain=domain)
    videos = write_videos(
        run,
        "em_fields/e_field",
        run.path_out / "frames",
        component=0,
        prefix="e1",
        max_frames=2,
        options_3d={"window_size": (300, 300)},
    )
    if MPI.COMM_WORLD.Get_rank() == 0:
        assert set(videos) == {"e1_xy", "e1_xz", "e1_yz", "e1_3d_cut", "e1_3d"}
        for files in videos.values():
            check_frames(files, 2)
    MPI.COMM_WORLD.Barrier()
