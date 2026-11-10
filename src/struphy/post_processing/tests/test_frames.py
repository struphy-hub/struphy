"""Frames of videos of FEEC fields (struphy.post_processing.frames) from a short run on a torus."""

import shutil

import numpy as np
import pytest
from matplotlib.image import imread
from maybempi import MPI

from struphy import BaseUnits, DerhamOptions, EnvironmentOptions, Output, Simulation, Time, domains, grids
from struphy.initial import perturbations
from struphy.post_processing.frames import (
    frame_steps,
    frames_to_video,
    saved_times,
    write_3d_frames,
    write_slice_frames,
)

N_SAVED = 4  # saved time steps of the run (the initial one and three time steps)


def short_torus_run(out_folder) -> Output:
    """A Maxwell run of three time steps on a coarse torus, with an m = 1, n = 1 electric field (2-form)."""
    from struphy.models import Maxwell

    model = Maxwell(base_units=BaseUnits())
    model.em_fields.e_field.save_data = True
    model.em_fields.b_field.save_data = True
    sim = Simulation(
        model=model,
        params_path=__file__,
        env=EnvironmentOptions(out_folders=out_folder, sim_folder="sim_frames"),
        time_opts=Time(dt=0.05, Tend=(N_SAVED - 1) * 0.05),
        domain=domains.HollowTorus(a1=0.2, a2=1.0, R0=3.0, tor_period=1),
        grid=grids.TensorProductGrid(num_elements=(4, 8, 4)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
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
    # points on the axis only: unchanged
    assert off_grid_azimuth(np.zeros((3, 3)), center, 0.3) == 0.3
