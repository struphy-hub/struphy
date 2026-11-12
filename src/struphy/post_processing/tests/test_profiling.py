"""Tests of the helpers for profiling results in struphy.post_processing.profiling (no simulation needed)."""

from types import SimpleNamespace

import numpy as np
import pytest

from struphy.post_processing.profiling import Profile, solver_iterations


@pytest.mark.mpi_skip
def test_solver_iterations_from_stdout(tmp_path):
    """The iterations of every propagator are read from struphy.out, in the order of the calls."""
    (tmp_path / "struphy.out").write_text(
        "Status     for ShearAlfven: True\n"
        "Iterations for ShearAlfven: 14\n"
        "Maxdiff up for ShearAlfven: 1e-3\n"
        "Iterations for ShearAlfvenCurlCurlPropagator: 9\n"
        "Iterations for ShearAlfven: 12\n"
    )
    # a log file is ignored when struphy.out exists
    (tmp_path / "struphy.log").write_text("Iterations for ShearAlfven: 99\n")
    assert solver_iterations(tmp_path) == {"ShearAlfven": [14, 12], "ShearAlfvenCurlCurlPropagator": [9]}


@pytest.mark.mpi_skip
def test_solver_iterations_from_rotated_logs(tmp_path, monkeypatch):
    """Without struphy.out, the log file and its rotated backups are read, oldest first."""
    monkeypatch.delenv("STRUPHY_LOG_FILE", raising=False)
    (tmp_path / "struphy.log.2").write_text("Iterations for Poisson: 1\n")
    (tmp_path / "struphy.log.1").write_text("Iterations for Poisson: 2\n")
    (tmp_path / "struphy.log").write_text("Iterations for Poisson: 3\n")
    assert solver_iterations(tmp_path) == {"Poisson": [1, 2, 3]}


@pytest.mark.mpi_skip
def test_solver_iterations_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("STRUPHY_LOG_FILE", raising=False)
    assert solver_iterations(tmp_path) == {}


@pytest.mark.mpi_skip
def test_step_times_max_over_ranks(tmp_path):
    """The duration of every call of the region is the maximum over the ranks (calls cut to the shortest rank)."""
    durations = {0: [1.0, 2.0, 3.0], 1: [1.5, 1.0, 4.0, 9.0]}

    class Region:
        ranks = list(durations)

        def __getitem__(self, rank):
            return SimpleNamespace(durations=durations[rank])

    profile = Profile(tmp_path / "profiling_data.h5")
    profile.__dict__["results"] = {"model.integrate": Region()}  # the cached results of scope-profiler
    np.testing.assert_allclose(profile.step_times(), [1.5, 2.0, 4.0])
