"""run_metadata.json records the code (Struphy version, git state) and the command line of a run."""

import json
import os
import re

import pytest
from maybempi import MPI

from struphy import DerhamOptions, EnvironmentOptions, Output, Simulation, Time, domains, grids
from struphy.models import Maxwell
from struphy.utils.utils import code_info


def _check_code_info(info: dict):
    assert set(info) == {"version", "git_commit", "git_dirty", "git_branch"}
    assert isinstance(info["version"], str) and info["version"]
    if info["git_commit"] is None:
        assert info["git_dirty"] is None and info["git_branch"] is None
    else:
        assert re.fullmatch(r"[0-9a-f]{40}", info["git_commit"])
        assert info["git_dirty"] in (True, False, None)
        assert info["git_branch"] is None or isinstance(info["git_branch"], str)


def test_run_metadata_records_code_and_command_line(mpi_out_folder: str):
    sim = Simulation(
        model=Maxwell(),
        env=EnvironmentOptions(out_folders=mpi_out_folder, sim_folder="sim_1"),
        time_opts=Time(dt=0.1, Tend=0.1),
        domain=domains.Cuboid(),
        grid=grids.TensorProductGrid(num_elements=(1, 1, 4)),
        derham_opts=DerhamOptions(degree=(1, 1, 1)),
    )
    sim.run()

    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        comm.Barrier()

    with open(os.path.join(sim.env.path_out, "run_metadata.json")) as stream:
        metadata = json.load(stream)
    assert metadata["mpi_ranks"] == comm.Get_size()
    command_line = metadata["command_line"]
    assert isinstance(command_line, list) and command_line
    assert all(isinstance(arg, str) for arg in command_line)
    _check_code_info(metadata["struphy"])

    # round trip: the new keys don't disturb restoring the run, and are exposed by Output
    restored = Simulation.from_output(sim.env.path_out)
    assert type(restored.model) is Maxwell
    out = Output(sim.env.path_out)
    assert out.code_info == metadata["struphy"]
    assert out.mpi_ranks == comm.Get_size()


@pytest.mark.mpi_skip
def test_code_info_of_installed_package():
    _check_code_info(code_info())


@pytest.mark.mpi_skip
def test_code_info_outside_git_checkout(tmp_path):
    (tmp_path / "__init__.py").write_text("")
    info = code_info(libpath=str(tmp_path))
    assert info["git_commit"] is None and info["git_dirty"] is None and info["git_branch"] is None
    assert isinstance(info["version"], str) and info["version"]


@pytest.mark.mpi_skip
def test_output_code_info_of_old_run(tmp_path):
    """Runs written before the code info was recorded give None."""
    (tmp_path / "run_metadata.json").write_text(json.dumps({"mpi_ranks": 1}))
    assert Output(tmp_path).code_info is None
