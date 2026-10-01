"""MPI tests for running a Simulation on a user-supplied communicator.

Run with::

    mpirun -n 2 pytest --with-mpi src/struphy/simulation/tests/test_sim_comm_mpi.py
"""

import os
import shutil
import tempfile

import pytest
from feectools.ddm.mpi import mpi as MPI

from struphy import EnvironmentOptions, Simulation, Time, grids
from struphy.models import Maxwell


@pytest.mark.mpi(min_size=2)
def test_simulations_on_split_comm_use_only_their_comm():
    """Two groups of COMM_WORLD run separate simulations; each must only use its own comm."""
    world = MPI.COMM_WORLD
    world_rank = world.Get_rank()

    # interleaved split: world rank 0 is not in group 1, so group 1 depends on its own rank 0
    color = world_rank % 2
    comm = world.Split(color, world_rank)

    out_folders = world.bcast(tempfile.mkdtemp() if world_rank == 0 else None, root=0)
    try:
        sim = Simulation(
            model=Maxwell(),
            env=EnvironmentOptions(out_folders=out_folders, sim_folder=f"group_{color}"),
            time_opts=Time(dt=0.05, Tend=0.05),
            grid=grids.TensorProductGrid(num_elements=(8, 4, 1)),
            comm=comm,
        )
        sim.run()

        assert sim.derham.comm.Get_size() == comm.Get_size()
        assert os.path.isdir(os.path.join(sim.env.path_out, "data"))

        sister = sim.spawn_sister()
        assert sister.comm is comm
        assert sister.rank == comm.Get_rank()
    finally:
        world.Barrier()
        comm.Free()
        if world_rank == 0:
            shutil.rmtree(out_folders, ignore_errors=True)
