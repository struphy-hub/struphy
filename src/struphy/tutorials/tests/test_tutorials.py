import os

import pytest
import yaml
from psydac.ddm.mpi import mpi as MPI

import struphy
from struphy.post_processing import pproc_struphy
from struphy.struphy import run

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

libpath = struphy.__path__[0]
i_path = os.path.join(libpath, "io", "inp")
o_path = os.path.join(libpath, "io", "out")


def test_tutorial_02():
    run(
        "LinearMHDVlasovCC",
        os.path.join(i_path, "tutorials", "params_02.yml"),
        os.path.join(o_path, "tutorial_02"),
        supress_out=True,
    )


def test_tutorial_03():
    run(
        "LinearMHD",
        os.path.join(i_path, "tutorials", "params_03.yml"),
        os.path.join(o_path, "tutorial_03"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_03"), physical=True)


def test_tutorial_04(fast):
    run(
        "Maxwell",
        os.path.join(i_path, "tutorials", "params_04a.yml"),
        os.path.join(o_path, "tutorial_04a"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_04a"))

    run(
        "LinearMHD",
        os.path.join(i_path, "tutorials", "params_04b.yml"),
        os.path.join(o_path, "tutorial_04b"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_04b"))

    if not fast:
        run(
            "VariationalMHD",
            os.path.join(i_path, "tutorials", "params_04c.yml"),
            os.path.join(o_path, "tutorial_04c"),
            supress_out=True,
        )

        comm.Barrier()
        if rank == 0:
            pproc_struphy.main(os.path.join(o_path, "tutorial_04c"))


def test_tutorial_05():
    run(
        "Vlasov",
        os.path.join(i_path, "tutorials", "params_05a.yml"),
        os.path.join(o_path, "tutorial_05a"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_05a"))

    run(
        "Vlasov",
        os.path.join(i_path, "tutorials", "params_05b.yml"),
        os.path.join(o_path, "tutorial_05b"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_05b"))

    run(
        "GuidingCenter",
        os.path.join(i_path, "tutorials", "params_05c.yml"),
        os.path.join(o_path, "tutorial_05c"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_05c"))

    run(
        "GuidingCenter",
        os.path.join(i_path, "tutorials", "params_05d.yml"),
        os.path.join(o_path, "tutorial_05d"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_05d"))

    run(
        "GuidingCenter",
        os.path.join(i_path, "tutorials", "params_05e.yml"),
        os.path.join(o_path, "tutorial_05e"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_05e"))

    run(
        "GuidingCenter",
        os.path.join(i_path, "tutorials", "params_05f.yml"),
        os.path.join(o_path, "tutorial_05f"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_05f"))


def test_tutorial_12():
    run(
        "Vlasov",
        os.path.join(i_path, "tutorials", "params_12a.yml"),
        os.path.join(o_path, "tutorial_12a"),
        save_step=100,
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_12a"))

    run(
        "GuidingCenter",
        os.path.join(i_path, "tutorials", "params_12b.yml"),
        os.path.join(o_path, "tutorial_12b"),
        save_step=10,
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_12b"))


if __name__ == "__main__":
    test_tutorial_04(True)
