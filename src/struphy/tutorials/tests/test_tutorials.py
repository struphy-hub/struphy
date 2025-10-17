import os

import pytest
import yaml
from psydac.ddm.mpi import mpi as MPI

import struphy
from struphy.main import main
from struphy.post_processing import pproc_struphy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

libpath = struphy.__path__[0]
i_path = os.path.join(libpath, "io", "inp")
o_path = os.path.join(libpath, "io", "out")


def test_tutorial_02():
    main(
        "LinearMHDVlasovCC",
        os.path.join(i_path, "tutorials", "params_02.yml"),
        os.path.join(o_path, "tutorial_02"),
        supress_out=True,
    )


def test_tutorial_03():
    main(
        "LinearMHD",
        os.path.join(i_path, "tutorials", "params_03.yml"),
        os.path.join(o_path, "tutorial_03"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_03"), physical=True)


def test_tutorial_04(fast):
    main(
        "Maxwell",
        os.path.join(i_path, "tutorials", "params_04a.yml"),
        os.path.join(o_path, "tutorial_04a"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_04a"))

    main(
        "LinearMHD",
        os.path.join(i_path, "tutorials", "params_04b.yml"),
        os.path.join(o_path, "tutorial_04b"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_04b"))

    if not fast:
        main(
            "VariationalMHD",
            os.path.join(i_path, "tutorials", "params_04c.yml"),
            os.path.join(o_path, "tutorial_04c"),
            supress_out=True,
        )

        comm.Barrier()
        if rank == 0:
            pproc_struphy.main(os.path.join(o_path, "tutorial_04c"))


def test_tutorial_05():
    main(
        "Vlasov",
        os.path.join(i_path, "tutorials", "params_05a.yml"),
        os.path.join(o_path, "tutorial_05a"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_05a"))

    main(
        "Vlasov",
        os.path.join(i_path, "tutorials", "params_05b.yml"),
        os.path.join(o_path, "tutorial_05b"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_05b"))

    main(
        "GuidingCenter",
        os.path.join(i_path, "tutorials", "params_05c.yml"),
        os.path.join(o_path, "tutorial_05c"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_05c"))

    main(
        "GuidingCenter",
        os.path.join(i_path, "tutorials", "params_05d.yml"),
        os.path.join(o_path, "tutorial_05d"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_05d"))

    main(
        "GuidingCenter",
        os.path.join(i_path, "tutorials", "params_05e.yml"),
        os.path.join(o_path, "tutorial_05e"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_05e"))

    main(
        "GuidingCenter",
        os.path.join(i_path, "tutorials", "params_05f.yml"),
        os.path.join(o_path, "tutorial_05f"),
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_05f"))


def test_tutorial_12():
    main(
        "Vlasov",
        os.path.join(i_path, "tutorials", "params_12a.yml"),
        os.path.join(o_path, "tutorial_12a"),
        save_step=100,
        supress_out=True,
    )

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, "tutorial_12a"))

    main(
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
