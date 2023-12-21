import pytest
from struphy.main import main
from struphy.post_processing import pproc_struphy

import os
import struphy
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

libpath = struphy.__path__[0]
i_path = os.path.join(libpath, 'io', 'inp')
o_path = os.path.join(libpath, 'io', 'out')

# These tests must be run without MPI (is called in subprocess) !!

@pytest.mark.mpi(min_size=2)
def test_tutorial_02():
    main('LinearMHDVlasovCC', 
         os.path.join(i_path, 'tutorials', 'params_02.yml'), 
         os.path.join(o_path, 'tutorial_02'))

@pytest.mark.mpi(min_size=2)
def test_tutorial_03():
    main('LinearMHD', 
         os.path.join(i_path, 'tutorials', 'params_03.yml'), 
         os.path.join(o_path, 'tutorial_03'))

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, 'tutorial_03'))

@pytest.mark.mpi(min_size=2)
def test_tutorial_04():
    main('Maxwell', 
         os.path.join(i_path, 'tutorials', 'params_04a.yml'), 
         os.path.join(o_path, 'tutorial_04a'))

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, 'tutorial_04a'))
    
    main('LinearMHD', 
         os.path.join(i_path, 'tutorials', 'params_04b.yml'), 
         os.path.join(o_path, 'tutorial_04b'))

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, 'tutorial_04b'))

@pytest.mark.mpi(min_size=2)
def test_tutorial_05():
    main('Vlasov', 
         os.path.join(i_path, 'tutorials', 'params_05a.yml'), 
         os.path.join(o_path, 'tutorial_05a'))

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, 'tutorial_05a'))
    
    main('DriftKinetic', 
         os.path.join(i_path, 'tutorials', 'params_05b.yml'), 
         os.path.join(o_path, 'tutorial_05b'))

    comm.Barrier()
    if rank == 0:
        pproc_struphy.main(os.path.join(o_path, 'tutorial_05b'))
