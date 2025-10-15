
from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING

class MockComm:
    
    @classmethod
    def Get_rank():
        return 0
    
    @classmethod
    def Get_size():
        return 1


@dataclass
class MockMPI:
    COMM_WORLD = MockComm
    
    def Wtime(self):
        return time()


class MPIwrapper:
    def __init__(self, use_mpi: bool = False):
        if use_mpi:
            try:
                from mpi4py import mpi as MPI 
            except ModuleNotFoundError:
                MPI = MockMPI()
        else:
            MPI = MockMPI()
            
        self._MPI = MPI
        
    @property
    def MPI(self):
        return self._MPI
    

# TODO: add environment variable for mpi use
mpi_wrapper = MPIwrapper()
    

    
    
from typing import TYPE_CHECKING

# TYPE_CHECKING is True when type checking (e.g., mypy), but False at runtime.
# This allows us to use autocompletion for xp (i.e., numpy/cupy) as if numpy was imported.
if TYPE_CHECKING:
    from mpi4py import MPI
    mpi = MPI
else:
    mpi = mpi_wrapper.MPI
