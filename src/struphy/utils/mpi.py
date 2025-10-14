
from dataclasses import dataclass
from time import time


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
mpi = mpi_wrapper.MPI       

    
    
    