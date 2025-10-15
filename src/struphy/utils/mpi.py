
from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING


# Might not be needed
class MPICommWrapper:
    def __init__(self, use_mpi=True):
        self.use_mpi = use_mpi
        if use_mpi:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = MockComm()

    def __getattr__(self, name):
        return getattr(self.comm, name)

class MockComm:
    def __getattr__(self, name):
        # Return a function that does nothing and returns None
        def dummy(*args, **kwargs):
            return None
        return dummy

    # Override some functions
    def Get_rank(self):
        return 0
    
    def Get_size(self):
        return 1


# class MockComm:
    
#     @classmethod
#     def Get_rank():
#         return 0
    
#     @classmethod
#     def Get_size():
#         return 1


# @dataclass
# class MockMPI:
#     COMM_WORLD = MockComm
    
#     def Wtime(self):
#         return time()


class MPIwrapper:
    def __init__(self, use_mpi: bool = False):
        self.use_mpi = use_mpi
        if use_mpi:
            from mpi4py import MPI
            self._MPI = MPI.COMM_WORLD
        else:
            self._MPI = MockMPI()

    @property
    def MPI(self):
        return self._MPI

class MockMPI:
    def __getattr__(self, name):
        # Return a function that does nothing and returns None
        def dummy(*args, **kwargs):
            return None
        return dummy

    # Override some functions
    @property
    def COMM_WORLD(self):
        return MockComm()

# TODO: add environment variable for mpi use
mpi_wrapper = MPIwrapper()
    

    
    
from typing import TYPE_CHECKING

# TYPE_CHECKING is True when type checking (e.g., mypy), but False at runtime.
if TYPE_CHECKING:
    from mpi4py import MPI
    mpi = MPI
else:
    mpi = mpi_wrapper.MPI
