import numpy as np
from dataclasses import dataclass


@dataclass
class TensorProductGrid:
    """Grid as a tensor product of 1d grids.

    Parameters
    ----------
    Nel : tuple[int]
        Number of elements in each direction.

    mpi_dims_mask: Tuple of bool
        True if the dimension is to be used in the domain decomposition (=default for each dimension).
        If mpi_dims_mask[i]=False, the i-th dimension will not be decomposed.
    """
    Nel: tuple = (16, 1, 1)
    mpi_dims_mask: tuple = (True, True, True)
