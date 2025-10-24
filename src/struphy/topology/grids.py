from dataclasses import dataclass

import numpy as np


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

    Nel: tuple = (24, 10, 1)
    mpi_dims_mask: tuple = (True, True, True)
