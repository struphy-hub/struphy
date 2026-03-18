import logging


from dataclasses import dataclass
from typing import Tuple

import numpy as np

from struphy.utils.utils import __dataclass_repr_no_defaults__, all_class_params_are_default

logger = logging.getLogger("struphy")

@dataclass
class TensorProductGrid:
    """Grid as a tensor product of 1d grids.

    Parameters
    ----------
    Nel : Tuple[int, int, int]
        Number of elements in each direction.

    mpi_dims_mask: Tuple[bool, bool, bool]
        True if the dimension is to be used in the domain decomposition (=default for each dimension).
        If mpi_dims_mask[i]=False, the i-th dimension will not be decomposed.
    """

    Nel: Tuple[int, int, int] = (24, 10, 1)
    mpi_dims_mask: Tuple[bool, bool, bool] = (True, True, True)

    def __str__(self):
        for k, v in self.__dict__.items():
            logger.info(f"{k}:".ljust(20), v)
        return ""

    def __repr_no_defaults__(self):
        return __dataclass_repr_no_defaults__(self)

    @property
    def is_default(self):
        return all_class_params_are_default(self)

    def to_dict(self) -> dict:
        dct = {
            "Nel": self.Nel,
            "mpi_dims_mask": self.mpi_dims_mask,
        }
        return dct

    @classmethod
    def from_dict(cls, dct) -> "TensorProductGrid":
        return cls(
            Nel=tuple(dct["Nel"]),
            mpi_dims_mask=tuple(dct["mpi_dims_mask"]),
        )
