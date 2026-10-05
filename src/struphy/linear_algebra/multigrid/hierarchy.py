"""Hierarchy of nested Derham sequences for geometric multigrid."""

import logging

import numpy as np
from feectools.ddm.cart import DomainDecomposition

from struphy.feec.psydac_derham import Derham
from struphy.topology.grids import TensorProductGrid

logger = logging.getLogger("struphy")


class MultiGridHierarchy:
    r"""Nested Derham sequences :math:`V_0 \supset V_1 \supset \dots \supset V_{L-1}` obtained by
    uniform coarsening of the user's (finest) Derham.

    From one level to the next, the number of elements is halved in every direction ``i`` where this
    is possible, i.e. where

    * ``num_elements[i]`` is even,
    * the coarse grid keeps at least ``max(min_cells, degree[i] + 1)`` elements,
    * the MPI decomposition stays aligned: every process owns exactly the coarse elements covering
      its fine elements (element starts and ends+1 are even), and owns at least ``degree[i]``
      coarse elements if the direction is split among several processes.

    Directions that cannot be halved are kept (semi-coarsening). Coarsening stops when no direction
    can be halved or when ``max_levels`` is reached. All coarse Derhams share the communicator,
    the process grid, the options (degree, boundary conditions, quadrature) and the domain of the finest one.

    Parameters
    ----------
    derham : Derham
        The finest level.

    max_levels : int | None
        Maximal number of levels (including the finest); None means as many as possible.

    min_cells : int
        Minimal number of elements per direction on the coarsest level.
    """

    def __init__(self, derham: Derham, *, max_levels: int | None = None, min_cells: int = 2):
        if derham.polar_splines:
            raise NotImplementedError("Multigrid is not yet implemented for polar splines.")
        assert max_levels is None or max_levels >= 1
        assert min_cells >= 1

        self._min_cells = min_cells
        self._derhams: list[Derham] = [derham]
        self._factors: list[tuple[int, int, int]] = []

        while max_levels is None or len(self._derhams) < max_levels:
            fine = self._derhams[-1]
            factors = self._coarsening_factors(fine)
            if all(f == 1 for f in factors):
                break

            ddm = fine.domain_decomposition.coarsen(factors)
            grid = TensorProductGrid(
                num_elements=tuple(int(n) for n in ddm.ncells),
                mpi_dims_mask=fine.grid.mpi_dims_mask,
            )
            coarse = Derham(grid, fine.options, comm=fine.comm, domain=fine.domain, domain_decomposition=ddm)

            self._derhams.append(coarse)
            self._factors.append(factors)

        logger.debug(f"Multigrid hierarchy: {[d.num_elements for d in self._derhams]}")

    def _coarsening_factors(self, derham: Derham) -> tuple[int, int, int]:
        """Return 2 in every direction that can be halved (see class docstring), else 1."""
        ddm: DomainDecomposition = derham.domain_decomposition
        factors = []
        for axis in range(3):
            n = derham.num_elements[axis]
            p = derham.degree[axis]
            starts = np.asarray(ddm.global_element_starts[axis])
            ends = np.asarray(ddm.global_element_ends[axis])
            ok = n % 2 == 0 and n // 2 >= max(self._min_cells, p + 1)
            ok = ok and np.all(starts % 2 == 0) and np.all((ends + 1) % 2 == 0)
            if ok and ddm.nprocs[axis] > 1:
                ok = np.all((ends - starts + 1) // 2 >= p)
            factors.append(2 if ok else 1)
        return tuple(factors)

    @property
    def derhams(self) -> list[Derham]:
        """Derham of each level, ``derhams[0]`` is the finest."""
        return self._derhams

    @property
    def n_levels(self) -> int:
        """Number of levels (including the finest)."""
        return len(self._derhams)

    @property
    def factors(self) -> list[tuple[int, int, int]]:
        """``factors[l]`` is the coarsening factor in each direction from level ``l`` to ``l+1``."""
        return self._factors

    def __getitem__(self, level: int) -> Derham:
        return self._derhams[level]

    def __len__(self) -> int:
        return self.n_levels
