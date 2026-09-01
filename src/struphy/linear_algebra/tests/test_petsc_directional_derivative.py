import cunumpy as xp
import pytest

pytest.importorskip("petsc4py")

from feectools.ddm.mpi import mpi as MPI

from struphy.feec.psydac_derham import Derham
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.petsc_solver import _directional_derivative_to_stencil_matrix
from struphy.topology.grids import TensorProductGrid


def _random_fill(v, seed):
    from feectools.linalg.block import BlockVector

    xp.random.seed(seed)
    if isinstance(v, BlockVector):
        for b in v.blocks:
            b._data[:] = xp.random.random(b._data.shape)
    else:
        v._data[:] = xp.random.random(v._data.shape)
    v.update_ghost_regions()
    return v


def test_directional_derivative_matches_grad_on_periodic_domain():
    """The StencilMatrix built by _directional_derivative_to_stencil_matrix must reproduce
    every block of derham.grad and derham.grad.T exactly, on a fully periodic domain.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    grid = TensorProductGrid(num_elements=[6, 6, 6])
    derham_opts = DerhamOptions(degree=[2, 2, 2], bcs=(None, None, None))
    derham = Derham(grid, derham_opts, comm=comm)

    for op, name in [(derham.grad, "grad"), (derham.grad.T, "grad.T")]:
        for i, j in op.nonzero_block_indices:
            block = op[i, j]
            M = _directional_derivative_to_stencil_matrix(block)

            v = _random_fill(block.domain.zeros(), seed=100 + i + 10 * j + rank)
            err = xp.linalg.norm((block.dot(v) - M.dot(v)).toarray())
            assert err < 1e-12, f"{name}[{i},{j}] mismatch: err={err:.3e}"


def test_directional_derivative_raises_on_nonperiodic_axis():
    """A non-periodic differentiation axis must raise NotImplementedError rather than
    silently produce wrong results (see the docstring of
    _directional_derivative_to_stencil_matrix for the unresolved root cause).
    """
    comm = MPI.COMM_WORLD

    grid = TensorProductGrid(num_elements=[6, 6, 6])
    derham_opts = DerhamOptions(degree=[2, 2, 2], bcs=(("free", "free"), None, None))
    derham = Derham(grid, derham_opts, comm=comm)

    op = derham.grad.T[0, 0]
    assert op.domain.periods[op._diffdir] is False

    with pytest.raises(NotImplementedError):
        _directional_derivative_to_stencil_matrix(op)


if __name__ == "__main__":
    test_directional_derivative_matches_grad_on_periodic_domain()
    test_directional_derivative_raises_on_nonperiodic_axis()
