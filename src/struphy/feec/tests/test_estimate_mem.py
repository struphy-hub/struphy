import pytest
from feectools.linalg.memory import stencil_matrix_memory
from feectools.linalg.stencil import StencilMatrix

from struphy import DerhamOptions, domains, grids
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.memory import linop_nbytes
from struphy.feec.psydac_derham import Derham


@pytest.fixture
def derham():
    return Derham(
        grids.TensorProductGrid(num_elements=(4, 5, 6)),
        DerhamOptions(degree=(2, 2, 3)),
    )


def test_stencil_matrix_dry_run(derham):
    """A dry-run StencilMatrix reports the same size as an allocated one, but allocates nothing."""
    V = derham.coeff_spaces["3"]

    n_before = stencil_matrix_memory.n_matrices
    dry = StencilMatrix(V, V, dry_run=True)
    assert dry.dry_run
    assert stencil_matrix_memory.n_matrices == n_before, "a dry-run matrix must not be registered"

    # the data array is not there, and saying so is part of the deal
    with pytest.raises(AttributeError, match="dry_run"):
        dry._data

    mat = StencilMatrix(V, V)
    assert not mat.dry_run
    assert mat.data_shape == dry.data_shape
    assert mat.nbytes == dry.nbytes == mat._data.nbytes


def test_mass_ops_estimate_mem_matches_allocation(derham):
    """The dry-run estimate of the standard mass matrices must match the real allocation."""
    mass_ops = WeightedMassOperators(derham, domains.Cuboid())

    names = ("M0", "M1", "M2", "M3", "Mv")
    estimated = mass_ops.estimate_mem(names=names)

    # nothing was created (nor allocated) by the estimate
    assert mass_ops.allocated_mem() == {}
    assert not mass_ops.dry_run

    # now allocate and assemble for real
    for name in names:
        getattr(mass_ops, name)

    allocated = mass_ops.allocated_mem()
    assert set(allocated) == set(names)
    for name in names:
        assert estimated[name] == allocated[name], f"{name}: {estimated[name]} != {allocated[name]}"
        assert estimated[name] > 0

    # and the estimate of an already allocated operator is its actual size
    assert mass_ops.estimate_mem(names=("M1",))["M1"] == allocated["M1"]


def test_mass_ops_zero_blocks_are_not_counted(derham):
    """On a Cartesian domain the metric is diagonal, so M1 must be a 3x3 block matrix with
    only 3 non-zero blocks -- the estimate has to see that, too."""
    mass_ops = WeightedMassOperators(derham, domains.Cuboid())

    estimated = mass_ops.estimate_mem(names=("M1",))["M1"]
    blocks = [block for row in mass_ops.M1._mat.blocks for block in row]

    assert sum(block is not None for block in blocks) == 3
    assert estimated == sum(block.nbytes for block in blocks if block is not None)


def test_derivative_matrices_are_matrix_free(derham):
    """The derivative operators of the Derham sequence do not store any matrix data."""
    assert sum(linop_nbytes(op) for op in (derham.grad, derham.curl, derham.div)) == 0
