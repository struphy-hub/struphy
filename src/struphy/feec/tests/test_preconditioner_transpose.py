import pytest


@pytest.mark.parametrize("dim_reduce", [0, 2])
@pytest.mark.parametrize("weight_reduction", ["midpoint", "average"])
@pytest.mark.parametrize("diagonal_scaling", [False, True])
def test_mass_preconditioner_transpose(dim_reduce, weight_reduction, diagonal_scaling):
    """The transposes of the mass-matrix preconditioners keep their class and options."""

    from struphy import domains
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import MassMatrixDiagonalPreconditioner, MassMatrixPreconditioner
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    grid = TensorProductGrid(num_elements=(3, 3, 2))
    derham = Derham(grid, DerhamOptions(degree=(1, 1, 1), bcs=(None, None, None)))
    mass_ops = WeightedMassOperators(derham, domains.Colella())

    pc = MassMatrixPreconditioner(
        mass_ops.M1,
        apply_bc=False,
        dim_reduce=dim_reduce,
        weight_reduction=weight_reduction,
        diagonal_scaling=diagonal_scaling,
    )
    pc_T = pc.transpose()
    assert type(pc_T) is MassMatrixPreconditioner
    assert pc_T._apply_bc is False
    assert pc_T._dim_reduce == dim_reduce
    assert pc_T.weight_reduction == weight_reduction
    assert pc_T.diagonal_scaling == diagonal_scaling

    pc_diag_T = MassMatrixDiagonalPreconditioner(mass_ops.M1, apply_bc=False).transpose()
    assert type(pc_diag_T) is MassMatrixDiagonalPreconditioner
    assert pc_diag_T._apply_bc is False


def test_mass_diagonal_preconditioner_assembles_1d_matrices_only(monkeypatch):
    """MassMatrixDiagonalPreconditioner assembles only 1d mass matrices (no 3d logical mass matrix);
    the diagonal of the logical mass matrix is the one of its Kronecker approximation."""

    from struphy import domains
    from struphy.feec.mass import WeightedMassOperator, WeightedMassOperators
    from struphy.feec.preconditioner import MassMatrixDiagonalPreconditioner, MassMatrixPreconditioner
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    grid = TensorProductGrid(num_elements=(3, 3, 2))
    derham = Derham(grid, DerhamOptions(degree=(1, 1, 1), bcs=(None, None, None)))
    mass_ops = WeightedMassOperators(derham, domains.Colella())

    M1 = mass_ops.M1
    created = []
    init = WeightedMassOperator.__init__

    def recording_init(self, derham, V, W, *args, **kwargs):
        created.append(V.ldim)
        init(self, derham, V, W, *args, **kwargs)

    monkeypatch.setattr(WeightedMassOperator, "__init__", recording_init)
    pc = MassMatrixDiagonalPreconditioner(M1)

    assert isinstance(pc, MassMatrixPreconditioner)
    assert pc.dim_reduce is None and pc.diagonal_scaling
    assert created and all(ldim == 1 for ldim in created)


def test_mass_diagonal_preconditioner_zero_diagonal_block():
    """A mass operator with a zero (None) diagonal block, e.g. after reassembly with a zero weight:
    construction and update_mass_operator must work; the scaling of that block is 1."""

    import cunumpy as xp

    from struphy import domains
    from struphy.feec.mass import WeightedMassOperator, WeightedMassOperators
    from struphy.feec.preconditioner import MassMatrixDiagonalPreconditioner
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    grid = TensorProductGrid(num_elements=(3, 3, 2))
    derham = Derham(grid, DerhamOptions(degree=(1, 1, 1), bcs=(None, None, None)))
    mass_ops = WeightedMassOperators(derham, domains.Colella())

    def one(e1, e2, e3):
        return xp.ones_like(e1, dtype=float)

    weights = [[one if i == j and i != 2 else None for j in range(3)] for i in range(3)]
    M = WeightedMassOperator(derham, derham.V1fem, derham.V1fem, weights_info=weights)
    M.assemble()
    assert M.matrix[2, 2] is None

    pc = MassMatrixDiagonalPreconditioner(mass_ops.M1)
    pc.update_mass_operator(M)
    scaling = pc._scaling[0]
    assert xp.allclose(scaling[2, 2]._data, 1.0)

    _, x = create_equal_random_arrays(derham.V1fem, seed=1)
    assert all(xp.all(xp.isfinite(b.toarray())) for b in pc.dot(x).blocks)


def test_mass_diagonal_preconditioner_small_periodic_direction():
    """Periodic direction with fewer points than the stencil width (1 element, degree 1): the main
    diagonal of the 1d factors must be at offset 0, otherwise the diagonal scaling vanishes."""

    import cunumpy as xp
    from feectools.linalg.solvers import inverse

    from struphy import domains
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import MassMatrixDiagonalPreconditioner, _local_diagonal
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    derham = Derham(grid=TensorProductGrid(num_elements=(4, 4, 1)), options=DerhamOptions(degree=(2, 2, 1)))
    mass_ops = WeightedMassOperators(derham=derham, domain=domains.Cuboid())

    pc = MassMatrixDiagonalPreconditioner(mass_ops.Mv)
    assert all(float(xp.min(d)) > 0.0 for d in _local_diagonal(pc.matrix))

    _, x = create_equal_random_arrays(derham.Vvfem, seed=1)
    inv = inverse(mass_ops.Mv, "pcg", pc=pc, tol=1e-12, maxiter=100)
    inv.dot(x)
    assert inv._info["success"]
