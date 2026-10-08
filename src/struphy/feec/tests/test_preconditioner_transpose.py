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
