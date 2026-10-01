import pytest


@pytest.mark.parametrize("dim_reduce", [0, 2])
def test_mass_preconditioner_transpose(dim_reduce):
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

    pc = MassMatrixPreconditioner(mass_ops.M1, apply_bc=False, dim_reduce=dim_reduce)
    pc_T = pc.transpose()
    assert type(pc_T) is MassMatrixPreconditioner
    assert pc_T._apply_bc is False
    assert pc_T._dim_reduce == dim_reduce

    pc_diag_T = MassMatrixDiagonalPreconditioner(mass_ops.M1, apply_bc=False).transpose()
    assert type(pc_diag_T) is MassMatrixDiagonalPreconditioner
    assert pc_diag_T._apply_bc is False


def test_mass_diagonal_preconditioner_assembles_diagonal_blocks_only(monkeypatch):
    """The logical mass matrix of MassMatrixDiagonalPreconditioner needs only the diagonal blocks."""

    from struphy import domains
    from struphy.feec.mass import WeightedMassOperator, WeightedMassOperators
    from struphy.feec.preconditioner import MassMatrixDiagonalPreconditioner
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    grid = TensorProductGrid(num_elements=(3, 3, 2))
    derham = Derham(grid, DerhamOptions(degree=(1, 1, 1), bcs=(None, None, None)))
    mass_ops = WeightedMassOperators(derham, domains.Colella())

    M1 = mass_ops.M1
    weights_infos = []
    init = WeightedMassOperator.__init__

    def recording_init(self, *args, weights_info=None, **kwargs):
        weights_infos.append(weights_info)
        init(self, *args, weights_info=weights_info, **kwargs)

    monkeypatch.setattr(WeightedMassOperator, "__init__", recording_init)
    MassMatrixDiagonalPreconditioner(M1)

    # the logical mass matrix is the only 3x3 weighted mass operator created
    (fun,) = [w for w in weights_infos if isinstance(w, list) and len(w) == 3]
    for i in range(3):
        for j in range(3):
            assert callable(fun[i][j]) == (i == j)
