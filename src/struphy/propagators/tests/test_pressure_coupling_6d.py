import cunumpy as xp
from feectools.linalg.basic import IdentityOperator

from struphy.feec.psydac_derham import Derham
from struphy.io.options import DerhamOptions
from struphy.propagators.pressure_coupling_6d import PressureCoupling6D
from struphy.topology.grids import TensorProductGrid


def test_gt_mat_g_transpose():
    """GT_MAT_G.transpose() must return a GT_MAT_G with flipped transposed flag (symmetric MAT)."""
    derham = Derham(TensorProductGrid(num_elements=[4, 3, 2]), DerhamOptions(degree=[2, 1, 1]))
    Id = IdentityOperator(derham.V1)
    MAT = [[Id, Id, Id], [Id, Id, Id], [Id, Id, Id]]

    op = PressureCoupling6D.GT_MAT_G(derham, MAT)
    opT = op.transpose()

    assert not op.transposed
    assert isinstance(opT, PressureCoupling6D.GT_MAT_G)
    assert opT.transposed
    assert not opT.transpose().transposed

    v = derham.Vv.zeros()
    for vi in v:
        vi._data[:] = xp.random.rand(*vi._data.shape)
    assert xp.allclose(op.dot(v).copy().toarray(), opT.dot(v).toarray())


if __name__ == "__main__":
    test_gt_mat_g_transpose()
