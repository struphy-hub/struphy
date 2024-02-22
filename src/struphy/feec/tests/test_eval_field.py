import pytest
from mpi4py import MPI
import numpy as np


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 9, 10]])
@pytest.mark.parametrize('p', [[3, 2, 4]])
@pytest.mark.parametrize('spl_kind', [[False, False, True], [False, True, False], [True, False, False]])
def test_eval_field(Nel, p, spl_kind):
    '''Compares distributed array spline evaluation in Field object with legacy code.'''

    from struphy.geometry.base import Domain
    from struphy.feec.psydac_derham import Derham

    from struphy.feec.utilities import compare_arrays
    from struphy.bsplines.evaluation_kernels_3d import evaluate_matrix

    comm = MPI.COMM_WORLD
    assert comm.size >= 2
    rank = comm.Get_rank()

    # derham object
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # fem field objects
    p0 = derham.create_field('pressure', 'H1')
    E1 = derham.create_field('e_field', 'Hcurl')
    B2 = derham.create_field('b_field', 'Hdiv')
    n3 = derham.create_field('density', 'L2')
    uv = derham.create_field('velocity', 'H1vec')

    # initialize fields with sin/cos
    comps = {'pressure':  '0',
             'e_field': ['1', '1', '1'],
             'b_field': ['2', '2', '2'],
             'density':  '3',
             'velocity': ['v', 'v', 'v']}

    init_params = {'type': 'ModesCos',
                   'ModesCos': {'comps': comps,
                                'ls': {'pressure': [0],
                                       'e_field': [[0], [0], [0]],
                                       'b_field': [[0], [0], [0]],
                                       'density': [0],
                                       'velocity': [[0], [0], [0]], },
                                'ms': {'pressure': [0],
                                       'e_field': [[0], [0], [0]],
                                       'b_field': [[0], [0], [0]],
                                       'density': [0],
                                       'velocity': [[0], [0], [0]], },
                                'ns': {'pressure': [1],
                                       'e_field': [[1], [1], [1]],
                                       'b_field': [[1], [1], [1]],
                                       'density': [1],
                                       'velocity': [[1], [1], [1]], },
                                'amps': {'pressure': [5.],
                                         'e_field': [[5.], [5.], [5.]],
                                         'b_field': [[5.], [5.], [5.]],
                                         'density': [5.],
                                         'velocity': [[5.], [5.], [5.]], }}}

    p0.initialize_coeffs(init_params)
    E1.initialize_coeffs(init_params)
    B2.initialize_coeffs(init_params)
    n3.initialize_coeffs(init_params)
    uv.initialize_coeffs(init_params)

    # evaluation points
    eta1 = np.linspace(0, 1, 11)
    eta2 = np.linspace(0, 1, 14)
    eta3 = np.linspace(0, 1, 17)

    # arrays for legacy evaluation
    arr1, arr2, arr3, is_sparse_meshgrid = Domain.prepare_eval_pts(
        eta1, eta2, eta3)
    tmp = np.zeros_like(arr1)

    ######
    # V0 #
    ######
    # create legacy arrays with same coeffs
    coeffs_loc = np.reshape(p0.vector.toarray(), p0.nbasis)
    coeffs = np.zeros_like(coeffs_loc)
    comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(p0.vector, coeffs, rank)

    # legacy evaluation
    evaluate_matrix(derham.Vh_fem['0'].knots[0], derham.Vh_fem['0'].knots[1], derham.Vh_fem['0'].knots[2],
                    p[0], p[1], p[2],
                    derham.indN[0], derham.indN[1], derham.indN[2],
                    coeffs, arr1, arr2, arr3, tmp, 0)
    val_legacy = np.squeeze(tmp.copy())
    tmp[:] = 0

    # distributed evaluation and comparison
    val = p0(eta1, eta2, eta3, squeeze_output=True)
    assert np.allclose(val, val_legacy)

    ######
    # V1 #
    ######
    # create legacy arrays with same coeffs
    coeffs_loc = np.reshape(E1.vector[0].toarray(), E1.nbasis[0])
    coeffs = np.zeros_like(coeffs_loc)
    comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(E1.vector[0], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(derham.Vh_fem['3'].knots[0], derham.Vh_fem['0'].knots[1], derham.Vh_fem['0'].knots[2],
                    p[0] - 1, p[1], p[2],
                    derham.indD[0], derham.indN[1], derham.indN[2],
                    coeffs, arr1, arr2, arr3, tmp, 11)
    val_legacy_1 = np.squeeze(tmp.copy())
    tmp[:] = 0

    # create legacy arrays with same coeffs
    coeffs_loc = np.reshape(E1.vector[1].toarray(), E1.nbasis[1])
    coeffs = np.zeros_like(coeffs_loc)
    comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(E1.vector[1], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(derham.Vh_fem['0'].knots[0], derham.Vh_fem['3'].knots[1], derham.Vh_fem['0'].knots[2],
                    p[0], p[1] - 1, p[2],
                    derham.indN[0], derham.indD[1], derham.indN[2],
                    coeffs, arr1, arr2, arr3, tmp, 12)
    val_legacy_2 = np.squeeze(tmp.copy())
    tmp[:] = 0

    # create legacy arrays with same coeffs
    coeffs_loc = np.reshape(E1.vector[2].toarray(), E1.nbasis[2])
    coeffs = np.zeros_like(coeffs_loc)
    comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(E1.vector[2], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(derham.Vh_fem['0'].knots[0], derham.Vh_fem['0'].knots[1], derham.Vh_fem['3'].knots[2],
                    p[0], p[1], p[2] - 1,
                    derham.indN[0], derham.indN[1], derham.indD[2],
                    coeffs, arr1, arr2, arr3, tmp, 13)
    val_legacy_3 = np.squeeze(tmp.copy())
    tmp[:] = 0

    # distributed evaluation and comparison
    val1, val2, val3 = E1(eta1, eta2, eta3, squeeze_output=True)
    assert np.allclose(val1, val_legacy_1)
    assert np.allclose(val2, val_legacy_2)
    assert np.allclose(val3, val_legacy_3)

    ######
    # V2 #
    ######
    # create legacy arrays with same coeffs
    coeffs_loc = np.reshape(B2.vector[0].toarray(), B2.nbasis[0])
    coeffs = np.zeros_like(coeffs_loc)
    comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(B2.vector[0], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(derham.Vh_fem['0'].knots[0], derham.Vh_fem['3'].knots[1], derham.Vh_fem['3'].knots[2],
                    p[0], p[1] - 1, p[2] - 1,
                    derham.indN[0], derham.indD[1], derham.indD[2],
                    coeffs, arr1, arr2, arr3, tmp, 21)
    val_legacy_1 = np.squeeze(tmp.copy())
    tmp[:] = 0

    # create legacy arrays with same coeffs
    coeffs_loc = np.reshape(B2.vector[1].toarray(), B2.nbasis[1])
    coeffs = np.zeros_like(coeffs_loc)
    comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(B2.vector[1], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(derham.Vh_fem['3'].knots[0], derham.Vh_fem['0'].knots[1], derham.Vh_fem['3'].knots[2],
                    p[0] - 1, p[1], p[2] - 1,
                    derham.indD[0], derham.indN[1], derham.indD[2],
                    coeffs, arr1, arr2, arr3, tmp, 22)
    val_legacy_2 = np.squeeze(tmp.copy())
    tmp[:] = 0

    # create legacy arrays with same coeffs
    coeffs_loc = np.reshape(B2.vector[2].toarray(), B2.nbasis[2])
    coeffs = np.zeros_like(coeffs_loc)
    comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(B2.vector[2], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(derham.Vh_fem['3'].knots[0], derham.Vh_fem['3'].knots[1], derham.Vh_fem['0'].knots[2],
                    p[0] - 1, p[1] - 1, p[2],
                    derham.indD[0], derham.indD[1], derham.indN[2],
                    coeffs, arr1, arr2, arr3, tmp, 23)
    val_legacy_3 = np.squeeze(tmp.copy())
    tmp[:] = 0

    # distributed evaluation and comparison
    val1, val2, val3 = B2(eta1, eta2, eta3, squeeze_output=True)
    assert np.allclose(val1, val_legacy_1)
    assert np.allclose(val2, val_legacy_2)
    assert np.allclose(val3, val_legacy_3)

    ######
    # V3 #
    ######
    # create legacy arrays with same coeffs
    coeffs_loc = np.reshape(n3.vector.toarray(), n3.nbasis)
    coeffs = np.zeros_like(coeffs_loc)
    comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(n3.vector, coeffs, rank)

    # legacy evaluation
    evaluate_matrix(derham.Vh_fem['3'].knots[0], derham.Vh_fem['3'].knots[1], derham.Vh_fem['3'].knots[2],
                    p[0] - 1, p[1] - 1, p[2] - 1,
                    derham.indD[0], derham.indD[1], derham.indD[2],
                    coeffs, arr1, arr2, arr3, tmp, 3)
    val_legacy = np.squeeze(tmp.copy())
    tmp[:] = 0

    # distributed evaluation and comparison
    val = n3(eta1, eta2, eta3, squeeze_output=True)
    assert np.allclose(val, val_legacy)

    #########
    # V0vec #
    #########
    # create legacy arrays with same coeffs
    coeffs_loc = np.reshape(uv.vector[0].toarray(), uv.nbasis[0])
    coeffs = np.zeros_like(coeffs_loc)
    comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(uv.vector[0], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(derham.Vh_fem['0'].knots[0], derham.Vh_fem['0'].knots[1], derham.Vh_fem['0'].knots[2],
                    p[0], p[1], p[2],
                    derham.indN[0], derham.indN[1], derham.indN[2],
                    coeffs, arr1, arr2, arr3, tmp, 0)
    val_legacy_1 = np.squeeze(tmp.copy())
    tmp[:] = 0

    # create legacy arrays with same coeffs
    coeffs_loc = np.reshape(uv.vector[1].toarray(), uv.nbasis[1])
    coeffs = np.zeros_like(coeffs_loc)
    comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(uv.vector[1], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(derham.Vh_fem['0'].knots[0], derham.Vh_fem['0'].knots[1], derham.Vh_fem['0'].knots[2],
                    p[0], p[1], p[2],
                    derham.indN[0], derham.indN[1], derham.indN[2],
                    coeffs, arr1, arr2, arr3, tmp, 0)
    val_legacy_2 = np.squeeze(tmp.copy())
    tmp[:] = 0

    # create legacy arrays with same coeffs
    coeffs_loc = np.reshape(uv.vector[2].toarray(), uv.nbasis[2])
    coeffs = np.zeros_like(coeffs_loc)
    comm.Allreduce(coeffs_loc, coeffs, op=MPI.SUM)
    compare_arrays(uv.vector[2], coeffs, rank)

    # legacy evaluation
    evaluate_matrix(derham.Vh_fem['0'].knots[0], derham.Vh_fem['0'].knots[1], derham.Vh_fem['0'].knots[2],
                    p[0], p[1], p[2],
                    derham.indN[0], derham.indN[1], derham.indN[2],
                    coeffs, arr1, arr2, arr3, tmp, 0)
    val_legacy_3 = np.squeeze(tmp.copy())
    tmp[:] = 0

    # distributed evaluation and comparison
    val1, val2, val3 = uv(eta1, eta2, eta3, squeeze_output=True)
    assert np.allclose(val1, val_legacy_1)
    assert np.allclose(val2, val_legacy_2)
    assert np.allclose(val3, val_legacy_3)


if __name__ == '__main__':
    test_eval_field([8, 9, 10], [3, 2, 4], [False, False, True])
