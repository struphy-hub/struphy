from sys import int_info
import pytest

from mpi4py import MPI
import numpy as np
from time import sleep


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 9, 10]])
@pytest.mark.parametrize('p', [[1, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, False, True], [False, True, False], [True, False, False]])
def test_eval_kernels(Nel, p, spl_kind, n_markers=10):
    '''Compares evaluation_kernel_3d with eval_spline_mpi_kernel.'''

    from struphy.feec.psydac_derham import Derham

    from struphy.feec.utilities import create_equal_random_arrays as cera
    from struphy.bsplines import bsplines_kernels as bsp
    from struphy.bsplines.evaluation_kernels_3d import evaluation_kernel_3d as eval3d
    from struphy.bsplines.evaluation_kernels_3d import eval_spline_mpi_kernel as eval3d_mpi

    comm = MPI.COMM_WORLD
    assert comm.size >= 2
    rank = comm.Get_rank()

    # Psydac discrete Derham sequence
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # derham attributes
    pn = np.array(derham.p)
    tn1, tn2, tn3 = derham.Vh_fem['0'].knots
    indN = derham.indN
    indD = derham.indD
    dims0 = derham.Vh['0'].npts
    dims1 = [space.vector_space.npts for space in derham.Vh_fem['1'].spaces]
    dims2 = [space.vector_space.npts for space in derham.Vh_fem['2'].spaces]
    dims3 = derham.Vh['3'].npts

    # Random spline coeffs_loc
    x0, x0_psy = cera(derham.Vh_fem['0'])
    x1, x1_psy = cera(derham.Vh_fem['1'])
    x2, x2_psy = cera(derham.Vh_fem['2'])
    x3, x3_psy = cera(derham.Vh_fem['3'])

    # Random points in domain of process
    dom = derham.domain_array[rank]
    eta1s = np.random.rand(n_markers)*(dom[1] - dom[0]) + dom[0]
    eta2s = np.random.rand(n_markers)*(dom[4] - dom[3]) + dom[3]
    eta3s = np.random.rand(n_markers)*(dom[7] - dom[6]) + dom[6]

    for eta1, eta2, eta3 in zip(eta1s, eta2s, eta3s):

        comm.Barrier()
        sleep(.02*(rank + 1))
        print(f'rank {rank} | eta1 = {eta1}')
        print(f'rank {rank} | eta2 = {eta2}')
        print(f'rank {rank} | eta3 = {eta3}\n')
        comm.Barrier()

        # spans (i.e. index for non-vanishing basis functions)
        span1 = bsp.find_span(tn1, derham.p[0], eta1)
        span2 = bsp.find_span(tn2, derham.p[1], eta2)
        span3 = bsp.find_span(tn3, derham.p[2], eta3)

        # non-zero spline values at eta
        bn1 = np.empty(derham.p[0] + 1, dtype=float)
        bn2 = np.empty(derham.p[1] + 1, dtype=float)
        bn3 = np.empty(derham.p[2] + 1, dtype=float)

        bd1 = np.empty(derham.p[0], dtype=float)
        bd2 = np.empty(derham.p[1], dtype=float)
        bd3 = np.empty(derham.p[2], dtype=float)

        bsp.b_d_splines_slim(tn1, derham.p[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, derham.p[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, derham.p[2], eta3, span3, bn3, bd3)

        # Non-vanishing B- and D-spline indices at eta (needed for the non-mpi routines)
        ie1 = span1 - derham.p[0]
        ie2 = span2 - derham.p[1]
        ie3 = span3 - derham.p[2]

        ind_n1 = indN[0][ie1]
        ind_n2 = indN[1][ie2]
        ind_n3 = indN[2][ie3]

        ind_d1 = indD[0][ie1]
        ind_d2 = indD[1][ie2]
        ind_d3 = indD[2][ie3]

        # compare spline evaluation routines in V0
        val = eval3d(*derham.p, bn1, bn2, bn3, ind_n1, ind_n2, ind_n3, x0[0])
        val_mpi = eval3d_mpi(*derham.p, bn1, bn2, bn3, span1,
                             span2, span3, x0_psy._data, np.array(x0_psy.starts))
        assert np.allclose(val, val_mpi)

        # compare spline evaluation routines in V1
        val = eval3d(derham.p[0] - 1, derham.p[1], derham.p[2], bd1,
                     bn2, bn3, ind_d1, ind_n2, ind_n3, x1[0])
        val_mpi = eval3d_mpi(derham.p[0] - 1, derham.p[1], derham.p[2], bd1, bn2, bn3,
                             span1, span2, span3, x1_psy[0]._data, np.array(x1_psy[0].starts))
        assert np.allclose(val, val_mpi)

        val = eval3d(derham.p[0], derham.p[1] - 1, derham.p[2], bn1,
                     bd2, bn3, ind_n1, ind_d2, ind_n3, x1[1])
        val_mpi = eval3d_mpi(derham.p[0], derham.p[1] - 1, derham.p[2], bn1, bd2, bn3,
                             span1, span2, span3, x1_psy[1]._data, np.array(x1_psy[1].starts))
        assert np.allclose(val, val_mpi)

        val = eval3d(derham.p[0], derham.p[1], derham.p[2] - 1, bn1,
                     bn2, bd3, ind_n1, ind_n2, ind_d3, x1[2])
        val_mpi = eval3d_mpi(derham.p[0], derham.p[1], derham.p[2] - 1, bn1, bn2, bd3,
                             span1, span2, span3, x1_psy[2]._data, np.array(x1_psy[2].starts))
        assert np.allclose(val, val_mpi)

        # compare spline evaluation routines in V2
        val = eval3d(derham.p[0], derham.p[1] - 1, derham.p[2] - 1, bn1,
                     bd2, bd3, ind_n1, ind_d2, ind_d3, x2[0])
        val_mpi = eval3d_mpi(derham.p[0], derham.p[1] - 1, derham.p[2] - 1, bn1, bd2, bd3,
                             span1, span2, span3, x2_psy[0]._data, np.array(x2_psy[0].starts))
        assert np.allclose(val, val_mpi)

        val = eval3d(derham.p[0] - 1, derham.p[1], derham.p[2] - 1, bd1,
                     bn2, bd3, ind_d1, ind_n2, ind_d3, x2[1])
        val_mpi = eval3d_mpi(derham.p[0] - 1, derham.p[1], derham.p[2] - 1, bd1, bn2, bd3,
                             span1, span2, span3, x2_psy[1]._data, np.array(x2_psy[1].starts))
        assert np.allclose(val, val_mpi)

        val = eval3d(derham.p[0] - 1, derham.p[1] - 1, derham.p[2], bd1,
                     bd2, bn3, ind_d1, ind_d2, ind_n3, x2[2])
        val_mpi = eval3d_mpi(derham.p[0] - 1, derham.p[1] - 1, derham.p[2], bd1, bd2, bn3,
                             span1, span2, span3, x2_psy[2]._data, np.array(x2_psy[2].starts))
        assert np.allclose(val, val_mpi)

        # compare spline evaluation routines in V3
        val = eval3d(derham.p[0] - 1, derham.p[1] - 1, derham.p[2] - 1,
                     bd1, bd2, bd3, ind_d1, ind_d2, ind_d3, x3[0])
        val_mpi = eval3d_mpi(derham.p[0] - 1, derham.p[1] - 1, derham.p[2] - 1, bd1, bd2,
                             bd3, span1, span2, span3, x3_psy._data, np.array(x3_psy.starts))
        assert np.allclose(val, val_mpi)


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
    comps = {'pressure':  True,
             'e_field': [True, True, True],
             'b_field': [True, True, True],
             'density':  True,
             'velocity': [True, True, True]}

    init_params = {'type': 'ModesCos', 'ModesCos': {'coords': 'logical',
                                                    'comps': comps, 'ls': [0], 'ms': [0], 'ns': [1], 'amps': [5.]}}

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
    #test_eval_kernels([8, 9, 10], [2, 3, 4], [False, False, True], n_markers=1)
    test_eval_field([8, 9, 10], [2, 3, 4], [False, True, True])
