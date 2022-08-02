from sys import int_info
import pytest

from mpi4py import MPI
import numpy as np
from time import sleep


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 9, 10]])
@pytest.mark.parametrize('p', [[2, 3, 4]])
@pytest.mark.parametrize('spl_kind', [[False, False, True], [False, True, False], [True, False, False]])
@pytest.mark.parametrize('mapping', [
    ['cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}], ])
def test_psydac_eval(Nel, p, spl_kind, mapping, n_markers=10):
    '''Compares ``evaluation_kernel_3d`` with ``eval_spline_mpi_3d``.'''

    from struphy.geometry.domain_3d import Domain
    from struphy.psydac_api.psydac_derham import Derham

    from struphy.psydac_api.utilities import create_equal_random_arrays as cera
    from struphy.feec import bsplines_kernels as bsp
    from struphy.feec.basics.spline_evaluation_3d import evaluation_kernel_3d as eval3d
    from struphy.feec.basics.spline_evaluation_3d import eval_spline_mpi_3d as eval3d_mpi

    comm = MPI.COMM_WORLD
    assert comm.size >= 2
    rank = comm.Get_rank()

    # Domain object
    map = mapping[0]
    params_map = mapping[1]

    DOMAIN = Domain(map, params_map)

    # Psydac discrete Derham sequence
    DR = Derham(Nel, p, spl_kind, comm=comm)

    # DR attributes
    pn = np.array(DR.p)
    tn1, tn2, tn3 = DR.V0.knots
    indN = DR.indN
    indD = DR.indD
    dims0 = DR.V0.vector_space.npts
    dims1 = [space.vector_space.npts for space in DR.V1.spaces]
    dims2 = [space.vector_space.npts for space in DR.V2.spaces]
    dims3 = DR.V3.vector_space.npts

    # Random spline coeffs
    x0, x0_psy = cera(DR.V0)
    x1, x1_psy = cera(DR.V1)
    x2, x2_psy = cera(DR.V2)
    x3, x3_psy = cera(DR.V3)

    # Random points in domain of process 
    dom = DR.domain_array[rank]
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
        span1 = bsp.find_span(tn1, DR.p[0], eta1)
        span2 = bsp.find_span(tn2, DR.p[1], eta2)
        span3 = bsp.find_span(tn3, DR.p[2], eta3)

        # non-zero spline values at eta
        bn1 = np.empty( DR.p[0] + 1, dtype=float)
        bn2 = np.empty( DR.p[1] + 1, dtype=float)
        bn3 = np.empty( DR.p[2] + 1, dtype=float)

        bd1 = np.empty( DR.p[0], dtype=float)
        bd2 = np.empty( DR.p[1], dtype=float)
        bd3 = np.empty( DR.p[2], dtype=float)

        bsp.b_d_splines_slim(tn1, DR.p[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, DR.p[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, DR.p[2], eta3, span3, bn3, bd3)

        # Non-vanishing B- and D-spline indices at eta (needed for the non-mpi routines)
        ie1 = span1 - DR.p[0]
        ie2 = span2 - DR.p[1]
        ie3 = span3 - DR.p[2]

        ind_n1 = indN[0][ie1]
        ind_n2 = indN[1][ie2]
        ind_n3 = indN[2][ie3]

        ind_d1 = indD[0][ie1]
        ind_d2 = indD[1][ie2]
        ind_d3 = indD[2][ie3]

        # compare spline evaluation routines in V0
        val = eval3d(*DR.p, bn1, bn2, bn3, ind_n1, ind_n2, ind_n3, x0[0])
        val_mpi = eval3d_mpi(*DR.p, bn1, bn2, bn3, span1, span2, span3, x0_psy._data, np.array(x0_psy.starts), pn)
        assert np.allclose(val, val_mpi)

        # compare spline evaluation routines in V1
        val = eval3d(DR.p[0] - 1, DR.p[1], DR.p[2], bd1, bn2, bn3, ind_d1, ind_n2, ind_n3, x1[0])
        val_mpi = eval3d_mpi(DR.p[0] - 1, DR.p[1], DR.p[2], bd1, bn2, bn3, span1, span2, span3, x1_psy[0]._data, np.array(x1_psy[0].starts), pn)
        assert np.allclose(val, val_mpi)

        val = eval3d(DR.p[0], DR.p[1] - 1, DR.p[2], bn1, bd2, bn3, ind_n1, ind_d2, ind_n3, x1[1])
        val_mpi = eval3d_mpi(DR.p[0], DR.p[1] - 1, DR.p[2], bn1, bd2, bn3, span1, span2, span3, x1_psy[1]._data, np.array(x1_psy[1].starts), pn)
        assert np.allclose(val, val_mpi)

        val = eval3d(DR.p[0], DR.p[1], DR.p[2] - 1, bn1, bn2, bd3, ind_n1, ind_n2, ind_d3, x1[2])
        val_mpi = eval3d_mpi(DR.p[0], DR.p[1], DR.p[2] - 1, bn1, bn2, bd3, span1, span2, span3, x1_psy[2]._data, np.array(x1_psy[2].starts), pn)
        assert np.allclose(val, val_mpi)

        # compare spline evaluation routines in V2
        val = eval3d(DR.p[0], DR.p[1] - 1, DR.p[2] - 1, bn1, bd2, bd3, ind_n1, ind_d2, ind_d3, x2[0])
        val_mpi = eval3d_mpi(DR.p[0], DR.p[1] - 1, DR.p[2] - 1, bn1, bd2, bd3, span1, span2, span3, x2_psy[0]._data, np.array(x2_psy[0].starts), pn)
        assert np.allclose(val, val_mpi)

        val = eval3d(DR.p[0] - 1, DR.p[1], DR.p[2] - 1, bd1, bn2, bd3, ind_d1, ind_n2, ind_d3, x2[1])
        val_mpi = eval3d_mpi(DR.p[0] - 1, DR.p[1], DR.p[2] - 1, bd1, bn2, bd3, span1, span2, span3, x2_psy[1]._data, np.array(x2_psy[1].starts), pn)
        assert np.allclose(val, val_mpi)

        val = eval3d(DR.p[0] - 1, DR.p[1] - 1, DR.p[2], bd1, bd2, bn3, ind_d1, ind_d2, ind_n3, x2[2])
        val_mpi = eval3d_mpi(DR.p[0] - 1, DR.p[1] - 1, DR.p[2], bd1, bd2, bn3, span1, span2, span3, x2_psy[2]._data, np.array(x2_psy[2].starts), pn)
        assert np.allclose(val, val_mpi)

        # compare spline evaluation routines in V3
        val = eval3d(DR.p[0] - 1, DR.p[1] - 1, DR.p[2] - 1, bd1, bd2, bd3, ind_d1, ind_d2, ind_d3, x3[0])
        val_mpi = eval3d_mpi(DR.p[0] - 1, DR.p[1] - 1, DR.p[2] - 1, bd1, bd2, bd3, span1, span2, span3, x3_psy._data, np.array(x3_psy.starts), pn)
        assert np.allclose(val, val_mpi)




if __name__ == '__main__':
    test_psydac_eval([8, 9, 10], [2, 3, 4], [False, False, True], ['cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}], n_markers=1)