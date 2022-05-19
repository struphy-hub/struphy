import pytest

import struphy.feec.bsplines_kernels as bsp
from   struphy.feec                  import spline_space 

import numpy as np

@pytest.mark.parametrize('Nel', [[8,5,6], [4, 4, 128]])
@pytest.mark.parametrize('p', [[2,3,2], [1,1,4]])
@pytest.mark.parametrize('spl_kind', [[True, True, True], [False, False, True]])
@pytest.mark.parametrize('N', [100])
def test_Bsplines_Kernels(Nel, p, spl_kind, N):

    # ========================================================================================= 
    # FEEC SPACES Object & related quantities
    # =========================================================================================

    spaces_FEM_1 = spline_space.Spline_space_1d(Nel[0], p[0], spl_kind[0]) 
    spaces_FEM_2 = spline_space.Spline_space_1d(Nel[1], p[1], spl_kind[1])
    spaces_FEM_3 = spline_space.Spline_space_1d(Nel[2], p[2], spl_kind[2])

    SPACES = spline_space.Tensor_spline_space([spaces_FEM_1, spaces_FEM_2, spaces_FEM_3])

    # To what precision equality of legacy and slim functions should be tested
    decimals = 12

    # ========================================================================================= 
    # Test Scaling, find_span, B-splines (both legacy and slim), and compare them
    # =========================================================================================
    for k in range(3):
        t  = SPACES.T[k]
        pn = SPACES.p[k]

        for j in range(N):
            eta = np.random.rand()

            # Test find_span
            span = bsp.find_span(t, pn, eta)

            # Test Scaling
            values = np.empty(pn+1, dtype=float)
            bsp.scaling(t, pn, span, values)

            bn_leg = np.empty( pn + 1, dtype=float )
            left   = np.empty( pn    , dtype=float )
            right  = np.empty( pn    , dtype=float )

            # Test basis_funs
            bsp.basis_funs(t, pn, eta, span, left, right, bn_leg)

            assert np.isnan(bn_leg).any() == False
            assert np.isinf(bn_leg).any() == False

            assert np.isnan(left).any() == False
            assert np.isinf(left).any() == False
            
            assert np.isnan(right).any() == False
            assert np.isinf(right).any() == False


            bn = np.empty( pn + 1, dtype=float )

            # Test b_splines_slim
            bsp.b_splines_slim(t, pn, eta, span, bn)

            assert np.isnan(bn).any() == False
            assert np.isinf(bn).any() == False

            # Assert legacy and slim versions give the same result
            assert np.sum(np.abs(bn - bn_leg)) == 0.

            del(bn)

            bn_all_leg = np.empty( (pn+1, pn+1), dtype=float )
            diff       = np.empty( pn          , dtype=float )

            # Test basis_funs_all
            bsp.basis_funs_all(t, pn, eta, span, left, right, bn_all_leg, diff)

            assert np.isnan(bn_all_leg).any() == False
            assert np.isinf(bn_all_leg).any() == False

            assert np.isnan(diff).any() == False
            assert np.isinf(diff).any() == False

            assert np.isnan(left).any() == False
            assert np.isinf(left).any() == False
            
            assert np.isnan(right).any() == False
            assert np.isinf(right).any() == False

            bd   = np.empty( pn, dtype=float )
            span = bsp.find_span(t, pn, eta)

            bsp.d_splines_slim(t, pn, eta, span, bd)

            assert np.round( np.sum( np.abs( bd - diff * bn_all_leg[pn-1,:pn] ) ), decimals ) == 0.
            
            del(bd)
            del(left)
            del(right)
            del(bn_all_leg)
            del(diff)

            # Test derivatives
            left       = np.empty( pn          , dtype=float )
            right      = np.empty( pn          , dtype=float )
            bn_all_leg = np.empty( (pn+1, pn+1), dtype=float )
            ders_leg   = np.empty( pn+1        , dtype=float )
            diff       = np.empty( pn          , dtype=float )

            bsp.basis_funs_and_der(t, pn, eta, span, left, right, bn_all_leg, diff, ders_leg)

            assert np.isnan(bn_all_leg).any() == False
            assert np.isinf(bn_all_leg).any() == False

            assert np.isnan(ders_leg).any() == False
            assert np.isinf(ders_leg).any() == False

            ders = np.empty( pn+1, dtype=float )

            bsp.b_spl_1st_der_slim(t, pn, eta, span, ders)

            assert np.isnan(ders).any() == False
            assert np.isinf(ders).any() == False

            assert np.round(np.sum(np.abs( ders - ders_leg )), decimals) == 0.

    # TODO: Tests for piecewise and convolution functions (Yingzhe)

    print('test_bsplines_kernels passed!')


if __name__ == '__main__':
    Nel         = [14,5,6]
    p           = [6,2,3]
    spl_kind    = [True, False, True]
    N           = 100
    test_Bsplines_Kernels(Nel, p, spl_kind, N)