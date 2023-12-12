import pytest
import numpy as np

import struphy.bsplines.bsplines_kernels as bsp
from struphy.eigenvalue_solvers import spline_space


@pytest.mark.parametrize('Nel', [[8, 5, 6], [4, 3, 128]])
@pytest.mark.parametrize('p', [[3, 2, 1], [1, 1, 4]])
@pytest.mark.parametrize('spl_kind', [[True, True, True], [False, False, True]])
@pytest.mark.parametrize('N', [100])
def test_bsplines_kernels(Nel, p, spl_kind, N):
    """
    TODO
    """

    # =========================================================================================
    # FEEC SPACES Object & related quantities
    # =========================================================================================

    spaces_FEM_1 = spline_space.Spline_space_1d(Nel[0], p[0], spl_kind[0])
    spaces_FEM_2 = spline_space.Spline_space_1d(Nel[1], p[1], spl_kind[1])
    spaces_FEM_3 = spline_space.Spline_space_1d(Nel[2], p[2], spl_kind[2])

    SPACES = spline_space.Tensor_spline_space(
        [spaces_FEM_1, spaces_FEM_2, spaces_FEM_3])

    # =========================================================================================
    # Test Scaling, find_span, B-splines (both legacy and slim), and compare them
    # =========================================================================================
    for tn, pn in zip(SPACES.T, SPACES.p):
        for j in range(N):
            eta = np.random.rand()

            # Test find_span
            span = bsp.find_span(tn, pn, eta)

            # Test Scaling
            values = np.empty(pn + 1, dtype=float)
            bsp.scaling(tn, pn, span, values)

            bn_leg = np.empty(pn + 1, dtype=float)
            left = np.empty(pn, dtype=float)
            right = np.empty(pn, dtype=float)

            # Test basis_funs
            bsp.basis_funs(tn, pn, eta, span, left, right, bn_leg)

            assert np.isnan(bn_leg).any() == False
            assert np.isinf(bn_leg).any() == False

            assert np.isnan(left).any() == False
            assert np.isinf(left).any() == False

            assert np.isnan(right).any() == False
            assert np.isinf(right).any() == False

            bn = np.empty(pn + 1, dtype=float)
            bn2 = bn.copy()

            # Test b_splines_slim
            bsp.b_splines_slim(tn, pn, eta, span, bn)

            assert np.isnan(bn).any() == False
            assert np.isinf(bn).any() == False

            # Assert legacy and slim versions give the same result
            assert np.allclose(bn, bn_leg)

            bn_all_leg = np.empty((pn + 1, pn + 1), dtype=float)
            diff = np.empty(pn, dtype=float)

            # Test basis_funs_all
            bsp.basis_funs_all(tn, pn, eta, span, left,
                               right, bn_all_leg, diff)

            assert np.isnan(bn_all_leg).any() == False
            assert np.isinf(bn_all_leg).any() == False

            assert np.isnan(diff).any() == False
            assert np.isinf(diff).any() == False

            assert np.isnan(left).any() == False
            assert np.isinf(left).any() == False

            assert np.isnan(right).any() == False
            assert np.isinf(right).any() == False

            bd = np.empty(pn, dtype=float)
            bd2 = bd.copy()

            bsp.d_splines_slim(tn, pn, eta, span, bd)

            assert np.allclose(bd, diff * bn_all_leg[pn - 1, :pn])

            bsp.b_d_splines_slim(tn, pn, eta, span, bn2, bd2)

            assert np.allclose(bn, bn2)
            assert np.allclose(bd, bd2)

            # Test derivatives
            left = np.empty(pn, dtype=float)
            right = np.empty(pn, dtype=float)
            bn_all_leg = np.empty((pn + 1, pn + 1), dtype=float)
            ders_leg = np.empty(pn + 1, dtype=float)
            diff = np.empty(pn, dtype=float)

            bsp.basis_funs_and_der(
                tn, pn, eta, span, left, right, bn_all_leg, diff, ders_leg)

            assert np.isnan(bn_all_leg).any() == False
            assert np.isinf(bn_all_leg).any() == False

            assert np.isnan(ders_leg).any() == False
            assert np.isinf(ders_leg).any() == False

            ders = np.empty(pn+1, dtype=float)

            bsp.b_spl_1st_der_slim(tn, pn, eta, span, ders)

            assert np.isnan(ders).any() == False
            assert np.isinf(ders).any() == False

            assert np.allclose(ders, ders_leg)

    # TODO: Tests for piecewise and convolution functions (Yingzhe)

    print('test_bsplines_kernels passed!')


@pytest.mark.parametrize('Nel', [10, 11, 12, 13])
@pytest.mark.parametrize('p', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('spl_kind', [False])
def test_spline_derivatives(Nel, p, spl_kind):
    '''Test spline evaluations and 1st/2nd derivatives for clamped splines.'''

    space0 = spline_space.Spline_space_1d(Nel, p, spl_kind)
    knots = space0.T
    dim = space0.NbaseN

    s = np.random.rand(30)

    if p == 1:
        for i in range(dim):
            coef = np.zeros(dim)
            coef[i] = 1.
            assert np.all(space0.evaluate_N(s, coef, 3) == 0.*s)
        return
    elif p == 2:
        return

    space1 = spline_space.Spline_space_1d(Nel, p - 1, spl_kind)
    space2 = spline_space.Spline_space_1d(Nel, p - 2, spl_kind)

    s = np.random.rand(30)

    print(knots)
    print(space1.T)
    print(space2.T)

    # weights for first derivative
    w1 = [None]
    for i in range(1, dim):
        w1 += [p / (knots[i + p] - knots[i])] 

    # weights for second derivative
    w2 = [None]
    for i in range(1, space1.NbaseN):
        w2 += [(p - 1) / (space1.T[i + p - 1] - space1.T[i])] 

    for i in range(dim):
        coef = np.zeros(dim)
        coef[i] = 1.

        coef1 = np.zeros(dim - 1)

        coef2 = np.zeros(dim - 2)

        if i == 0:
            coef1[i] = - w1[i + 1]

            coef2[i] = w1[i + 1] * w2[i + 1]
        elif i == 1:
            coef1[i - 1] = w1[i]
            coef1[i] = -w1[i + 1]

            coef2[i - 1] = - w1[i] * w2[i] - w1[i + 1] * w2[i]
            coef2[i] = w1[i + 1] * w2[i + 1]
        elif i == dim - 2:
            coef1[i - 1] = w1[i]
            coef1[i] = -w1[i + 1]

            coef2[i - 2] = w1[i] * w2[i-1]
            coef2[i - 1] = - w1[i] * w2[i] - w1[i + 1] * w2[i]
        elif i == dim - 1:
            coef1[i - 1] = w1[i]

            coef2[i - 2] = w1[i] * w2[i-1]
        else:
            coef1[i - 1] = w1[i]
            coef1[i] = -w1[i + 1]

            coef2[i - 2] = w1[i] * w2[i-1]
            coef2[i - 1] = - w1[i] * w2[i] - w1[i + 1] * w2[i]
            coef2[i] = w1[i + 1] * w2[i + 1] 
            
        vals1 = space1.evaluate_N(s.flatten(), coef1).reshape(s.shape)
        vals2 = space2.evaluate_N(s.flatten(), coef2).reshape(s.shape)

        assert np.allclose(space0.evaluate_N(s, coef, 2), vals1)
        assert np.allclose(space0.evaluate_N(s, coef, 3), vals2)


if __name__ == '__main__':
    # Nel = [14, 5, 6]
    # p = [6, 2, 3]
    # spl_kind = [True, False, True]
    # N = 100
    # test_bsplines_kernels(Nel, p, spl_kind, N)
    test_spline_derivatives(10, 3, True)
