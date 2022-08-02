import pytest

from  struphy.feec import spline_space

import struphy.feec.bsplines_kernels            as bsp
import struphy.feec.basics.spline_evaluation_2d as eva_2d
import struphy.feec.basics.spline_evaluation_3d as eva_3d

import numpy as np

@pytest.mark.parametrize('Nel', [[8,5,6], [4, 4, 128]])
@pytest.mark.parametrize('p', [[2,3,2], [1,1,4]])
@pytest.mark.parametrize('spl_kind', [[True, True, True], [False, False, True]])
@pytest.mark.parametrize('N', [100])
def test_Spline_Evaluation(Nel, p, spl_kind, N):
    # ========================================================================================= 
    # FEEC SPACES Object & related quantities
    # =========================================================================================

    spaces_FEM_1 = spline_space.Spline_space_1d(Nel[0], p[0], spl_kind[0]) 
    spaces_FEM_2 = spline_space.Spline_space_1d(Nel[1], p[1], spl_kind[1])
    spaces_FEM_3 = spline_space.Spline_space_1d(Nel[2], p[2], spl_kind[2])

    SPACES = spline_space.Tensor_spline_space([spaces_FEM_1, spaces_FEM_2, spaces_FEM_3])

    # To what precision equality of legacy and slim functions should be tested
    decimals = 1

    ind_n1 = SPACES.indN[0]
    ind_n2 = SPACES.indN[1]
    ind_n3 = SPACES.indN[2]
    
    ind_d1 = SPACES.indD[0]
    ind_d2 = SPACES.indD[1]
    ind_d3 = SPACES.indD[2]

    # B-spline knot vectors
    t1  = SPACES.T[0]
    t2  = SPACES.T[1]
    t3  = SPACES.T[2]

    # D-spline knot vectors
    T1  = SPACES.t[0]
    T2  = SPACES.t[1]
    T3  = SPACES.t[2]

    pn1 = SPACES.p[0]
    pn2 = SPACES.p[1]
    pn3 = SPACES.p[2]

    nbase_n1 = SPACES.NbaseN[0]
    nbase_n2 = SPACES.NbaseN[1]
    nbase_n3 = SPACES.NbaseN[2]

    nbase_d1 = SPACES.NbaseD[0]
    nbase_d2 = SPACES.NbaseD[1]
    nbase_d3 = SPACES.NbaseD[2]

    pn1 = SPACES.p[0]
    pn2 = SPACES.p[1]
    pn3 = SPACES.p[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1


    # ========================================================================================= 
    # Test evaluation kernels (legacy and new)
    # =========================================================================================
    for j in range(N):
        eta = np.random.rand(3)

        eta1 = eta[0]
        eta2 = eta[1]
        eta3 = eta[2]

        span1 = bsp.find_span(t1, pn1, eta1)
        span2 = bsp.find_span(t2, pn2, eta2)
        span3 = bsp.find_span(t3, pn3, eta3)

        ie1 = span1 - pn1
        ie2 = span2 - pn2
        ie3 = span3 - pn3

        bn1 = np.empty( pn1+1, dtype=float )
        bn2 = np.empty( pn2+1, dtype=float )
        bn3 = np.empty( pn3+1, dtype=float )

        bsp.b_splines_slim(t1, pn1, eta1, span1, bn1)
        bsp.b_splines_slim(t2, pn2, eta2, span2, bn2)
        bsp.b_splines_slim(t3, pn3, eta3, span3, bn3)

        # 2d case
        coeff2d = np.random.rand( nbase_n1+1, nbase_n2+1 )

        kern_leg = eva_2d.evaluation_kernel_2d(pn1, pn2, bn1, bn2, span1, span2, ind_n1, ind_n2, coeff2d)

        assert np.isnan(kern_leg) == False
        assert np.isinf(kern_leg) == False

        kern = eva_2d.evaluation_kernel_2d_slim(pn1, pn2, bn1, bn2, ind_n1[ie1,:], ind_n2[ie2,:], coeff2d)

        assert np.isnan(kern) == False
        assert np.isinf(kern) == False

        assert np.round(kern - kern_leg, decimals) == 0. , 'value: '+str(kern)+'  legacy: '+str(kern_leg)

        # 3d case
        coeff3d = np.random.rand( nbase_n1+1, nbase_n2+1, nbase_n3+1 )

        kern_leg = eva_3d.evaluation_kernel_3d(pn1, pn2, pn3, bn1, bn2, bn3, span1, span2, span3, ind_n1, ind_n2, ind_n3, coeff3d)

        assert np.isnan(kern_leg) == False
        assert np.isinf(kern_leg) == False

        kern = eva_3d.evaluation_kernel_3d_slim(pn1, pn2, pn3, bn1, bn2, bn3, ind_n1[ie1,:], ind_n2[ie2,:], ind_n3[ie3,:],coeff3d)

        assert np.isnan(kern) == False
        assert np.isinf(kern) == False

        assert np.round(kern - kern_leg, decimals) == 0. , 'value: '+str(kern)+'  legacy: '+str(kern_leg)

        del(bn1)
        del(bn2)
        del(bn3)
        del(coeff2d)
        del(coeff3d)


    # ========================================================================================= 
    # Test different evaluate functions (2d)
    # =========================================================================================
    for j in range(N):
        eta = np.random.rand(2)

        eta1 = eta[0]
        eta2 = eta[1]

        span1 = bsp.find_span(t1, pn1, eta1)
        span2 = bsp.find_span(t2, pn2, eta2)

        ie1 = span1 - pn1
        ie2 = span2 - pn2

        bn1 = np.empty( pn1+1, dtype=float )
        bn2 = np.empty( pn2+1, dtype=float )

        bd1 = np.empty( pn1  , dtype=float )
        bd2 = np.empty( pn2  , dtype=float )

        bsp.b_d_splines_slim(t1, pn1, eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(t2, pn2, eta2, span2, bn2, bd2)

        coeff2d = np.random.rand( nbase_n1+1, nbase_n2+1 )

        # n_n
        value_leg = eva_2d.evaluate(1, 1, t1, t2, pn1, pn2, ind_n1, ind_n2, coeff2d, eta1, eta2)

        assert np.isnan(value_leg) == False
        assert np.isinf(value_leg) == False

        value = eva_2d.evaluate_slim(1, 1, t1, t2, pn1, pn2, ind_n1[ie1, :], ind_n2[ie2, :], coeff2d, eta1, eta2)

        assert np.isnan(value) == False
        assert np.isinf(value) == False

        assert np.round(value - value_leg, decimals) == 0. , 'value: '+str(value)+'  legacy: '+str(value_leg)


        # diffn_n
        value_leg = eva_2d.evaluate(3, 1, t1, t2, pn1, pn2, ind_n1, ind_n2, coeff2d, eta1, eta2)

        assert np.isnan(value_leg) == False
        assert np.isinf(value_leg) == False

        value = eva_2d.evaluate_slim(3, 1, t1, t2, pn1, pn2, ind_n1[ie1, :], ind_n2[ie2, :], coeff2d, eta1, eta2)

        assert np.isnan(value) == False
        assert np.isinf(value) == False

        assert np.round(value - value_leg, decimals) == 0. , 'value: '+str(value)+'  legacy: '+str(value_leg)


        # n_diffn
        value_leg = eva_2d.evaluate(1, 3, t1, t2, pn1, pn2, ind_n1, ind_n2, coeff2d, eta1, eta2)

        assert np.isnan(value_leg) == False
        assert np.isinf(value_leg) == False

        value = eva_2d.evaluate_slim(1, 3, t1, t2, pn1, pn2, ind_n1[ie1, :], ind_n2[ie2, :], coeff2d, eta1, eta2)

        assert np.isnan(value) == False
        assert np.isinf(value) == False

        assert np.round(value - value_leg, decimals) == 0. , 'value: '+str(value)+'  legacy: '+str(value_leg)


        ## d_n
        #value_leg = eva_2d.evaluate(2, 1, T1, t2, pd1, pn2, ind_d1, ind_n2, coeff2d, eta1, eta2)
#
        #assert np.isnan(value_leg) == False
        #assert np.isinf(value_leg) == False
#
        #value = eva_2d.evaluate_slim(2, 1, T1, t2, pd1, pn2, ind_d1[ie1, :], ind_n2[ie2, :], coeff2d, eta1, eta2)
#
        #assert np.isnan(value) == False
        #assert np.isinf(value) == False
#
        #assert np.round(value - value_leg, decimals) == 0. , 'value: '+str(value)+'  legacy: '+str(value_leg)
#
#
        ## n_d
        #value_leg = eva_2d.evaluate(1, 2, t1, T2, pn1, pd2, ind_n1, ind_d2, coeff2d, eta1, eta2)
#
        #assert np.isnan(value_leg) == False
        #assert np.isinf(value_leg) == False
#
        #value = eva_2d.evaluate_slim(1, 2, t1, T2, pn1, pd2, ind_n1[ie1, :], ind_d2[ie2, :], coeff2d, eta1, eta2)
#
        #assert np.isnan(value) == False
        #assert np.isinf(value) == False
#
        #assert np.round(value - value_leg, decimals) == 0. , 'value: '+str(value)+'  legacy: '+str(value_leg)
#
#
        ## d_d
        #value_leg = eva_2d.evaluate(2, 2, T1, T2, pd1, pd2, ind_d1, ind_d2, coeff2d, eta1, eta2)
#
        #assert np.isnan(value_leg) == False
        #assert np.isinf(value_leg) == False
#
        #value = eva_2d.evaluate_slim(2, 2, T1, T2, pd1, pd2, ind_d1[ie1, :], ind_d2[ie2, :], coeff2d, eta1, eta2)
#
        #assert np.isnan(value) == False
        #assert np.isinf(value) == False
#
        #assert np.round(value - value_leg, decimals) == 0. , 'value: '+str(value)+'  legacy: '+str(value_leg)
    

    for j in range(N):
        pass
        # TODO write test for eval_tensor_product and evaluate_matrix



    # ========================================================================================= 
    # Test different evaluate functions (3d)
    # =========================================================================================
    for j in range(N):
        eta = np.random.rand(3)

        eta1 = eta[0]
        eta2 = eta[1]
        eta3 = eta[2]

        span1 = bsp.find_span(t1, pn1, eta1)
        span2 = bsp.find_span(t2, pn2, eta2)
        span3 = bsp.find_span(t3, pn3, eta3)

        ie1 = span1 - pn1
        ie2 = span2 - pn2
        ie3 = span3 - pn3

        bn1 = np.empty( pn1+1, dtype=float )
        bn2 = np.empty( pn2+1, dtype=float )
        bn3 = np.empty( pn3+1, dtype=float )

        bd1 = np.empty( pn1  , dtype=float )
        bd2 = np.empty( pn2  , dtype=float )
        bd3 = np.empty( pn3  , dtype=float )

        bsp.b_d_splines_slim(t1, pn1, eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(t2, pn2, eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(t3, pn3, eta3, span3, bn3, bd3)

        coeff3d = np.random.rand( nbase_n1+1, nbase_n2+1, nbase_n3+1 )


        # n_n_n
        value_leg = eva_3d.evaluate(1, 1, 1, t1, t2, t3, pn1, pn2, pn3, ind_n1, ind_n2, ind_n3, coeff3d, eta1, eta2, eta3)

        assert np.isnan(value_leg) == False
        assert np.isinf(value_leg) == False

        value = eva_3d.evaluate_slim(1, 1, 1, t1, t2, t3, pn1, pn2, pn3, ind_n1[ie1, :], ind_n2[ie2, :], ind_n3[ie3, :], coeff3d, eta1, eta2, eta3)

        assert np.isnan(value) == False
        assert np.isinf(value) == False

        assert np.round(value - value_leg, decimals) == 0. , 'value: '+str(value)+'  legacy: '+str(value_leg)


        # diffn_n_n
        value_leg = eva_3d.evaluate(3, 1, 1, t1, t2, t3, pn1, pn2, pn3, ind_n1, ind_n2, ind_n3, coeff3d, eta1, eta2, eta3)

        assert np.isnan(value_leg) == False
        assert np.isinf(value_leg) == False

        value = eva_3d.evaluate_slim(3, 1, 1, t1, t2, t3, pn1, pn2, pn3, ind_n1[ie1, :], ind_n2[ie2, :], ind_n3[ie3, :], coeff3d, eta1, eta2, eta3)

        assert np.isnan(value) == False
        assert np.isinf(value) == False

        assert np.round(value - value_leg, decimals) == 0. , 'value: '+str(value)+'  legacy: '+str(value_leg)


        # n_diffn_n
        value_leg = eva_3d.evaluate(1, 3, 1, t1, t2, t3, pn1, pn2, pn3, ind_n1, ind_n2, ind_n3, coeff3d, eta1, eta2, eta3)

        assert np.isnan(value_leg) == False
        assert np.isinf(value_leg) == False

        value = eva_3d.evaluate_slim(1, 3, 1, t1, t2, t3, pn1, pn2, pn3, ind_n1[ie1, :], ind_n2[ie2, :], ind_n3[ie3, :], coeff3d, eta1, eta2, eta3)

        assert np.isnan(value) == False
        assert np.isinf(value) == False

        assert np.round(value - value_leg, decimals) == 0. , 'value: '+str(value)+'  legacy: '+str(value_leg)


        # n_n_diffn
        value_leg = eva_3d.evaluate(1, 1, 3, t1, t2, t3, pn1, pn2, pn3, ind_n1, ind_n2, ind_n3, coeff3d, eta1, eta2, eta3)

        assert np.isnan(value_leg) == False
        assert np.isinf(value_leg) == False

        value = eva_3d.evaluate_slim(1, 1, 3, t1, t2, t3, pn1, pn2, pn3, ind_n1[ie1, :], ind_n2[ie2, :], ind_n3[ie3, :], coeff3d, eta1, eta2, eta3)

        assert np.isnan(value) == False
        assert np.isinf(value) == False

        assert np.round(value - value_leg, decimals) == 0. , 'value: '+str(value)+'  legacy: '+str(value_leg)


        # d_n_n
        value_leg = eva_3d.evaluate(2, 1, 1, T1, t2, t3, pd1, pn2, pn3, ind_d1, ind_n2, ind_n3, coeff3d, eta1, eta2, eta3)

        assert np.isnan(value_leg) == False
        assert np.isinf(value_leg) == False

        value = eva_3d.evaluate_slim(2, 1, 1, T1, t2, t3, pd1, pn2, pn3, ind_d1[ie1, :], ind_n2[ie2, :], ind_n3[ie3, :], coeff3d, eta1, eta2, eta3)

        assert np.isnan(value) == False
        assert np.isinf(value) == False

        assert np.round(value - value_leg, decimals) == 0. , 'value: '+str(value)+'  legacy: '+str(value_leg)


        # n_d_n
        value_leg = eva_3d.evaluate(1, 2, 1, t1, T2, t3, pn1, pd2, pn3, ind_n1, ind_d2, ind_n3, coeff3d, eta1, eta2, eta3)

        assert np.isnan(value_leg) == False
        assert np.isinf(value_leg) == False

        value = eva_3d.evaluate_slim(1, 2, 1, t1, T2, t3, pn1, pd2, pn3, ind_n1[ie1, :], ind_d2[ie2, :], ind_n3[ie3, :], coeff3d, eta1, eta2, eta3)

        assert np.isnan(value) == False
        assert np.isinf(value) == False

        assert np.round(value - value_leg, decimals) == 0. , 'value: '+str(value)+'  legacy: '+str(value_leg)


        # n_n_d
        value_leg = eva_3d.evaluate(1, 1, 2, t1, t2, T3, pn1, pn2, pd3, ind_n1, ind_n2, ind_d3, coeff3d, eta1, eta2, eta3)

        assert np.isnan(value_leg) == False
        assert np.isinf(value_leg) == False

        value = eva_3d.evaluate_slim(1, 1, 2, t1, t2, T3, pn1, pn2, pd3, ind_n1[ie1, :], ind_n2[ie2, :], ind_d3[ie3, :], coeff3d, eta1, eta2, eta3)

        assert np.isnan(value) == False
        assert np.isinf(value) == False

        assert np.round(value - value_leg, decimals) == 0. , 'value: '+str(value)+'  legacy: '+str(value_leg)


        # d_d_n
        value_leg = eva_3d.evaluate(2, 2, 1, T1, T2, t3, pd1, pd2, pn3, ind_d1, ind_d2, ind_n3, coeff3d, eta1, eta2, eta3)

        assert np.isnan(value_leg) == False
        assert np.isinf(value_leg) == False

        value = eva_3d.evaluate_slim(2, 2, 1, T1, T2, t3, pd1, pd2, pn3, ind_d1[ie1, :], ind_d2[ie2, :], ind_n3[ie3, :], coeff3d, eta1, eta2, eta3)

        assert np.isnan(value) == False
        assert np.isinf(value) == False

        assert np.round(value - value_leg, decimals) == 0. , 'value: '+str(value)+'  legacy: '+str(value_leg)


        # d_n_d
        value_leg = eva_3d.evaluate(2, 1, 2, T1, t2, T3, pd1, pn2, pd3, ind_d1, ind_n2, ind_d3, coeff3d, eta1, eta2, eta3)

        assert np.isnan(value_leg) == False
        assert np.isinf(value_leg) == False

        value = eva_3d.evaluate_slim(2, 1, 2, T1, t2, T3, pd1, pn2, pd3, ind_d1[ie1, :], ind_n2[ie2, :], ind_d3[ie3, :], coeff3d, eta1, eta2, eta3)

        assert np.isnan(value) == False
        assert np.isinf(value) == False

        assert np.round(value - value_leg, decimals) == 0. , 'value: '+str(value)+'  legacy: '+str(value_leg)


        # n_d_d
        value_leg = eva_3d.evaluate(1, 2, 2, t1, T2, T3, pn1, pd2, pd3, ind_n1, ind_d2, ind_d3, coeff3d, eta1, eta2, eta3)

        assert np.isnan(value_leg) == False
        assert np.isinf(value_leg) == False

        value = eva_3d.evaluate_slim(1, 2, 2, t1, T2, T3, pn1, pd2, pd3, ind_n1[ie1, :], ind_d2[ie2, :], ind_d3[ie3, :], coeff3d, eta1, eta2, eta3)

        assert np.isnan(value) == False
        assert np.isinf(value) == False

        assert np.round(value - value_leg, decimals) == 0. , 'value: '+str(value)+'  legacy: '+str(value_leg)


        # d_d_d
        value_leg = eva_3d.evaluate(2, 2, 2, T1, T2, T3, pd1, pd2, pd3, ind_d1, ind_d2, ind_d3, coeff3d, eta1, eta2, eta3)

        assert np.isnan(value_leg) == False
        assert np.isinf(value_leg) == False

        value = eva_3d.evaluate_slim(2, 2, 2, T1, T2, T3, pd1, pd2, pd3, ind_d1[ie1, :], ind_d2[ie2, :], ind_d3[ie3, :], coeff3d, eta1, eta2, eta3)

        assert np.isnan(value) == False
        assert np.isinf(value) == False

        assert np.round(value - value_leg, decimals) == 0. , 'value: '+str(value)+'  legacy: '+str(value_leg)
    

    for j in range(N):
        pass
        # TODO write test for eval_tensor_product, evaluate_matrix, and evaluate_sparse


    print('test_spline_evaluation passed!')


if __name__ == '__main__':
    Nel         = [8,15,64]
    p           = [2,6,8]
    spl_kind    = [True, False, True]
    N           = 100
    test_Spline_Evaluation(Nel, p, spl_kind, N)