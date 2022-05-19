import pytest

import struphy.feec.bsplines_kernels as bsp

from struphy.feec   import spline_space 
from struphy.pic    import mat_vec_filler

import numpy as np

@pytest.mark.parametrize('Nel', [[8,5,6], [4, 4, 128]])
@pytest.mark.parametrize('p', [[2,3,2], [1,1,4]])
@pytest.mark.parametrize('spl_kind', [[True, True, True], [False, False, True]])
def test_Mat_Vec_Filler(Nel, p, spl_kind):
    # ========================================================================================= 
    # FEEC SPACES Object & related quantities
    # =========================================================================================
    
    spaces_FEM_1 = spline_space.Spline_space_1d(Nel[0], p[0], spl_kind[0])
    spaces_FEM_2 = spline_space.Spline_space_1d(Nel[1], p[1], spl_kind[1])
    spaces_FEM_3 = spline_space.Spline_space_1d(Nel[2], p[2], spl_kind[2])

    SPACES = spline_space.Tensor_spline_space([spaces_FEM_1, spaces_FEM_2, spaces_FEM_3])

    t1, t2, t3 = SPACES.T

    indn1, indn2, indn3 = SPACES.indN[0], SPACES.indN[1], SPACES.indN[2]


    # ========================================================================================= 
    # Test _b Functions at Random Position
    # =========================================================================================
    eta1, eta2, eta3 = np.random.rand(3)

    # test diagonal and antisymmetric matrices
    kinds  = ['diag', 'asym']
    spaces = ['1', '2']
    
    matrix = [0, 0, 0] # since for diag und asym there are only 3 independent matrix entries, it suffices to only have 3 matrix entries like a vector
    vector = [0, 0, 0]
    
    for space in spaces:
        for a in range(3):
            
            Nbase_form = getattr(SPACES, 'Nbase_'+space+'form')

            Ni = Nbase_form[a]

            vector[a] = np.zeros((Ni[0], Ni[1], Ni[2]), dtype=float)
            vector[a][:,:,:] = 0.
            
            matrix[a] = np.zeros( (Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
            matrix[a][:,:,:,:,:,:] = 0.

        filling_m = 10. - 20.*np.random.rand(3)
        filling_v = 10. - 20.*np.random.rand(3)


        for kind in kinds:
            function_m  = getattr(mat_vec_filler, 'mat_fill_b_v'+space+'_'+kind)
            function_mv = getattr(mat_vec_filler, 'm_v_fill_b_v'+space+'_'+kind)

            function_m(np.array(p), t1, t2, t3, indn1, indn2, indn3, eta1, eta2, eta3, matrix[0], matrix[1], matrix[2], filling_m[0], filling_m[1], filling_m[2])

            for a in range(3):
                assert np.isnan(matrix[a]).any() == False
                assert np.isinf(matrix[a]).any() == False

            function_mv(np.array(p), t1, t2, t3, indn1, indn2, indn3, eta1, eta2, eta3, matrix[0], matrix[1], matrix[2], filling_m[0], filling_m[1], filling_m[2], vector[0], vector[1], vector[2], filling_v[0], filling_v[1], filling_v[2])
            
            for a in range(3):
                assert np.isnan(matrix[a]).any() == False
                assert np.isinf(matrix[a]).any() == False
                assert np.isnan(vector[a]).any() == False
                assert np.isinf(vector[a]).any() == False

    del(matrix)
    del(vector)
    del(filling_m)
    del(filling_v)
    del(function_m)
    del(function_mv)


    # test symmetric matrices
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    vector =  [0, 0, 0]

    for space in spaces:
        for a in range(3):
            
            Nbase_form = getattr(SPACES, 'Nbase_'+space+'form')

            Ni = Nbase_form[a]

            vector[a] = np.zeros((Ni[0], Ni[1], Ni[2]), dtype=float)
            vector[a][:,:,:] = 0.
            
            for b in range(a, 3):

                matrix[a][b] = np.zeros( (Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
                matrix[a][b][:,:,:,:,:,:] = 0.

        filling_m = 10.*np.random.rand(3,3)
        filling_v = 10.*np.random.rand(3)


        function_m  = getattr(mat_vec_filler, 'mat_fill_b_v'+space+'_symm')
        function_mv = getattr(mat_vec_filler, 'm_v_fill_b_v'+space+'_symm')

        function_m(np.array(p), t1, t2, t3, indn1, indn2, indn3, eta1, eta2, eta3, matrix[0][0], matrix[0][1], matrix[0][2], matrix[1][1], matrix[1][2], matrix[2][2], filling_m[0][0], filling_m[0][1], filling_m[0][2], filling_m[1][1], filling_m[1][2], filling_m[2][2])

        for a in range(3):
            for b in range(a,3):
                assert np.isnan(matrix[a][b]).any() == False
                assert np.isinf(matrix[a][b]).any() == False


        function_mv(np.array(p), t1, t2, t3, indn1, indn2, indn3, eta1, eta2, eta3, matrix[0][0], matrix[0][1], matrix[0][2], matrix[1][1], matrix[1][2], matrix[2][2], filling_m[0][0], filling_m[0][1], filling_m[0][2], filling_m[1][1], filling_m[1][2], filling_m[2][2], vector[0], vector[1], vector[2], filling_v[0], filling_v[1], filling_v[2])
        
        for a in range(3):
            assert np.isnan(vector[a]).any() == False
            assert np.isinf(vector[a]).any() == False

            for b in range(a,3):
                assert np.isnan(matrix[a][b]).any() == False
                assert np.isinf(matrix[a][b]).any() == False

    del(matrix)
    del(vector)
    del(filling_m)
    del(filling_v)
    del(function_m)
    del(function_mv)


    # test full matrices
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    vector =  [0, 0, 0]

    for space in spaces:
        for a in range(3):
            
            Nbase_form = getattr(SPACES, 'Nbase_'+space+'form')

            Ni = Nbase_form[a]

            vector[a] = np.zeros((Ni[0], Ni[1], Ni[2]), dtype=float)
            vector[a][:,:,:] = 0.
            
            for b in range(3):

                matrix[a][b] = np.zeros( (Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
                matrix[a][b][:,:,:,:,:,:] = 0.

        filling_m = 10.*np.random.rand(3,3)
        filling_v = 10.*np.random.rand(3)

        filling_m = 10.*np.random.rand(3,3)
        filling_v = 10.*np.random.rand(3)

        function_m  = getattr(mat_vec_filler, 'mat_fill_b_v'+space+'_full')
        function_mv = getattr(mat_vec_filler, 'm_v_fill_b_v'+space+'_full')

        function_m(np.array(p), t1, t2, t3, indn1, indn2, indn3, eta1, eta2, eta3, matrix[0][0], matrix[0][1], matrix[0][2], matrix[1][0], matrix[1][1], matrix[1][2], matrix[2][0], matrix[2][1], matrix[2][2], filling_m[0][0], filling_m[0][1], filling_m[0][2], filling_m[1][0], filling_m[1][1], filling_m[1][2], filling_m[2][0], filling_m[2][1], filling_m[2][2])

        for a in range(3):
            for b in range(a,3):
                assert np.isnan(matrix[a][b]).any() == False
                assert np.isinf(matrix[a][b]).any() == False


        function_mv(np.array(p), t1, t2, t3, indn1, indn2, indn3, eta1, eta2, eta3, matrix[0][0], matrix[0][1], matrix[0][2], matrix[1][0], matrix[1][1], matrix[1][2], matrix[2][0], matrix[2][1], matrix[2][2], filling_m[0][0], filling_m[0][1], filling_m[0][2], filling_m[1][0], filling_m[1][1], filling_m[1][2], filling_m[2][0], filling_m[2][1], filling_m[2][2], vector[0], vector[1], vector[2], filling_v[0], filling_v[1], filling_v[2])
        
        for a in range(3):
            assert np.isnan(vector[a]).any() == False
            assert np.isinf(vector[a]).any() == False

            for b in range(3):
                assert np.isnan(matrix[a][b]).any() == False
                assert np.isinf(matrix[a][b]).any() == False

    del(matrix)
    del(vector)
    del(filling_m)
    del(filling_v)
    del(function_m)
    del(function_mv)







    # ========================================================================================= 
    # Test non _b Functions at Random Position
    # =========================================================================================
    eta1, eta2, eta3 = np.random.rand(3)

    # degrees of the basis functions : B-splines (pn) and D-splines(pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing N-splines at particle position
    bn1 = np.zeros( pn1 + 1, dtype=float)
    bn2 = np.zeros( pn2 + 1, dtype=float)
    bn3 = np.zeros( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = np.zeros( pd1 + 1, dtype=float)
    bd2 = np.zeros( pd2 + 1, dtype=float)
    bd3 = np.zeros( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(t1, pn1, eta1)
    span2 = bsp.find_span(t2, pn2, eta2)
    span3 = bsp.find_span(t3, pn3, eta3)
    span = [ span1, span2, span3 ]

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(t1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(t2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(t3, pn3, eta3, span3, bn3, bd3)

    # test diagonal and antisymmetric matrices
    kinds  = ['diag', 'asym']
    spaces = ['1', '2']

    matrix = [0, 0, 0] # since for diag und asym there are only 3 independent matrix entries, it suffices to only have 3 matrix entries like a vector
    vector = [0, 0, 0]


    for space in spaces:
        for a in range(3):
            
            Nbase_form = getattr(SPACES, 'Nbase_'+space+'form')

            Ni = Nbase_form[a]

            vector[a] = np.zeros((Ni[0], Ni[1], Ni[2]), dtype=float)
            vector[a][:,:,:] = 0.
            
            matrix[a] = np.zeros( (Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
            matrix[a][:,:,:,:,:,:] = 0.

        for kind in kinds:

            filling_m = 10. - 20.*np.random.rand(3)
            
            function_m  = getattr(mat_vec_filler, 'mat_fill_v'+space+'_'+kind)
            function_mv = getattr(mat_vec_filler, 'm_v_fill_v'+space+'_'+kind)

            function_m(np.array(p), np.array(span), bn1, bn2, bn3, bd1, bd2, bd3, indn1, indn2, indn3, matrix[1], matrix[1], matrix[2], filling_m[0], filling_m[1], filling_m[2])

            for a in range(3):
                assert np.isnan(matrix[a]).any() == False
                assert np.isinf(matrix[a]).any() == False

            filling_m = 10. - 20.*np.random.rand(3)
            filling_v = 10. - 20.*np.random.rand(3)
            
            function_mv(np.array(p), np.array(span), bn1, bn2, bn3, bd1, bd2, bd3, indn1, indn2, indn3, matrix[0], matrix[1], matrix[2], filling_m[0], filling_m[1], filling_m[2], vector[0], vector[1], vector[2], filling_v[0], filling_v[1], filling_v[2])

            for a in range(3):    
                assert np.isnan(matrix[a]).any() == False
                assert np.isinf(matrix[a]).any() == False
                assert np.isnan(vector[a]).any() == False
                assert np.isinf(vector[a]).any() == False

    del(matrix)
    del(vector)
    del(filling_m)
    del(filling_v)
    del(function_m)
    del(function_mv)




    # test symmetric matrices
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    vector =  [0, 0, 0]

    for space in spaces:
        for a in range(3):
            
            Nbase_form = getattr(SPACES, 'Nbase_'+space+'form')

            Ni = Nbase_form[a]

            vector[a] = np.zeros((Ni[0], Ni[1], Ni[2]), dtype=float)
            vector[a][:,:,:] = 0.
            
            for b in range(a, 3):

                matrix[a][b] = np.zeros( (Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
                matrix[a][b][:,:,:,:,:,:] = 0.

                matrix[a][b] = np.empty( (Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
                matrix[a][b][:,:,:,:,:,:] = 0.

        function_m  = getattr(mat_vec_filler, 'mat_fill_v'+space+'_symm')
        function_mv = getattr(mat_vec_filler, 'm_v_fill_v'+space+'_symm')

        filling_m = 10. - 20.*np.random.rand(3,3)

        function_m(np.array(p), np.array(span), bn1, bn2, bn3, bd1, bd2, bd3, indn1, indn2, indn3, matrix[0][0], matrix[0][1], matrix[0][2], matrix[1][1], matrix[1][2], matrix[2][2], filling_m[0][0], filling_m[0][1], filling_m[0][2], filling_m[1][1], filling_m[1][2], filling_m[2][2])

        for a in range(3):
            for b in range(a,3):
                assert np.isnan(matrix[a][b]).any() == False
                assert np.isinf(matrix[a][b]).any() == False


        filling_m = 10. - 20.*np.random.rand(3,3)
        filling_v = 10. - 20.*np.random.rand(3)

        function_mv(np.array(p), np.array(span), bn1, bn2, bn3, bd1, bd2, bd3, indn1, indn2, indn3, matrix[0][0], matrix[0][1], matrix[0][2], matrix[1][1], matrix[1][2], matrix[2][2], filling_m[0][0], filling_m[0][1], filling_m[0][2], filling_m[1][1], filling_m[1][2], filling_m[2][2], vector[0], vector[1], vector[2], filling_v[0], filling_v[1], filling_v[2])
        
        for a in range(3):
            assert np.isnan(vector[a]).any() == False
            assert np.isinf(vector[a]).any() == False

            for b in range(a,3):
                assert np.isnan(matrix[a][b]).any() == False
                assert np.isinf(matrix[a][b]).any() == False

    del(matrix)
    del(vector)
    del(filling_m)
    del(filling_v)
    del(function_m)
    del(function_mv)



    # test full matrices
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    vector =  [0, 0, 0]

    for space in spaces:
        for a in range(3):
            
            Nbase_form = getattr(SPACES, 'Nbase_'+space+'form')

            Ni = Nbase_form[a]

            vector[a] = np.zeros((Ni[0], Ni[1], Ni[2]), dtype=float)
            vector[a][:,:,:] = 0.
            
            for b in range(3):

                matrix[a][b] = np.zeros( (Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
                matrix[a][b][:,:,:,:,:,:] = 0.

        function_m  = getattr(mat_vec_filler, 'mat_fill_v'+space+'_full')
        function_mv = getattr(mat_vec_filler, 'm_v_fill_v'+space+'_full')

        filling_m = 10. - 20.*np.random.rand(3,3)

        function_m(np.array(p), np.array(span), bn1, bn2, bn3, bd1, bd2, bd3, indn1, indn2, indn3, matrix[0][0], matrix[0][1], matrix[0][2], matrix[1][0], matrix[1][1], matrix[1][2], matrix[2][0], matrix[2][1], matrix[2][2], filling_m[0][0], filling_m[0][1], filling_m[0][2], filling_m[1][0], filling_m[1][1], filling_m[1][2], filling_m[2][0], filling_m[2][1], filling_m[2][2])

        for a in range(3):
            for b in range(3):
                assert np.isnan(matrix[a][b]).any() == False
                assert np.isinf(matrix[a][b]).any() == False

        filling_m = 10. - 20.*np.random.rand(3,3)
        filling_v = 10. - 20.*np.random.rand(3)

        function_mv(np.array(p), np.array(span), bn1, bn2, bn3, bd1, bd2, bd3, indn1, indn2, indn3, matrix[0][0], matrix[0][1], matrix[0][2], matrix[1][0], matrix[1][1], matrix[1][2], matrix[2][0], matrix[2][1], matrix[2][2], filling_m[0][0], filling_m[0][1], filling_m[0][2], filling_m[1][0], filling_m[1][1], filling_m[1][2], filling_m[2][0], filling_m[2][1], filling_m[2][2], vector[0], vector[1], vector[2], filling_v[0], filling_v[1], filling_v[2])
        
        for a in range(3):
            assert np.isnan(vector[a]).any() == False
            assert np.isinf(vector[a]).any() == False

            for b in range(3):
                assert np.isnan(matrix[a][b]).any() == False
                assert np.isinf(matrix[a][b]).any() == False

    del(matrix)
    del(vector)
    del(filling_m)
    del(filling_v)
    del(function_m)
    del(function_mv)


        # TODO: v0 and v3 space

    print('test_mat_vec_filler passed!')

if __name__ == '__main__':
    Nel         = [4,5,6]
    p           = [2,2,3]
    spl_kind    = [True, True, True]
    test_Mat_Vec_Filler(Nel, p, spl_kind)
