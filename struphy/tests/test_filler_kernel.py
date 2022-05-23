import pytest

import struphy.feec.bsplines_kernels as bsp
from   struphy.feec                  import spline_space 
from   struphy.pic                   import filler_kernel

import numpy as np


@pytest.mark.parametrize('Nel', [[8,5,6], [4, 4, 128]])
@pytest.mark.parametrize('p', [[2,3,2], [1,1,4]])
@pytest.mark.parametrize('spl_kind', [[True, True, True], [False, False, True]])
def test_Filler_Kernel(Nel, p, spl_kind):
    # ========================================================================================= 
    # FEEC SPACES Object & related quantities
    # =========================================================================================

    spaces_FEM_1 = spline_space.Spline_space_1d(Nel[0], p[0], spl_kind[0]) 
    spaces_FEM_2 = spline_space.Spline_space_1d(Nel[1], p[1], spl_kind[1])
    spaces_FEM_3 = spline_space.Spline_space_1d(Nel[2], p[2], spl_kind[2])

    SPACES = spline_space.Tensor_spline_space([spaces_FEM_1, spaces_FEM_2, spaces_FEM_3])

    t1 = SPACES.T[0]
    t2 = SPACES.T[1]
    t3 = SPACES.T[2]

    indn1, indn2, indn3 = SPACES.indN[0], SPACES.indN[1], SPACES.indN[2]
    indd1, indd2, indd3 = SPACES.indD[0], SPACES.indD[1], SPACES.indD[2]



    # ========================================================================================= 
    # Compute non-vanishing Basis Functions at Random Position
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
    bn1 = np.empty( pn1 + 1, dtype=float)
    bn2 = np.empty( pn2 + 1, dtype=float)
    bn3 = np.empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = np.empty( pd1 + 1, dtype=float)
    bd2 = np.empty( pd2 + 1, dtype=float)
    bd3 = np.empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(t1, pn1, eta1)
    span2 = bsp.find_span(t2, pn2, eta2)
    span3 = bsp.find_span(t3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(t1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(t2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(t3, pn3, eta3, span3, bn3, bd3)

    # global indices of non-vanishing basis functions
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3


    # ========================================================================================= 
    # Test filler_kernel
    # =========================================================================================

    # test v1 functions
    for a in range(3):
        
        Ni = SPACES.Nbase_1form[a]

        if a+1 == 1:
            b1 = bd1
            b2 = bn2
            b3 = bn3

            ind1 = indd1[ie1,:]
            ind2 = indn2[ie2,:]
            ind3 = indn3[ie3,:]

        elif a+1 == 2:
            b1 = bn1
            b2 = bd2
            b3 = bn3

            ind1 = indn1[ie1,:]
            ind2 = indd2[ie2,:]
            ind3 = indn3[ie3,:]

        elif a+1 == 3:
            b1 = bn1
            b2 = bn2
            b3 = bd3

            ind1 = indn1[ie1,:]
            ind2 = indn2[ie2,:]
            ind3 = indd3[ie3,:]
        
        else:
            raise ValueError('something is wrong, I can feel it')

        vector = np.empty((Ni[0], Ni[1], Ni[2]), dtype=float)
        vector[:,:,:] = 0.
        
        filling_v = 10. - 20.*np.random.rand()

        # test vector functions
        function_v  = getattr(filler_kernel, 'fill_vec'+str(a+1)+'_v1')
        function_v(np.array(p), b1, b2, b3 , ind1, ind2, ind3, vector, filling_v)
        
        assert np.isnan(vector).any() == False
        assert np.isinf(vector).any() == False
        
        for b in range(3):
            
            matrix = np.empty( (Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
            matrix[:,:,:,:,:,:] = 0.

            function_m  = getattr(filler_kernel, 'fill_mat'+str(a+1)+str(b+1)+'_v1')
            function_mv = getattr(filler_kernel, 'fill_mat'+str(a+1)+str(b+1)+'_vec'+str(a+1)+'_v1')

            if a == b:

                filling_m = 10. - 20.*np.random.rand()
    
                function_m(np.array(p), b1, b2, b3, ind1, ind2, ind3, matrix, filling_m)
                
                assert np.isnan(matrix).any() == False
                assert np.isinf(matrix).any() == False

                filling_m = 10. - 20.*np.random.rand()
                filling_v = 10. - 20.*np.random.rand()

                function_mv(np.array(p), b1, b2, b3, ind1, ind2, ind3, matrix, filling_m, vector, filling_v)

                assert np.isnan(matrix).any() == False
                assert np.isinf(matrix).any() == False
                assert np.isnan(vector).any() == False
                assert np.isinf(vector).any() == False

            else:
                if not 3 in [a,b]:
                    b1 = bn1
                    b2 = bd1
                    b3 = bn2
                    b4 = bd2
                    b5 = bn3

                elif not 2 in [a,b]:
                    b1 = bn1
                    b2 = bd1
                    b3 = bn2
                    b4 = bn3
                    b5 = bd3

                elif not 1 in [a,b]:
                    b1 = bn1
                    b2 = bn2
                    b3 = bd2
                    b4 = bn3
                    b5 = bd3
                
                else:
                    raise ValueError('off-diagonal is not quite right')

                filling_m = 10. - 20.*np.random.rand()

                function_m(np.array(p), b1, b2, b3, b4, b5, ind1, ind2, ind3, matrix, filling_m)

                assert np.isnan(matrix).any() == False
                assert np.isinf(matrix).any() == False

                filling_m = 10. - 20.*np.random.rand()
                filling_v = 10. - 20.*np.random.rand()

                function_mv(np.array(p), b1, b2, b3, b4, b5, ind1, ind2, ind3, matrix, filling_m, vector, filling_v)

                assert np.isnan(matrix).any() == False
                assert np.isinf(matrix).any() == False
                assert np.isnan(vector).any() == False
                assert np.isinf(vector).any() == False
                
        del(matrix)
        del(vector)
        del(filling_m)
        del(filling_v)
        del(function_m)
        del(function_mv)
        del(function_v)



    # test v2 functions
    for a in range(3):
        
        Ni = SPACES.Nbase_2form[a]

        if a+1 == 1:
            b1 = bn1
            b2 = bd2
            b3 = bd3

            ind1 = indn1[ie1,:]
            ind2 = indd2[ie2,:]
            ind3 = indd3[ie3,:]

        elif a+1 == 2:
            b1 = bd1
            b2 = bn2
            b3 = bd3

            ind1 = indd1[ie1,:]
            ind2 = indn2[ie2,:]
            ind3 = indd3[ie3,:]

        elif a+1 == 3:
            b1 = bd1
            b2 = bd2
            b3 = bn3

            ind1 = indd1[ie1,:]
            ind2 = indn2[ie2,:]
            ind3 = indn3[ie3,:]
        
        else:
            raise ValueError('something is wrong, I can feel it')

        vector = np.empty((Ni[0], Ni[1], Ni[2]), dtype=float)
        vector[:,:,:] = 0.
        
        filling_v = 10. - 20.*np.random.rand()

        # test vector functions
        function_v  = getattr(filler_kernel, 'fill_vec'+str(a+1)+'_v2')
        function_v(np.array(p), b1, b2, b3 , ind1, ind2, ind3, vector, filling_v)

        assert np.isnan(vector).any() == False
        assert np.isinf(vector).any() == False

        for b in range(3):
            
            matrix = np.empty( (Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
            matrix[:,:,:,:,:,:] = 0.

            function_m  = getattr(filler_kernel, 'fill_mat'+str(a+1)+str(b+1)+'_v2')
            function_mv = getattr(filler_kernel, 'fill_mat'+str(a+1)+str(b+1)+'_vec'+str(a+1)+'_v2')

            if a == b:

                filling_m = 10. - 20.*np.random.rand()

                function_m(np.array(p), b1, b2, b3, ind1, ind2, ind3, matrix, filling_m)

                assert np.isnan(matrix).any() == False
                assert np.isinf(matrix).any() == False

                filling_m = 10. - 20.*np.random.rand()
                filling_v = 10. - 20.*np.random.rand()

                function_mv(np.array(p), b1, b2, b3, ind1, ind2, ind3, matrix, filling_m, vector, filling_v)

                assert np.isnan(matrix).any() == False
                assert np.isinf(matrix).any() == False
                assert np.isnan(vector).any() == False
                assert np.isinf(vector).any() == False

            else:
                if not 3 in [a,b]:
                    b1 = bn1
                    b2 = bd1
                    b3 = bn2
                    b4 = bd2
                    b5 = bd3

                elif not 2 in [a,b]:
                    b1 = bn1
                    b2 = bd1
                    b3 = bd2
                    b4 = bn3
                    b5 = bd3

                elif not 1 in [a,b]:
                    b1 = bd1
                    b2 = bn2
                    b3 = bd2
                    b4 = bn3
                    b5 = bd3
                
                else:
                    raise ValueError('off-diagonal is not quite right')

                filling_m = 10. - 20.*np.random.rand()

                function_m(np.array(p), b1, b2, b3, b4, b5, ind1, ind2, ind3, matrix, filling_m)

                assert np.isnan(matrix).any() == False
                assert np.isinf(matrix).any() == False

                filling_m = 10. - 20.*np.random.rand()
                filling_v = 10. - 20.*np.random.rand()
                
                function_mv(np.array(p), b1, b2, b3, b4, b5, ind1, ind2, ind3, matrix, filling_m, vector, filling_v)
                
                assert np.isnan(matrix).any() == False
                assert np.isinf(matrix).any() == False
                assert np.isnan(vector).any() == False
                assert np.isinf(vector).any() == False

        del(matrix)
        del(vector)
        del(filling_m)
        del(filling_v)
        del(function_m)
        del(function_mv)
        del(function_v)

    print('test_filler_kernel passed!')

if __name__ == '__main__':
    Nel         = [4,5,6]
    p           = [2,2,3]
    spl_kind    = [True, True, True]
    test_Filler_Kernel(Nel, p, spl_kind)
