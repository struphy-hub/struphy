import struphy.feec.bsplines_kernels as bsp
import struphy.pic.filler_kernels as fk

def _docstring():
    """
    MODULE DOCSTRING for **struphy.pic.mat_vec_filler**.

    The module contains pyccelized functions to add one particle to a FE matrix and vector in accumulation step.

    Computed are only the independent components of the matrix (e.g. m12,m13,m23 for antisymmetric).
    Matrix fillings carry 2 indices (mu-nu) while vector fillings only carry one index (mu).

    Naming conventions:

    1) mat_ adds only to a matrix of the respective space, m_v_ adds to a matrix and a vector.

    2) The functions with _b compute the pn+1 non-vanishing basis functions Lambda^p_ijk(eta) at the point eta.
    In case Lambda^p_ijk(eta) has already been computed for the filling, it can be passed (functions without _b).

    3) vn with n=0,1,2,3 denotes the discrete space from the 3d Derham sequence the matrix/vector belongs to.

    4) diag/asym/symm/full refer to the property of the block matrix (for v1 or v2) and define which independent components are computed.

    5) functions u0 and u3 are not geometric form coefficient vectors, but three vectors with each component in V0 or V3. (replacing the basis_u
        case distinction in mhd codes.)

    __all__ = [ 'mat_fill_b_v1_diag',
                'm_v_fill_b_v1_diag',
                'mat_fill_b_v2_diag',
                'm_v_fill_b_v2_diag',
                'mat_fill_b_v1_asym',
                'm_v_fill_b_v1_asym',
                'mat_fill_b_v2_asym',
                'm_v_fill_b_v2_asym',
                'mat_fill_b_v1_symm',
                'm_v_fill_b_v1_symm',
                'mat_fill_b_v2_symm',
                'm_v_fill_b_v2_symm',
                'mat_fill_b_v1_full',
                'm_v_fill_b_v1_full',
                'mat_fill_b_v2_full',
                'm_v_fill_b_v2_full',
                'mat_fill_v1_diag',
                'm_v_fill_v1_diag',
                'mat_fill_v2_diag',
                'm_v_fill_v2_diag',
                'mat_fill_v1_asym',
                'm_v_fill_v1_asym',
                'mat_fill_v2_asym',
                'm_v_fill_v2_asym',
                'mat_fill_v1_symm',
                'm_v_fill_v1_symm',
                'mat_fill_v2_symm',
                'm_v_fill_v2_symm',
                'mat_fill_v1_full',
                'm_v_fill_v1_full',
                'mat_fill_v2_full',
                'm_v_fill_v2_full',
                'mat_fill_b_v0',
                'm_v_fill_b_v0',
                'mat_fill_b_v3',
                'm_v_fill_b_v3',
                'mat_fill_v0',
                'm_v_fill_v0',
                'mat_fill_v3',
                'm_v_fill_v3',
                'mat_fill_b_u0_diag',
                'm_v_fill_b_u0_diag',
                'mat_fill_b_u3_diag',
                'm_v_fill_b_u3_diag',
                'mat_fill_b_u0_asym',
                'm_v_fill_b_u0_asym',
                'mat_fill_b_u3_asym',
                'm_v_fill_b_u3_asym',
                'mat_fill_b_u0_symm',
                'm_v_fill_b_u0_symm',
                'mat_fill_b_u3_symm',
                'm_v_fill_b_u3_symm',
                'mat_fill_b_u0_full',
                'm_v_fill_b_u0_full',
                'mat_fill_b_u3_full',
                'm_v_fill_b_u3_full',
                'mat_fill_u0_diag',
                'm_v_fill_u0_diag',
                'mat_fill_u3_diag',
                'm_v_fill_u3_diag',
                'mat_fill_u0_asym',
                'm_v_fill_u0_asym',
                'mat_fill_u3_asym',
                'm_v_fill_u3_asym',
                'mat_fill_u0_symm',
                'm_v_fill_u0_symm',
                'mat_fill_u3_symm',
                'm_v_fill_u3_symm',
                'mat_fill_u0_full',
                'm_v_fill_u0_full',
                'mat_fill_u3_full',
                'm_v_fill_u3_full',
                ]
    """
    
    print('This is just the docstring function.')


def mat_fill_b_v1_diag(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat22 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : 'float[:,:,:,:,:,:]' : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat11_v1(pn, bd1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat22_v1(pn, bn1, bd2, bn3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat33_v1(pn, bn1, bn2, bd3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_b_v1_diag(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat22 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat11_vec1_v1(pn, bd1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat22_vec2_v1(pn, bn1, bd2, bn3, ie1, ie2, ie3, starts2, mat22, filling22, vec2, filling2)
    fk.fill_mat33_vec3_v1(pn, bn1, bn2, bd3, ie1, ie2, ie3, starts3, mat33, filling33, vec3, filling3)


def mat_fill_b_v2_diag(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat22 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V2 -> V2 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V2 -> V2 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V2 -> V2 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V2 and written to mat11

        filling22 : float
            number that will be multiplied by the basis functions of V2 and written to mat22

        filling33 : float
            number that will be multiplied by the basis functions of V2 and written to mat33
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat11_v2(pn, bn1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat22_v2(pn, bd1, bn2, bd3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat33_v2(pn, bd1, bd2, bn3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_b_v2_diag(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat22 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V2 -> V2 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V2 -> V2 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V2 -> V2 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V2 and written to mat11

        filling22 : float
            number that will be multiplied by the basis functions of V2 and written to mat22

        filling33 : float
            number that will be multiplied by the basis functions of V2 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V2 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V2 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V2 and written to vec3
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat11_vec1_v2(pn, bn1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat22_vec2_v2(pn, bd1, bn2, bd3, ie1, ie2, ie3, starts2, mat22, filling22, vec2, filling2)
    fk.fill_mat33_vec3_v2(pn, bd1, bd2, bn3, ie1, ie2, ie3, starts3, mat33, filling33, vec3, filling3)


def mat_fill_b_v1_asym(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float'):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V1 -> V1. The result is returned in mat12, mat13 and mat23.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat12_v1(pn, bn1, bd1, bn2, bd2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v1(pn, bn1, bd1, bn2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat23_v1(pn, bn1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)


def m_v_fill_b_v1_asym(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V1 -> V1. The result is returned in mat12, mat13 and mat23.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat12_vec1_v1(pn, bn1, bd1, bn2, bd2, bn3, ie1, ie2, ie3, starts1, mat12, filling12, vec1, filling1)
    fk.fill_mat13_v1(pn, bn1, bd1, bn2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat23_vec2_v1(pn, bn1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23, vec2, filling2)
    fk.fill_vec3_v1(pn, bn1, bn2, bd3, ie1, ie2, ie3, starts3, vec3, filling3)


def mat_fill_b_v2_asym(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float'):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V2 -> V2. The result is returned in mat12, mat13 and mat23.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3

        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat12 : array
            mu=1, nu=2 element of the block matrix V2 -> V2 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V2 -> V2 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V2 -> V2 that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V2 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V2 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V2 and written to mat23
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat12_v2(pn, bn1, bd1, bn2, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v2(pn, bn1, bd1, bd2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat23_v2(pn, bd1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)


def m_v_fill_b_v2_asym(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V2 -> V2. The result is returned in mat12, mat13 and mat23.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat12 : array
            mu=1, nu=2 element of the block matrix V2 -> V2 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V2 -> V2 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V2 -> V2 that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V2 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V2 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V2 and written to mat23
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V2 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V2 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V2 and written to vec3
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat12_vec1_v2(pn, bn1, bd1, bn2, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12, vec1, filling1)
    fk.fill_mat13_v2(pn, bn1, bd1, bd2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat23_vec2_v2(pn, bd1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23, vec2, filling2)
    fk.fill_vec3_v2(pn, bd1, bd2, bn3, ie1, ie2, ie3, starts3, vec3, filling3)


def mat_fill_b_v1_symm(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat11_v1(pn, bd1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat12_v1(pn, bn1, bd1, bn2, bd2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v1(pn, bn1, bd1, bn2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat22_v1(pn, bn1, bd2, bn3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat23_v1(pn, bn1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat33_v1(pn, bn1, bn2, bd3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_b_v1_symm(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat11_vec1_v1(pn, bd1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat12_v1(pn, bn1, bd1, bn2, bd2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v1(pn, bn1, bd1, bn2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat22_vec2_v1(pn, bn1, bd2, bn3, ie1, ie2, ie3, starts2, mat22, filling22, vec2, filling2)
    fk.fill_mat23_v1(pn, bn1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat33_vec3_v1(pn, bn1, bn2, bd3, ie1, ie2, ie3, starts3, mat33, filling33, vec3, filling3)


def mat_fill_b_v2_symm(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V2 -> V2 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V2 -> V2 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V2 -> V2 that is written to
        
        mat22 : array
            mu=2, nu=2 element of the block matrix V2 -> V2 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V2 -> V2 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V2 -> V2 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V2 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V2 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V2 and written to mat13

        filling22 : float
            number that will be multiplied by the basis functions of V2 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V2 and written to mat23

        filling33 : float
            number that will be multiplied by the basis functions of V2 and written to mat33
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat11_v2(pn, bn1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat12_v2(pn, bn1, bd1, bn2, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v2(pn, bn1, bd1, bd2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat22_v2(pn, bd1, bn2, bd3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat23_v2(pn, bd1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat33_v2(pn, bd1, bd2, bn3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_b_v2_symm(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V2 -> V2 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V2 -> V2 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V2 -> V2 that is written to
        
        mat22 : array
            mu=2, nu=2 element of the block matrix V2 -> V2 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V2 -> V2 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V2 -> V2 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V2 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V2 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V2 and written to mat13

        filling22 : float
            number that will be multiplied by the basis functions of V2 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V2 and written to mat23

        filling33 : float
            number that will be multiplied by the basis functions of V2 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V2 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V2 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V2 and written to vec3
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat11_vec1_v2(pn, bn1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat12_v2(pn, bn1, bd1, bn2, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v2(pn, bn1, bd1, bd2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat22_vec2_v2(pn, bd1, bn2, bd3, ie1, ie2, ie3, starts2, mat22, filling22, vec2, filling2)
    fk.fill_mat23_v2(pn, bd1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat33_vec3_v2(pn, bd1, bd2, bn3, ie1, ie2, ie3, starts3, mat33, filling33, vec3, filling3)


def mat_fill_b_v1_full(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat21 : array
            mu=2, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat31 : array
            mu=3, nu=1 element of the block matrix V1 -> V1 that is written to

        mat32 : array
            mu=3, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling21 : float
            number that will be multiplied by the basis functions of V1 and written to mat21

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling31 : float
            number that will be multiplied by the basis functions of V1 and written to mat31

        filling32 : float
            number that will be multiplied by the basis functions of V1 and written to mat32

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat11_v1(pn, bd1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat12_v1(pn, bn1, bd1, bn2, bd2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v1(pn, bn1, bd1, bn2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat21_v1(pn, bn1, bd1, bn2, bd2, bn3, ie1, ie2, ie3, starts2, mat21, filling21)
    fk.fill_mat22_v1(pn, bn1, bd2, bn3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat23_v1(pn, bn1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat31_v1(pn, bn1, bd1, bn2, bn3, bd3, ie1, ie2, ie3, starts3, mat31, filling31)
    fk.fill_mat32_v1(pn, bn1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts3, mat32, filling32)
    fk.fill_mat33_v1(pn, bn1, bn2, bd3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_b_v1_full(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]',  filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat21 : array
            mu=2, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat31 : array
            mu=3, nu=1 element of the block matrix V1 -> V1 that is written to

        mat32 : array
            mu=3, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling21 : float
            number that will be multiplied by the basis functions of V1 and written to mat21

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling31 : float
            number that will be multiplied by the basis functions of V1 and written to mat31

        filling32 : float
            number that will be multiplied by the basis functions of V1 and written to mat32

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat11_vec1_v1(pn, bd1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat12_v1(pn, bn1, bd1, bn2, bd2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v1(pn, bn1, bd1, bn2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat21_vec2_v1(pn, bn1, bd1, bn2, bd2, bn3, ie1, ie2, ie3, starts2, mat21, filling21, vec2, filling2)
    fk.fill_mat22_v1(pn, bn1, bd2, bn3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat23_v1(pn, bn1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat31_vec3_v1(pn, bn1, bd1, bn2, bn3, bd3, ie1, ie2, ie3, starts3, mat31, filling31, vec3, filling3)
    fk.fill_mat32_v1(pn, bn1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts3, mat32, filling32)
    fk.fill_mat33_v1(pn, bn1, bn2, bd3, ie1, ie2, ie3, starts3, mat33, filling33)


def mat_fill_b_v2_full(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V2 -> V2 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V2 -> V2 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V2 -> V2 that is written to
        
        mat21 : array
            mu=2, nu=1 element of the block matrix V2 -> V2 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V2 -> V2 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V2 -> V2 that is written to
        
        mat31 : array
            mu=3, nu=1 element of the block matrix V2 -> V2 that is written to

        mat32 : array
            mu=3, nu=2 element of the block matrix V2 -> V2 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V2 -> V2 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V2 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V2 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V2 and written to mat13

        filling21 : float
            number that will be multiplied by the basis functions of V2 and written to mat21

        filling22 : float
            number that will be multiplied by the basis functions of V2 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V2 and written to mat23

        filling31 : float
            number that will be multiplied by the basis functions of V2 and written to mat31

        filling32 : float
            number that will be multiplied by the basis functions of V2 and written to mat32

        filling33 : float
            number that will be multiplied by the basis functions of V2 and written to mat33
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat11_v2(pn, bn1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat12_v2(pn, bn1, bd1, bn2, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v2(pn, bn1, bd1, bd2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat21_v2(pn, bn1, bd1, bn2, bd2, bd3, ie1, ie2, ie3, starts2, mat21, filling21)
    fk.fill_mat22_v2(pn, bd1, bn2, bd3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat23_v2(pn, bd1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat31_v2(pn, bn1, bd1, bd2, bn3, bd3, ie1, ie2, ie3, starts3, mat31, filling31)
    fk.fill_mat32_v2(pn, bd1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts3, mat32, filling32)
    fk.fill_mat33_v2(pn, bd1, bd2, bn3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_b_v2_full(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]',  filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V2 -> V2 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V2 -> V2 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V2 -> V2 that is written to
        
        mat21 : array
            mu=2, nu=1 element of the block matrix V2 -> V2 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V2 -> V2 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V2 -> V2 that is written to
        
        mat31 : array
            mu=3, nu=1 element of the block matrix V2 -> V2 that is written to

        mat32 : array
            mu=3, nu=2 element of the block matrix V2 -> V2 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V2 -> V2 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V2 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V2 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V2 and written to mat13

        filling21 : float
            number that will be multiplied by the basis functions of V2 and written to mat21

        filling22 : float
            number that will be multiplied by the basis functions of V2 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V2 and written to mat23

        filling31 : float
            number that will be multiplied by the basis functions of V2 and written to mat31

        filling32 : float
            number that will be multiplied by the basis functions of V2 and written to mat32

        filling33 : float
            number that will be multiplied by the basis functions of V2 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V2 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V2 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V2 and written to vec3
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat11_vec1_v2(pn, bn1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat12_v2(pn, bn1, bd1, bn2, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v2(pn, bn1, bd1, bd2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat21_vec2_v2(pn, bn1, bd1, bn2, bd2, bd3, ie1, ie2, ie3, starts2, mat21, filling21, vec2, filling2)
    fk.fill_mat22_v2(pn, bd1, bn2, bd3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat23_v2(pn, bd1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat31_vec3_v2(pn, bn1, bd1, bd2, bn3, bd3, ie1, ie2, ie3, starts3, mat31, filling31, vec3, filling3)
    fk.fill_mat32_v2(pn, bd1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts3, mat32, filling32)
    fk.fill_mat33_v2(pn, bd1, bd2, bn3, ie1, ie2, ie3, starts3, mat33, filling33)


def mat_fill_v1_diag(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat11 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat22 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span1: 'int', span2: 'int', span3: 'int' : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat11_v1(pn, bd1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat22_v1(pn, bn1, bd2, bn3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat33_v1(pn, bn1, bn2, bd3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_v1_diag(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat11 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat22 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat11_vec1_v1(pn, bd1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat22_vec2_v1(pn, bn1, bd2, bn3, ie1, ie2, ie3, starts2, mat22, filling22, vec2, filling2)
    fk.fill_mat33_vec3_v1(pn, bn1, bn2, bd3, ie1, ie2, ie3, starts3, mat33, filling33, vec3, filling3)


def mat_fill_v2_diag(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat11 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat22 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V2 -> V2 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V2 -> V2 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V2 -> V2 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V2 and written to mat11

        filling22 : float
            number that will be multiplied by the basis functions of V2 and written to mat22

        filling33 : float
            number that will be multiplied by the basis functions of V2 and written to mat33
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat11_v2(pn, bn1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat22_v2(pn, bd1, bn2, bd3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat33_v2(pn, bd1, bd2, bn3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_v2_diag(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat11 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat22 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V2 -> V2 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V2 -> V2 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V2 -> V2 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V2 and written to mat11

        filling22 : float
            number that will be multiplied by the basis functions of V2 and written to mat22

        filling33 : float
            number that will be multiplied by the basis functions of V2 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V2 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V2 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V2 and written to vec3
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat11_vec1_v2(pn, bn1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat22_vec2_v2(pn, bd1, bn2, bd3, ie1, ie2, ie3, starts2, mat22, filling22, vec2, filling2)
    fk.fill_mat33_vec3_v2(pn, bd1, bd2, bn3, ie1, ie2, ie3, starts3, mat33, filling33, vec3, filling3)


def mat_fill_v1_asym(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float'):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V1 -> V1. The result is returned in mat12, mat13 and mat23.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat12_v1(pn, bn1, bd1, bn2, bd2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v1(pn, bn1, bd1, bn2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat23_v1(pn, bn1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)


def m_v_fill_v1_asym(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V1 -> V1. The result is returned in mat12, mat13 and mat23.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat12_vec1_v1(pn, bn1, bd1, bn2, bd2, bn3, ie1, ie2, ie3, starts1, mat12, filling12, vec1, filling1)
    fk.fill_mat13_v1(pn, bn1, bd1, bn2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat23_vec2_v1(pn, bn1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23, vec2, filling2)
    fk.fill_vec3_v1(pn, bn1, bn2, bd3, ie1, ie2, ie3, starts3, vec3, filling3)


def mat_fill_v2_asym(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float'):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V2 -> V2. The result is returned in mat12, mat13 and mat23.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat12 : array
            mu=1, nu=2 element of the block matrix V2 -> V2 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V2 -> V2 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V2 -> V2 that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V2 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V2 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V2 and written to mat23
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat12_v2(pn, bn1, bd1, bn2, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v2(pn, bn1, bd1, bd2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat23_v2(pn, bd1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)


def m_v_fill_v2_asym(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V2 -> V2. The result is returned in mat12, mat13 and mat23.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat12 : array
            mu=1, nu=2 element of the block matrix V2 -> V2 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V2 -> V2 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V2 -> V2 that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V2 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V2 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V2 and written to mat23
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V2 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V2 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V2 and written to vec3
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat12_vec1_v2(pn, bn1, bd1, bn2, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12, vec1, filling1)
    fk.fill_mat13_v2(pn, bn1, bd1, bd2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat23_vec2_v2(pn, bd1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23, vec2, filling2)
    fk.fill_vec3_v2(pn, bd1, bd2, bn3, ie1, ie2, ie3, starts3, vec3, filling3)


def mat_fill_v1_symm(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat11_v1(pn, bd1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat12_v1(pn, bn1, bd1, bn2, bd2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v1(pn, bn1, bd1, bn2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat22_v1(pn, bn1, bd2, bn3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat23_v1(pn, bn1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat33_v1(pn, bn1, bn2, bd3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_v1_symm(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat11_vec1_v1(pn, bd1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat12_v1(pn, bn1, bd1, bn2, bd2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v1(pn, bn1, bd1, bn2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat22_vec2_v1(pn, bn1, bd2, bn3, ie1, ie2, ie3, starts2, mat22, filling22, vec2, filling2)
    fk.fill_mat23_v1(pn, bn1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat33_vec3_v1(pn, bn1, bn2, bd3, ie1, ie2, ie3, starts3, mat33, filling33, vec3, filling3)


def mat_fill_v2_symm(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V2 -> V2 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V2 -> V2 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V2 -> V2 that is written to
        
        mat22 : array
            mu=2, nu=2 element of the block matrix V2 -> V2 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V2 -> V2 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V2 -> V2 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V2 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V2 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V2 and written to mat13

        filling22 : float
            number that will be multiplied by the basis functions of V2 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V2 and written to mat23

        filling33 : float
            number that will be multiplied by the basis functions of V2 and written to mat33
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat11_v2(pn, bn1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat12_v2(pn, bn1, bd1, bn2, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v2(pn, bn1, bd1, bd2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat22_v2(pn, bd1, bn2, bd3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat23_v2(pn, bd1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat33_v2(pn, bd1, bd2, bn3, ie1, ie2, ie3, starts3, mat33, filling33)

    
def m_v_fill_v2_symm(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',  mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V2 -> V2 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V2 -> V2 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V2 -> V2 that is written to
        
        mat22 : array
            mu=2, nu=2 element of the block matrix V2 -> V2 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V2 -> V2 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V2 -> V2 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V2 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V2 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V2 and written to mat13

        filling22 : float
            number that will be multiplied by the basis functions of V2 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V2 and written to mat23

        filling33 : float
            number that will be multiplied by the basis functions of V2 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V2 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V2 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V2 and written to vec3
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat11_vec1_v2(pn, bn1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat12_v2(pn, bn1, bd1, bn2, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v2(pn, bn1, bd1, bd2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat22_vec2_v2(pn, bd1, bn2, bd3, ie1, ie2, ie3, starts2, mat22, filling22, vec2, filling2)
    fk.fill_mat23_v2(pn, bd1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat33_vec3_v2(pn, bd1, bd2, bn3, ie1, ie2, ie3, starts3, mat33, filling33, vec3, filling3)


def mat_fill_v1_full(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',  mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat21 : array
            mu=2, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat31 : array
            mu=3, nu=1 element of the block matrix V1 -> V1 that is written to

        mat32 : array
            mu=3, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling21 : float
            number that will be multiplied by the basis functions of V1 and written to mat21

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling31 : float
            number that will be multiplied by the basis functions of V1 and written to mat31

        filling32 : float
            number that will be multiplied by the basis functions of V1 and written to mat32

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat11_v1(pn, bd1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat12_v1(pn, bn1, bd1, bn2, bd2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v1(pn, bn1, bd1, bn2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat21_v1(pn, bn1, bd1, bn2, bd2, bn3, ie1, ie2, ie3, starts2, mat21, filling21)
    fk.fill_mat22_v1(pn, bn1, bd2, bn3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat23_v1(pn, bn1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat31_v1(pn, bn1, bd1, bn2, bn3, bd3, ie1, ie2, ie3, starts3, mat31, filling31)
    fk.fill_mat32_v1(pn, bn1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts3, mat32, filling32)
    fk.fill_mat33_v1(pn, bn1, bn2, bd3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_v1_full(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',  mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]',  filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat21 : array
            mu=2, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat31 : array
            mu=3, nu=1 element of the block matrix V1 -> V1 that is written to

        mat32 : array
            mu=3, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling21 : float
            number that will be multiplied by the basis functions of V1 and written to mat21

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling31 : float
            number that will be multiplied by the basis functions of V1 and written to mat31

        filling32 : float
            number that will be multiplied by the basis functions of V1 and written to mat32

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat11_vec1_v1(pn, bd1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat12_v1(pn, bn1, bd1, bn2, bd2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v1(pn, bn1, bd1, bn2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat21_vec2_v1(pn, bn1, bd1, bn2, bd2, bn3, ie1, ie2, ie3, starts2, mat21, filling21, vec2, filling2)
    fk.fill_mat22_v1(pn, bn1, bd2, bn3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat23_v1(pn, bn1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat31_vec3_v1(pn, bn1, bd1, bn2, bn3, bd3, ie1, ie2, ie3, starts3, mat31, filling31, vec3, filling3)
    fk.fill_mat32_v1(pn, bn1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts3, mat32, filling32)
    fk.fill_mat33_v1(pn, bn1, bn2, bd3, ie1, ie2, ie3, starts3, mat33, filling33)


def mat_fill_v2_full(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',  mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V2 -> V2 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V2 -> V2 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V2 -> V2 that is written to
        
        mat21 : array
            mu=2, nu=1 element of the block matrix V2 -> V2 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V2 -> V2 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V2 -> V2 that is written to
        
        mat31 : array
            mu=3, nu=1 element of the block matrix V2 -> V2 that is written to

        mat32 : array
            mu=3, nu=2 element of the block matrix V2 -> V2 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V2 -> V2 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V2 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V2 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V2 and written to mat13

        filling21 : float
            number that will be multiplied by the basis functions of V2 and written to mat21

        filling22 : float
            number that will be multiplied by the basis functions of V2 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V2 and written to mat23

        filling31 : float
            number that will be multiplied by the basis functions of V2 and written to mat31

        filling32 : float
            number that will be multiplied by the basis functions of V2 and written to mat32

        filling33 : float
            number that will be multiplied by the basis functions of V2 and written to mat33
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat11_v2(pn, bn1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat12_v2(pn, bn1, bd1, bn2, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v2(pn, bn1, bd1, bd2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat21_v2(pn, bn1, bd1, bn2, bd2, bd3, ie1, ie2, ie3, starts2, mat21, filling21)
    fk.fill_mat22_v2(pn, bd1, bn2, bd3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat23_v2(pn, bd1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat31_v2(pn, bn1, bd1, bd2, bn3, bd3, ie1, ie2, ie3, starts3, mat31, filling31)
    fk.fill_mat32_v2(pn, bd1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts3, mat32, filling32)
    fk.fill_mat33_v2(pn, bd1, bd2, bn3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_v2_full(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',  mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]',  filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V2 -> V2 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V2 -> V2 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V2 -> V2 that is written to
        
        mat21 : array
            mu=2, nu=1 element of the block matrix V2 -> V2 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V2 -> V2 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V2 -> V2 that is written to
        
        mat31 : array
            mu=3, nu=1 element of the block matrix V2 -> V2 that is written to

        mat32 : array
            mu=3, nu=2 element of the block matrix V2 -> V2 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V2 -> V2 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V2 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V2 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V2 and written to mat13

        filling21 : float
            number that will be multiplied by the basis functions of V2 and written to mat21

        filling22 : float
            number that will be multiplied by the basis functions of V2 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V2 and written to mat23

        filling31 : float
            number that will be multiplied by the basis functions of V2 and written to mat31

        filling32 : float
            number that will be multiplied by the basis functions of V2 and written to mat32

        filling33 : float
            number that will be multiplied by the basis functions of V2 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V2 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V2 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V2 and written to vec3
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat11_vec1_v2(pn, bn1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat12_v2(pn, bn1, bd1, bn2, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat13_v2(pn, bn1, bd1, bd2, bn3, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat21_vec2_v2(pn, bn1, bd1, bn2, bd2, bd3, ie1, ie2, ie3, starts2, mat21, filling21, vec2, filling2)
    fk.fill_mat22_v2(pn, bd1, bn2, bd3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat23_v2(pn, bd1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat31_vec3_v2(pn, bn1, bd1, bd2, bn3, bd3, ie1, ie2, ie3, starts3, mat31, filling31, vec3, filling3)
    fk.fill_mat32_v2(pn, bd1, bn2, bd2, bn3, bd3, ie1, ie2, ie3, starts3, mat32, filling32)
    fk.fill_mat33_v2(pn, bd1, bd2, bn3, ie1, ie2, ie3, starts3, mat33, filling33)


def mat_fill_b_v0(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat : 'float[:,:,:,:,:,:]', filling : 'float'):
    """
    Adds the contribution of one particle to the elements of an accumulation block matrix V0 -> V0. The result is returned in mat.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat : array
            block matrix V0 -> V0 that is written to

        filling : float
            number that will be multiplied by the basis functions of V0 and written to mat
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat, filling)


def m_v_fill_b_v0(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat : 'float[:,:,:,:,:,:]', filling_m : 'float', vec : 'float[:,:,:]', filling_v : 'float'):
    """
    Adds the contribution of one particle to the elements of an accumulation block matrix V0 -> V0. The result is returned in mat.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat : array
            block matrix V0 -> V0 that is written to

        filling_m : float
            number that will be multiplied by the basis functions of V0 and written to mat
        
        vec : array
            vector that is written to
        
        filling_v : float
            number that is multiplied by the basis functions of V0 and written to vec
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat, filling_m, vec, filling_v)


def mat_fill_b_v3(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat : 'float[:,:,:,:,:,:]', filling : 'float'):
    """
    Adds the contribution of one particle to the elements of an accumulation block matrix V3 -> V3. The result is returned in mat.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat : array
            block matrix V3 -> V3 that is written to

        filling : float
            number that will be multiplied by the basis functions of V3 and written to mat
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat, filling)


def m_v_fill_b_v3(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat : 'float[:,:,:,:,:,:]', filling_m : 'float', vec : 'float[:,:,:]', filling_v : 'float'):
    """
    Adds the contribution of one particle to the elements of an accumulation block matrix V3 -> V3. The result is returned in mat.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat : array
            block matrix V3 -> V3 that is written to

        filling_m : float
            number that will be multiplied by the basis functions of V3 and written to mat
        
        vec : array
            vector that is written to
        
        filling_v : float
            number that is multiplied by the basis functions of V3 and written to vec
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_vec_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat, filling_m, vec, filling_v)


def mat_fill_v0(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', starts1: 'int[:]', mat : 'float[:,:,:,:,:,:]', filling : 'float'):
    """
    Adds the contribution of one particle to the elements of an accumulation block matrix V0 -> V0. The result is returned in mat.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat : array
            block matrix V0 -> V0 that is written to

        filling : float
            number that will be multiplied by the basis functions of V0 and written to mat
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat, filling)


def m_v_fill_v0(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', starts1: 'int[:]', mat : 'float[:,:,:,:,:,:]', filling_m : 'float', vec : 'float[:,:,:]', filling_v : 'float'):
    """
    Adds the contribution of one particle to the elements of an accumulation block matrix V0 -> V0. The result is returned in mat.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat : array
            block matrix V0 -> V0 that is written to

        filling_m : float
            number that will be multiplied by the basis functions of V0 and written to mat

        vec : array
            vector that is written to
        
        filling_v : float
            number that will be multiplied by the basis functions of V0 and written to vec
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat, filling_m, vec, filling_v)


def mat_fill_v3(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', mat : 'float[:,:,:,:,:,:]', filling : 'float'):
    """
    Adds the contribution of one particle to the elements of an accumulation block matrix V3 -> V3. The result is returned in mat.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : 'float[:]' : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat : array
            block matrix V3 -> V3 that is written to

        filling : float
            number that will be multiplied by the basis functions of V3 and written to mat
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat, filling)


def m_v_fill_v3(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', mat : 'float[:,:,:,:,:,:]', filling_m : 'float', vec : 'float[:,:,:]', filling_v : 'float'):
    """
    Adds the contribution of one particle to the elements of an accumulation block matrix V3 -> V3. The result is returned in mat.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat : array
            block matrix V3 -> V3 that is written to

        filling_m : float
            number that will be multiplied by the basis functions of V3 and written to mat

        vec : array
            vector that is written to
        
        filling_v : float
            number that will be multiplied by the basis functions of V3 and written to vec
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat_vec_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat, filling_m, vec, filling_v)


def mat_fill_b_u0_diag(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix for three-vectors of V0 -> V0. The result is returned in mat11, mat22 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : 'float[:,:,:,:,:,:]' : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsp.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsp.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_b_u0_diag(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix for three-vectors of V0 -> V0. The result is returned in mat11, mat22 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsp.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsp.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat22, filling22, vec2, filling2)
    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat33, filling33, vec3, filling3)


def mat_fill_b_u3_diag(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix for three-vectors of V3 -> V3. The result is returned in mat11, mat22 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : 'float[:,:,:,:,:,:]' : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.d_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsp.d_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsp.d_splines_slim(tn3, pn3, eta3, span3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_b_u3_diag(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix for three-vectors of V3 -> V3. The result is returned in mat11, mat22 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.d_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsp.d_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsp.d_splines_slim(tn3, pn3, eta3, span3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_vec_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat_vec_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat22, filling22, vec2, filling2)
    fk.fill_mat_vec_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat33, filling33, vec3, filling3)


def mat_fill_b_u0_asym(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float'):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix for a three-vector V0 -> V0. The result is returned in mat12, mat13 and mat23.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsp.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsp.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat23, filling23)


def m_v_fill_b_u0_asym(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix for three-vectors V0 -> V0. The result is returned in mat12, mat13 and mat23.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsp.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsp.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat12, filling12, vec1, filling1)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat23, filling23, vec2, filling2)
    fk.fill_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, vec3, filling3)


def mat_fill_b_u3_asym(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float'):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix for a three-vector V0 -> V0. The result is returned in mat12, mat13 and mat23.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bd1 = empty( pn1, dtype=float)
    bd2 = empty( pn2, dtype=float)
    bd3 = empty( pn3, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.d_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsp.d_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsp.d_splines_slim(tn3, pn3, eta3, span3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat23, filling23)


def m_v_fill_b_u3_asym(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix for three-vectors V0 -> V0. The result is returned in mat12, mat13 and mat23.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing D-splines at particle position
    bd1 = empty( pn1, dtype=float)
    bd2 = empty( pn2, dtype=float)
    bd3 = empty( pn3, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.d_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsp.d_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsp.d_splines_slim(tn3, pn3, eta3, span3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_vec_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12, vec1, filling1)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_vec_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat23, filling23, vec2, filling2)
    fk.fill_vec_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, vec3, filling3)


def mat_fill_b_u0_symm(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix for a three-vectors V0 -> V0. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsp.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsp.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_b_u0_symm(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix for three-vectors V0 -> V0. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsp.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsp.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat22, filling22, vec2, filling2)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat33, filling33, vec3, filling3)


def mat_fill_b_u3_symm(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix for a three-vectors V3 -> V3. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing D-splines at particle position
    bd1 = empty( pn1, dtype=float)
    bd2 = empty( pn2, dtype=float)
    bd3 = empty( pn3, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.d_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsp.d_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsp.d_splines_slim(tn3, pn3, eta3, span3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_b_u3_symm(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix for three-vectors V3 -> V3. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing D-splines at particle position
    bd1 = empty( pn1, dtype=float)
    bd2 = empty( pn2, dtype=float)
    bd3 = empty( pn3, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsp.b_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsp.b_splines_slim(tn3, pn3, eta3, span3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_vec_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_vec_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat22, filling22, vec2, filling2)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat_vec_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat33, filling33, vec3, filling3)


def mat_fill_b_u0_full(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix for a three-vector V0 -> V0. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat21 : array
            mu=2, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat31 : array
            mu=3, nu=1 element of the block matrix V1 -> V1 that is written to

        mat32 : array
            mu=3, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling21 : float
            number that will be multiplied by the basis functions of V1 and written to mat21

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling31 : float
            number that will be multiplied by the basis functions of V1 and written to mat31

        filling32 : float
            number that will be multiplied by the basis functions of V1 and written to mat32

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsp.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsp.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat21, filling21)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat31, filling31)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat32, filling32)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_b_u0_full(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]',  filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix for a three-vector V0 -> V0. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat21 : array
            mu=2, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat31 : array
            mu=3, nu=1 element of the block matrix V1 -> V1 that is written to

        mat32 : array
            mu=3, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling21 : float
            number that will be multiplied by the basis functions of V1 and written to mat21

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling31 : float
            number that will be multiplied by the basis functions of V1 and written to mat31

        filling32 : float
            number that will be multiplied by the basis functions of V1 and written to mat32

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsp.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsp.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat21, filling21, vec2, filling2)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat31, filling31, vec3, filling3)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat32, filling32)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat33, filling33)


def mat_fill_b_u3_full(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix for a three-vector V3 -> V3. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat21 : array
            mu=2, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat31 : array
            mu=3, nu=1 element of the block matrix V1 -> V1 that is written to

        mat32 : array
            mu=3, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling21 : float
            number that will be multiplied by the basis functions of V1 and written to mat21

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling31 : float
            number that will be multiplied by the basis functions of V1 and written to mat31

        filling32 : float
            number that will be multiplied by the basis functions of V1 and written to mat32

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing D-splines at particle position
    bd1 = empty( pn1, dtype=float)
    bd2 = empty( pn2, dtype=float)
    bd3 = empty( pn3, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.d_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsp.d_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsp.d_splines_slim(tn3, pn3, eta3, span3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat21, filling21)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat31, filling31)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat32, filling32)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_b_u3_full(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]',  filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix for a three-vector V3 -> V3. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat21 : array
            mu=2, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat31 : array
            mu=3, nu=1 element of the block matrix V1 -> V1 that is written to

        mat32 : array
            mu=3, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling21 : float
            number that will be multiplied by the basis functions of V1 and written to mat21

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling31 : float
            number that will be multiplied by the basis functions of V1 and written to mat31

        filling32 : float
            number that will be multiplied by the basis functions of V1 and written to mat32

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing D-splines at particle position
    bd1 = empty( pn1, dtype=float)
    bd2 = empty( pn2, dtype=float)
    bd3 = empty( pn3, dtype=float)

    # spans (i.e. index for non-vanishing basis functions)
    span1 = bsp.find_span(tn1, pn1, eta1)
    span2 = bsp.find_span(tn2, pn2, eta2)
    span3 = bsp.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.d_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsp.d_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsp.d_splines_slim(tn3, pn3, eta3, span3, bd3)

    # element index of the particle in each direction
    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat_vec_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_vec_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat21, filling21, vec2, filling2)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat_vec_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat31, filling31, vec3, filling3)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat32, filling32)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat33, filling33)


def mat_fill_u0_asym(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float'):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix for a three-vector V0 -> V0. The result is returned in mat12, mat13 and mat23.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat13, filling13)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat23, filling23)


def m_v_fill_u0_asym(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix for a three-vector V0 -> V0. The result is returned in mat12, mat13 and mat23.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat12, filling12, vec1, filling1)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat23, filling23, vec2, filling2)
    fk.fill_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, vec3, filling3)


def mat_fill_u3_asym(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float'):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix for a three-vector V3 -> V3. The result is returned in mat12, mat13 and mat23.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat13, filling13)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat23, filling23)


def m_v_fill_u3_asym(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix for a three-vector V3 -> V3. The result is returned in mat12, mat13 and mat23.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat_vec_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12, vec1, filling1)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_vec_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat23, filling23, vec2, filling2)
    fk.fill_vec_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, vec3, filling3)


def mat_fill_u0_symm(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix for three-vectors V0 -> V0. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_u0_symm(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix for three-vectors V0 -> V0. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3

        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat22, filling22, vec2, filling2)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat33, filling33, vec3, filling3)


def mat_fill_u3_symm(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix for three-vectors V3 -> V3. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_u3_symm(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix for three-vectors V3 -> V3. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span1, span2, span3 : int
            the three values of the span index in each direction
        
        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat_vec_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_vec_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat22, filling22, vec2, filling2)
    fk.fill_mat_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat_vec_u0(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat33, filling33, vec3, filling3)


def mat_fill_u0_full(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',  mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix for three-vector V0 -> V0. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span1, span2, span3 : int
            the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat21 : array
            mu=2, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat31 : array
            mu=3, nu=1 element of the block matrix V1 -> V1 that is written to

        mat32 : array
            mu=3, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling21 : float
            number that will be multiplied by the basis functions of V1 and written to mat21

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling31 : float
            number that will be multiplied by the basis functions of V1 and written to mat31

        filling32 : float
            number that will be multiplied by the basis functions of V1 and written to mat32

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat21, filling21)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat31, filling31)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat32, filling32)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_u0_full(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',  mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]',  filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix for three-vector V0 -> V0. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span1, span2, span3 : int
            the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat21 : array
            mu=2, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat31 : array
            mu=3, nu=1 element of the block matrix V1 -> V1 that is written to

        mat32 : array
            mu=3, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling21 : float
            number that will be multiplied by the basis functions of V1 and written to mat21

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling31 : float
            number that will be multiplied by the basis functions of V1 and written to mat31

        filling32 : float
            number that will be multiplied by the basis functions of V1 and written to mat32

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat21, filling21, vec2, filling2)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat_vec_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat31, filling31, vec3, filling3)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat32, filling32)
    fk.fill_mat_u0(pn, bn1, bn2, bn3, ie1, ie2, ie3, starts3, mat33, filling33)


def mat_fill_u3_full(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',  mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float'):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix for three-vector V3 -> V3. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span1, span2, span3 : int
            the three values of the span index in each direction
        
        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat21 : array
            mu=2, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat31 : array
            mu=3, nu=1 element of the block matrix V1 -> V1 that is written to

        mat32 : array
            mu=3, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling21 : float
            number that will be multiplied by the basis functions of V1 and written to mat21

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling31 : float
            number that will be multiplied by the basis functions of V1 and written to mat31

        filling32 : float
            number that will be multiplied by the basis functions of V1 and written to mat32

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat21, filling21)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat31, filling31)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat32, filling32)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat33, filling33)


def m_v_fill_u3_full(pn: 'int[:]', span1: 'int', span2: 'int', span3: 'int', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',  mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]',  filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix for three-vector V3 -> V3. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters : 
    ------------
        pn: array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span1, span2, span3 : int
            the three values of the span index in each direction
        
        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        tn1: array
            the knot vector in direction 1

        tn2: array
            the knot vector in direction 2

        tn3: array
            the knot vector in direction 3
        
        start1, start2, start3 : int
            start index of the current process in each direction
        
        pad1, pad2, pad3 : int
            paddings of the current process in each direction
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1, nu=1 element of the block matrix V1 -> V1 that is written to

        mat12 : array
            mu=1, nu=2 element of the block matrix V1 -> V1 that is written to

        mat13 : array
            mu=1, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat21 : array
            mu=2, nu=1 element of the block matrix V1 -> V1 that is written to

        mat22 : array
            mu=2, nu=2 element of the block matrix V1 -> V1 that is written to

        mat23 : array
            mu=2, nu=3 element of the block matrix V1 -> V1 that is written to
        
        mat31 : array
            mu=3, nu=1 element of the block matrix V1 -> V1 that is written to

        mat32 : array
            mu=3, nu=2 element of the block matrix V1 -> V1 that is written to

        mat33 : array
            mu=3, nu=3 element of the block matrix V1 -> V1 that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling21 : float
            number that will be multiplied by the basis functions of V1 and written to mat21

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23

        filling31 : float
            number that will be multiplied by the basis functions of V1 and written to mat31

        filling32 : float
            number that will be multiplied by the basis functions of V1 and written to mat32

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
        
        vec1 : array
            mu=1 element of the vector that is written to

        vec2 : array
            mu=2 element of the vector that is written to
            
        vec3 : array
            mu=3 element of the vector that is written to
            
        filling1 : float
            number that will be multplied by the basis functions of V1 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V1 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V1 and written to vec3
    """

    # element index of the particle in each direction
    ie1 = span1 - pn[0]
    ie2 = span2 - pn[1]
    ie3 = span3 - pn[2]

    fk.fill_mat_vec_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat11, filling11, vec1, filling1)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat12, filling12)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts1, mat13, filling13)
    fk.fill_mat_vec_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat21, filling21, vec2, filling2)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat22, filling22)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts2, mat23, filling23)
    fk.fill_mat_vec_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat31, filling31, vec3, filling3)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat32, filling32)
    fk.fill_mat_u3(pn, bd1, bd2, bd3, ie1, ie2, ie3, starts3, mat33, filling33)

