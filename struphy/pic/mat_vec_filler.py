"""
pyccel function to accumulate a matix and matrix vector product in accumulation step.
Computed are only the independent mu-nu-components of the matrix (e.g. m12,m13,m23 for antisymmetric).
Symmetric/antisymmetric/diagonal refer to the mu-nu property of the matrix to be computed.

matrix fillings carry 2 indices (mu-nu) while vector fillings only carry one index (mu)

the functions with _b compute the non-vanishing basis functions at the point eta themselves (if this knowledge is not needed for the filling)
"""

# import modules for B-spline evaluation
import struphy.feec.bsplines_kernels as bsp

# import Filler kernel
import struphy.pic.filler_kernel as fk


# =====================================================================================================
def mat_fill_b_v1_diag(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a diagonal matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
     mat11 : 'float[:,:,:,:,:,:]' : array
            mu=1,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    from numpy import empty

    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    fk.fill_mat11_v1(p, bd1, bn2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat11, filling11)
    fk.fill_mat22_v1(p, bn1, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat22, filling22)
    fk.fill_mat33_v1(p, bn1, bn2, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat33, filling33)


# =====================================================================================================
def m_v_fill_b_v1_diag(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a diagonal matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec1[:,:,:] = 0.
    vec2[:,:,:] = 0.
    vec3[:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    fk.fill_mat11_vec1_v1(p, bd1, bn2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat11, filling11, vec1, filling1)
    fk.fill_mat22_vec2_v1(p, bn1, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat22, filling22, vec2, filling2)
    fk.fill_mat33_vec3_v1(p, bn1, bn2, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat33, filling33, vec3, filling3)


# =====================================================================================================
def mat_fill_b_v2_diag(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a diagonal matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V2 and written to mat11

        filling22 : float
            number that will be multiplied by the basis functions of V2 and written to mat22

        filling33 : float
            number that will be multiplied by the basis functions of V2 and written to mat33
    """

    from numpy import empty

    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    fk.fill_mat11_v2(p, bn1, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat11, filling11)
    fk.fill_mat22_v2(p, bd1, bn2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat22, filling22)
    fk.fill_mat33_v2(p, bd1, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat33, filling33)


# =====================================================================================================
def m_v_fill_b_v2_diag(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a diagonal matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec1[:,:,:] = 0.
    vec2[:,:,:] = 0.
    vec3[:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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
    span1 = bsp.find_span(t1, pn1, eta1)
    span2 = bsp.find_span(t2, pn2, eta2)
    span3 = bsp.find_span(t3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(t1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(t2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(t3, pn3, eta3, span3, bn3, bd3)

    ie1 = span1 - pn1
    ie2 = span2 - pn2
    ie3 = span3 - pn3

    fk.fill_mat11_vec1_v2(p, bn1, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat11, filling11, vec1, filling1)
    fk.fill_mat22_vec2_v2(p, bd1, bn2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat22, filling22, vec2, filling2)
    fk.fill_mat33_vec3_v2(p, bd1, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat33, filling33, vec3, filling3)


# =====================================================================================================
def mat_fill_b_v1_asym(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of an antisymmetric matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23
    """

    from numpy import empty

    # make sure the matrices that are written to are empty
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    fk.fill_mat12_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat12, filling12)
    fk.fill_mat13_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat13, filling13)
    fk.fill_mat23_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat23, filling23)


# =====================================================================================================
def m_v_fill_b_v1_asym(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of an antisymmetric matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
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

    # make sure the matrices that are written to are empty
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec1[:,:,:] = 0.
    vec2[:,:,:] = 0.
    vec3[:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    fk.fill_mat12_vec1_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat12, filling12, vec1, filling1)
    fk.fill_mat13_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat13, filling13)
    fk.fill_mat23_vec2_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat23, filling23, vec2, filling2)
    fk.fill_vec3_v1(p, bn1, bn2, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], vec3, filling3)


# =====================================================================================================
def mat_fill_b_v2_asym(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of an antisymmetric matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V2 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V2 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V2 and written to mat23
    """

    from numpy import empty

    # make sure the matrices that are written to are empty
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    fk.fill_mat12_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat12, filling12)
    fk.fill_mat13_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat13, filling13)
    fk.fill_mat23_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat23, filling23)


# =====================================================================================================
def m_v_fill_b_v2_asym(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of an antisymmetric matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
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

    # make sure the matrices that are written to are empty
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec1[:,:,:] = 0.
    vec2[:,:,:] = 0.
    vec3[:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    fk.fill_mat12_vec1_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat12, filling12, vec1, filling1)
    fk.fill_mat13_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat13, filling13)
    fk.fill_mat23_vec2_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat23, filling23, vec2, filling2)
    fk.fill_vec3_v2(p, bd1, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], vec3, filling3)


# =====================================================================================================
def mat_fill_b_v1_symm(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a symmetric matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    fk.fill_mat11_v1(p, bd1, bn2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat11, filling11)
    fk.fill_mat12_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat12, filling12)
    fk.fill_mat13_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat13, filling13)
    fk.fill_mat22_v1(p, bn1, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat22, filling22)
    fk.fill_mat23_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat23, filling23)
    fk.fill_mat33_v1(p, bn1, bn2, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat33, filling33)


# =====================================================================================================
def m_v_fill_b_v1_symm(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a symmetric matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec1[:,:,:] = 0.
    vec2[:,:,:] = 0.
    vec3[:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    fk.fill_mat11_vec1_v1(p, bd1, bn2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat11, filling11, vec1, filling1)
    fk.fill_mat12_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat12, filling12)
    fk.fill_mat13_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat13, filling13)
    fk.fill_mat22_vec2_v1(p, bn1, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat22, filling22, vec2, filling2)
    fk.fill_mat23_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat23, filling23)
    fk.fill_mat33_vec3_v1(p, bn1, bn2, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat33, filling33, vec3, filling3)


# =====================================================================================================
def mat_fill_b_v2_symm(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a symmetric matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    fk.fill_mat11_v2(p, bn1, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat11, filling11)
    fk.fill_mat12_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat12, filling12)
    fk.fill_mat13_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat13, filling13)
    fk.fill_mat22_v2(p, bd1, bn2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat22, filling22)
    fk.fill_mat23_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat23, filling23)
    fk.fill_mat33_v2(p, bd1, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat33, filling33)


# =====================================================================================================
def m_v_fill_b_v2_symm(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a symmetric matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec1[:,:,:] = 0.
    vec2[:,:,:] = 0.
    vec3[:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    fk.fill_mat11_vec1_v2(p, bn1, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat11, filling11, vec1, filling1)
    fk.fill_mat12_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat12, filling12)
    fk.fill_mat13_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat13, filling13)
    fk.fill_mat22_vec2_v2(p, bd1, bn2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat22, filling22, vec2, filling2)
    fk.fill_mat23_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat23, filling23)
    fk.fill_mat33_vec3_v2(p, bd1, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat33, filling33, vec3, filling3)


# =====================================================================================================
def mat_fill_b_v1_full(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a general matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat21 : array
            mu=2,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
        mat31 : array
            mu=3,nu=1 element of the matrix that is written to

        mat32 : array
            mu=3,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat21[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.
    mat31[:,:,:,:,:,:] = 0.
    mat32[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    fk.fill_mat11_v1(p, bd1, bn2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat11, filling11)
    fk.fill_mat12_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat12, filling12)
    fk.fill_mat13_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat13, filling13)
    fk.fill_mat21_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat21, filling21)
    fk.fill_mat22_v1(p, bn1, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat22, filling22)
    fk.fill_mat23_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat23, filling23)
    fk.fill_mat31_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat31, filling31)
    fk.fill_mat32_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat32, filling32)
    fk.fill_mat33_v1(p, bn1, bn2, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat33, filling33)


# =====================================================================================================
def m_v_fill_b_v1_full(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]',  filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a general matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat21 : array
            mu=2,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
        mat31 : array
            mu=3,nu=1 element of the matrix that is written to

        mat32 : array
            mu=3,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat21[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.
    mat31[:,:,:,:,:,:] = 0.
    mat32[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec1[:,:,:] = 0.
    vec2[:,:,:] = 0.
    vec3[:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    fk.fill_mat11_vec1_v1(p, bd1, bn2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat11, filling11, vec1, filling1)
    fk.fill_mat12_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat12, filling12)
    fk.fill_mat13_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat13, filling13)
    fk.fill_mat21_vec2_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat21, filling21, vec2, filling2)
    fk.fill_mat22_v1(p, bn1, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat22, filling22)
    fk.fill_mat23_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat23, filling23)
    fk.fill_mat31_vec3_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat31, filling31, vec3, filling3)
    fk.fill_mat32_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat32, filling32)
    fk.fill_mat33_v1(p, bn1, bn2, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat33, filling33)


# =====================================================================================================
def mat_fill_b_v2_full(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a general matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat21 : array
            mu=2,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
        mat31 : array
            mu=3,nu=1 element of the matrix that is written to

        mat32 : array
            mu=3,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat21[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.
    mat31[:,:,:,:,:,:] = 0.
    mat32[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    fk.fill_mat11_v2(p, bn1, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat11, filling11)
    fk.fill_mat12_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat12, filling12)
    fk.fill_mat13_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat13, filling13)
    fk.fill_mat21_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat21, filling21)
    fk.fill_mat22_v2(p, bd1, bn2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat22, filling22)
    fk.fill_mat23_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat23, filling23)
    fk.fill_mat31_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat31, filling31)
    fk.fill_mat32_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat32, filling32)
    fk.fill_mat33_v2(p, bd1, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat33, filling33)


# =====================================================================================================
def m_v_fill_b_v2_full(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]',  filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a general matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat21 : array
            mu=2,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
        mat31 : array
            mu=3,nu=1 element of the matrix that is written to

        mat32 : array
            mu=3,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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
            number that will be multplied by the basis functions of V2 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V2 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V2 and written to vec3
    """

    from numpy import empty

    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat21[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.
    mat31[:,:,:,:,:,:] = 0.
    mat32[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec1[:,:,:] = 0.
    vec2[:,:,:] = 0.
    vec3[:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    fk.fill_mat11_vec1_v2(p, bn1, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat11, filling11, vec1, filling1)
    fk.fill_mat12_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat12, filling12)
    fk.fill_mat13_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat13, filling13)
    fk.fill_mat21_vec2_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat21, filling21, vec2, filling2)
    fk.fill_mat22_v2(p, bd1, bn2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat22, filling22)
    fk.fill_mat23_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat23, filling23)
    fk.fill_mat31_vec3_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat31, filling31, vec3, filling3)
    fk.fill_mat32_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat32, filling32)
    fk.fill_mat33_v2(p, bd1, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat33, filling33)


# =====================================================================================================
def mat_fill_v1_diag(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', mat11 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a diagonal matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
  span : 'int[:]' : array
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
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V1 and written to mat11

        filling22 : float
            number that will be multiplied by the basis functions of V1 and written to mat22

        filling33 : float
            number that will be multiplied by the basis functions of V1 and written to mat33
    """

    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    fk.fill_mat11_v1(p, bd1, bn2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat11, filling11)
    fk.fill_mat22_v1(p, bn1, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat22, filling22)
    fk.fill_mat33_v1(p, bn1, bn2, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat33, filling33)


# =====================================================================================================
def m_v_fill_v1_diag(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', mat11 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a diagonal matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
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
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec1[:,:,:] = 0.
    vec2[:,:,:] = 0.
    vec3[:,:,:] = 0.

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    fk.fill_mat11_vec1_v1(p, bd1, bn2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat11, filling11, vec1, filling1)
    fk.fill_mat22_vec2_v1(p, bn1, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat22, filling22, vec2, filling2)
    fk.fill_mat33_vec3_v1(p, bn1, bn2, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat33, filling33, vec3, filling3)


# =====================================================================================================
def mat_fill_v2_diag(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', mat11 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a diagonal matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
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
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
        filling11 : float
            number that will be multiplied by the basis functions of V2 and written to mat11

        filling22 : float
            number that will be multiplied by the basis functions of V2 and written to mat22

        filling33 : float
            number that will be multiplied by the basis functions of V2 and written to mat33
    """

    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    fk.fill_mat11_v2(p, bn1, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat11, filling11)
    fk.fill_mat22_v2(p, bd1, bn2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat22, filling22)
    fk.fill_mat33_v2(p, bd1, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat33, filling33)


# =====================================================================================================
def m_v_fill_v2_diag(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', mat11 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling22 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a diagonal matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
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
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec1[:,:,:] = 0.
    vec2[:,:,:] = 0.
    vec3[:,:,:] = 0.

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    fk.fill_mat11_vec1_v2(p, bn1, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat11, filling11, vec1, filling1)
    fk.fill_mat22_vec2_v2(p, bd1, bn2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat22, filling22, vec2, filling2)
    fk.fill_mat33_vec3_v2(p, bd1, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat33, filling33, vec3, filling3)


# =====================================================================================================
def mat_fill_v1_asym(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of an antisymmetric matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
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
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V1 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V1 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V1 and written to mat23
    """

    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    # make sure the matrices that are written to are empty
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    fk.fill_mat12_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat12, filling12)
    fk.fill_mat13_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat13, filling13)
    fk.fill_mat23_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat23, filling23)


# =====================================================================================================
def m_v_fill_v1_asym(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of an antisymmetric matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
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
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
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

    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    # make sure the matrices that are written to are empty
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec1[:,:,:] = 0.
    vec2[:,:,:] = 0.
    vec3[:,:,:] = 0.

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    fk.fill_mat12_vec1_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat12, filling12, vec1, filling1)
    fk.fill_mat13_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat13, filling13)
    fk.fill_mat23_vec2_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat23, filling23, vec2, filling2)
    fk.fill_vec3_v1(p, bn1, bn2, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], vec3, filling3)


# =====================================================================================================
def mat_fill_v2_asym(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of an antisymmetric matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
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
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
        filling12 : float
            number that will be multiplied by the basis functions of V2 and written to mat12

        filling13 : float
            number that will be multiplied by the basis functions of V2 and written to mat13

        filling23 : float
            number that will be multiplied by the basis functions of V2 and written to mat23
    """

    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    # make sure the matrices that are written to are empty
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    fk.fill_mat12_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat12, filling12)
    fk.fill_mat13_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat13, filling13)
    fk.fill_mat23_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat23, filling23)


# =====================================================================================================
def m_v_fill_v2_asym(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', filling12 : 'float', filling13 : 'float', filling23 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of an antisymmetric matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
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
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
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

    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    # make sure the matrices that are written to are empty
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec1[:,:,:] = 0.
    vec2[:,:,:] = 0.
    vec3[:,:,:] = 0.

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    fk.fill_mat12_vec1_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat12, filling12, vec1, filling1)
    fk.fill_mat13_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat13, filling13)
    fk.fill_mat23_vec2_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat23, filling23, vec2, filling2)
    fk.fill_vec3_v2(p, bd1, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], vec3, filling3)


# =====================================================================================================
def mat_fill_v1_symm(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a symmetric matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
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
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    fk.fill_mat11_v1(p, bd1, bn2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat11, filling11)
    fk.fill_mat12_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat12, filling12)
    fk.fill_mat13_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat13, filling13)
    fk.fill_mat22_v1(p, bn1, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat22, filling22)
    fk.fill_mat23_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat23, filling23)
    fk.fill_mat33_v1(p, bn1, bn2, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat33, filling33)


# =====================================================================================================
def m_v_fill_v1_symm(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a symmetric matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
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
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec1[:,:,:] = 0.
    vec2[:,:,:] = 0.
    vec3[:,:,:] = 0.

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    fk.fill_mat11_vec1_v1(p, bd1, bn2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat11, filling11, vec1, filling1)
    fk.fill_mat12_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat12, filling12)
    fk.fill_mat13_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat13, filling13)
    fk.fill_mat22_vec2_v1(p, bn1, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat22, filling22, vec2, filling2)
    fk.fill_mat23_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat23, filling23)
    fk.fill_mat33_vec3_v1(p, bn1, bn2, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat33, filling33, vec3, filling3)


# =====================================================================================================
def mat_fill_v2_symm(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a symmetric matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
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
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    fk.fill_mat11_v2(p, bn1, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat11, filling11)
    fk.fill_mat12_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat12, filling12)
    fk.fill_mat13_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat13, filling13)
    fk.fill_mat22_v2(p, bd1, bn2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat22, filling22)
    fk.fill_mat23_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat23, filling23)
    fk.fill_mat33_v2(p, bd1, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat33, filling33)

    
# =====================================================================================================
def m_v_fill_v2_symm(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]',  mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling22 : 'float', filling23 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a symmetric matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
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
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec1[:,:,:] = 0.
    vec2[:,:,:] = 0.
    vec3[:,:,:] = 0.

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    fk.fill_mat11_vec1_v2(p, bn1, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat11, filling11, vec1, filling1)
    fk.fill_mat12_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat12, filling12)
    fk.fill_mat13_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat13, filling13)
    fk.fill_mat22_vec2_v2(p, bd1, bn2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat22, filling22, vec2, filling2)
    fk.fill_mat23_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat23, filling23)
    fk.fill_mat33_vec3_v2(p, bd1, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat33, filling33, vec3, filling3)


# =====================================================================================================
def mat_fill_v1_full(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]',  mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a general matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat21 : array
            mu=2,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
        mat31 : array
            mu=3,nu=1 element of the matrix that is written to

        mat32 : array
            mu=3,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat21[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.
    mat31[:,:,:,:,:,:] = 0.
    mat32[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    fk.fill_mat11_v1(p, bd1, bn2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat11, filling11)
    fk.fill_mat12_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat12, filling12)
    fk.fill_mat13_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat13, filling13)
    fk.fill_mat21_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat21, filling21)
    fk.fill_mat22_v1(p, bn1, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat22, filling22)
    fk.fill_mat23_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat23, filling23)
    fk.fill_mat31_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat31, filling31)
    fk.fill_mat32_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat32, filling32)
    fk.fill_mat33_v1(p, bn1, bn2, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat33, filling33)


# =====================================================================================================
def m_v_fill_v1_full(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]',  mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]',  filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a general matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat21 : array
            mu=2,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
        mat31 : array
            mu=3,nu=1 element of the matrix that is written to

        mat32 : array
            mu=3,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat21[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.
    mat31[:,:,:,:,:,:] = 0.
    mat32[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec1[:,:,:] = 0.
    vec2[:,:,:] = 0.
    vec3[:,:,:] = 0.

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    fk.fill_mat11_vec1_v1(p, bd1, bn2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat11, filling11, vec1, filling1)
    fk.fill_mat12_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat12, filling12)
    fk.fill_mat13_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat13, filling13)
    fk.fill_mat21_vec2_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat21, filling21, vec2, filling2)
    fk.fill_mat22_v1(p, bn1, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat22, filling22)
    fk.fill_mat23_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat23, filling23)
    fk.fill_mat31_vec3_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat31, filling31, vec3, filling3)
    fk.fill_mat32_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat32, filling32)
    fk.fill_mat33_v1(p, bn1, bn2, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat33, filling33)


# =====================================================================================================
def mat_fill_v2_full(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]',  mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a general matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat21 : array
            mu=2,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
        mat31 : array
            mu=3,nu=1 element of the matrix that is written to

        mat32 : array
            mu=3,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat21[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.
    mat31[:,:,:,:,:,:] = 0.
    mat32[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    fk.fill_mat11_v2(p, bn1, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat11, filling11)
    fk.fill_mat12_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat12, filling12)
    fk.fill_mat13_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat13, filling13)
    fk.fill_mat21_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat21, filling21)
    fk.fill_mat22_v2(p, bd1, bn2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat22, filling22)
    fk.fill_mat23_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat23, filling23)
    fk.fill_mat31_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat31, filling31)
    fk.fill_mat32_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat32, filling32)
    fk.fill_mat33_v2(p, bd1, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat33, filling33)


# =====================================================================================================
def m_v_fill_v2_full(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]',  mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]',  vec3 : 'float[:,:,:]',  filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a general matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3 
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat21 : array
            mu=2,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
        mat31 : array
            mu=3,nu=1 element of the matrix that is written to

        mat32 : array
            mu=3,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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
            number that will be multplied by the basis functions of V2 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V2 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V2 and written to vec3
    """

    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0.
    mat12[:,:,:,:,:,:] = 0.
    mat13[:,:,:,:,:,:] = 0.
    mat21[:,:,:,:,:,:] = 0.
    mat22[:,:,:,:,:,:] = 0.
    mat23[:,:,:,:,:,:] = 0.
    mat31[:,:,:,:,:,:] = 0.
    mat32[:,:,:,:,:,:] = 0.
    mat33[:,:,:,:,:,:] = 0.

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    fk.fill_mat11_vec1_v2(p, bn1, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat11, filling11, vec1, filling1)
    fk.fill_mat12_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat12, filling12)
    fk.fill_mat13_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat13, filling13)
    fk.fill_mat21_vec2_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat21, filling21, vec2, filling2)
    fk.fill_mat22_v2(p, bd1, bn2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat22, filling22)
    fk.fill_mat23_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat23, filling23)
    fk.fill_mat31_vec3_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat31, filling31, vec3, filling3)
    fk.fill_mat32_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat32, filling32)
    fk.fill_mat33_v2(p, bd1, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat33, filling33)


# =====================================================================================================
def mat_fill_b_v1_full(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a general matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3
        
        indd1 : array
            indD[0] from TensorSpline class, contains the global indices of non-zero D-splines in direction 1

        indd2 : array
            indD[1] from TensorSpline class, contains the global indices of non-zero D-splines in direction 2

        indd3 : array
            indD[2] from TensorSpline class, contains the global indices of non-zero D-splines in direction 3
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat21 : array
            mu=2,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
        mat31 : array
            mu=3,nu=1 element of the matrix that is written to

        mat32 : array
            mu=3,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0
    mat12[:,:,:,:,:,:] = 0
    mat13[:,:,:,:,:,:] = 0
    mat21[:,:,:,:,:,:] = 0
    mat22[:,:,:,:,:,:] = 0
    mat23[:,:,:,:,:,:] = 0
    mat31[:,:,:,:,:,:] = 0
    mat32[:,:,:,:,:,:] = 0
    mat33[:,:,:,:,:,:] = 0

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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
    span1 = bsp.find_span(t1, pn1, eta1)
    span2 = bsp.find_span(t2, pn2, eta2)
    span3 = bsp.find_span(t3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(t1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(t2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(t3, pn3, eta3, span3, bn3, bd3)
    
    # global indices of non-vanishing basis functions
    ie1 = span1 - p[0]
    ie2 = span2 - p[1]
    ie3 = span3 - p[2]

    fk.fill_mat11_v1(p, bd1, bn2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat11, filling11)
    fk.fill_mat12_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat12, filling12)
    fk.fill_mat13_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat13, filling13)
    fk.fill_mat21_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat21, filling21)
    fk.fill_mat22_v1(p, bn1, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat22, filling22)
    fk.fill_mat23_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat23, filling23)
    fk.fill_mat31_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat31, filling31)
    fk.fill_mat32_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat32, filling32)
    fk.fill_mat33_v1(p, bn1, bn2, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat33, filling33)


# =====================================================================================================
def m_v_fill_b_v1_full(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]', vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a general matrix with basis functions in V1 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3
        
        indd1 : array
            indD[0] from TensorSpline class, contains the global indices of non-zero D-splines in direction 1

        indd2 : array
            indD[1] from TensorSpline class, contains the global indices of non-zero D-splines in direction 2

        indd3 : array
            indD[2] from TensorSpline class, contains the global indices of non-zero D-splines in direction 3
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat21 : array
            mu=2,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
        mat31 : array
            mu=3,nu=1 element of the matrix that is written to

        mat32 : array
            mu=3,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0
    mat12[:,:,:,:,:,:] = 0
    mat13[:,:,:,:,:,:] = 0
    mat21[:,:,:,:,:,:] = 0
    mat22[:,:,:,:,:,:] = 0
    mat23[:,:,:,:,:,:] = 0
    mat31[:,:,:,:,:,:] = 0
    mat32[:,:,:,:,:,:] = 0
    mat33[:,:,:,:,:,:] = 0

    # make sure the vector that is written to is empty
    vec1[:,:,:] = 0
    vec2[:,:,:] = 0
    vec3[:,:,:] = 0

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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
    span1 = bsp.find_span(t1, pn1, eta1)
    span2 = bsp.find_span(t2, pn2, eta2)
    span3 = bsp.find_span(t3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(t1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(t2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(t3, pn3, eta3, span3, bn3, bd3)
    
    # global indices of non-vanishing basis functions
    ie1 = span1 - p[0]
    ie2 = span2 - p[1]
    ie3 = span3 - p[2]

    fk.fill_mat11_vec1_v1(p, bd1, bn2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat11, filling11, vec1, filling1)
    fk.fill_mat12_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat12, filling12)
    fk.fill_mat13_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:], mat13, filling13)
    fk.fill_mat21_vec2_v1(p, bn1, bd1, bn2, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat21, filling21, vec2, filling2)
    fk.fill_mat22_v1(p, bn1, bd2, bn3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat22, filling22)
    fk.fill_mat23_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:], mat23, filling23)
    fk.fill_mat31_vec3_v1(p, bn1, bd1, bn2, bn3, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat31, filling31, vec3, filling3)
    fk.fill_mat32_v1(p, bn1, bn2, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat32, filling32)
    fk.fill_mat33_v1(p, bn1, bn2, bd3, indn1[ie1,:], indn2[ie2,:], indn3[ie3,:pn3], mat33, filling33)


# =====================================================================================================
def mat_fill_b_v2_full(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a general matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3
        
        indd1 : array
            indD[0] from TensorSpline class, contains the global indices of non-zero D-splines in direction 1

        indd2 : array
            indD[1] from TensorSpline class, contains the global indices of non-zero D-splines in direction 2

        indd3 : array
            indD[2] from TensorSpline class, contains the global indices of non-zero D-splines in direction 3
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat21 : array
            mu=2,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
        mat31 : array
            mu=3,nu=1 element of the matrix that is written to

        mat32 : array
            mu=3,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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

    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0
    mat12[:,:,:,:,:,:] = 0
    mat13[:,:,:,:,:,:] = 0
    mat21[:,:,:,:,:,:] = 0
    mat22[:,:,:,:,:,:] = 0
    mat23[:,:,:,:,:,:] = 0
    mat31[:,:,:,:,:,:] = 0
    mat32[:,:,:,:,:,:] = 0
    mat33[:,:,:,:,:,:] = 0

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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
    span1 = bsp.find_span(t1, pn1, eta1)
    span2 = bsp.find_span(t2, pn2, eta2)
    span3 = bsp.find_span(t3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(t1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(t2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(t3, pn3, eta3, span3, bn3, bd3)
    
    # global indices of non-vanishing basis functions
    ie1 = span1 - p[0]
    ie2 = span2 - p[1]
    ie3 = span3 - p[2]

    fk.fill_mat11_v2(p, bn1, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat11, filling11)
    fk.fill_mat12_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat12, filling12)
    fk.fill_mat13_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat13, filling13)
    fk.fill_mat21_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat21, filling21)
    fk.fill_mat22_v2(p, bd1, bn2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat22, filling22)
    fk.fill_mat23_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat23, filling23)
    fk.fill_mat31_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat31, filling31)
    fk.fill_mat32_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat32, filling32)
    fk.fill_mat33_v2(p, bd1, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat33, filling33)


# =====================================================================================================
def m_v_fill_b_v2_full(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat11 : 'float[:,:,:,:,:,:]', mat12 : 'float[:,:,:,:,:,:]', mat13 : 'float[:,:,:,:,:,:]', mat21 : 'float[:,:,:,:,:,:]', mat22 : 'float[:,:,:,:,:,:]', mat23 : 'float[:,:,:,:,:,:]', mat31 : 'float[:,:,:,:,:,:]', mat32 : 'float[:,:,:,:,:,:]', mat33 : 'float[:,:,:,:,:,:]', filling11 : 'float', filling12 : 'float', filling13 : 'float', filling21 : 'float', filling22 : 'float', filling23 : 'float', filling31 : 'float', filling32 : 'float', filling33 : 'float', vec1 : 'float[:,:,:]', vec2 : 'float[:,:,:]', vec3 : 'float[:,:,:]', filling1 : 'float', filling2 : 'float', filling3 : 'float'):
    """
    fills the independent elements (each "element" has size of N_k x (p+1)) of a general matrix with basis functions in V2 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3
        
        indd1 : array
            indD[0] from TensorSpline class, contains the global indices of non-zero D-splines in direction 1

        indd2 : array
            indD[1] from TensorSpline class, contains the global indices of non-zero D-splines in direction 2

        indd3 : array
            indD[2] from TensorSpline class, contains the global indices of non-zero D-splines in direction 3
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the matrix that is written to

        mat12 : array
            mu=1,nu=2 element of the matrix that is written to

        mat13 : array
            mu=1,nu=3 element of the matrix that is written to
        
        mat21 : array
            mu=2,nu=1 element of the matrix that is written to

        mat22 : array
            mu=2,nu=2 element of the matrix that is written to

        mat23 : array
            mu=2,nu=3 element of the matrix that is written to
        
        mat31 : array
            mu=3,nu=1 element of the matrix that is written to

        mat32 : array
            mu=3,nu=2 element of the matrix that is written to

        mat33 : array
            mu=3,nu=3 element of the matrix that is written to
        
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
            number that will be multplied by the basis functions of V2 and written to vec1

        filling2 : float
            number that will be multplied by the basis functions of V2 and written to vec2

        filling3 : float
            number that will be multplied by the basis functions of V2 and written to vec3
    """

    from numpy import empty

    # make sure the matrices that are written to are empty
    mat11[:,:,:,:,:,:] = 0
    mat12[:,:,:,:,:,:] = 0
    mat13[:,:,:,:,:,:] = 0
    mat21[:,:,:,:,:,:] = 0
    mat22[:,:,:,:,:,:] = 0
    mat23[:,:,:,:,:,:] = 0
    mat31[:,:,:,:,:,:] = 0
    mat32[:,:,:,:,:,:] = 0
    mat33[:,:,:,:,:,:] = 0

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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
    span1 = bsp.find_span(t1, pn1, eta1)
    span2 = bsp.find_span(t2, pn2, eta2)
    span3 = bsp.find_span(t3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsp.b_d_splines_slim(t1, pn1, eta1, span1, bn1, bd1)
    bsp.b_d_splines_slim(t2, pn2, eta2, span2, bn2, bd2)
    bsp.b_d_splines_slim(t3, pn3, eta3, span3, bn3, bd3)
    
    # global indices of non-vanishing basis functions
    ie1 = span1 - p[0]
    ie2 = span2 - p[1]
    ie3 = span3 - p[2]

    fk.fill_mat11_vec1_v2(p, bn1, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat11, filling11, vec1, filling1)
    fk.fill_mat12_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat12, filling12)
    fk.fill_mat13_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:], indn2[ie2,:pn2], indn3[ie3,:pn3], mat13, filling13)
    fk.fill_mat21_vec2_v2(p, bn1, bd1, bn2, bd2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat21, filling21, vec2, filling2)
    fk.fill_mat22_v2(p, bd1, bn2, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat22, filling22)
    fk.fill_mat23_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:], indn3[ie3,:pn3], mat23, filling23)
    fk.fill_mat31_vec3_v2(p, bn1, bd1, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat31, filling31, vec3, filling3)
    fk.fill_mat32_v2(p, bd1, bn2, bd2, bn3, bd3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat32, filling32)
    fk.fill_mat33_v2(p, bd1, bd2, bn3, indn1[ie1,:pn1], indn2[ie2,:pn2], indn3[ie3,:], mat33, filling33)


# =====================================================================================================
def mat_fill_b_v0(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat : 'float[:,:,:,:,:,:]', filling : 'float'):
    """
    fills the independent element (of size N_k x (p+1)) of a "matrix" with basis functions in V0 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat : array
            matrix that is written to

        filling : float
            number that will be multiplied by the basis functions of V0 and written to mat
    """

    from numpy import empty

    # make sure the matrix that is written to is empty
    mat[:,:,:,:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    # (NNN NNN)
    for il1 in range(pn1 + 1):
        i1  = indn1[ie1,il1]
        bi1 = bn1[il1] * filling
        for il2 in range(pn2 + 1):
            i2  = indn2[ie2,il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[ie3,il3]
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


# =====================================================================================================
def mat_vec_fill_b_v0(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat : 'float[:,:,:,:,:,:]', filling_m : 'float', vec : 'float[:,:,:]', filling_v : 'float'):
    """
    fills the independent element (of size N_k x (p+1)) of a "matrix" with basis functions in V0 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indN : array
            indN from TensorSpline class, contains the global indices of non-zero B-splines in all directions
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat : array
            matrix that is written to

        filling_m : float
            number that will be multiplied by the basis functions of V0 and written to mat
        
        vec : array
            vector that is written to
        
        filling_v : float
            number that is multiplied by the basis functions of V0 and written to vec
    """

    from numpy import empty

    # make sure the matrix that is written to is empty
    mat[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec[:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    # (NNN NNN)
    for il1 in range(pn1 + 1):
        i1  = indn1[ie1,il1]
        bi1 = bn1[il1]
        for il2 in range(pn2 + 1):
            i2  = indn2[ie2,il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[ie3,il3]
                bi3 = bi2 * bn3[il3]

                vec[i1, i2, i3] += bi3 * filling_v

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling_m
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


# =====================================================================================================
def mat_fill_b_v3(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indd1 : 'int[:,:]', indd2 : 'int[:,:]', indd3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat : 'float[:,:,:,:,:,:]', filling : 'float'):
    """
    fills the independent element (of size N_k x (p+1)) of a "matrix" with basis functions in V3 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indD : array
            indN from TensorSpline class, contains the global indices of non-zero D-splines in all directions
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat : array
            matrix that is written to

        filling : float
            number that will be multiplied by the basis functions of V3 and written to mat
    """

    from numpy import empty

    # make sure the matrix that is written to is empty
    mat[:,:,:,:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    # (DDD DDD)
    for il1 in range(pd1 + 1):
        i1  = indd1[ie1,il1]
        bi1 = bd1[il1] * filling
        for il2 in range(pd2 + 1):
            i2  = indd2[ie2,il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[ie3,il3]
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat[i1, i2, i3, pd1 + jl1 - il1, pd2 + jl2 - il2, pd3 + jl3 - il3] += bj3


# =====================================================================================================
def mat_vec_fill_b_v3(p : 'int[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', indd1 : 'int[:,:]', indd2 : 'int[:,:]', indd3 : 'int[:,:]', eta1 : 'float', eta2 : 'float', eta3 : 'float', mat : 'float[:,:,:,:,:,:]', filling_m : 'float', vec : 'float[:,:,:]', filling_v : 'float'):
    """
    fills the independent element (of size N_k x (p+1)) of a "matrix" with basis functions in V3 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        t1 : array
            the knot vector in direction 1

        t2 : array
            the knot vector in direction 2

        t3 : array
            the knot vector in direction 3
        
        indD : array
            indN from TensorSpline class, contains the global indices of non-zero D-splines in all directions
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat : array
            matrix that is written to

        filling_m : float
            number that will be multiplied by the basis functions of V3 and written to mat
        
        vec : array
            vector that is written to
        
        filling_v : float
            number that is multiplied by the basis functions of V3 and written to vec
    """

    from numpy import empty

    # make sure the matrix that is written to is empty
    mat[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec[:,:,:] = 0.

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

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

    # (DDD DDD)
    for il1 in range(pd1 + 1):
        i1  = indd1[ie1,il1]
        bi1 = bd1[il1]
        for il2 in range(pd2 + 1):
            i2  = indd2[ie2,il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[ie3,il3]
                bi3 = bi2 * bd3[il3]

                vec[i1, i2, i3] += bi3 * filling_v

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling_m
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat[i1, i2, i3, pd1 + jl1 - il1, pd2 + jl2 - il2, pd3 + jl3 - il3] += bj3


# =====================================================================================================
def mat_fill_v0(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', mat : 'float[:,:,:,:,:,:]', filling : 'float'):
    """
    fills the element (of size N_k x (p+1)) of a "matrix" with basis functions in V0 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        indN : array
            indN from TensorSpline class, contains the global indices of non-zero B-splines in all directions
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat : array
            matrix that is written to

        filling : float
            number that will be multiplied by the basis functions of V0 and written to mat
    """

    # make sure the matrix that is written to is empty
    mat[:,:,:,:,:,:] = 0.

    # degrees of the basis functions in each direction
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    # (NNN NNN)
    for il1 in range(pn1 + 1):
        i1  = indn1[ie1,il1]
        bi1 = bn1[il1] * filling
        for il2 in range(pn2 + 1):
            i2  = indn2[ie2,il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[ie3,il3]
                bi3 = bi2 * bn3[il3]

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1]
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


# =====================================================================================================
def mat_vec_fill_v0(p : 'int[:]', span : 'int[:]', bn1 : 'float[:]', bn2 : 'float[:]', bn3 : 'float[:]', indn1 : 'int[:,:]', indn2 : 'int[:,:]', indn3 : 'int[:,:]', mat : 'float[:,:,:,:,:,:]', filling_m : 'float', vec : 'float[:,:,:]', filling_v : 'float'):
    """
    fills the element (of size N_k x (p+1)) of a "matrix" with basis functions in V0 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bn1 : array
            contains the values of non-vanishing B-splines in direction 1

        bn2 : array
            contains the values of non-vanishing B-splines in direction 2

        bn3 : array
            contains the values of non-vanishing B-splines in direction 3
        
        indN : array
            indN from TensorSpline class, contains the global indices of non-zero B-splines in all directions
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat : array
            matrix that is written to

        filling_m : float
            number that will be multiplied by the basis functions of V0 and written to mat

        vec : array
            vector that is written to
        
        filling_v : float
            number that will be multiplied by the basis functions of V0 and written to vec
    """

    # make sure the matrix that is written to is empty
    mat[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec[:,:,:] = 0.

    # degrees of the basis functions in each direction
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    # (NNN NNN)
    for il1 in range(pn1 + 1):
        i1  = indn1[ie1,il1]
        bi1 = bn1[il1]
        for il2 in range(pn2 + 1):
            i2  = indn2[ie2,il2]
            bi2 = bi1 * bn2[il2]
            for il3 in range(pn3 + 1):
                i3  = indn3[ie3,il3]
                bi3 = bi2 * bn3[il3]

                vec[i1, i2, i3] = bi3 * filling_v

                for jl1 in range(pn1 + 1):
                    bj1 = bi3 * bn1[jl1] * filling_m
                    for jl2 in range(pn2 + 1):
                        bj2 =  bj1 * bn2[jl2]
                        for jl3 in range(pn3 + 1):
                            bj3 = bj2 * bn3[jl3]

                            mat[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3


# =====================================================================================================
def mat_fill_v3(p : 'int[:]', span : 'int[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indd1 : 'int[:,:]', indd2 : 'int[:,:]', indd3 : 'int[:,:]', mat : 'float[:,:,:,:,:,:]', filling : 'float'):
    """
    fills the element (of size N_k x (p+1)) of a "matrix" with basis functions in V3 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
    bn1 : 'float[:]' : array
            contains the values of non-vanishing D-splines in direction 1

        bn2 : array
            contains the values of non-vanishing D-splines in direction 2

        bn3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        indd1 : array
            indD[0] from TensorSpline class, contains the global indices of non-zero D-splines in direction 1

        indd2 : array
            indD[1] from TensorSpline class, contains the global indices of non-zero D-splines in direction 2

        indd3 : array
            indD[2] from TensorSpline class, contains the global indices of non-zero D-splines in direction 3
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat : array
            matrix that is written to

        filling : float
            number that will be multiplied by the basis functions of V3 and written to mat
    """

    # make sure the matrix that is written to is empty
    mat[:,:,:,:,:,:] = 0.

    # degrees of the basis functions in each direction
    pd1 = p[0] - 1
    pd2 = p[1] - 1
    pd3 = p[2] - 1

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    # (DDD DDD)
    for il1 in range(pd1 + 1):
        i1  = indd1[ie1,il1]
        bi1 = bd1[il1] * filling
        for il2 in range(pd2 + 1):
            i2  = indd2[ie2,il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[ie3,il3]
                bi3 = bi2 * bd3[il3]

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1]
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat[i1, i2, i3, pd1 + jl1 - il1, pd2 + jl2 - il2, pd3 + jl3 - il3] += bj3


# =====================================================================================================
def mat_vec_fill_v3(p : 'int[:]', span : 'int[:]', bd1 : 'float[:]', bd2 : 'float[:]', bd3 : 'float[:]', indd1 : 'int[:,:]', indd2 : 'int[:,:]', indd3 : 'int[:,:]', mat : 'float[:,:,:,:,:,:]', filling_m : 'float', vec : 'float[:,:,:]', filling_v : 'float'):
    """
    fills the element (of size N_k x (p+1)) of a "matrix" with basis functions in V3 times the filling

    Parameters : 
    ------------
        p : array of integers
            contains 3 values of the degrees of the B-splines in each direction
        
        span : array
            contains the three values of the span index in each direction
        
        bd1 : array
            contains the values of non-vanishing D-splines in direction 1

        bd2 : array
            contains the values of non-vanishing D-splines in direction 2

        bd3 : array
            contains the values of non-vanishing D-splines in direction 3
        
        indd1 : array
            indD[0] from TensorSpline class, contains the global indices of non-zero D-splines in direction 1

        indd2 : array
            indD[1] from TensorSpline class, contains the global indices of non-zero D-splines in direction 2

        indd3 : array
            indD[2] from TensorSpline class, contains the global indices of non-zero D-splines in direction 3
        
        eta1 : float
            (logical) position of the particle in direction 1

        eta2 : float
            (logical) position of the particle in direction 2

        eta3 : float
            (logical) position of the particle in direction 3
        
        mat : array
            matrix that is written to

        filling_m : float
            number that will be multiplied by the basis functions of V3 and written to mat

        vec : array
            vector that is written to
        
        filling_v : float
            number that will be multiplied by the basis functions of V3 and written to vec
    """

    # make sure the matrix that is written to is empty
    mat[:,:,:,:,:,:] = 0.

    # make sure the vector that is written to is empty
    vec[:,:,:] = 0.

    # degrees of the basis functions in each direction
    pd1 = p[0] - 1
    pd2 = p[1] - 1
    pd3 = p[2] - 1

    # global indices of non-vanishing basis functions
    ie1 = span[0] - p[0]
    ie2 = span[1] - p[1]
    ie3 = span[2] - p[2]

    # (DDD DDD)
    for il1 in range(pd1 + 1):
        i1  = indd1[ie1,il1]
        bi1 = bd1[il1]
        for il2 in range(pd2 + 1):
            i2  = indd2[ie2,il2]
            bi2 = bi1 * bd2[il2]
            for il3 in range(pd3 + 1):
                i3  = indd3[ie3,il3]
                bi3 = bi2 * bd3[il3]

                vec[i1, i2, i3] = bi3 * filling_v

                for jl1 in range(pd1 + 1):
                    bj1 = bi3 * bd1[jl1] * filling_m
                    for jl2 in range(pd2 + 1):
                        bj2 =  bj1 * bd2[jl2]
                        for jl3 in range(pd3 + 1):
                            bj3 = bj2 * bd3[jl3]

                            mat[i1, i2, i3, pd1 + jl1 - il1, pd2 + jl2 - il2, pd3 + jl3 - il3] += bj3

