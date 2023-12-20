from pyccel.decorators import pure, stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.pic.accumulation.filler_kernels as filler_kernels


def _docstring():
    """
    MODULE DOCSTRING for **struphy.pic.accumulation.particle_to_mat_kernels**.

    The module contains pyccelized functions to add one particle to a FE matrix and vector in accumulation step.

    Computed are only the independent components of the matrix (e.g. m12,m13,m23 for antisymmetric).
    Matrix fillings carry 2 indices (mu-nu) while vector fillings only carry one index (mu).

    Naming conventions:

    1) mat_ adds only to a matrix of the respective space, m_v_ adds to a matrix and a vector.

    2) The functions with _b compute the pn+1 non-vanishing basis functions Lambda^p_ijk(eta) at the point eta.
    In case Lambda^p_ijk(eta) has already been computed for the filling, it can be passed (functions without _b).

    3) vn with n=0,1,2,3 denotes the discrete space from the 3d Derham sequence the matrix/vector belongs to. v0vec and v3vec denote discrete spaces of vector-valued functions where every component lives in v0 resp. v3.

    4) diag/asym/symm/full refer to the property of the block matrix (for v1 or v2) and define which independent components are computed.

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
                'mat_fill_b_v0vec_diag',
                'm_v_fill_b_v0vec_diag',
                'mat_fill_b_v3vec_diag',
                'm_v_fill_b_v3vec_diag',
                'mat_fill_b_v0vec_asym',
                'm_v_fill_b_v0vec_asym',
                'mat_fill_b_v3vec_asym',
                'm_v_fill_b_v3vec_asym',
                'mat_fill_b_v0vec_symm',
                'm_v_fill_b_v0vec_symm',
                'mat_fill_b_v3vec_symm',
                'm_v_fill_b_v3vec_symm',
                'mat_fill_b_v0vec_full',
                'm_v_fill_b_v0vec_full',
                'mat_fill_b_v3vec_full',
                'm_v_fill_b_v3vec_full',
                'mat_fill_v0vec_diag',
                'm_v_fill_v0vec_diag',
                'mat_fill_v3vec_diag',
                'm_v_fill_v3vec_diag',
                'mat_fill_v0vec_asym',
                'm_v_fill_v0vec_asym',
                'mat_fill_v3vec_asym',
                'm_v_fill_v3vec_asym',
                'mat_fill_v0vec_symm',
                'm_v_fill_v0vec_symm',
                'mat_fill_v3vec_symm',
                'm_v_fill_v3vec_symm',
                'mat_fill_v0vec_full',
                'm_v_fill_v0vec_full',
                'mat_fill_v3vec_full',
                'm_v_fill_v3vec_full',
                'm_v_fill_v1_pressure_full',
                'm_v_fill_v1_pressure',
                'vec_fill_v1'
                'scalar_fill_v0'
                ]
    """

    print('This is just the docstring function.')


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def mat_fill_b_v1_diag(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill22: float, fill33: float):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat22 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat22, mat33 : array[float]
        (mu=1, nu=1)-, (mu=2, nu=2)- and (mu=3, nu=3)-block of the block matrix V1 -> V1 that is written to.

    fill11, fill22, fill33 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat11, mat22 and mat33.
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pn2, pn3, pd1, pn2, pn3,
                            bd1, bn2, bn3, bd1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pn1, pd2, pn3, pn1, pd2, pn3,
                            bn1, bd2, bn3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pn1, pn2, pd3, pn1, pn2, pd3,
                            bn1, bn2, bd3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def m_v_fill_b_v1_diag(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill22: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat22 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat22, mat33 : array[float]
        (mu=1, nu=1)-, (mu=2, nu=2)- and (mu=3, nu=3)-block of the block matrix V1 -> V1 that is written to.

    fill11, fill22, fill33 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat11, mat22 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2, and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V1 and written to vec1, vec2 and vec3.
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill matrix and vector entries
    filler_kernels.fill_mat_vec(pd1, pn2, pn3, pd1, pn2, pn3,
                                bd1, bn2, bn3, bd1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pn1, pd2, pn3, pn1, pd2, pn3,
                                bn1, bd2, bn3, bn1, bd2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pn1, pn2, pd3, pn1, pn2, pd3,
                                bn1, bn2, bd3, bn1, bn2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def mat_fill_b_v2_diag(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill22: float, fill33: float):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat22 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts2 : array[int]
        Start indices of the current process in space V2.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat22, mat33 : array[float]
        (mu=1, nu=1)-, (mu=2, nu=2)- and (mu=3, nu=3)-block of the block matrix V2 -> V2 that is written to.

    fill11, fill22, fill33 : float
        Numbers that will be multiplied by the basis functions of V2 and written to mat11, mat22 and mat33.
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pd2, pd3, pn1, pd2, pd3,
                            bn1, bd2, bd3, bn1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pd1, pn2, pd3, pd1, pn2, pd3,
                            bd1, bn2, bd3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pd1, pd2, pn3, pd1, pd2, pn3,
                            bd1, bd2, bn3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def m_v_fill_b_v2_diag(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill22: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat22 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts2 : array[int]
        Start indices of the current process in space V2.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat22, mat33 : array[float]
        (mu=1, nu=1)-, (mu=2, nu=2)- and (mu=3, nu=3)-block of the block matrix V2 -> V2 that is written to.

    fill11, fill22, fill33 : float
        Numbers that will be multiplied by the basis functions of V2 and written to mat11, mat22 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V2 and written to vec1, vec2 and vec3.
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill matrix and vector entries
    filler_kernels.fill_mat_vec(pn1, pd2, pd3, pn1, pd2, pd3,
                                bn1, bd2, bd3, bn1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pd1, pn2, pd3, pd1, pn2, pd3,
                                bd1, bn2, bd3, bd1, bn2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pd1, pd2, pn3, pd1, pd2, pn3,
                                bd1, bd2, bn3, bd1, bd2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def mat_fill_b_v1_asym(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', fill12: float, fill13: float, fill23: float):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V1 -> V1. The result is returned in mat12, mat13 and mat23.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat12, mat13, mat23 : array[float]
        (mu=1, nu=2)-, (mu=1, nu=3)- and (mu=2, nu=3)-block of the block matrix V1 -> V1 that is written to.

    fill12, fill13, fill23 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat12, mat13 and mat23.
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pd2, pn3,
                            bd1, bn2, bn3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pn2, pd3,
                            bd1, bn2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pd2, pn3, pn1, pn2, pd3,
                            bn1, bd2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def m_v_fill_b_v1_asym(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', fill12: float, fill13: float, fill23: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V1 -> V1. The result is returned in mat12, mat13 and mat23.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat12, mat13, mat23 : array[float]
        (mu=1, nu=2)-, (mu=1, nu=3)-, and (mu=2, nu=3)-block of the block matrix V1 -> V1 that is written to.

    fill12, fill13, fill23 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat12, mat13 and mat23.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V1 and written to vec1, vec2 and vec3.
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat_vec(pd1, pn2, pn3, pn1, pd2, pn3,
                                bd1, bn2, bn3, bn1, bd2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat12, fill12, vec1, fill1)

    filler_kernels.fill_mat_vec(pn1, pd2, pn3, pn1, pn2, pd3,
                                bn1, bd2, bn3, bn1, bn2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat23, fill23, vec2, fill2)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pn2, pd3,
                            bd1, bn2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_vec(pn1, pn2, pd3,
                            bn1, bn2, bd3,
                            span1, span2, span3,
                            starts,
                            vec3, fill3)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def mat_fill_b_v2_asym(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', fill12: float, fill13: float, fill23: float):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V2 -> V2. The result is returned in mat12, mat13 and mat23.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts2 : array[int]
        Start indices of the current process in space V2.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat12, mat13, mat23 : array[float]
        (mu=1, nu=2)-, (mu=1, nu=3)- and (mu=2, nu=3)-block of the block matrix V2 -> V2 that is written to.

    fill12, fill13, fill23 : float
        Numbers that will be multiplied by the basis functions of V2 and written to mat12, mat13 and mat23.
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pn2, pd3,
                            bn1, bd2, bd3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pd2, pn3,
                            bn1, bd2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pn2, pd3, pd1, pd2, pn3,
                            bd1, bn2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def m_v_fill_b_v2_asym(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', fill12: float, fill13: float, fill23: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V2 -> V2. The result is returned in mat12, mat13 and mat23.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts2 : array[int]
        Start indices of the current process in space V2.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat12, mat13, mat23 : array[float]
        (mu=1, nu=2)-, (mu=1, nu=3)-, and (mu=2, nu=3)-block of the block matrix V2 -> V2 that is written to.

    fill12, fill13, fill23 : float
        Numbers that will be multiplied by the basis functions of V2 and written to mat12, mat13 and mat23.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V2 and written to vec1, vec2 and vec3.
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat_vec(pn1, pd2, pd3, pd1, pn2, pd3,
                                bn1, bd2, bd3, bd1, bn2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat12, fill12, vec1, fill1)

    filler_kernels.fill_mat_vec(pd1, pn2, pd3, pd1, pd2, pn3,
                                bd1, bn2, bd3, bd1, bd2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat23, fill23, vec2, fill2)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pd2, pn3,
                            bn1, bd2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_vec(pd1, pd2, pn3,
                            bd1, bd2, bn3,
                            span1, span2, span3,
                            starts,
                            vec3, fill3)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def mat_fill_b_v1_symm(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat12, mat13, mat22, mat23, mat33 : array[float]
        (mu=1, nu=1)-, (mu=1, nu=2)-, (mu=1, nu=3)-, (mu=2, nu=2)-, (mu=2, nu=3)- and (mu=3, nu=3)-block of the block matrix V1 -> V1 that is written to.

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat11, mat12, mat13, mat22, mat23 and mat33.
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pn2, pn3, pd1, pn2, pn3,
                            bd1, bn2, bn3, bd1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pd2, pn3,
                            bd1, bn2, bn3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pn2, pd3,
                            bd1, bn2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pd2, pn3, pn1, pd2, pn3,
                            bn1, bd2, bn3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pn1, pd2, pn3, pn1, pn2, pd3,
                            bn1, bd2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pn1, pn2, pd3, pn1, pn2, pd3,
                            bn1, bn2, bd3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def m_v_fill_b_v1_symm(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat12, mat13, mat22, mat23, mat33 : array[float]
        (mu=1, nu=1)-, (mu=1, nu=2)-, (mu=1, nu=3), (mu=2, nu=2)-, (mu=2, nu=3)- and (mu=3, nu=3)-block of the block matrix V1 -> V1 that is written to.

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat11, mat12, mat13, mat22, mat23 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V1 and written to vec1, vec2 and vec3.
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat_vec(pd1, pn2, pn3, pd1, pn2, pn3,
                                bd1, bn2, bn3, bd1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pn1, pd2, pn3, pn1, pd2, pn3,
                                bn1, bd2, bn3, bn1, bd2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pn1, pn2, pd3, pn1, pn2, pd3,
                                bn1, bn2, bd3, bn1, bn2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pd2, pn3,
                            bd1, bn2, bn3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pn2, pd3,
                            bd1, bn2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pd2, pn3, pn1, pn2, pd3,
                            bn1, bd2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def mat_fill_b_v2_symm(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts2 : array[int]
        Start indices of the current process in space V2.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat12, mat13, mat22, mat23, mat33 : array[float]
        (mu=1, nu=1)-, (mu=1, nu=2)-, (mu=1, nu=3)-, (mu=2, nu=2)-, (mu=2, nu=3)- and (mu=3, nu=3)-block of the block matrix V2 -> V2 that is written to.

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat11, mat12, mat13, mat22, mat23 and mat33.
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pd2, pd3, pn1, pd2, pd3,
                            bn1, bd2, bd3, bn1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pn2, pd3,
                            bn1, bd2, bd3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pd2, pn3,
                            bn1, bd2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pn2, pd3, pd1, pn2, pd3,
                            bd1, bn2, bd3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pd1, pn2, pd3, pd1, pd2, pn3,
                            bd1, bn2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pd1, pd2, pn3, pd1, pd2, pn3,
                            bd1, bd2, bn3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def m_v_fill_b_v2_symm(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts2 : array[int]
        Start indices of the current process in space V2.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat12, mat13, mat22, mat23, mat33 : array[float]
        (mu=1, nu=1)-, (mu=1, nu=2)-, (mu=1, nu=3), (mu=2, nu=2)-, (mu=2, nu=3)- and (mu=3, nu=3)-block of the block matrix V2 -> V2 that is written to.

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Numbers that will be multiplied by the basis functions of V2 and written to mat11, mat12, mat13, mat22, mat23 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V1 and written to vec1, vec2 and vec3.
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat_vec(pn1, pd2, pd3, pn1, pd2, pd3,
                                bn1, bd2, bd3, bn1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pd1, pn2, pd3, pd1, pn2, pd3,
                                bd1, bn2, bd3, bd1, bn2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pd1, pd2, pn3, pd1, pd2, pn3,
                                bd1, bd2, bn3, bd1, bd2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pn2, pd3,
                            bn1, bd2, bd3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pd2, pn3,
                            bn1, bd2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pn2, pd3, pd1, pd2, pn3,
                            bd1, bn2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def mat_fill_b_v1_full(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat21: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat31: 'float[:,:,:,:,:,:]', mat32: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill21: float, fill22: float, fill23: float, fill31: float, fill32: float, fill33: float):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat12, mat13, mat12, mat22, mat23, mat13, mat23, mat33 : array[float]
        All 9 blocks of the block matrix V1 -> V1 that is written to.

    fill11, fill12, fill13, fill12, fill23, fill23, fill31, fill32, fill33 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat11, mat12, mat13, mat12, mat22, mat23, mat13, mat23 and mat33.
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pn2, pn3, pd1, pn2, pn3,
                            bd1, bn2, bn3, bd1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pd2, pn3,
                            bd1, bn2, bn3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pn2, pd3,
                            bd1, bn2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pd2, pn3, pd1, pn2, pn3,
                            bn1, bd2, bn3, bd1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat21, fill21)

    filler_kernels.fill_mat(pn1, pd2, pn3, pn1, pd2, pn3,
                            bn1, bd2, bn3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pn1, pd2, pn3, pn1, pn2, pd3,
                            bn1, bd2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pn1, pn2, pd3, pd1, pn2, pn3,
                            bn1, bn2, bd3, bd1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat31, fill31)

    filler_kernels.fill_mat(pn1, pn2, pd3, pn1, pd2, pn3,
                            bn1, bn2, bd3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat32, fill32)

    filler_kernels.fill_mat(pn1, pn2, pd3, pn1, pn2, pd3,
                            bn1, bn2, bd3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def m_v_fill_b_v1_full(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat21: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat31: 'float[:,:,:,:,:,:]', mat32: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill21: float, fill22: float, fill23: float, fill31: float, fill32: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32, mat33 : array[float]
        All 9 blocks of the block matrix V1 -> V1 that is written to.

    fill11, fill12, fill13, fill21, fill22, fill23, fill31, fill32, fill33 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V1 and written to vec1, vec2 and vec3.
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat_vec(pd1, pn2, pn3, pd1, pn2, pn3,
                                bd1, bn2, bn3, bd1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pn1, pd2, pn3, pn1, pd2, pn3,
                                bn1, bd2, bn3, bn1, bd2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pn1, pn2, pd3, pn1, pn2, pd3,
                                bn1, bn2, bd3, bn1, bn2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pd2, pn3,
                            bd1, bn2, bn3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pn2, pd3,
                            bd1, bn2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pd2, pn3, pd1, pn2, pn3,
                            bn1, bd2, bn3, bd1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat21, fill21)

    filler_kernels.fill_mat(pn1, pd2, pn3, pn1, pn2, pd3,
                            bn1, bd2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pn1, pn2, pd3, pd1, pn2, pn3,
                            bn1, bn2, bd3, bd1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat31, fill31)

    filler_kernels.fill_mat(pn1, pn2, pd3, pn1, pd2, pn3,
                            bn1, bn2, bd3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat32, fill32)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def mat_fill_b_v2_full(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat21: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat31: 'float[:,:,:,:,:,:]', mat32: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill21: float, fill22: float, fill23: float, fill31: float, fill32: float, fill33: float):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts2 : array[int]
        Start indices of the current process in space V2.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32, mat33 : array[float]
        All 9 blocks of the block matrix V2 -> V2 that is written to.

    fill11, fill12, fill13, fill21, fill22, fill23, fill31, fill32, fill33 : float
        Numbers that will be multiplied by the basis functions of V2 and written to mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V2 and written to vec1, vec2 and vec3.
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pd2, pd3, pn1, pd2, pd3,
                            bn1, bd2, bd3, bn1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pn2, pd3,
                            bn1, bd2, bd3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pd2, pn3,
                            bn1, bd2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pn2, pd3, pn1, pd2, pd3,
                            bd1, bn2, bd3, bn1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat21, fill21)

    filler_kernels.fill_mat(pd1, pn2, pd3, pd1, pn2, pd3,
                            bd1, bn2, bd3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pd1, pn2, pd3, pd1, pd2, pn3,
                            bd1, bn2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pd1, pd2, pn3, pn1, pd2, pd3,
                            bd1, bd2, bn3, bn1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat31, fill31)

    filler_kernels.fill_mat(pd1, pd2, pn3, pd1, pn2, pd3,
                            bd1, bd2, bn3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat32, fill32)

    filler_kernels.fill_mat(pd1, pd2, pn3, pd1, pd2, pn3,
                            bd1, bd2, bn3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def m_v_fill_b_v2_full(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat21: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat31: 'float[:,:,:,:,:,:]', mat32: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill21: float, fill22: float, fill23: float, fill31: float, fill32: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]',  fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts2 : array[int]
        Start indices of the current process in space V2.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32, mat33 : array[float]
        All 9 blocks of the block matrix V2 -> V2 that is written to.

    fill11, fill12, fill13, fill21, fill22, fill23, fill31, fill32, fill33 : float
        Numbers that will be multiplied by the basis functions of V2 and written to mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V2 and written to vec1, vec2 and vec3.
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat_vec(pn1, pd2, pd3, pn1, pd2, pd3,
                                bn1, bd2, bd3, bn1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pd1, pn2, pd3, pd1, pn2, pd3,
                                bd1, bn2, bd3, bd1, bn2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pd1, pd2, pn3, pd1, pd2, pn3,
                                bd1, bd2, bn3, bd1, bd2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pn2, pd3,
                            bn1, bd2, bd3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pd2, pn3,
                            bn1, bd2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pn2, pd3, pn1, pd2, pd3,
                            bd1, bn2, bd3, bn1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat21, fill21)

    filler_kernels.fill_mat(pd1, pn2, pd3, pd1, pd2, pn3,
                            bd1, bn2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pd1, pd2, pn3, pn1, pd2, pd3,
                            bd1, bd2, bn3, bn1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat31, fill31)

    filler_kernels.fill_mat(pd1, pd2, pn3, pd1, pn2, pd3,
                            bd1, bd2, bn3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat32, fill32)


@pure
def mat_fill_v1_diag(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill22: float, fill33: float):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat22 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    mat11, mat22, mat33 : array[float]
        (mu=1, nu=1)-, (mu=2, nu=2)- and (mu=3, nu=3)-block of the block matrix V1 -> V1 that is written to.

    fill11, fill22, fill33 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat11, mat22 and mat33.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pn2, pn3, pd1, pn2, pn3,
                            bd1, bn2, bn3, bd1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pn1, pd2, pn3, pn1, pd2, pn3,
                            bn1, bd2, bn3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pn1, pn2, pd3, pn1, pn2, pd3,
                            bn1, bn2, bd3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
def m_v_fill_v1_diag(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill22: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat22 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    mat11, mat22, mat33 : array[float]
        (mu=1, nu=1)-, (mu=2, nu=2)- and (mu=3, nu=3)-block of the block matrix V1 -> V1 that is written to.

    fill11, fill22, fill33 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat11, mat22 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2, and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V1 and written to vec1, vec2 and vec3.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix and vector entries
    filler_kernels.fill_mat_vec(pd1, pn2, pn3, pd1, pn2, pn3,
                                bd1, bn2, bn3, bd1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pn1, pd2, pn3, pn1, pd2, pn3,
                                bn1, bd2, bn3, bn1, bd2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pn1, pn2, pd3, pn1, pn2, pd3,
                                bn1, bn2, bd3, bn1, bn2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)


@pure
def mat_fill_v2_diag(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill22: float, fill33: float):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat22 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts2 : array[int]
        Start indices of the current process in space V2.

    mat11, mat22, mat33 : array[float]
        (mu=1, nu=1)-, (mu=2, nu=2)- and (mu=3, nu=3)-block of the block matrix V2 -> V2 that is written to.

    fill11, fill22, fill33 : float
        Numbers that will be multiplied by the basis functions of V2 and written to mat11, mat22 and mat33.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pd2, pd3, pn1, pd2, pd3,
                            bn1, bd2, bd3, bn1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pd1, pn2, pd3, pd1, pn2, pd3,
                            bd1, bn2, bd3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pd1, pd2, pn3, pd1, pd2, pn3,
                            bd1, bd2, bn3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
def m_v_fill_v2_diag(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill22: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat22 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts2 : array[int]
        Start indices of the current process in space V2.

    mat11, mat22, mat33 : array[float]
        (mu=1, nu=1)-, (mu=2, nu=2)- and (mu=3, nu=3)-block of the block matrix V2 -> V2 that is written to.

    fill11, fill22, fill33 : float
        Numbers that will be multiplied by the basis functions of V2 and written to mat11, mat22 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V2 and written to vec1, vec2 and vec3.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix and vector entries
    filler_kernels.fill_mat_vec(pn1, pd2, pd3, pn1, pd2, pd3,
                                bn1, bd2, bd3, bn1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pd1, pn2, pd3, pd1, pn2, pd3,
                                bd1, bn2, bd3, bd1, bn2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pd1, pd2, pn3, pd1, pd2, pn3,
                                bd1, bd2, bn3, bd1, bd2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)


@pure
def mat_fill_v1_asym(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', fill12: float, fill13: float, fill23: float):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V1 -> V1. The result is returned in mat12, mat13 and mat23.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    mat12, mat13, mat23 : array[float]
        (mu=1, nu=2)-, (mu=1, nu=3)- and (mu=2, nu=3)-block of the block matrix V1 -> V1 that is written to.

    fill12, fill13, fill23 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat12, mat13 and mat23.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pd2, pn3,
                            bd1, bn2, bn3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pn2, pd3,
                            bd1, bn2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pd2, pn3, pn1, pn2, pd3,
                            bn1, bd2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)


@pure
def m_v_fill_v1_asym(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', fill12: float, fill13: float, fill23: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V1 -> V1. The result is returned in mat12, mat13 and mat23.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    mat12, mat13, mat23 : array[float]
        (mu=1, nu=2)-, (mu=1, nu=3)-, and (mu=2, nu=3)-block of the block matrix V1 -> V1 that is written to.

    fill12, fill13, fill23 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat12, mat13 and mat23.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V1 and written to vec1, vec2 and vec3.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat_vec(pd1, pn2, pn3, pn1, pd2, pn3,
                                bd1, bn2, bn3, bn1, bd2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat12, fill12, vec1, fill1)

    filler_kernels.fill_mat_vec(pn1, pd2, pn3, pn1, pn2, pd3,
                                bn1, bd2, bn3, bn1, bn2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat23, fill23, vec2, fill2)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pn2, pd3,
                            bd1, bn2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_vec(pn1, pn2, pd3,
                            bn1, bn2, bd3,
                            span1, span2, span3,
                            starts,
                            vec3, fill3)


@pure
def mat_fill_v2_asym(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', fill12: float, fill13: float, fill23: float):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V2 -> V2. The result is returned in mat12, mat13 and mat23.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts2 : array[int]
        Start indices of the current process in space V2.

    mat12, mat13, mat23 : array[float]
        (mu=1, nu=2)-, (mu=1, nu=3)- and (mu=2, nu=3)-block of the block matrix V2 -> V2 that is written to.

    fill12, fill13, fill23 : float
        Numbers that will be multiplied by the basis functions of V2 and written to mat12, mat13 and mat23.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pn2, pd3,
                            bn1, bd2, bd3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pd2, pn3,
                            bn1, bd2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pn2, pd3, pd1, pd2, pn3,
                            bd1, bn2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)


@pure
def m_v_fill_v2_asym(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', fill12: float, fill13: float, fill23: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V2 -> V2. The result is returned in mat12, mat13 and mat23.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts2 : array[int]
        Start indices of the current process in space V2.

    mat12, mat13, mat23 : array[float]
        (mu=1, nu=2)-, (mu=1, nu=3)-, and (mu=2, nu=3)-block of the block matrix V2 -> V2 that is written to.

    fill12, fill13, fill23 : float
        Numbers that will be multiplied by the basis functions of V2 and written to mat12, mat13 and mat23.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V2 and written to vec1, vec2 and vec3.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat_vec(pn1, pd2, pd3, pd1, pn2, pd3,
                                bn1, bd2, bd3, bd1, bn2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat12, fill12, vec1, fill1)

    filler_kernels.fill_mat_vec(pd1, pn2, pd3, pd1, pd2, pn3,
                                bd1, bn2, bd3, bd1, bd2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat23, fill23, vec2, fill2)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pd2, pn3,
                            bn1, bd2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_vec(pd1, pd2, pn3,
                            bd1, bd2, bn3,
                            span1, span2, span3,
                            starts,
                            vec3, fill3)


@pure
def mat_fill_v1_symm(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    mat11, mat12, mat13, mat22, mat23, mat33 : array[float]
        (mu=1, nu=1)-, (mu=1, nu=2)-, (mu=1, nu=3)-, (mu=2, nu=2)-, (mu=2, nu=3)- and (mu=3, nu=3)-block of the block matrix V1 -> V1 that is written to.

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat11, mat12, mat13, mat22, mat23 and mat33.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pn2, pn3, pd1, pn2, pn3,
                            bd1, bn2, bn3, bd1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pd2, pn3,
                            bd1, bn2, bn3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pn2, pd3,
                            bd1, bn2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pd2, pn3, pn1, pd2, pn3,
                            bn1, bd2, bn3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pn1, pd2, pn3, pn1, pn2, pd3,
                            bn1, bd2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pn1, pn2, pd3, pn1, pn2, pd3,
                            bn1, bn2, bd3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
def m_v_fill_v1_symm(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    mat11, mat12, mat13, mat22, mat23, mat33 : array[float]
        (mu=1, nu=1)-, (mu=1, nu=2)-, (mu=1, nu=3), (mu=2, nu=2)-, (mu=2, nu=3)- and (mu=3, nu=3)-block of the block matrix V1 -> V1 that is written to.

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat11, mat12, mat13, mat22, mat23 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V1 and written to vec1, vec2 and vec3.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat_vec(pd1, pn2, pn3, pd1, pn2, pn3,
                                bd1, bn2, bn3, bd1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pn1, pd2, pn3, pn1, pd2, pn3,
                                bn1, bd2, bn3, bn1, bd2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pn1, pn2, pd3, pn1, pn2, pd3,
                                bn1, bn2, bd3, bn1, bn2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pd2, pn3,
                            bd1, bn2, bn3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pn2, pd3,
                            bd1, bn2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pd2, pn3, pn1, pn2, pd3,
                            bn1, bd2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)


@pure
def mat_fill_v2_symm(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts2 : array[int]
        Start indices of the current process in space V2.

    mat11, mat12, mat13, mat22, mat23, mat33 : array[float]
        (mu=1, nu=1)-, (mu=1, nu=2)-, (mu=1, nu=3)-, (mu=2, nu=2)-, (mu=2, nu=3)- and (mu=3, nu=3)-block of the block matrix V2 -> V2 that is written to.

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat11, mat12, mat13, mat22, mat23 and mat33.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pd2, pd3, pn1, pd2, pd3,
                            bn1, bd2, bd3, bn1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pn2, pd3,
                            bn1, bd2, bd3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pd2, pn3,
                            bn1, bd2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pn2, pd3, pd1, pn2, pd3,
                            bd1, bn2, bd3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pd1, pn2, pd3, pd1, pd2, pn3,
                            bd1, bn2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pd1, pd2, pn3, pd1, pd2, pn3,
                            bd1, bd2, bn3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
def m_v_fill_v2_symm(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]',  mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts2 : array[int]
        Start indices of the current process in space V2.

    mat11, mat12, mat13, mat22, mat23, mat33 : array[float]
        (mu=1, nu=1)-, (mu=1, nu=2)-, (mu=1, nu=3), (mu=2, nu=2)-, (mu=2, nu=3)- and (mu=3, nu=3)-block of the block matrix V2 -> V2 that is written to.

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Numbers that will be multiplied by the basis functions of V2 and written to mat11, mat12, mat13, mat22, mat23 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V1 and written to vec1, vec2 and vec3.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat_vec(pn1, pd2, pd3, pn1, pd2, pd3,
                                bn1, bd2, bd3, bn1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pd1, pn2, pd3, pd1, pn2, pd3,
                                bd1, bn2, bd3, bd1, bn2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pd1, pd2, pn3, pd1, pd2, pn3,
                                bd1, bd2, bn3, bd1, bd2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pn2, pd3,
                            bn1, bd2, bd3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pd2, pn3,
                            bn1, bd2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pn2, pd3, pd1, pd2, pn3,
                            bd1, bn2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)


@pure
def mat_fill_v1_full(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]',  mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat21: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat31: 'float[:,:,:,:,:,:]', mat32: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill21: float, fill22: float, fill23: float, fill31: float, fill32: float, fill33: float):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    mat11, mat12, mat13, mat12, mat22, mat23, mat13, mat23, mat33 : array[float]
        All 9 blocks of the block matrix V1 -> V1 that is written to.

    fill11, fill12, fill13, fill12, fill23, fill23, fill31, fill32, fill33 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat11, mat12, mat13, mat12, mat22, mat23, mat13, mat23 and mat33.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pn2, pn3, pd1, pn2, pn3,
                            bd1, bn2, bn3, bd1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pd2, pn3,
                            bd1, bn2, bn3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pn2, pd3,
                            bd1, bn2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pd2, pn3, pd1, pn2, pn3,
                            bn1, bd2, bn3, bd1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat21, fill21)

    filler_kernels.fill_mat(pn1, pd2, pn3, pn1, pd2, pn3,
                            bn1, bd2, bn3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pn1, pd2, pn3, pn1, pn2, pd3,
                            bn1, bd2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pn1, pn2, pd3, pd1, pn2, pn3,
                            bn1, bn2, bd3, bd1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat31, fill31)

    filler_kernels.fill_mat(pn1, pn2, pd3, pn1, pd2, pn3,
                            bn1, bn2, bd3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat32, fill32)

    filler_kernels.fill_mat(pn1, pn2, pd3, pn1, pn2, pd3,
                            bn1, bn2, bd3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
def m_v_fill_v1_full(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]',  mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat21: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat31: 'float[:,:,:,:,:,:]', mat32: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill21: float, fill22: float, fill23: float, fill31: float, fill32: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]',  fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V1 -> V1. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32, mat33 : array[float]
        All 9 blocks of the block matrix V1 -> V1 that is written to.

    fill11, fill12, fill13, fill21, fill22, fill23, fill31, fill32, fill33 : float
        Numbers that will be multiplied by the basis functions of V1 and written to mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V1 and written to vec1, vec2 and vec3.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat_vec(pd1, pn2, pn3, pd1, pn2, pn3,
                                bd1, bn2, bn3, bd1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pn1, pd2, pn3, pn1, pd2, pn3,
                                bn1, bd2, bn3, bn1, bd2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pn1, pn2, pd3, pn1, pn2, pd3,
                                bn1, bn2, bd3, bn1, bn2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pd2, pn3,
                            bd1, bn2, bn3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pn2, pn3, pn1, pn2, pd3,
                            bd1, bn2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pd2, pn3, pd1, pn2, pn3,
                            bn1, bd2, bn3, bd1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat21, fill21)

    filler_kernels.fill_mat(pn1, pd2, pn3, pn1, pn2, pd3,
                            bn1, bd2, bn3, bn1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pn1, pn2, pd3, pd1, pn2, pn3,
                            bn1, bn2, bd3, bd1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat31, fill31)

    filler_kernels.fill_mat(pn1, pn2, pd3, pn1, pd2, pn3,
                            bn1, bn2, bd3, bn1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat32, fill32)


@pure
def mat_fill_v2_full(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]',  mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat21: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat31: 'float[:,:,:,:,:,:]', mat32: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill21: float, fill22: float, fill23: float, fill31: float, fill32: float, fill33: float):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts2 : array[int]
        Start indices of the current process in space V2.

    mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32, mat33 : array[float]
        All 9 blocks of the block matrix V2 -> V2 that is written to.

    fill11, fill12, fill13, fill21, fill22, fill23, fill31, fill32, fill33 : float
        Numbers that will be multiplied by the basis functions of V2 and written to mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32 and mat33.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pd2, pd3, pn1, pd2, pd3,
                            bn1, bd2, bd3, bn1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pn2, pd3,
                            bn1, bd2, bd3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pd2, pn3,
                            bn1, bd2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pn2, pd3, pn1, pd2, pd3,
                            bd1, bn2, bd3, bn1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat21, fill21)

    filler_kernels.fill_mat(pd1, pn2, pd3, pd1, pn2, pd3,
                            bd1, bn2, bd3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pd1, pn2, pd3, pd1, pd2, pn3,
                            bd1, bn2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pd1, pd2, pn3, pn1, pd2, pd3,
                            bd1, bd2, bn3, bn1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat31, fill31)

    filler_kernels.fill_mat(pd1, pd2, pn3, pd1, pn2, pd3,
                            bd1, bd2, bn3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat32, fill32)

    filler_kernels.fill_mat(pd1, pd2, pn3, pd1, pd2, pn3,
                            bd1, bd2, bn3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
def m_v_fill_v2_full(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]',  mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat21: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat31: 'float[:,:,:,:,:,:]', mat32: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill21: float, fill22: float, fill23: float, fill31: float, fill32: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]',  fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V2 -> V2. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts2 : array[int]
        Start indices of the current process in space V2.

    mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32, mat33 : array[float]
        All 9 blocks of the block matrix V2 -> V2 that is written to.

    fill11, fill12, fill13, fill21, fill22, fill23, fill31, fill32, fill33 : float
        Numbers that will be multiplied by the basis functions of V2 and written to mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V2 and written to vec1, vec2 and vec3.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat_vec(pn1, pd2, pd3, pn1, pd2, pd3,
                                bn1, bd2, bd3, bn1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pd1, pn2, pd3, pd1, pn2, pd3,
                                bd1, bn2, bd3, bd1, bn2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pd1, pd2, pn3, pd1, pd2, pn3,
                                bd1, bd2, bn3, bd1, bd2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pn2, pd3,
                            bn1, bd2, bd3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pd2, pd3, pd1, pd2, pn3,
                            bn1, bd2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pn2, pd3, pn1, pd2, pd3,
                            bd1, bn2, bd3, bn1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat21, fill21)

    filler_kernels.fill_mat(pd1, pn2, pd3, pd1, pd2, pn3,
                            bd1, bn2, bd3, bd1, bd2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pd1, pd2, pn3, pn1, pd2, pd3,
                            bd1, bd2, bn3, bn1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat31, fill31)

    filler_kernels.fill_mat(pd1, pd2, pn3, pd1, pn2, pd3,
                            bd1, bd2, bn3, bd1, bn2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat32, fill32)


@pure
@stack_array('bn1', 'bn2', 'bn3')
def mat_fill_b_v0(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat: 'float[:,:,:,:,:,:]', fill: float):
    """
    Adds the contribution of one particle to the elements of an accumulation matrix V0 -> V0. The result is returned in mat.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat : array[float]
        Matrix V0 -> V0 that is written to.

    fill : float
        Number that will be multiplied by the basis functions of V0 and written to mat.
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, i.e. values for non-vanishing B-splines at position eta
    bsplines_kernels.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsplines_kernels.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsplines_kernels.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat, fill)


@pure
@stack_array('bn1', 'bn2', 'bn3')
def m_v_fill_b_v0(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat: 'float[:,:,:,:,:,:]', fill_m: float, vec: 'float[:,:,:]', fill_v: float):
    """
    Adds the contribution of one particle to the elements of an accumulation matrix V0 -> V0. The result is returned in mat.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat : array[float]
        Matrix V0 -> V0 that is written to.

    fill_m : float
        Number that will be multiplied by the basis functions of V0 and written to mat.

    vec : array[float]
        Vector that is written to.

    fill_v : float
        Number that is multiplied by the basis functions of V0 and written to vec.
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, i.e. values for non-vanishing B-splines at position eta
    bsplines_kernels.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsplines_kernels.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsplines_kernels.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # fill matrix and vector entries
    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat, fill_m, vec, fill_v)


@pure
@stack_array('bd1', 'bd2', 'bd3')
def mat_fill_b_v3(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat: 'float[:,:,:,:,:,:]', fill: float):
    """
    Adds the contribution of one particle to the elements of an accumulation matrix V3 -> V3. The result is returned in mat.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat : array[float]
        Matrix V3 -> V3 that is written to.

    fill : float
        Number that will be multiplied by the basis functions of V3 and written to mat.
    """

    from numpy import empty

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bd, i.e. values for non-vanishing D-splines at position eta
    bsplines_kernels.d_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsplines_kernels.d_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsplines_kernels.d_splines_slim(tn3, pn3, eta3, span3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat, fill)


@pure
@stack_array('bd1', 'bd2', 'bd3')
def m_v_fill_b_v3(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat: 'float[:,:,:,:,:,:]', fill_m: float, vec: 'float[:,:,:]', fill_v: float):
    """
    Adds the contribution of one particle to the elements of an accumulation matrix V3 -> V3. The result is returned in mat.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat : array[float]
        Matrix V3 -> V3 that is written to.

    fill_m : float
        Number that will be multiplied by the basis functions of V3 and written to mat.

    vec : array[float]
        Vector that is written to.

    fill_v : float
            Number that is multiplied by the basis functions of V3 and written to vec.
    """

    from numpy import empty

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bd, i.e. values for non-vanishing D-splines at position eta
    bsplines_kernels.d_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsplines_kernels.d_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsplines_kernels.d_splines_slim(tn3, pn3, eta3, span3, bd3)

    # fill matrix and vector entries
    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat, fill_m, vec, fill_v)


@pure
def mat_fill_v0(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', starts: 'int[:]', mat: 'float[:,:,:,:,:,:]', fill: float):
    """
    Adds the contribution of one particle to the elements of an accumulation matrix V0 -> V0. The result is returned in mat.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    mat : array[float]
        Matrix V0 -> V0 that is written to.

    fill : float
        Number that will be multiplied by the basis functions of V0 and written to mat.
    """

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat, fill)


@pure
def m_v_fill_v0(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', starts: 'int[:]', mat: 'float[:,:,:,:,:,:]', fill_m: float, vec: 'float[:,:,:]', fill_v: float):
    """
    Adds the contribution of one particle to the elements of an accumulation matrix V0 -> V0. The result is returned in mat.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    mat : array[float]
        Matrix V0 -> V0 that is written to.

    fill_m : float
        Number that will be multiplied by the basis functions of V0 and written to mat.

    vec : array[float]
        Vector that is written to.

    fill_v : float
        Number that is multiplied by the basis functions of V0 and written to vec.
    """

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # fill matrix and vector entries
    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat, fill_m, vec, fill_v)


@pure
def mat_fill_v3(pn: 'int[:]', span1: int, span2: int, span3: int, bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat: 'float[:,:,:,:,:,:]', fill: float):
    """
    Adds the contribution of one particle to the elements of an accumulation block matrix V3 -> V3. The result is returned in mat.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    mat : array[float]
        Matrix V3 -> V3 that is written to.

    fill : float
        Number that will be multiplied by the basis functions of V3 and written to mat.
    """

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn[0] - 1
    pd2 = pn[1] - 1
    pd3 = pn[2] - 1

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat, fill)


@pure
def m_v_fill_v3(pn: 'int[:]', span1: int, span2: int, span3: int, bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat: 'float[:,:,:,:,:,:]', fill_m: float, vec: 'float[:,:,:]', fill_v: float):
    """
    Adds the contribution of one particle to the elements of an accumulation matrix V3 -> V3. The result is returned in mat.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    mat : array[float]
        Matrix V3 -> V3 that is written to.

    fill_m : float
        Number that will be multiplied by the basis functions of V3 and written to mat.

    vec : array[float]
        Vector that is written to.

    fill_v : float
        Number that is multiplied by the basis functions of V3 and written to vec.
    """

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn[0] - 1
    pd2 = pn[1] - 1
    pd3 = pn[2] - 1

    # fill matrix and vector entries
    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat, fill_m, vec, fill_v)


@pure
@stack_array('bn1', 'bn2', 'bn3')
def mat_fill_b_v0vec_diag(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill22: float, fill33: float):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V0vec -> V0vec. The result is returned in mat11, mat22 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat22, mat33 : array[float]
        (mu=1, nu=1)-, (mu=2, nu=2)- and (mu=3, nu=3)-block of the block matrix V0vec -> V0vec that is written to.

    fill11, fill22, fill33 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to mat11, mat22 and mat33.
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, i.e. values for non-vanishing B-splines at position eta
    bsplines_kernels.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsplines_kernels.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsplines_kernels.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
@stack_array('bn1', 'bn2', 'bn3')
def m_v_fill_b_v0vec_diag(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill22: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V0vec -> V0vec. The result is returned in mat11, mat22 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat22, mat33 : array[float]
        (mu=1, nu=1)-, (mu=2, nu=2)- and (mu=3, nu=3)-block of the block matrix V0vec -> V0vec that is written to.

    fill11, fill22, fill33 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to mat11, mat22 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2, and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to vec1, vec2 and vec3.
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, i.e. values for non-vanishing B-splines at position eta
    bsplines_kernels.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsplines_kernels.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsplines_kernels.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # fill matrix entries
    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)


@pure
@stack_array('bd1', 'bd2', 'bd3')
def mat_fill_b_v3vec_diag(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill22: float, fill33: float):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V3vec -> V3vec. The result is returned in mat11, mat22 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat22, mat33 : array[float]
        (mu=1, nu=1)-, (mu=2, nu=2)- and (mu=3, nu=3)-block of the block matrix V3vec -> V3vec that is written to.

    fill11, fill22, fill33 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to mat11, mat22 and mat33.
    """

    from numpy import empty

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn[0] - 1
    pd2 = pn[1] - 1
    pd3 = pn[2] - 1

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bd, i.e. values for non-vanishing D-splines at position eta
    bsplines_kernels.d_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsplines_kernels.d_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsplines_kernels.d_splines_slim(tn3, pn3, eta3, span3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
@stack_array('bd1', 'bd2', 'bd3')
def m_v_fill_b_v3vec_diag(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill22: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V3vec -> V3vec. The result is returned in mat11, mat22 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat22, mat33 : array[float]
        (mu=1, nu=1)-, (mu=2, nu=2)- and (mu=3, nu=3)-block of the block matrix V3vec -> V3vec that is written to.

    fill11, fill22, fill33 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to mat11, mat22 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to vec1, vec2 and vec3.
    """

    from numpy import empty

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bd, i.e. values for non-vanishing D-splines at position eta
    bsplines_kernels.d_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsplines_kernels.d_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsplines_kernels.d_splines_slim(tn3, pn3, eta3, span3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)


@pure
@stack_array('bn1', 'bn2', 'bn3')
def mat_fill_b_v0vec_asym(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', fill12: float, fill13: float, fill23: float):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V0vec -> V0vec. The result is returned in mat12, mat13 and mat23.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat12, mat13, mat23 : array[float]
        (mu=1, nu=2)-, (mu=1, nu=3)- and (mu=2, nu=3)-block of the block matrix V0vec -> V0vec that is written to.

    fill12, fill13, fill23 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to mat12, mat13 and mat23.
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, i.e. values for non-vanishing B-splines at position eta
    bsplines_kernels.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsplines_kernels.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsplines_kernels.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)


@pure
@stack_array('bn1', 'bn2', 'bn3')
def m_v_fill_b_v0vec_asym(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', fill12: float, fill13: float, fill23: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V0vec -> V0vec. The result is returned in mat12, mat13 and mat23.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat12, mat13, mat23 : array[float]
        (mu=1, nu=2)-, (mu=1, nu=3)-, and (mu=2, nu=3)-block of the block matrix V0vec -> V0vec that is written to.

    fill12, fill13, fill23 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to mat12, mat13 and mat23.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to vec1, vec2 and vec3.
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty(pn[0], dtype=float)
    bn2 = empty(pn[1], dtype=float)
    bn3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, i.e. values for non-vanishing B-splines at position eta
    bsplines_kernels.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsplines_kernels.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsplines_kernels.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # fill matrix entries
    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat12, fill12, vec1, fill1)

    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat23, fill23, vec2, fill2)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_vec(pn1, pn2, pn3,
                            bn1, bn2, bn3,
                            span1, span2, span3,
                            starts,
                            vec3, fill3)


@pure
@stack_array('bd1', 'bd2', 'bd3')
def mat_fill_b_v3vec_asym(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', fill12: float, fill13: float, fill23: float):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V3vec -> V3vec. The result is returned in mat12, mat13 and mat23.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat12, mat13, mat23 : array[float]
        (mu=1, nu=2)-, (mu=1, nu=3)- and (mu=2, nu=3)-block of the block matrix V3vec -> V3vec that is written to.

    fill12, fill13, fill23 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to mat12, mat13 and mat23.
    """

    from numpy import empty

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bd, i.e. values for non-vanishing D-splines at position eta
    bsplines_kernels.d_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsplines_kernels.d_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsplines_kernels.d_splines_slim(tn3, pn3, eta3, span3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)


@pure
@stack_array('bd1', 'bd2', 'bd3')
def m_v_fill_b_v3vec_asym(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', fill12: float, fill13: float, fill23: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V3vec -> V3vec. The result is returned in mat12, mat13 and mat23.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat12, mat13, mat23 : array[float]
        (mu=1, nu=2)-, (mu=1, nu=3)-, and (mu=2, nu=3)-block of the block matrix V3vec -> V3vec that is written to.

    fill12, fill13, fill23 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to mat12, mat13 and mat23.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to vec1, vec2 and vec3.
    """

    from numpy import empty

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bd, i.e. values for non-vanishing D-splines at position eta
    bsplines_kernels.d_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsplines_kernels.d_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsplines_kernels.d_splines_slim(tn3, pn3, eta3, span3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat12, fill12, vec1, fill1)

    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat23, fill23, vec2, fill2)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_vec(pd1, pd2, pd3,
                            bd1, bd2, bd3,
                            span1, span2, span3,
                            starts,
                            vec3, fill3)


@pure
@stack_array('bn1', 'bn2', 'bn3')
def mat_fill_b_v0vec_symm(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V0vec -> V0vec. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat12, mat13, mat22, mat23, mat33 : array[float]
        (mu=1, nu=1)-, (mu=1, nu=2)-, (mu=1, nu=3)-, (mu=2, nu=2)-, (mu=2, nu=3)- and (mu=3, nu=3)-block of the block matrix V0vec -> V0vec that is written to.

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to mat11, mat12, mat13, mat22, mat23 and mat33.
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, i.e. values for non-vanishing B-splines at position eta
    bsplines_kernels.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsplines_kernels.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsplines_kernels.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
@stack_array('bn1', 'bn2', 'bn3')
def m_v_fill_b_v0vec_symm(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V0vec -> V0vec. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat12, mat13, mat22, mat23, mat33 : array[float]
        (mu=1, nu=1)-, (mu=1, nu=2)-, (mu=1, nu=3), (mu=2, nu=2)-, (mu=2, nu=3)- and (mu=3, nu=3)-block of the block matrix V0vec -> V0vec that is written to.

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to mat11, mat12, mat13, mat22, mat23 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to vec1, vec2 and vec3.
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, i.e. values for non-vanishing B-splines at position eta
    bsplines_kernels.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsplines_kernels.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsplines_kernels.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # fill matrix entries
    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)


@pure
@stack_array('bd1', 'bd2', 'bd3')
def mat_fill_b_v3vec_symm(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V3vec -> V3vec. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat12, mat13, mat22, mat23, mat33 : array[float]
        (mu=1, nu=1)-, (mu=1, nu=2)-, (mu=1, nu=3)-, (mu=2, nu=2)-, (mu=2, nu=3)- and (mu=3, nu=3)-block of the block matrix V3vec -> V3vec that is written to.

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to mat11, mat12, mat13, mat22, mat23 and mat33.
    """

    from numpy import empty

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bd, i.e. values for non-vanishing D-splines at position eta
    bsplines_kernels.d_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsplines_kernels.d_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsplines_kernels.d_splines_slim(tn3, pn3, eta3, span3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
@stack_array('bd1', 'bd2', 'bd3')
def m_v_fill_b_v3vec_symm(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V3vec -> V3vec. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat12, mat13, mat22, mat23, mat33 : array[float]
        (mu=1, nu=1)-, (mu=1, nu=2)-, (mu=1, nu=3), (mu=2, nu=2)-, (mu=2, nu=3)- and (mu=3, nu=3)-block of the block matrix V3vec -> V3vec that is written to.

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to mat11, mat12, mat13, mat22, mat23 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to vec1, vec2 and vec3.
    """

    from numpy import empty

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bd, i.e. values for non-vanishing D-splines at position eta
    bsplines_kernels.d_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsplines_kernels.d_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsplines_kernels.d_splines_slim(tn3, pn3, eta3, span3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)


@pure
@stack_array('bn1', 'bn2', 'bn3')
def mat_fill_b_v0vec_full(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat21: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat31: 'float[:,:,:,:,:,:]', mat32: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill21: float, fill22: float, fill23: float, fill31: float, fill32: float, fill33: float):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V0vec -> V0vec. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat12, mat13, mat12, mat22, mat23, mat13, mat23, mat33 : array[float]
        All 9 blocks of the block matrix V0vec -> V0vec that is written to.

    fill11, fill12, fill13, fill12, fill23, fill23, fill31, fill32, fill33 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to mat11, mat12, mat13, mat12, mat22, mat23, mat13, mat23 and mat33.
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, i.e. values for non-vanishing B-splines at position eta
    bsplines_kernels.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsplines_kernels.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsplines_kernels.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat21, fill21)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat31, fill31)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat32, fill32)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
@stack_array('bn1', 'bn2', 'bn3')
def m_v_fill_b_v0vec_full(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat21: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat31: 'float[:,:,:,:,:,:]', mat32: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill21: float, fill22: float, fill23: float, fill31: float, fill32: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]',  fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V0vec -> V0vec. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32, mat33 : array[float]
        All 9 blocks of the block matrix V0vec -> V0vec that is written to.

    fill11, fill12, fill13, fill21, fill22, fill23, fill31, fill32, fill33 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to vec1, vec2 and vec3.
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, i.e. values for non-vanishing B-splines at position eta
    bsplines_kernels.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsplines_kernels.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsplines_kernels.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # fill matrix entries
    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat21, fill21)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat31, fill31)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat32, fill32)


@pure
@stack_array('bd1', 'bd2', 'bd3')
def mat_fill_b_v3vec_full(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat21: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat31: 'float[:,:,:,:,:,:]', mat32: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill21: float, fill22: float, fill23: float, fill31: float, fill32: float, fill33: float):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V3vec -> V3vec. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32, mat33 : array[float]
        All 9 blocks of the block matrix V3vec -> V3vec that is written to.

    fill11, fill12, fill13, fill21, fill22, fill23, fill31, fill32, fill33 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to vec1, vec2 and vec3.
    """

    from numpy import empty

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bd, i.e. values for non-vanishing D-splines at position eta
    bsplines_kernels.d_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsplines_kernels.d_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsplines_kernels.d_splines_slim(tn3, pn3, eta3, span3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat21, fill21)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat31, fill31)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat32, fill32)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
@stack_array('bd1', 'bd2', 'bd3')
def m_v_fill_b_v3vec_full(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat21: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat31: 'float[:,:,:,:,:,:]', mat32: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill21: float, fill22: float, fill23: float, fill31: float, fill32: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]',  fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V3vec -> V3vec. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    eta1, eta2, eta3 : float
        (logical) position of the particle in each direction.

    mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32, mat33 : array[float]
        All 9 blocks of the block matrix V3vec -> V3vec that is written to.

    fill11, fill12, fill13, fill21, fill22, fill23, fill31, fill32, fill33 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to vec1, vec2 and vec3.
    """

    from numpy import empty

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bd, i.e. values for non-vanishing D-splines at position eta
    bsplines_kernels.d_splines_slim(tn1, pn1, eta1, span1, bd1)
    bsplines_kernels.d_splines_slim(tn2, pn2, eta2, span2, bd2)
    bsplines_kernels.d_splines_slim(tn3, pn3, eta3, span3, bd3)

    # fill matrix entries
    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat21, fill21)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat31, fill31)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat32, fill32)


@pure
def mat_fill_v0vec_diag(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', starts: 'int[:]', mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill22: float, fill33: float):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V0vec -> V0vec. The result is returned in mat11, mat22 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    mat11, mat22, mat33 : array[float]
        (mu=1, nu=1)-, (mu=2, nu=2)- and (mu=3, nu=3)-block of the block matrix V0vec -> V0vec that is written to.

    fill11, fill22, fill33 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to mat11, mat22 and mat33.
    """

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
def m_v_fill_v0vec_diag(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', starts: 'int[:]', mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill22: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V0vec -> V0vec. The result is returned in mat11, mat22 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    mat11, mat22, mat33 : array[float]
        (mu=1, nu=1)-, (mu=2, nu=2)- and (mu=3, nu=3)-block of the block matrix V0vec -> V0vec that is written to.

    fill11, fill22, fill33 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to mat11, mat22 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2, and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to vec1, vec2 and vec3.
    """

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # fill matrix and vector entries
    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)


@pure
def mat_fill_v3vec_diag(pn: 'int[:]', span1: int, span2: int, span3: int, bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill22: float, fill33: float):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V3vec -> V3vec. The result is returned in mat11, mat22 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    mat11, mat22, mat33 : array[float]
        (mu=1, nu=1)-, (mu=2, nu=2)- and (mu=3, nu=3)-block of the block matrix V3vec -> V3vec that is written to.

    fill11, fill22, fill33 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to mat11, mat22 and mat33.
    """

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
def m_v_fill_v3vec_diag(pn: 'int[:]', span1: int, span2: int, span3: int, bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat11: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill22: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the diagonal elements (mu,nu)=(1,1), (mu,nu)=(2,2) and (mu,nu)=(3,3) of an accumulation block matrix V3vec -> V3vec. The result is returned in mat11, mat22 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    mat11, mat22, mat33 : array[float]
        (mu=1, nu=1)-, (mu=2, nu=2)- and (mu=3, nu=3)-block of the block matrix V3vec -> V3vec that is written to.

    fill11, fill22, fill33 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to mat11, mat22 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to vec1, vec2 and vec3.
    """

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix and vector entries
    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)


@pure
def mat_fill_v0vec_asym(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', starts: 'int[:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', fill12: float, fill13: float, fill23: float):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V0vec -> V0vec. The result is returned in mat12, mat13 and mat23.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    mat12, mat13, mat23 : array[float]
        (mu=1, nu=2)-, (mu=1, nu=3)- and (mu=2, nu=3)-block of the block matrix V0vec -> V0vec that is written to.

    fill12, fill13, fill23 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to mat12, mat13 and mat23.
    """

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)


@pure
def m_v_fill_v0vec_asym(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', starts: 'int[:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', fill12: float, fill13: float, fill23: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V0vec -> V0vec. The result is returned in mat12, mat13 and mat23.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    mat12, mat13, mat23 : array[float]
        (mu=1, nu=2)-, (mu=1, nu=3)-, and (mu=2, nu=3)-block of the block matrix V0vec -> V0vec that is written to.

    fill12, fill13, fill23 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to mat12, mat13 and mat23.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to vec1, vec2 and vec3.
    """

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # fill matrix entries
    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat12, fill12, vec1, fill1)

    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat23, fill23, vec2, fill2)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_vec(pn1, pn2, pn3,
                            bn1, bn2, bn3,
                            span1, span2, span3,
                            starts,
                            vec3, fill3)


@pure
def mat_fill_v3vec_asym(pn: 'int[:]', span1: int, span2: int, span3: int, bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', fill12: float, fill13: float, fill23: float):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V3 -> V3. The result is returned in mat12, mat13 and mat23.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    mat12, mat13, mat23 : array[float]
        (mu=1, nu=2)-, (mu=1, nu=3)- and (mu=2, nu=3)-block of the block matrix V3vec -> V3vec that is written to.

    fill12, fill13, fill23 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to mat12, mat13 and mat23.
    """

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)


@pure
def m_v_fill_v3vec_asym(pn: 'int[:]', span1: int, span2: int, span3: int, bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', fill12: float, fill13: float, fill23: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the antisymmetric elements (mu,nu)=(1,2), (mu,nu)=(1,3) and (mu,nu)=(2,3) of an accumulation block matrix V3vec -> V3vec. The result is returned in mat12, mat13 and mat23.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    mat12, mat13, mat23 : array[float]
        (mu=1, nu=2)-, (mu=1, nu=3)-, and (mu=2, nu=3)-block of the block matrix V3vec -> V3vec that is written to.

    fill12, fill13, fill23 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to mat12, mat13 and mat23.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to vec1, vec2 and vec3.
    """

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat12, fill12, vec1, fill1)

    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat23, fill23, vec2, fill2)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_vec(pd1, pd2, pd3,
                            bd1, bd2, bd3,
                            span1, span2, span3,
                            starts,
                            vec3, fill3)


@pure
def mat_fill_v0vec_symm(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', starts: 'int[:]', mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V0vec -> V0vec. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    mat11, mat12, mat13, mat22, mat23, mat33 : array[float]
        (mu=1, nu=1)-, (mu=1, nu=2)-, (mu=1, nu=3)-, (mu=2, nu=2)-, (mu=2, nu=3)- and (mu=3, nu=3)-block of the block matrix V0vec -> V0vec that is written to.

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to mat11, mat12, mat13, mat22, mat23 and mat33.
    """

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
def m_v_fill_v0vec_symm(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', starts: 'int[:]', mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V0vec -> V0vec. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    mat11, mat12, mat13, mat22, mat23, mat33 : array[float]
        (mu=1, nu=1)-, (mu=1, nu=2)-, (mu=1, nu=3), (mu=2, nu=2)-, (mu=2, nu=3)- and (mu=3, nu=3)-block of the block matrix V0vec -> V0vec that is written to.

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to mat11, mat12, mat13, mat22, mat23 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to vec1, vec2 and vec3.
    """

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # fill matrix entries
    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)


@pure
def mat_fill_v3vec_symm(pn: 'int[:]', span1: int, span2: int, span3: int, bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V3vec -> V3vec. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    mat11, mat12, mat13, mat22, mat23, mat33 : array[float]
        (mu=1, nu=1)-, (mu=1, nu=2)-, (mu=1, nu=3)-, (mu=2, nu=2)-, (mu=2, nu=3)- and (mu=3, nu=3)-block of the block matrix V3vec -> V3vec that is written to.

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to mat11, mat12, mat13, mat22, mat23 and mat33.
    """

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
def m_v_fill_v3vec_symm(pn: 'int[:]', span1: int, span2: int, span3: int, bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V3vec -> V3vec. The result is returned in mat11, mat12, mat13, mat22, mat23 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    mat11, mat12, mat13, mat22, mat23, mat33 : array[float]
        (mu=1, nu=1)-, (mu=1, nu=2)-, (mu=1, nu=3), (mu=2, nu=2)-, (mu=2, nu=3)- and (mu=3, nu=3)-block of the block matrix V3vec -> V3vec that is written to.

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to mat11, mat12, mat13, mat22, mat23 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to vec1, vec2 and vec3.
    """

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)


@pure
def mat_fill_v0vec_full(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', starts: 'int[:]',  mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat21: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat31: 'float[:,:,:,:,:,:]', mat32: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill21: float, fill22: float, fill23: float, fill31: float, fill32: float, fill33: float):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V0vec -> V0vec. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters
    ----------
    pn : array[int] 
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    mat11, mat12, mat13, mat12, mat22, mat23, mat13, mat23, mat33 : array[float]
        All 9 blocks of the block matrix V0vec -> V0vec that is written to.

    fill11, fill12, fill13, fill12, fill23, fill23, fill31, fill32, fill33 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to mat11, mat12, mat13, mat12, mat22, mat23, mat13, mat23 and mat33.
    """

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # fill matrix entries
    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat21, fill21)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat31, fill31)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat32, fill32)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
def m_v_fill_v0vec_full(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', starts: 'int[:]',  mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat21: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat31: 'float[:,:,:,:,:,:]', mat32: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill21: float, fill22: float, fill23: float, fill31: float, fill32: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]',  fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V0vec -> V0vec. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V0.

    mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32, mat33 : array[float]
        All 9 blocks of the block matrix V0vec -> V0vec that is written to.

    fill11, fill12, fill13, fill21, fill22, fill23, fill31, fill32, fill33 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V0vec and written to vec1, vec2 and vec3.
    """

    # degrees of the basis functions : B-splines (pn)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # fill matrix entries
    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pn1, pn2, pn3, pn1, pn2, pn3,
                                bn1, bn2, bn3, bn1, bn2, bn3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat21, fill21)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat31, fill31)

    filler_kernels.fill_mat(pn1, pn2, pn3, pn1, pn2, pn3,
                            bn1, bn2, bn3, bn1, bn2, bn3,
                            span1, span2, span3,
                            starts, pn,
                            mat32, fill32)


@pure
def mat_fill_v3vec_full(pn: 'int[:]', span1: int, span2: int, span3: int, bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]',  mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat21: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat31: 'float[:,:,:,:,:,:]', mat32: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill21: float, fill22: float, fill23: float, fill31: float, fill32: float, fill33: float):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V3vec -> V3vec. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32, mat33 : array[float]
        All 9 blocks of the block matrix V3vec -> V3vec that is written to.

    fill11, fill12, fill13, fill21, fill22, fill23, fill31, fill32, fill33 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to vec1, vec2 and vec3.
    """

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat11, fill11)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat21, fill21)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat22, fill22)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat31, fill31)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat32, fill32)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat33, fill33)


@pure
def m_v_fill_v3vec_full(pn: 'int[:]', span1: int, span2: int, span3: int, bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]',  mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat21: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat31: 'float[:,:,:,:,:,:]', mat32: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill21: float, fill22: float, fill23: float, fill31: float, fill32: float, fill33: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]',  fill1: float, fill2: float, fill3: float):
    """
    Adds the contribution of one particle to the generic elements (mu,nu) of an accumulation block matrix V3vec -> V3vec. The result is returned in mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32 and mat33.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts : array[int]
        Start indices of the current process in space V3.

    mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32, mat33 : array[float]
        All 9 blocks of the block matrix V3vec -> V3vec that is written to.

    fill11, fill12, fill13, fill21, fill22, fill23, fill31, fill32, fill33 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to mat11, mat12, mat13, mat12, mat22, mat23, mat31, mat32 and mat33.

    vec1, vec2, vec3 : array[float]
        mu=1, mu=2 and mu=3-component of the vector that is written to.

    fill1, fill2, fill3 : float
        Numbers that will be multiplied by the basis functions of V3vec and written to vec1, vec2 and vec3.
    """

    # degrees of the basis functions : D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix entries
    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat11, fill11, vec1, fill1)

    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat22, fill22, vec2, fill2)

    filler_kernels.fill_mat_vec(pd1, pd2, pd3, pd1, pd2, pd3,
                                bd1, bd2, bd3, bd1, bd2, bd3,
                                span1, span2, span3,
                                starts, pn,
                                mat33, fill33, vec3, fill3)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat12, fill12)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat13, fill13)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat21, fill21)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat23, fill23)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat31, fill31)

    filler_kernels.fill_mat(pd1, pd2, pd3, pd1, pd2, pd3,
                            bd1, bd2, bd3, bd1, bd2, bd3,
                            span1, span2, span3,
                            starts, pn,
                            mat32, fill32)


@pure
def m_v_fill_v1_pressure_full(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat11_11: 'float[:,:,:,:,:,:]', mat12_11: 'float[:,:,:,:,:,:]', mat13_11: 'float[:,:,:,:,:,:]', mat22_11: 'float[:,:,:,:,:,:]', mat23_11: 'float[:,:,:,:,:,:]', mat33_11: 'float[:,:,:,:,:,:]', mat11_12: 'float[:,:,:,:,:,:]', mat12_12: 'float[:,:,:,:,:,:]', mat13_12: 'float[:,:,:,:,:,:]', mat22_12: 'float[:,:,:,:,:,:]', mat23_12: 'float[:,:,:,:,:,:]', mat33_12: 'float[:,:,:,:,:,:]', mat11_13: 'float[:,:,:,:,:,:]', mat12_13: 'float[:,:,:,:,:,:]', mat13_13: 'float[:,:,:,:,:,:]', mat22_13: 'float[:,:,:,:,:,:]', mat23_13: 'float[:,:,:,:,:,:]', mat33_13: 'float[:,:,:,:,:,:]', mat11_22: 'float[:,:,:,:,:,:]', mat12_22: 'float[:,:,:,:,:,:]', mat13_22: 'float[:,:,:,:,:,:]', mat22_22: 'float[:,:,:,:,:,:]', mat23_22: 'float[:,:,:,:,:,:]', mat33_22: 'float[:,:,:,:,:,:]', mat11_23: 'float[:,:,:,:,:,:]', mat12_23: 'float[:,:,:,:,:,:]', mat13_23: 'float[:,:,:,:,:,:]', mat22_23: 'float[:,:,:,:,:,:]', mat23_23: 'float[:,:,:,:,:,:]', mat33_23: 'float[:,:,:,:,:,:]', mat11_33: 'float[:,:,:,:,:,:]', mat12_33: 'float[:,:,:,:,:,:]', mat13_33: 'float[:,:,:,:,:,:]', mat22_33: 'float[:,:,:,:,:,:]', mat23_33: 'float[:,:,:,:,:,:]', mat33_33: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float, vec1_1: 'float[:,:,:]', vec2_1: 'float[:,:,:]',  vec3_1: 'float[:,:,:]', vec1_2: 'float[:,:,:]', vec2_2: 'float[:,:,:]',  vec3_2: 'float[:,:,:]', vec1_3: 'float[:,:,:]', vec2_3: 'float[:,:,:]',  vec3_3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float, vx: float, vy: float, vz: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. 
    The result is returned in mat11_xy, mat12_xy, mat13_xy, mat22_xy, mat23_xy, mat33_xy, vec1_x, vec2_x, vec3_x (x and y denotes components of velocity for the accumulation of the pressure tensor).

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    mat.._.. : array[float]
        (mu, nu)-th element (mu, nu=1,2,3) of the block matrix corresponding to the pressure term with velocity components v_a and v_b (a,b=x,y,z). 

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Number that will be multiplied by the basis functions of V1 and written to mat.._..

    vec._. : array[float]
        mu-th element (mu=1,2,3) of the vector corresponding to the pressure term with velocity component v_a (a=x,y,z).

    fill1, fill2, fill3 : float
        Number that will be multplied by the basis functions of V1 and written to vec._.

    vx, vy, vz : float
        Component of the particle velocity.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix and vector entries
    filler_kernels.fill_mat_vec_pressure_full(pd1, pn2, pn3, pd1, pn2, pn3,
                                              bd1, bn2, bn3, bd1, bn2, bn3,
                                              span1, span2, span3,
                                              starts, pn,
                                              mat11_11, mat11_12, mat11_13, mat11_22, mat11_23, mat11_33, fill11,
                                              vec1_1, vec1_2, vec1_3, fill1,
                                              vx, vy, vz)

    filler_kernels.fill_mat_vec_pressure_full(pn1, pd2, pn3, pn1, pd2, pn3,
                                              bn1, bd2, bn3, bn1, bd2, bn3,
                                              span1, span2, span3,
                                              starts, pn,
                                              mat22_11, mat22_12, mat22_13, mat22_22, mat22_23, mat22_33, fill22,
                                              vec2_1, vec2_2, vec2_3, fill2,
                                              vx, vy, vz)

    filler_kernels.fill_mat_vec_pressure_full(pn1, pn2, pd3, pn1, pn2, pd3,
                                              bn1, bn2, bd3, bn1, bn2, bd3,
                                              span1, span2, span3,
                                              starts, pn,
                                              mat33_11, mat33_12, mat33_13, mat33_22, mat33_23, mat33_33, fill33,
                                              vec3_1, vec3_2, vec3_3, fill3,
                                              vx, vy, vz)

    filler_kernels.fill_mat_pressure_full(pd1, pn2, pn3, pn1, pd2, pn3,
                                          bd1, bn2, bn3, bn1, bd2, bn3,
                                          span1, span2, span3,
                                          starts, pn,
                                          mat12_11, mat12_12, mat12_13, mat12_22, mat12_23, mat12_33, fill12,
                                          vx, vy, vz)

    filler_kernels.fill_mat_pressure_full(pd1, pn2, pn3, pn1, pn2, pd3,
                                          bd1, bn2, bn3, bn1, bn2, bd3,
                                          span1, span2, span3,
                                          starts, pn,
                                          mat13_11, mat13_12, mat13_13, mat13_22, mat13_23, mat13_33, fill13,
                                          vx, vy, vz)

    filler_kernels.fill_mat_pressure_full(pn1, pd2, pn3, pn1, pn2, pd3,
                                          bn1, bd2, bn3, bn1, bn2, bd3,
                                          span1, span2, span3,
                                          starts, pn,
                                          mat23_11, mat23_12, mat23_13, mat23_22, mat23_23, mat23_33, fill23,
                                          vx, vy, vz)


@pure
def m_v_fill_v1_pressure(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', mat11_11: 'float[:,:,:,:,:,:]', mat12_11: 'float[:,:,:,:,:,:]', mat13_11: 'float[:,:,:,:,:,:]', mat22_11: 'float[:,:,:,:,:,:]', mat23_11: 'float[:,:,:,:,:,:]', mat33_11: 'float[:,:,:,:,:,:]', mat11_12: 'float[:,:,:,:,:,:]', mat12_12: 'float[:,:,:,:,:,:]', mat13_12: 'float[:,:,:,:,:,:]', mat22_12: 'float[:,:,:,:,:,:]', mat23_12: 'float[:,:,:,:,:,:]', mat33_12: 'float[:,:,:,:,:,:]', mat11_22: 'float[:,:,:,:,:,:]', mat12_22: 'float[:,:,:,:,:,:]', mat13_22: 'float[:,:,:,:,:,:]', mat22_22: 'float[:,:,:,:,:,:]', mat23_22: 'float[:,:,:,:,:,:]', mat33_22: 'float[:,:,:,:,:,:]', fill11: float, fill12: float, fill13: float, fill22: float, fill23: float, fill33: float, vec1_1: 'float[:,:,:]', vec2_1: 'float[:,:,:]',  vec3_1: 'float[:,:,:]', vec1_2: 'float[:,:,:]', vec2_2: 'float[:,:,:]',  vec3_2: 'float[:,:,:]', fill1: float, fill2: float, fill3: float, vx: float, vy: float):
    """
    Adds the contribution of one particle to the symmetric elements (mu,nu)=(1,1), (mu,nu)=(1,2), (mu,nu)=(1,3), (mu,nu)=(2,2), (mu,nu)=(2,3) and (mu,nu)=(3,3) of an accumulation block matrix V1 -> V1. 
    The result is returned in mat11_xy, mat12_xy, mat13_xy, mat22_xy, mat23_xy, mat33_xy, vec1_x, vec2_x, vec3_x (x and y denotes components of velocity for the accumulation of the pressure tensor).

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    span1, span2, span3 : int
        Spline knot span indices in each direction.

    bn1, bn2, bn3 : array[float]
        Evaluated B-splines at particle position in each direction.

    bd1, bd2, bd3 : array[float]
        Evaluated D-splines at particle position in each direction.

    starts1 : array[int]
        Start indices of the current process in space V1.

    mat.._.. : array[float]
        (mu, nu)-th element (mu, nu=1,2,3) of the block matrix corresponding to the pressure term with velocity components v_a and v_b (a,b=x,y,z). 

    fill11, fill12, fill13, fill22, fill23, fill33 : float
        Number that will be multiplied by the basis functions of V1 and written to mat.._..

    vec._. : array[float]
        mu-th element (mu=1,2,3) of the vector corresponding to the pressure term with velocity component v_a (a=x,y,z).

    fill1, fill2, fill3 : float
        Number that will be multplied by the basis functions of V1 and written to vec._.

    vx, vy, vz : float
        Component of the particle velocity.
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill matrix and vector entries
    filler_kernels.fill_mat_vec_pressure(pd1, pn2, pn3, pd1, pn2, pn3,
                                         bd1, bn2, bn3, bd1, bn2, bn3,
                                         span1, span2, span3,
                                         starts, pn,
                                         mat11_11, mat11_12, mat11_22, fill11,
                                         vec1_1, vec1_2, fill1,
                                         vx, vy)

    filler_kernels.fill_mat_vec_pressure(pn1, pd2, pn3, pn1, pd2, pn3,
                                         bn1, bd2, bn3, bn1, bd2, bn3,
                                         span1, span2, span3,
                                         starts, pn,
                                         mat22_11, mat22_12, mat22_22, fill22,
                                         vec2_1, vec2_2, fill2,
                                         vx, vy)

    filler_kernels.fill_mat_vec_pressure(pn1, pn2, pd3, pn1, pn2, pd3,
                                         bn1, bn2, bd3, bn1, bn2, bd3,
                                         span1, span2, span3,
                                         starts, pn,
                                         mat33_11, mat33_12, mat33_22, fill33,
                                         vec3_1, vec3_2, fill3,
                                         vx, vy)

    filler_kernels.fill_mat_pressure(pd1, pn2, pn3, pn1, pd2, pn3,
                                     bd1, bn2, bn3, bn1, bd2, bn3,
                                     span1, span2, span3,
                                     starts, pn,
                                     mat12_11, mat12_12, mat12_22, fill12,
                                     vx, vy)

    filler_kernels.fill_mat_pressure(pd1, pn2, pn3, pn1, pn2, pd3,
                                     bd1, bn2, bn3, bn1, bn2, bd3,
                                     span1, span2, span3,
                                     starts, pn,
                                     mat13_11, mat13_12, mat13_22, fill13,
                                     vx, vy)

    filler_kernels.fill_mat_pressure(pn1, pd2, pn3, pn1, pn2, pd3,
                                     bn1, bd2, bn3, bn1, bn2, bd3,
                                     span1, span2, span3,
                                     starts, pn,
                                     mat23_11, mat23_12, mat23_22, fill23,
                                     vx, vy)


@pure
def hybrid_density(Nel: 'int[:]', pn: 'int[:]', cell_left: 'int[:]', cell_number: 'int[:]', span1: 'int', span2: 'int', span3: 'int', starts: 'int[:]', ie1: 'int', ie2: 'int', ie3: 'int', temp1: 'float[:]', temp4: 'float[:]', quad: 'int[:]', quad_pts_x: 'float[:]', quad_pts_y: 'float[:]', quad_pts_z: 'float[:]', compact: 'float[:]', eta1: 'float', eta2: 'float', eta3: 'float', mat: 'float[:,:,:,:,:,:]', weight: 'float', p_shape: 'int[:]', p_size: 'float[:]', grids_shapex: 'float[:]', grids_shapey: 'float[:]', grids_shapez: 'float[:]'):

    filler_kernels.hy_density(Nel, pn, cell_left, cell_number, span1, span2, span3, starts, ie1, ie2, ie3, temp1, temp4, quad, quad_pts_x,
                              quad_pts_y, quad_pts_z, compact, eta1, eta2, eta3, mat, weight, p_shape, p_size, grids_shapex, grids_shapey, grids_shapez)


@pure
def vec_fill_v0(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', starts: 'int[:]', vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """TODO
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # fill vector entries
    filler_kernels.fill_vec(pn1, pn2, pn3,
                            bn1, bn2, bn3,
                            span1, span2, span3,
                            starts,
                            vec1, fill1)

    filler_kernels.fill_vec(pn1, pn2, pn3,
                            bn1, bn2, bn3,
                            span1, span2, span3,
                            starts,
                            vec2, fill2)

    filler_kernels.fill_vec(pn1, pn2, pn3,
                            bn1, bn2, bn3,
                            span1, span2, span3,
                            starts,
                            vec3, fill3)


@pure
def vec_fill_v1(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """TODO
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill vector entries
    filler_kernels.fill_vec(pd1, pn2, pn3,
                            bd1, bn2, bn3,
                            span1, span2, span3,
                            starts,
                            vec1, fill1)

    filler_kernels.fill_vec(pn1, pd2, pn3,
                            bn1, bd2, bn3,
                            span1, span2, span3,
                            starts,
                            vec2, fill2)

    filler_kernels.fill_vec(pn1, pn2, pd3,
                            bn1, bn2, bd3,
                            span1, span2, span3,
                            starts,
                            vec3, fill3)


@pure
def vec_fill_v2(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', bd1: 'float[:]', bd2: 'float[:]', bd3: 'float[:]', starts: 'int[:]', vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """TODO
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # fill vector entries
    filler_kernels.fill_vec(pn1, pd2, pd3,
                            bn1, bd2, bd3,
                            span1, span2, span3,
                            starts,
                            vec1, fill1)

    filler_kernels.fill_vec(pd1, pn2, pd3,
                            bd1, bn2, bd3,
                            span1, span2, span3,
                            starts,
                            vec2, fill2)

    filler_kernels.fill_vec(pd1, pd2, pn3,
                            bd1, bd2, bn3,
                            span1, span2, span3,
                            starts,
                            vec3, fill3)


@pure
def scalar_fill_v0(pn: 'int[:]', span1: int, span2: int, span3: int, bn1: 'float[:]', bn2: 'float[:]', bn3: 'float[:]', starts: 'int[:]', vec: 'float[:,:,:]', fill: float):
    """TODO
    """

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    filler_kernels.fill_vec(pn1, pn2, pn3, bn1, bn2, bn3, span1,
                            span2, span3, starts, vec, fill)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def vec_fill_b_v0(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """TODO
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_splines_slim(tn1, pn1, eta1, span1, bn1)
    bsplines_kernels.b_splines_slim(tn2, pn2, eta2, span2, bn2)
    bsplines_kernels.b_splines_slim(tn3, pn3, eta3, span3, bn3)

    # fill vector entries
    filler_kernels.fill_vec(pn1, pn2, pn3,
                            bn1, bn2, bn3,
                            span1, span2, span3,
                            starts,
                            vec1, fill1)

    filler_kernels.fill_vec(pn1, pn2, pn3,
                            bn1, bn2, bn3,
                            span1, span2, span3,
                            starts,
                            vec2, fill2)

    filler_kernels.fill_vec(pn1, pn2, pn3,
                            bn1, bn2, bn3,
                            span1, span2, span3,
                            starts,
                            vec3, fill3)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def vec_fill_b_v1(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """TODO
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill vector entries
    filler_kernels.fill_vec(pd1, pn2, pn3,
                            bd1, bn2, bn3,
                            span1, span2, span3,
                            starts,
                            vec1, fill1)

    filler_kernels.fill_vec(pn1, pd2, pn3,
                            bn1, bd2, bn3,
                            span1, span2, span3,
                            starts,
                            vec2, fill2)

    filler_kernels.fill_vec(pn1, pn2, pd3,
                            bn1, bn2, bd3,
                            span1, span2, span3,
                            starts,
                            vec3, fill3)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def vec_fill_b_v2(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, vec1: 'float[:,:,:]', vec2: 'float[:,:,:]',  vec3: 'float[:,:,:]', fill1: float, fill2: float, fill3: float):
    """TODO
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
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    # fill vector entries
    filler_kernels.fill_vec(pn1, pd2, pd3,
                            bn1, bd2, bd3,
                            span1, span2, span3,
                            starts,
                            vec1, fill1)

    filler_kernels.fill_vec(pd1, pn2, pd3,
                            bd1, bn2, bd3,
                            span1, span2, span3,
                            starts,
                            vec2, fill2)

    filler_kernels.fill_vec(pd1, pd2, pn3,
                            bd1, bd2, bn3,
                            span1, span2, span3,
                            starts,
                            vec3, fill3)


@pure
@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def scalar_fill_b_v0(pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', starts: 'int[:]', eta1: float, eta2: float, eta3: float, vec: 'float[:,:,:]', fill: float):
    """TODO
    """

    from numpy import empty

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # spans (i.e. index for non-vanishing B-spline basis functions)
    span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
    span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
    span3 = bsplines_kernels.find_span(tn3, pn3, eta3)

    # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
    bsplines_kernels.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
    bsplines_kernels.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
    bsplines_kernels.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

    filler_kernels.fill_vec(pn1, pn2, pn3, bn1, bn2, bn3, span1,
                            span2, span3, starts, vec, fill)
