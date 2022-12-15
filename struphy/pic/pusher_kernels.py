from pyccel.decorators import stack_array

import struphy.linear_algebra.core as linalg
import struphy.geometry.map_eval as map_eval
import struphy.feec.bsplines_kernels as bsp
import struphy.feec.basics.spline_evaluation_3d as eval_3d
import struphy.kinetic_background.background_eval as background_eval
from struphy.pic.pusher_utilities import aux_fun_x_v_stat_e

from numpy import zeros, empty, shape, sqrt, cos, sin


#@stack_array('df', 'df_inv', 'df_inv_t', 'e_form', 'e_cart', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_v_with_efield(markers: 'float[:,:]', dt: float, stage: int,
                       pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                       starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                       kind_map: int, params_map: 'float[:]',
                       p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                       ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                       cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                       e1_1: 'float[:,:,:]', e1_2: 'float[:,:,:]', e1_3: 'float[:,:,:]'):
    r'''Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = DF^{-\top} \hat{\mathbf E}^1(\eta^n_p)

    for each marker :math:`p` in markers array, where :math:`\hat{\mathbf E}^1 in H(\textnormal{curl})`.

    Parameters
    ----------
        e1_1, e1_2, e1_3: array[float]
            3d array of FE coeffs of B-field as 2-form.
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    # allocate for field evaluations (1-form and Cartesian components)
    e_form = empty(3, dtype=float)
    e_cart = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, df, dfinv, dfinv_t, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, e_form, e_cart)
    #$ omp for
    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)
        linalg.matrix_inv_with_det(df, det_df, dfinv)
        linalg.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # electric field: 1-form components
        e_form[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, e1_1, starts1[0])
        e_form[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, e1_2, starts1[1])
        e_form[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, e1_3, starts1[2])

        # electric field: Cartesian components
        linalg.matrix_vector(dfinv_t, e_form, e_cart)

        # update velocities
        markers[ip, 3:6] += dt*e_cart

    #$ omp end parallel

#@stack_array('df', 'b_form', 'b_cart', 'b_norm', 'e', 'v', 'vperp', 'vxb_norm', 'b_normxvperp', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_vxb_analytic(markers: 'float[:,:]', dt: float, stage: int,
                      pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                      starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                      kind_map: int, params_map: 'float[:]',
                      p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                      ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                      cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                      b2_1: 'float[:,:,:]', b2_2: 'float[:,:,:]', b2_3: 'float[:,:,:]'):
    r'''Solves exactly the rotation

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \mathbf v_p(t) \times \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}

    for each marker :math:`p` in markers array, with fixed rotation vector.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)

    # allocate for field evaluations (2-form components, Cartesian components and normalized Cartesian components)
    b_form = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    b_norm = empty(3, dtype=float)

    # particle position and velocity
    e = empty(3, dtype=float)
    v = empty(3, dtype=float)
    
    # perpendicular velocity, v x b_norm and b_norm x vperp
    vperp = empty(3, dtype=float)
    vxb_norm = empty(3, dtype=float)
    b_normxvperp = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (ip, e, v, df, det_df, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b_form, b_cart, b_abs, b_norm, vpar, vxb_norm, vperp, b_normxvperp)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue
            
        e[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in df
        map_eval.df(e[0], e[1], e[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # magnetic field: 2-form components
        b_form[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts2[0])
        b_form[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2[1])
        b_form[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts2[2])

        # magnetic field: Cartesian components
        linalg.matrix_vector(df, b_form, b_cart)
        b_cart[:] = b_cart/det_df

        # normalized magnetic field direction
        b_abs = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)

        if b_abs != 0.:
            b_norm[:] = b_cart/b_abs
        else:
            b_norm[:] = b_cart

        # parallel velocity v.b_norm
        vpar = linalg.scalar_dot(v, b_norm)

        # first component of perpendicular velocity
        linalg.cross(v, b_norm, vxb_norm)
        linalg.cross(b_norm, vxb_norm, vperp)

        # second component of perpendicular velocity
        linalg.cross(b_norm, vperp, b_normxvperp)

        # analytic rotation
        markers[ip, 3:6] = vpar*b_norm + \
            cos(b_abs*dt)*vperp - sin(b_abs*dt)*b_normxvperp

    #$ omp end parallel

#@stack_array('df', 'b_form', 'b_cart', 'b_prod', 'e', 'v', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'identity', 'rhs', 'lhs', 'lhs_inv', 'vec', 'res')
def push_vxb_implicit(markers: 'float[:,:]', dt: float, stage: int,
                      pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                      starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                      kind_map: int, params_map: 'float[:]',
                      p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                      ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                      cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                      b2_1: 'float[:,:,:]', b2_2: 'float[:,:,:]', b2_3: 'float[:,:,:]'):
    r'''Solves the rotation

    .. math::

        \frac{\textnormal d \mathbf v_p(t)}{\textnormal d t} =  \mathbf v_p(t) \times \frac{DF\, \hat{\mathbf B}^2}{\sqrt g}

    with the Crank-Nicolson method for each marker :math:`p` in markers array, with fixed rotation vector.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)

    # allocate for field evaluations (2-form components, Cartesian components and rotation matrix such that vxB = B_prod.v)
    b_form = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    b_prod = zeros((3, 3), dtype=float)

    # particle position and velocity
    e = empty(3, dtype=float)
    v = empty(3, dtype=float)
    
    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # identity matrix
    identity = zeros((3, 3), dtype=float)
    
    identity[0, 0] = 1.
    identity[1, 1] = 1.
    identity[2, 2] = 1.

    # right-hand side and left-hand side of Crank-Nicolson scheme
    rhs = empty((3, 3), dtype=float)
    lhs = empty((3, 3), dtype=float)
    
    lhs_inv = empty((3, 3), dtype=float)
    
    vec = empty(3, dtype=float)
    res = empty(3, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel firstprivate(b_prod) private (ip, e, v, df, det_df, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b_form, b_cart, rhs, lhs, lhs_inv, vec, res)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in df
        map_eval.df(e[0], e[1], e[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # magnetic field: 2-form components
        b_form[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts2[0])
        b_form[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2[1])
        b_form[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts2[2])

        # magnetic field: Cartesian components
        linalg.matrix_vector(df, b_form, b_cart)
        b_cart[:] = b_cart/det_df

        # magnetic field: rotation matrix
        b_prod[0, 1] =  b_cart[2]
        b_prod[0, 2] = -b_cart[1]

        b_prod[1, 0] = -b_cart[2]
        b_prod[1, 2] =  b_cart[0]

        b_prod[2, 0] =  b_cart[1]
        b_prod[2, 1] = -b_cart[0]

        # solve 3x3 system
        rhs[:, :] = identity + dt/2*b_prod
        lhs[:, :] = identity - dt/2*b_prod
        
        linalg.matrix_inv(lhs, lhs_inv)
        
        linalg.matrix_vector(rhs, v, vec)
        linalg.matrix_vector(lhs_inv, vec, res)
        
        markers[ip, 3:6] = res

    #$ omp end parallel

#@stack_array('df', 'b_form', 'u_form', 'b_cart', 'u_cart', 'e_cart', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_bxu_Hdiv(markers: 'float[:,:]', dt: float, stage: int,
                  pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                  starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                  kind_map: int, params_map: 'float[:]',
                  p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                  ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                  cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                  b2_1: 'float[:,:,:]', b2_2: 'float[:,:,:]', b2_3: 'float[:,:,:]',
                  u2_1: 'float[:,:,:]', u2_2: 'float[:,:,:]', u2_3: 'float[:,:,:]'):
    r'''Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = DF^{-\top} \left(  \hat{\mathbf B}^2 \times \frac{\hat{\mathbf U}^2}{\sqrt g}  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\hat{\mathbf U}^2 \in H(\textnormal{div})`.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.

        u2_1, u2_2, u2_3: array[float]
            3d array of FE coeffs of U-field as 2-form.
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)

    # allocate for field evaluations (2-form and Cartesian components)
    b_form = empty(3, dtype=float)
    u_form = empty(3, dtype=float)
    
    b_cart = empty(3, dtype=float)
    u_cart = empty(3, dtype=float)
    
    e_cart = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, df, det_df, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b_form, b_cart, u_form, u_cart, e_cart)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker data
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # magnetic field: 2-form components
        b_form[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts2[0])
        b_form[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2[1])
        b_form[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts2[2])
        
        # magnetic field: Cartesian components
        linalg.matrix_vector(df, b_form, b_cart)
        b_cart[:] = b_cart/det_df

        # velocity field: 2-form components
        u_form[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u2_1, starts2[0])
        u_form[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u2_2, starts2[1])
        u_form[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u2_3, starts2[2])
        
        linalg.matrix_vector(df, u_form, u_cart)
        u_cart[:] = u_cart/det_df

        # electric field E = B x U
        linalg.cross(b_cart, u_cart, e_cart)

        # update velocities
        markers[ip, 3:6] += dt*e_cart

    #$ omp end parallel

#@stack_array('df', 'dfinv', 'dfinv_t', 'b_form', 'u_form', 'b_cart', 'u_cart', 'e_cart', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_bxu_Hcurl(markers: 'float[:,:]', dt: float, stage: int,
                   pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                   starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                   kind_map: int, params_map: 'float[:]',
                   p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                   ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                   cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                   b2_1: 'float[:,:,:]', b2_2: 'float[:,:,:]', b2_3: 'float[:,:,:]',
                   u1_1: 'float[:,:,:]', u1_2: 'float[:,:,:]', u1_3: 'float[:,:,:]'):
    r'''Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = DF^{-\top} \left(  \hat{\mathbf B}^2 \times G^{-1}\hat{\mathbf U}^1  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\hat{\mathbf U}^1 \in H(\textnormal{curl})`.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.

        u1_1, u1_2, u1_3: array[float]
            3d array of FE coeffs of U-field as 1-form.
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    # allocate for field evaluations (2-form and Cartesian components)
    b_form = empty(3, dtype=float)
    u_form = empty(3, dtype=float)
    
    b_cart = empty(3, dtype=float)
    u_cart = empty(3, dtype=float)
    
    e_cart = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, df, det_df, dfinv, dfinv_t, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b_form, b_cart, u_form, u_cart, e_cart)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker data
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)
        linalg.matrix_inv_with_det(df, det_df, dfinv)
        linalg.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # magnetic field: 2-form components
        b_form[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts2[0])
        b_form[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2[1])
        b_form[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts2[2])
        
        # magnetic field: Cartesian components
        linalg.matrix_vector(df, b_form, b_cart)
        b_cart[:] = b_cart/det_df

        # velocity field: 1-form components
        u_form[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u1_1, starts1[0])
        u_form[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u1_2, starts1[1])
        u_form[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, u1_3, starts1[2])

        # velocity field: Cartesian components
        linalg.matrix_vector(dfinv_t, u_form, u_cart)
        
        # electric field E = B x U
        linalg.cross(b_cart, u_cart, e_cart)

        # update velocities
        markers[ip, 3:6] += dt*e_cart

    #$ omp end parallel

#@stack_array('df', 'b_form', 'u_form', 'b_cart', 'u_cart', 'e_cart', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_bxu_H1vec(markers: 'float[:,:]', dt: float, stage: int,
                   pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                   starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                   kind_map: int, params_map: 'float[:]',
                   p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                   ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                   cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                   b2_1: 'float[:,:,:]', b2_2: 'float[:,:,:]', b2_3: 'float[:,:,:]',
                   uv_1: 'float[:,:,:]', uv_2: 'float[:,:,:]', uv_3: 'float[:,:,:]'):
    r'''Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = DF^{-\top} \left(  \hat{\mathbf B}^2 \times \hat{\mathbf U}  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\hat{\mathbf U}` is a vector-field (dual to 1-form) in :math:`(H^1)^3`.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.

        uv_1, uv_2, uv_3: array[float]
            3d array of FE coeffs of U-field as vector field in (H^1)^3.
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)

    # allocate for field evaluations (2-form and Cartesian components)
    b_form = empty(3, dtype=float)
    u_form = empty(3, dtype=float)
    
    b_cart = empty(3, dtype=float)
    u_cart = empty(3, dtype=float)
    
    e_cart = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, df, det_df, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b_form, b_cart, u_form, u_cart, e_cart)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker data
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # magnetic field: 2-form components
        b_form[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts2[0])
        b_form[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2[1])
        b_form[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts2[2])
        
        # magnetic field: Cartesian components
        linalg.matrix_vector(df, b_form, b_cart)
        b_cart[:] = b_cart/det_df

        # velocity field: vector field components
        u_form[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, uv_1, starts0)
        u_form[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, uv_2, starts0)
        u_form[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, uv_3, starts0)
        
        # velocity field: Cartesian components
        linalg.matrix_vector(df, u_form, u_cart)

        # electric field E = B x U
        linalg.cross(b_cart, u_cart, e_cart)

        # update velocities
        markers[ip, 3:6] += dt*e_cart

    #$ omp end parallel

#@stack_array('df', 'dfinv', 'dfinv_t', 'b_form', 'u_form', 'b_diff', 'b_cart', 'u_cart', 'b_grad', 'e_cart', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'der1', 'der2', 'der3')
def push_bxu_Hdiv_pauli(markers: 'float[:,:]', dt: float, stage: int,
                        pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                        starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                        kind_map: int, params_map: 'float[:]',
                        p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                        ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                        cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                        b2_1: 'float[:,:,:]', b2_2: 'float[:,:,:]', b2_3: 'float[:,:,:]',
                        u2_1: 'float[:,:,:]', u2_2: 'float[:,:,:]', u2_3: 'float[:,:,:]',
                        b0: 'float[:,:,:]',
                        mu: 'float[:]'):
    r'''Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = DF^{-\top} \left(  \hat{\mathbf B}^2 \times \frac{\hat{\mathbf U}^2}{\sqrt g} - \mu\,\nabla \hat{|\mathbf B|}^0  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\hat{\mathbf U}^2 \in H(\textnormal{div})` and :math:`\hat{|\mathbf B|}^0 \in H^1`.

    Parameters
    ----------
        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.

        u2_1, u2_2, u2_3: array[float]
            3d array of FE coeffs of U-field as 2-form.

        b0 : array[float]
            3d array of FE coeffs of abs(B) as 0-form.

        mu : array[float]
            1d array of size n_markers holding particle magnetic moments.
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    # allocate for field evaluations (2-form and Cartesian components)
    b_form = empty(3, dtype=float)
    u_form = empty(3, dtype=float)
    b_diff = empty(3, dtype=float)
    
    b_cart = empty(3, dtype=float)
    u_cart = empty(3, dtype=float)
    b_grad = empty(3, dtype=float)
    
    e_cart = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)
    
    der1 = empty(pn[0] + 1, dtype=float)
    der2 = empty(pn[1] + 1, dtype=float)
    der3 = empty(pn[2] + 1, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta1, eta2, eta3, df, det_df, dfinv, dfinv_t, span1, span2, span3, bn1, bn2, bn3, der1, der2, der3, bd1, bd2, bd3, b_form, b_cart, b_diff, b_grad, u_form, u_cart, e_cart)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        # marker data
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)
        linalg.matrix_inv_with_det(df, det_df, dfinv)
        linalg.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_der_splines_slim(tn1, pn[0], eta1, span1, bn1, der1)
        bsp.b_der_splines_slim(tn2, pn[1], eta2, span2, bn2, der2)
        bsp.b_der_splines_slim(tn3, pn[2], eta3, span3, bn3, der3)

        bsp.d_splines_slim(tn1, pn[0], eta1, span1, bd1)
        bsp.d_splines_slim(tn2, pn[1], eta2, span2, bd2)
        bsp.d_splines_slim(tn3, pn[2], eta3, span3, bd3)

        # magnetic field: 2-form components
        b_form[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts2[0])
        b_form[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2[1])
        b_form[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts2[2])
        
        # magnetic field: Cartesian components
        linalg.matrix_vector(df, b_form, b_cart)
        b_cart[:] = b_cart/det_df
        
        # magnetic field: evaluation of gradient (1-form)
        b_diff[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], der1, bn2, bn3, span1, span2, span3, b0, starts1[0])
        b_diff[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, der2, bn3, span1, span2, span3, b0, starts1[1])
        b_diff[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, der3, span1, span2, span3, b0, starts1[2])
        
        # magnetic field: evaluation of gradient (Cartesian components)
        linalg.matrix_vector(dfinv_t, b_diff, b_grad)

        # velocity field: 2-form components
        u_form[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u2_1, starts2[0])
        u_form[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u2_2, starts2[1])
        u_form[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u2_3, starts2[2])
        
        linalg.matrix_vector(df, u_form, u_cart)
        u_cart[:] = u_cart/det_df

        # electric field E = B x U
        linalg.cross(b_cart, u_cart, e_cart)
        
        # additional artificial electric field of Pauli markers
        e_cart[:] = e_cart - mu[ip]*b_grad

        # update velocities
        markers[ip, 3:6] += dt*e_cart

    #$ omp end parallel

#@stack_array('df', 'dfinv', 'dfinv_t', 'e', 'e_cart', 'v', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_pc_GXu_full(markers: 'float[:,:]', dt: float, stage: int,
                     pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                     starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                     kind_map: int, params_map: 'float[:]',
                     p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                     ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                     cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                     GXu_11: 'float[:,:,:]', GXu_12: 'float[:,:,:]', GXu_13: 'float[:,:,:]',
                     GXu_21: 'float[:,:,:]', GXu_22: 'float[:,:,:]', GXu_23: 'float[:,:,:]',
                     GXu_31: 'float[:,:,:]', GXu_32: 'float[:,:,:]', GXu_33: 'float[:,:,:]'):
    r'''Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = - DF^{-\top} \left(  \boldsymbol \Lambda^1 \mathbb G \mathcal X(\mathbf u, \mathbf v)  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\mathbf u` 
    are the coefficients of the mhd velocity field (either 1-form or 2-form) and :math:`\mathcal X`
    is either the MHD operator :meth:`struphy.psydac_api.basis_projection_ops.MHDOperators.assemble_X1` (if u is 1-form)
    or :meth:`struphy.psydac_api.basis_projection_ops.MHDOperators.assemble_X2` (if u is 2-form).

    Parameters
    ----------
        grad_Xu_ij: array[float]
            3d array of FE coeffs of :math:`\nabla_j(\mathcal X \cdot \mathbf u)_i`. i,j=1,2,3.
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    # allocate for field evaluations
    e = empty(3, dtype=float)
    e_cart = empty(3, dtype=float)

    # particle velocity
    v = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        linalg.matrix_inv(df, dfinv)
        linalg.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # Evaluate grad(X(u, v)) at the particle positions
        e[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, GXu_11 * v[0] + GXu_21 * v[1] + GXu_31 * v[2], starts1[0])
        e[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, GXu_12 * v[0] + GXu_22 * v[1] + GXu_32 * v[2], starts1[1])
        e[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, GXu_13 * v[0] + GXu_23 * v[1] + GXu_33 * v[2], starts1[2])

        # electric field in Cartesian coordinates
        linalg.matrix_vector(dfinv_t, e, e_cart)

        # update velocities
        markers[ip, 3:6] -= dt*e_cart[:]/2.

#@stack_array('df', 'dfinv', 'dfinv_t', 'e', 'e_cart', 'v', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_pc_GXu(markers: 'float[:,:]', dt: float, stage: int,
                pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                kind_map: int, params_map: 'float[:]',
                p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                GXu_11: 'float[:,:,:]', GXu_12: 'float[:,:,:]', GXu_13: 'float[:,:,:]',
                GXu_21: 'float[:,:,:]', GXu_22: 'float[:,:,:]', GXu_23: 'float[:,:,:]',
                GXu_31: 'float[:,:,:]', GXu_32: 'float[:,:,:]', GXu_33: 'float[:,:,:]'):
    r'''Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = - DF^{-\top} \left(  \boldsymbol \Lambda^1 \mathbb G \mathcal X(\mathbf u, \mathbf v)  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\mathbf u` 
    are the coefficients of the mhd velocity field (either 1-form or 2-form) and :math:`\mathcal X`
    is either the MHD operator :meth:`struphy.psydac_api.basis_projection_ops.MHDOperators.assemble_X1` (if u is 1-form)
    or :meth:`struphy.psydac_api.basis_projection_ops.MHDOperators.assemble_X2` (if u is 2-form).

    Parameters
    ----------
        grad_Xu_ij: array[float]
            3d array of FE coeffs of :math:`\nabla_j(\mathcal X \cdot \mathbf u)_i`. i,j=1,2,3.
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    # allocate for field evaluations
    e = empty(3, dtype=float)
    e_cart = empty(3, dtype=float)

    # particle velocity
    v = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        linalg.matrix_inv(df, dfinv)
        linalg.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # Evaluate grad(X(u, v)) at the particle positions
        e[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, GXu_11 * v[0] + GXu_21 * v[1], starts1[0])
        e[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, GXu_12 * v[0] + GXu_22 * v[1], starts1[1])
        e[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, GXu_13 * v[0] + GXu_23 * v[1], starts1[2])

        # electric field in Cartesian coordinates
        linalg.matrix_vector(dfinv_t, e, e_cart)

        # update velocities
        markers[ip, 3:6] -= dt*e_cart/2.

#@stack_array('df', 'dfinv', 'e', 'v', 'k')
def push_eta_stage(markers: 'float[:,:]', dt: float, stage: int,
                   pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                   starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                   kind_map: int, params_map: 'float[:]',
                   p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                   ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                   cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                   a: 'float[:]', b: 'float[:]', c: 'float[:]'):
    r'''Single stage of a s-stage Runge-Kutta solve of 

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant.
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)

    # marker position e and velocity v
    e = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]
    
    # get number of stages
    n_stages = shape(b)[0]
    
    if stage == n_stages - 1:
        last = 1.
    else:
        last = 0.

    #$ omp parallel private(ip, e, v, df, dfinv, k)
    #$ omp for
    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

        # evaluate Jacobian, result in df
        map_eval.df(e[0], e[1], e[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # evaluate inverse Jacobian matrix
        linalg.matrix_inv(df, dfinv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k)
        
        # accumulation for last stage
        markers[ip, 12:15] += dt*b[stage]*k

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = markers[ip, 9:12] + dt*a[stage]*k + last*markers[ip, 12:15]

    #$ omp end parallel

#@stack_array('df', 'dfinv', 'dfinv_t', 'ginv', 'eta', 'v', 'u', 'k', 'k_v', 'k_u', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_pc_eta_rk4_Hcurl_full(markers: 'float[:,:]', dt: float, stage: int,
                               pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                               starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                               kind_map: int, params_map: 'float[:]',
                               p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                               ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                               cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                               u_1: 'float[:,:,:]', u_2: 'float[:,:,:]', u_3: 'float[:,:,:]'):
    r'''Fourth order Runge-Kutta solve of 

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and 

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
        u_1, u_2, u_3: array[float]
            3d array of FE coeffs of U-field, either as 1-form or as 2-form.

        u_basis : int
            U is 1-form (u_basis=1) or a 2-form (u_basis=2).
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker position and velocity
    eta = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)
    k_u = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.
    else: nk = 2.

    # which stage
    if stage == 3:
        last = 1.
        cont  = 0.
    elif stage == 2:
        last = 0.
        cont  = 2.
    else:
        last = 0.
        cont = 1.

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]
        v[:]   = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in df
        map_eval.df(eta[0], eta[1], eta[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        linalg.matrix_inv(df, dfinv)
        linalg.transpose(dfinv, dfinv_t)
        linalg.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta[0])
        span2 = bsp.find_span(tn2, pn[1], eta[1])
        span3 = bsp.find_span(tn3, pn[2], eta[2])

        bsp.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # U-field
        u[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u_1, starts1[0])
        u[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u_2, starts1[1])
        u[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, u_3, starts1[2])

        # transform to vector field
        linalg.matrix_vector(ginv, u, k_u)

        # sum contribs
        k[:] = k_v + k_u

        # accum k
        markers[ip, 12:15] += k*nk/6.
        
        # update markers for the next stage
        markers[ip, 0:3] = (markers[ip, 9:12] + dt*k/2 * cont + dt*markers[ip, 12:15]* last)

#@stack_array('df', 'dfinv', 'dfinv_t', 'ginv', 'eta', 'v', 'u', 'k', 'k_v', 'k_u', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_pc_eta_rk4_Hdiv_full(markers: 'float[:,:]', dt: float, stage: int,
                              pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                              starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                              kind_map: int, params_map: 'float[:]',
                              p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                              ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                              cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                              u_1: 'float[:,:,:]', u_2: 'float[:,:,:]', u_3: 'float[:,:,:]'):
    r'''Fourth order Runge-Kutta solve of 

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and 

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
        u_1, u_2, u_3: array[float]
            3d array of FE coeffs of U-field, either as 1-form or as 2-form.

        u_basis : int
            U is 1-form (u_basis=1) or a 2-form (u_basis=2).
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker position and velocity
    eta = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)
    k_u = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.
    else:
        nk = 2.

    # is it the last stage?
    if stage == 3:
        last = 1.
        cont  = 0.
    else:
        last = 0.
        cont  = 1.

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in df
        map_eval.df(eta[0], eta[1], eta[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)
        linalg.matrix_inv(df, dfinv)
        linalg.transpose(dfinv, dfinv_t)
        linalg.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta[0])
        span2 = bsp.find_span(tn2, pn[1], eta[1])
        span3 = bsp.find_span(tn3, pn[2], eta[2])

        bsp.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # U-field
        u[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u_1, starts2[0])
        u[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u_2, starts2[1])
        u[2] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u_3, starts2[2])

        # transform to vector field
        k_u[:] = u/det_df

        # sum contribs
        k[:] = k_v + k_u

        # accum k
        markers[ip, 12:15] += k*nk/6.
        
        # update markers for the next stage
        markers[ip, 0:3] = (markers[ip, 9:12] + dt*k/2 * cont + dt*markers[ip, 12:15]* last)

#@stack_array('df', 'dfinv', 'dfinv_t', 'ginv', 'eta', 'v', 'u', 'k', 'k_v', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_pc_eta_rk4_H1vec_full(markers: 'float[:,:]', dt: float, stage: int,
                               pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                               starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                               kind_map: int, params_map: 'float[:]',
                               p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                               ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                               cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                               u_1: 'float[:,:,:]', u_2: 'float[:,:,:]', u_3: 'float[:,:,:]'):
    r'''Fourth order Runge-Kutta solve of 

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and 

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
        u_1, u_2, u_3: array[float]
            3d array of FE coeffs of U-field, either as 1-form or as 2-form.

        u_basis : int
            U is 1-form (u_basis=1) or a 2-form (u_basis=2).
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker position and velocity
    eta = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.
    else: nk = 2.

    # which stage
    if stage == 3:
        last = 1.
        cont  = 0.
    elif stage == 2:
        last = 0.
        cont  = 2.
    else:
        last = 0.
        cont = 1.

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]
        v[:]   = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in df
        map_eval.df(eta[0], eta[1], eta[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        linalg.matrix_inv(df, dfinv)
        linalg.transpose(dfinv, dfinv_t)
        linalg.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta[0])
        span2 = bsp.find_span(tn2, pn[1], eta[1])
        span3 = bsp.find_span(tn3, pn[2], eta[2])

        bsp.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # U-field
        u[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u_1, starts0)
        u[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u_2, starts0)
        u[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u_3, starts0)

        # sum contribs
        k[:] = k_v + u

        # accum k
        markers[ip, 12:15] += k*nk/6.
        
        # update markers for the next stage
        markers[ip, 0:3] = (markers[ip, 9:12] + dt*k/2 * cont + dt*markers[ip, 12:15]* last)       

#@stack_array('df', 'dfinv', 'dfinv_t', 'ginv', 'eta', 'v', 'u', 'k', 'k_v', 'k_u', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_pc_eta_rk4_Hcurl(markers: 'float[:,:]', dt: float, stage: int,
                          pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                          starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                          kind_map: int, params_map: 'float[:]',
                          p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                          ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                          cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                          u_1: 'float[:,:,:]', u_2: 'float[:,:,:]', u_3: 'float[:,:,:]'):
    r'''Fourth order Runge-Kutta solve of 

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and 

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
        u_1, u_2, u_3: array[float]
            3d array of FE coeffs of U-field, either as 1-form or as 2-form.

        u_basis : int
            U is 1-form (u_basis=1) or a 2-form (u_basis=2).
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker position and velocity
    eta = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)
    k_u = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.
    else: nk = 2.

    # which stage
    if stage == 3:
        last = 1.
        cont  = 0.
    elif stage == 2:
        last = 0.
        cont  = 2.
    else:
        last = 0.
        cont = 1.

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]
        v[:]   = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in df
        map_eval.df(eta[0], eta[1], eta[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        linalg.matrix_inv(df, dfinv)
        linalg.transpose(dfinv, dfinv_t)
        linalg.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta[0])
        span2 = bsp.find_span(tn2, pn[1], eta[1])
        span3 = bsp.find_span(tn3, pn[2], eta[2])

        bsp.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # U-field
        u[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u_1, starts1[0])
        u[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u_2, starts1[1])
        u[2] = 0.

        # transform to vector field
        linalg.matrix_vector(ginv, u, k_u)

        # sum contribs
        k[:] = k_v + k_u

        # accum k
        markers[ip, 12:15] += k*nk/6.
        
        # update markers for the next stage
        markers[ip, 0:3] = (markers[ip, 9:12] + dt*k/2 * cont + dt*markers[ip, 12:15]* last)

#@stack_array('df', 'dfinv', 'dfinv_t', 'ginv', 'eta', 'v', 'u', 'k', 'k_v', 'k_u', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_pc_eta_rk4_Hdiv(markers: 'float[:,:]', dt: float, stage: int,
                              pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                              starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                              kind_map: int, params_map: 'float[:]',
                              p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                              ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                              cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                              u_1: 'float[:,:,:]', u_2: 'float[:,:,:]', u_3: 'float[:,:,:]'):
    r'''Fourth order Runge-Kutta solve of 

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and 

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
        u_1, u_2, u_3: array[float]
            3d array of FE coeffs of U-field, either as 1-form or as 2-form.

        u_basis : int
            U is 1-form (u_basis=1) or a 2-form (u_basis=2).
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker position and velocity
    eta = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)
    k_u = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.
    else:
        nk = 2.

    # is it the last stage?
    if stage == 3:
        last = 1.
        cont  = 0.
    else:
        last = 0.
        cont  = 1.

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]
        v[:] = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in df
        map_eval.df(eta[0], eta[1], eta[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)
        linalg.matrix_inv(df, dfinv)
        linalg.transpose(dfinv, dfinv_t)
        linalg.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta[0])
        span2 = bsp.find_span(tn2, pn[1], eta[1])
        span3 = bsp.find_span(tn3, pn[2], eta[2])

        bsp.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # U-field
        u[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u_1, starts2[0])
        u[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u_2, starts2[1])
        u[2] = 0.

        # transform to vector field
        k_u[:] = u/det_df

        # sum contribs
        k[:] = k_v + k_u

        # accum k
        markers[ip, 12:15] += k*nk/6.
        
        # update markers for the next stage
        markers[ip, 0:3] = (markers[ip, 9:12] + dt*k/2 * cont + dt*markers[ip, 12:15]* last)

#@stack_array('df', 'dfinv', 'dfinv_t', 'ginv', 'eta', 'v', 'u', 'k', 'k_v', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3')
def push_pc_eta_rk4_H1vec(markers: 'float[:,:]', dt: float, stage: int,
                          pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                          starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                          kind_map: int, params_map: 'float[:]',
                          p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                          ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                          cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                          u_1: 'float[:,:,:]', u_2: 'float[:,:,:]', u_3: 'float[:,:,:]'):
    r'''Fourth order Runge-Kutta solve of 

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant and 

    .. math::

        \textnormal{vec}( \hat{\mathbf U}^{1}) = G^{-1}\hat{\mathbf U}^{1}\,,\qquad \textnormal{vec}( \hat{\mathbf U}^{2}) = \frac{\hat{\mathbf U}^{2}}{\sqrt g}\,.

    Parameters
    ----------
        u_1, u_2, u_3: array[float]
            3d array of FE coeffs of U-field, either as 1-form or as 2-form.

        u_basis : int
            U is 1-form (u_basis=1) or a 2-form (u_basis=2).
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    ginv = empty((3, 3), dtype=float)

    # marker position and velocity
    eta = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # U-fiels
    u = empty(3, dtype=float)

    # intermediate stages in RK4
    k = empty(3, dtype=float)
    k_v = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    # assign factor of k for each stage
    if stage == 0 or stage == 3:
        nk = 1.
    else: nk = 2.

    # which stage
    if stage == 3:
        last = 1.
        cont  = 0.
    elif stage == 2:
        last = 0.
        cont  = 2.
    else:
        last = 0.
        cont = 1.

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]
        v[:]   = markers[ip, 3:6]

        # ----------------- stage n in Runge-Kutta method -------------------
        # evaluate Jacobian, result in df
        map_eval.df(eta[0], eta[1], eta[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        linalg.matrix_inv(df, dfinv)
        linalg.transpose(dfinv, dfinv_t)
        linalg.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k_v)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta[0])
        span2 = bsp.find_span(tn2, pn[1], eta[1])
        span3 = bsp.find_span(tn3, pn[2], eta[2])

        bsp.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # U-field
        u[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u_1, starts0)
        u[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u_2, starts0)
        u[2] = 0.

        # sum contribs
        k[:] = k_v + u

        # accum k
        markers[ip, 12:15] += k*nk/6.
        
        # update markers for the next stage
        markers[ip, 0:3] = (markers[ip, 9:12] + dt*k/2 * cont + dt*markers[ip, 12:15]* last)       

#@stack_array('bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'df', 'df_inv', 'x', 'v')
def push_weights_with_efield(markers: 'float[:,:]', dt: float, stage: int,
                             pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                             starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                             kind_map: int, params_map: 'float[:]',
                             p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                             ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                             cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                             e1_1: 'float[:,:,:]', e1_2: 'float[:,:,:]', e1_3: 'float[:,:,:]',
                             f0_spec: 'int', moms_spec: 'int[:]', f0_params: 'float[:]',
                             n_markers_tot: 'int'):
    r'''
    updates the single weights in the e_W substep of the linearized Vlasov Maxwell system;
    c.f. struphy.propagators.propagators.StepEfieldWeights

    Parameters :
    ------------
        e1_1, e1_2, e1_3: array[float]
            3d array of FE coeffs of E-field as 1-form.

        f0_spec : int
            Specifier for kinetic background, 0 -> maxwellian_6d. See Notes.

        moms_spec : array[int]
            Specifier for the seven moments n0, u0x, u0y, u0z, vth0x, vth0y, vth0z (in this order).
            Is 0 for constant moment, for more see :meth:`struphy.kinetic_background.moments_kernels.moments`.

        params : array[float]
            Parameters needed to specify the moments; the order is specified in :ref:`struphy.kinetic_background.moments_kernels`
            for the respective functions available.

        n_markers : int
            total number of particles
    '''

    # total number of basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing N-splines at particle position
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pn[0] + 1, dtype=float)
    bd2 = empty(pn[1] + 1, dtype=float)
    bd3 = empty(pn[2] + 1, dtype=float)

    df      = empty((3,3), dtype=float)
    df_inv  = empty((3,3), dtype=float)
    eta     = empty(3    , dtype=float)
    v       = empty(3    , dtype=float)
    prod    = empty(3    , dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private (ip, eta1, eta2, eta3, eta, v, prod, df, df_inv, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, f0, temp1, temp2, temp3, update)
    #$ omp for
    for ip in range(n_markers):
        if markers[ip, 0] == -1:
            continue

        # position
        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        eta  = [eta1, eta2, eta3]

        # velocity
        v[0] = markers[ip, 3]
        v[1] = markers[ip, 4]
        v[2] = markers[ip, 5]

        # spans (i.e. index for non-vanisle of manishing basis functions)
        span1 = bsp.find_span(tn1, pn1, eta1)
        span2 = bsp.find_span(tn2, pn2, eta2)
        span3 = bsp.find_span(tn3, pn3, eta3)

        # compute bn, bd, i.e. values for non-vanishing B-/D-splines at position eta
        bsp.b_d_splines_slim(tn1, pn1, eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn2, eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn3, eta3, span3, bn3, bd3)

        f0 = background_eval.f0(eta, v, f0_spec, moms_spec, f0_params)

        map_eval.df(eta1, eta2, eta3, kind_map, params_map, t1_map, t2_map, t3_map, p_map, ind1_map, ind2_map, ind3_map, cx, cy, cz, df)
        linalg.matrix_inv(df, df_inv)

        # E-field (1-form)
        temp1 = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3,
                                               span1, span2, span3, e1_1, starts1[0])
        temp2 = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3,
                                               span1, span2, span3, e1_2, starts1[1])
        temp3 = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3,
                                               span1, span2, span3, e1_3, starts1[2])

        linalg.matrix_vector(df_inv, v, prod)

        # w_{n+1} = w_n - dt/2 * 1/(N s_0) * sqrt(f) * ( DF^{-1} v ) * ( e_{n+1} + e_n )
        update = (prod[0] * temp1 + prod[1] * temp2 + prod[2] * temp3) * sqrt(f0) * dt / (2 * n_markers_tot * markers[ip, 7])
        markers[ip, 6] -= update 

    #$ omp end parallel

#@stack_array('particle')
def push_x_v_static_efield(markers: 'float[:,:]', dt: float, stage: int,
                           pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                           starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                           kind_map: int, params_map: 'float[:]',
                           p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                           ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                           cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                           loc1: 'float[:]', loc2: 'float[:]', loc3: 'float[:]',
                           weight1: 'float[:]', weight2: 'float[:]', weight3: 'float[:]',
                           e1_1: 'float[:,:,:]', e1_2: 'float[:,:,:]', e1_3: 'float[:,:,:]',
                           eps: 'float[:]', maxiter: int):
    r"""
    particle pusher for ODE
    
    .. math::
        \frac{\text{d} \mathbf{\eta}}{\text{d} t} & = DL^{-1} \mathbf{v}
        \frac{\text{d} \mathbf{v}}{\text{d} t} & = DL^{-T} \mathbf{E}_0(\mathbf{\eta})

    Parameters 
    ----------
        loc1, loc2, loc3 : array
            contain the positions of the Legendre-Gauss quadrature points of necessary order to integrate basis splines exactly in each direction
        
        weight1, weight2, weight3 : array
            contain the values of the weights for the Legendre-Gauss quadrature in each direction

        e1_1, e1_2, e1_3: array[float]
            3d array of FE coeffs of the background E-field as 1-form.

        eps: array
            determines the accuracy for the position (0th element) and velocity (1st element) with which the implicit scheme is executed

        maxiter : integer
            sets the maximum number of iterations for the iterative scheme
    """

    particle = zeros(9, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, run, temp, k, particle, dt2)
    #$ omp for
    for ip in range(n_markers):

        particle[:] = markers[ip, :]

        run = 1
        k   = 0

        while run != 0:
            k += 1
            if k == 5:
                print('Splitting the time steps into 4 has not been enough, aborting the iteration.')
                print()
                break

            run = 0

            dt2 = dt/k

            for _ in range(k):
                temp = aux_fun_x_v_stat_e(particle,
                                          pn, tn1, tn2, tn3,
                                          starts1[0], starts1[1], starts1[2],
                                          kind_map, params_map,
                                          p_map, t1_map, t2_map, t3_map,
                                          ind1_map, ind2_map, ind3_map,
                                          cx, cy, cz,
                                          dt2,
                                          loc1, loc2, loc3, weight1, weight2, weight3,
                                          e1_1, e1_2, e1_3,
                                          eps, maxiter)
                run = run + temp

        # write the results in the particles array
        markers[ip, :] = particle[:]

    #$ omp end parallel

#@stack_array('df', 'dfinv', 'g', 'g_inv', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'v', 'k')
def push_gc1_explicit_stage(markers: 'float[:,:]', dt: float, stage: int,
                           pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                           starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                           kind_map: int, params_map: 'float[:]',
                           p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                           ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                           cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                           epsilon: float,
                           b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                           norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                           norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                           curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                           grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]',
                           a: 'float[:]', b: 'float[:]', c: 'float[:]'):
    r'''Single stage of a s-stage Runge-Kutta solve of 

    .. math::

        \dot{\mathbf H}_p = \frac{\epsilon \mu_p}{|B^*_{p,\parallel}|}  G_p^{-1} \mathbb{b}_{p,0, \otimes}G_p^{-1} \hat \nabla |\hat B^0_{p,0}| \,,

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant.
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    g = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    grad_abs_b = empty(3, dtype=float)
    temp1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
    temp2 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)

    # marker position e
    e = empty(3, dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]
    
    # get number of stages
    n_stages = shape(b)[0]
    
    if stage == n_stages - 1:
        last = 1.
    else:
        last = 0.

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        v = markers[ip, 3]
        mu = markers[ip, 4]

        # evaluate Jacobian, result in df
        map_eval.df(e[0], e[1], e[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # evaluate inverse of G
        linalg.transpose(df, df_t)
        linalg.matrix_matrix(df_t, df, g)
        linalg.matrix_inv(g, g_inv)

        # metric coeffs
        det_df = linalg.det(df)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # grad_abs_b; 1form
        grad_abs_b[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # norm_b1; 1form
        norm_b1[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # norm_b2; 2form
        norm_b2[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts2[0])
        norm_b2[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts2[1])
        norm_b2[2] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts2[2])

        # b_star; 2form
        b_star[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1 + epsilon*v*curl_norm_b1, starts2[0])
        b_star[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2 + epsilon*v*curl_norm_b2, starts2[1])
        b_star[2] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3 + epsilon*v*curl_norm_b3, starts2[2])

        # transform to H1vec
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # calculate norm_b X grad_abs_b
        linalg.matrix_vector(g_inv, grad_abs_b, temp1)

        linalg.cross(norm_b2, temp1, temp2)

        linalg.matrix_vector(g_inv, temp2, temp1)

        # calculate k
        k[:] = epsilon*mu/abs_b_star_para*temp1
        
        # accumulation for last stage
        markers[ip, 13:16] += dt*b[stage]*k

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = markers[ip, 9:12] + dt*a[stage]*k + last*markers[ip, 13:16]

def push_gc2_explicit_stage(markers: 'float[:,:]', dt: float, stage: int,
                           pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                           starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                           kind_map: int, params_map: 'float[:]',
                           p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                           ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                           cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                           epsilon: float,
                           b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                           norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                           norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                           curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                           grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]',
                           a: 'float[:]', b: 'float[:]', c: 'float[:]'):
    r'''Single stage of a s-stage Runge-Kutta solve of 

    .. math::

        &\dot{\mathbf H}_p &= \frac{1}{|B^*_{p,\parallel}|} \left( \frac{1}{\sqrt{g}} \hat{\mathbf B}^{*2}_p \right) v_{p, \parallel} \,,
            
        &\dot v_{p, \parallel} &= -\frac{\mu}{|B^*_{p,\parallel}|}  \left( \frac{1}{\sqrt{g}} \hat{\mathbf B}^{*2}_p \right) \cdot \hat \nabla |\hat B^0_0|_p \,.

    for each marker :math:`p` in markers array.
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    grad_abs_b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)

    # marker position e
    e = empty(3, dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]
    
    # get number of stages
    n_stages = shape(b)[0]
    
    if stage == n_stages - 1:
        last = 1.
    else:
        last = 0.

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        v = markers[ip, 3]
        mu = markers[ip, 4]

        # evaluate Jacobian, result in df
        map_eval.df(e[0], e[1], e[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)
        
        # metric coeffs
        det_df = linalg.det(df)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # grad_abs_b; 1form
        grad_abs_b[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # norm_b1; 1form
        norm_b1[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # b_star; 2form
        b_star[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1 + epsilon*v*curl_norm_b1, starts2[0])
        b_star[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2 + epsilon*v*curl_norm_b2, starts2[1])
        b_star[2] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3 + epsilon*v*curl_norm_b3, starts2[2])

        # transform to H1vec
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # calculate k for X
        k[:] = b_star/abs_b_star_para*v

        # calculate k_v for v
        temp = linalg.scalar_dot(b_star, grad_abs_b)
        k_v = -1*mu/abs_b_star_para*temp
        
        # accumulation for last stage
        markers[ip, 13:16] += dt*b[stage]*k
        markers[ip, 16] += dt*b[stage]*k_v

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = markers[ip, 9:12] + dt*a[stage]*k + last*markers[ip, 13:16]
        markers[ip, 3] = markers[ip, 12] + dt*a[stage]*k_v + last*markers[ip, 16]

def push_gc_explicit_stage(markers: 'float[:,:]', dt: float, stage: int,
                           pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                           starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                           kind_map: int, params_map: 'float[:]',
                           p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                           ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                           cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                           epsilon: float,
                           b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                           norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                           norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                           curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                           grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]',
                           a: 'float[:]', b: 'float[:]', c: 'float[:]'):
    r'''Single stage of a s-stage Runge-Kutta solve of 

    .. math::

        &\dot{\mathbf H}_p = \frac{\epsilon \mu_p}{|B^*_{p,\parallel}|}  G_p^{-1} \mathbb{b}_{p,0, \otimes}G_p^{-1} \hat \nabla |\hat B^0_{p,0}| + \frac{1}{|B^*_{p,\parallel}|} \left( \frac{1}{\sqrt{g}} \hat{\mathbf B}^{*2}_p \right) v_{p, \parallel}\,,

        &\dot v_{p, \parallel} &= -\frac{\mu}{|B^*_{p,\parallel}|}  \left( \frac{1}{\sqrt{g}} \hat{\mathbf B}^{*2}_p \right) \cdot \hat \nabla |\hat B^0_0|_p \,.
        
    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant.
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    g = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    grad_abs_b = empty(3, dtype=float)
    temp1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
    temp2 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    temp3 = empty(3, dtype=float)

    # marker position e
    e = empty(3, dtype=float)

    # intermediate k-vector
    k = empty(3, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]
    
    # get number of stages
    n_stages = shape(b)[0]
    
    if stage == n_stages - 1:
        last = 1.
    else:
        last = 0.

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        v = markers[ip, 3]
        mu = markers[ip, 4]

        # evaluate Jacobian, result in df
        map_eval.df(e[0], e[1], e[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # evaluate inverse of G
        linalg.transpose(df, df_t)
        linalg.matrix_matrix(df_t, df, g)
        linalg.matrix_inv(g, g_inv)

        # metric coeffs
        det_df = linalg.det(df)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # grad_abs_b; 1form
        grad_abs_b[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # norm_b1; 1form
        norm_b1[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # norm_b2; 2form
        norm_b2[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts2[0])
        norm_b2[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts2[1])
        norm_b2[2] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts2[2])

        # b_star; 2form
        b_star[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1 + epsilon*v*curl_norm_b1, starts2[0])
        b_star[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2 + epsilon*v*curl_norm_b2, starts2[1])
        b_star[2] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3 + epsilon*v*curl_norm_b3, starts2[2])

        # transform to H1vec
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # calculate norm_b X grad_abs_b
        linalg.matrix_vector(g_inv, grad_abs_b, temp1)

        linalg.cross(norm_b2, temp1, temp2)

        linalg.matrix_vector(g_inv, temp2, temp3)

        # calculate k
        k[:] = (epsilon*mu*temp3 + b_star*v)/abs_b_star_para

        # calculate k_v for v
        temp = linalg.scalar_dot(b_star, grad_abs_b)

        k_v = -1*mu/abs_b_star_para*temp
        
        # accumulation for last stage
        markers[ip, 13:16] += dt*b[stage]*k
        markers[ip, 16] += dt*b[stage]*k_v

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = markers[ip, 9:12] + dt*a[stage]*k + last*markers[ip, 13:16]
        markers[ip, 3] = markers[ip, 12] + dt*a[stage]*k_v + last*markers[ip, 16]


def push_gc1_discrete_gradients_stage(markers: 'float[:,:]', dt: float, stage: int, tol: float,
                                      domain_array: 'float[:]',
                                      pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                      starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                                      kind_map: int, params_map: 'float[:]',
                                      p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                      ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                      cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                      epsilon: float,
                                      abs_b: 'float[:,:,:]',
                                      b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                                      norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                                      norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                                      curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                                      grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]'):
    r'''Single stage of the fixed-point iteration for the discrete gradient method

    .. math::

        {\mathbf H}^k_{n+1} = {\mathbf H}_n + dt*S1({\mathbf H}_n)*\bar{\nabla} I_1 ({\mathbf H}_n, {\mathbf H}^{k-1}_{n+1})

    where

    ..math::

        \bar{\nabla} I_1 ({\mathbf H}_n, {\mathbf H}_{n+1}) = \mu \nabla |\hat B^0_0({\mathbf H}_{n+1/2})| + ({\mathbf H}_{n+1} + {\mathbf H}_{n}) \frac{\mu |\hat B^0_0({\mathbf H}_{n+1})| - \mu |\hat B^0_0({\mathbf H}_n)| - ({\mathbf H}_{n+1} - {\mathbf H}_n)\cdot \mu \nabla |\hat B^0_0({\mathbf H}_{n+1/2})|}{||{\mathbf H}_{n+1} - {\mathbf H}_n||^2}

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant.
    '''
    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    g = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    temp = empty(3, dtype=float)
    S = empty((3, 3), dtype=float)
    grad_abs_b = empty(3, dtype=float)
    grad_I = empty(3, dtype=float)
    bcross = empty((3, 3), dtype=float)
    temp1 = empty((3, 3), dtype=float)
    norm_b2 = empty(3, dtype=float)
    temp2 = empty((3, 3), dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)

    # marker position e
    e = empty(3, dtype=float)
    e_mid = empty(3, dtype=float)
    e_diff = empty(3, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 21] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_mid[:] = (e[:] + markers[ip, 9:12])/2.
        e_diff[:] = e[:] - markers[ip, 9:12]
        v = markers[ip, 3]
        mu = markers[ip, 4]

        # if the particle just came from the other process (then mid-point eval has not been performed yet)
        if markers[ip, 20] == -1.:

            # evaluate Jacobian, result in df
            map_eval.df(e_mid[0], e_mid[1], e_mid[2],
                        kind_map, params_map,
                        t1_map, t2_map, t3_map, p_map,
                        ind1_map, ind2_map, ind3_map,
                        cx, cy, cz,
                        df)
    
            # evaluate inverse of G
            linalg.transpose(df, df_t)
            linalg.matrix_matrix(df_t, df, g)
            linalg.matrix_inv(g, g_inv)
    
            # metric coeffs
            det_df = linalg.det(df)
    
            # spline evaluation
            span1 = bsp.find_span(tn1, pn[0], e_mid[0])
            span2 = bsp.find_span(tn2, pn[1], e_mid[1])
            span3 = bsp.find_span(tn3, pn[2], e_mid[2])
    
            bsp.b_d_splines_slim(tn1, pn[0], e_mid[0], span1, bn1, bd1)
            bsp.b_d_splines_slim(tn2, pn[1], e_mid[1], span2, bn2, bd2)
            bsp.b_d_splines_slim(tn3, pn[2], e_mid[2], span3, bn3, bd3)
    
            # eval all the needed field
            # grad_abs_b; 1form
            grad_abs_b[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
            grad_abs_b[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
            grad_abs_b[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])
    
            # norm_b1; 1form
            norm_b1[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
            norm_b1[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
            norm_b1[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])
    
            # norm_b2; 2form
            norm_b2[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts2[0])
            norm_b2[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts2[1])
            norm_b2[2] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts2[2])
    
            # b_star; 2form
            b_star[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1 + epsilon*v*curl_norm_b1, starts2[0])
            b_star[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2 + epsilon*v*curl_norm_b2, starts2[1])
            b_star[2] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3 + epsilon*v*curl_norm_b3, starts2[2])
    
            # transform to H1vec
            b_star[:] = b_star/det_df
    
            # calculate abs_b_star_para
            abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)
    
            # assemble b cross (.) as 3x3 matrix
            bcross[:,:] = ( (     0.     ,  -norm_b2[2],   norm_b2[1]), 
                            (  norm_b2[2],      0.     ,  -norm_b2[0]), 
                            ( -norm_b2[1],   norm_b2[0],      0.     ) )
    
            # calculate G-1 b cross G-1 
            linalg.matrix_matrix(bcross, g_inv, temp1)
            linalg.matrix_matrix(g_inv, temp1, temp2)
    
            # calculate S
            S = (epsilon*temp2)/abs_b_star_para

            # save at the markers
            markers[ip, 13:16] = S[0,:]
            markers[ip, 16:18] = S[1,1:3]
            markers[ip, 18]    = S[2,2]
            markers[ip, 20:23] = mu*grad_abs_b[:]

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts0)

        # assemble S
        S[:,:] = ( ( markers[ip, 13],  markers[ip, 14], markers[ip, 15]), 
                   (-markers[ip, 14],  markers[ip, 16], markers[ip, 17]), 
                   (-markers[ip, 15], -markers[ip, 17], markers[ip, 18]) )

        # calculate grad_I
        temp_scalar = linalg.scalar_dot(e_diff[:], markers[ip,20:23])
        temp_scalar2 = e_diff[0]**2 + e_diff[1]**2 + e_diff[2]**2 

        grad_I[:] = markers[ip,20:23] + e_diff[:]*(abs_b0*mu - markers[ip,19] - temp_scalar)/temp_scalar2
        
        linalg.matrix_vector(S, grad_I, temp)
        
        markers[ip, 0:3] = markers[ip, 9:12] + dt*temp[:]

        diff = sqrt((e[0] - markers[ip, 0])**2 + (e[1] - markers[ip, 1])**2 + (e[2] - markers[ip, 2])**2)

        if diff < tol:
            markers[ip, 21] = -1.
            markers[ip, 22] = stage

            continue

        # mid-point eval for the next iteration
        e_mid[:] = (markers[ip, 0:3] + markers[ip, 9:12])/2

        # check whether the mid position is already outside of the proc domain
        if (e_mid[0] < domain_array[0]) or (e_mid[0] > domain_array[1]):
            markers[ip,20] = -1.
            continue
        if (e_mid[1] < domain_array[3]) or (e_mid[1] > domain_array[4]):
            markers[ip,20] = -1.
            continue
        if (e_mid[2] < domain_array[6]) or (e_mid[2] > domain_array[7]):
            markers[ip,20] = -1.
            continue

        # evaluate Jacobian, result in df
        map_eval.df(e_mid[0], e_mid[1], e_mid[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # evaluate inverse of G
        linalg.transpose(df, df_t)
        linalg.matrix_matrix(df_t, df, g)
        linalg.matrix_inv(g, g_inv)

        # metric coeffs
        det_df = linalg.det(df)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e_mid[0])
        span2 = bsp.find_span(tn2, pn[1], e_mid[1])
        span3 = bsp.find_span(tn3, pn[2], e_mid[2])

        bsp.b_d_splines_slim(tn1, pn[0], e_mid[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e_mid[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e_mid[2], span3, bn3, bd3)

        # eval all the needed field
        # grad_abs_b; 1form
        grad_abs_b[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # norm_b1; 1form
        norm_b1[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # norm_b2; 2form
        norm_b2[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts2[0])
        norm_b2[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts2[1])
        norm_b2[2] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts2[2])

        # b_star; 2form
        b_star[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1 + epsilon*v*curl_norm_b1, starts2[0])
        b_star[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2 + epsilon*v*curl_norm_b2, starts2[1])
        b_star[2] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3 + epsilon*v*curl_norm_b3, starts2[2])

        # transform to H1vec
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # assemble b cross (.) as 3x3 matrix
        bcross[:,:] = ( (     0.     ,  -norm_b2[2],   norm_b2[1]), 
                        (  norm_b2[2],      0.     ,  -norm_b2[0]), 
                        ( -norm_b2[1],   norm_b2[0],      0.     ) )

        # calculate G-1 b cross G-1 
        linalg.matrix_matrix(bcross, g_inv, temp1)
        linalg.matrix_matrix(g_inv, temp1, temp2)

        # calculate S
        S[:,:] = (epsilon*temp2)/abs_b_star_para

        # save at the markers
        markers[ip, 13:16] = S[0,:]
        markers[ip, 16:18] = S[1,1:3]
        markers[ip, 18]    = S[2,2]
        markers[ip, 20:23] = mu*grad_abs_b[:]


def push_gc2_discrete_gradients_stage(markers: 'float[:,:]', dt: float, stage: int, tol: float,
                                      domain_array: 'float[:]',
                                      pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                      starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                                      kind_map: int, params_map: 'float[:]',
                                      p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                      ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                      cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                      epsilon: float,
                                      abs_b: 'float[:,:,:]',
                                      b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                                      norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                                      norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                                      curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                                      grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]'):
    r'''Single stage of the fixed-point iteration for the discrete gradient method

    .. math::

        {\mathbf z}^k_{n+1} = {\mathbf z}_n + dt*S2({\mathbf z}_n)*\bar{\nabla} I_2 ({\mathbf z}_n, {\mathbf z}^{k-1}_{n+1})

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant.
    '''
    # allocate metric coeffs
    df = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    grad_abs_b = empty(3, dtype=float)
    grad_I = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)

    # marker position e
    e = empty(3, dtype=float)
    e_mid = empty(3, dtype=float)
    e_diff = empty(3, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 21] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_mid[:] = (e[:] + markers[ip, 9:12])/2.
        e_diff[:] = e[:] - markers[ip, 9:12]
        v = markers[ip, 3]
        v_old = markers[ip,12]
        v_mid = (markers[ip, 3] + markers[ip, 12])/2.
        mu = markers[ip, 4]

        # if the particle just came from the other process (then mid-point eval has not been performed yet)
        if markers[ip, 20] == -1.:

            # evaluate Jacobian, result in df
            map_eval.df(e_mid[0], e_mid[1], e_mid[2],
                        kind_map, params_map,
                        t1_map, t2_map, t3_map, p_map,
                        ind1_map, ind2_map, ind3_map,
                        cx, cy, cz,
                        df)

            # metric coeffs
            det_df = linalg.det(df)

            # spline evaluation
            span1 = bsp.find_span(tn1, pn[0], e_mid[0])
            span2 = bsp.find_span(tn2, pn[1], e_mid[1])
            span3 = bsp.find_span(tn3, pn[2], e_mid[2])

            bsp.b_d_splines_slim(tn1, pn[0], e_mid[0], span1, bn1, bd1)
            bsp.b_d_splines_slim(tn2, pn[1], e_mid[1], span2, bn2, bd2)
            bsp.b_d_splines_slim(tn3, pn[2], e_mid[2], span3, bn3, bd3)

            # eval all the needed field
            # grad_abs_b; 1form
            grad_abs_b[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
            grad_abs_b[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
            grad_abs_b[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

            # norm_b1; 1form
            norm_b1[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
            norm_b1[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
            norm_b1[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

            # b_star; 2form
            b_star[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1 + epsilon*v_mid*curl_norm_b1, starts2[0])
            b_star[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2 + epsilon*v_mid*curl_norm_b2, starts2[1])
            b_star[2] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3 + epsilon*v_mid*curl_norm_b3, starts2[2])

            # transform to H1vec
            b_star[:] = b_star/det_df

            # calculate abs_b_star_para
            abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

            # save at the markers
            markers[ip, 13:16] = b_star[:]/abs_b_star_para
            markers[ip, 20:23] = mu*grad_abs_b[:]

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts0)

        # calculate grad_I
        temp_scalar = linalg.scalar_dot(e_diff[:], markers[ip,20:23])
        temp_scalar2 = e_diff[0]**2 + e_diff[1]**2 + e_diff[2]**2 + (v - v_old)**2

        grad_I[:] = markers[ip,20:23] + e_diff*(abs_b0*mu - markers[ip,19] - temp_scalar)/temp_scalar2
        grad_Iv = v_mid + (v - v_old)*(abs_b0*mu - markers[ip,19] - temp_scalar)/temp_scalar2

        temp_scalar3 = linalg.scalar_dot(markers[ip, 13:16], grad_I)
        
        markers[ip, 0:3] = markers[ip, 9:12] + dt*markers[ip, 13:16]*grad_Iv
        markers[ip, 3] = markers[ip,12] - dt*temp_scalar3

        diff = sqrt((e[0] - markers[ip, 0])**2 + (e[1] - markers[ip, 1])**2 + (e[2] - markers[ip, 2])**2)
        vdiff = sqrt((v - markers[ip, 3])**2)

        if diff < tol and vdiff < tol:
            markers[ip, 21] = -1.
            markers[ip, 22] = stage
            continue

        # mid-point eval for the next iteration
        e_mid[:] = (markers[ip, 0:3] + markers[ip, 9:12])/2
        v_mid = (markers[ip, 3] + markers[ip, 12])/2

        # check whether the mid position is already outside of the proc domain
        if (e_mid[0] < domain_array[0]) or (e_mid[0] > domain_array[1]):
            markers[ip,20] = -1.
            continue
        if (e_mid[1] < domain_array[3]) or (e_mid[1] > domain_array[4]):
            markers[ip,20] = -1.
            continue
        if (e_mid[2] < domain_array[6]) or (e_mid[2] > domain_array[7]):
            markers[ip,20] = -1.
            continue

        # evaluate Jacobian, result in df
        map_eval.df(e_mid[0], e_mid[1], e_mid[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e_mid[0])
        span2 = bsp.find_span(tn2, pn[1], e_mid[1])
        span3 = bsp.find_span(tn3, pn[2], e_mid[2])

        bsp.b_d_splines_slim(tn1, pn[0], e_mid[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e_mid[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e_mid[2], span3, bn3, bd3)

        # eval all the needed field
        # grad_abs_b; 1form
        grad_abs_b[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # norm_b1; 1form
        norm_b1[0] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # b_star; 2form
        b_star[0] = eval_3d.eval_spline_mpi_kernel(pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1 + epsilon*v_mid*curl_norm_b1, starts2[0])
        b_star[1] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2 + epsilon*v_mid*curl_norm_b2, starts2[1])
        b_star[2] = eval_3d.eval_spline_mpi_kernel(pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3 + epsilon*v_mid*curl_norm_b3, starts2[2])

        # transform to H1vec
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # save at the markers
        markers[ip, 13:16] = b_star[:]/abs_b_star_para
        markers[ip, 20:23] = mu*grad_abs_b[:]