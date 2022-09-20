import struphy.linear_algebra.core as linalg
import struphy.geometry.map_eval as map_eval
import struphy.feec.bsplines_kernels as bsp
import struphy.feec.basics.spline_evaluation_3d as eval_3d

from numpy import zeros, empty, shape, sqrt, cos, sin


def push_v_with_efield(markers: 'float[:,:]', dt: 'float',
                       pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                       starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                       kind_map: 'int', params_map: 'float[:]',
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
        e_form[0] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, e1_1, starts1[0], pn)
        e_form[1] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, e1_2, starts1[1], pn)
        e_form[2] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, e1_3, starts1[2], pn)

        # electric field: Cartesian components
        linalg.matrix_vector(dfinv_t, e_form, e_cart)

        # update velocities
        markers[ip, 3:6] += dt*e_cart

    #$ omp end parallel

    

def push_vxb_analytic(markers: 'float[:,:]', dt: 'float',
                      pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                      starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                      kind_map: 'int', params_map: 'float[:]',
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

    # particle velocity (Cartesian, perpendicular, v x b_norm, b_norm x vperp)
    v = empty(3, dtype=float)
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

    #$ omp parallel private (ip, eta1, eta2, eta3, df, det_df, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b_form, b_cart, b_abs, b_norm, v, vpar, vxb_norm, vperp, b_normxvperp)
    #$ omp for
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
        det_df = linalg.det(df)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # magnetic field: 2-form components
        b_form[0] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts2[0], pn)
        b_form[1] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2[1], pn)
        b_form[2] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts2[2], pn)

        # magnetic field: Cartesian components
        linalg.matrix_vector(df, b_form, b_cart)
        b_cart[:] = b_cart/det_df

        # normalized magnetic field direction
        b_abs = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)
        b_norm[:] = b_cart/b_abs

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


    
def push_bxu_Hdiv(markers: 'float[:,:]', dt: 'float',
                  pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                  starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                  kind_map: 'int', params_map: 'float[:]',
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
        b_form[0] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts2[0], pn)
        b_form[1] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2[1], pn)
        b_form[2] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts2[2], pn)
        
        # magnetic field: Cartesian components
        linalg.matrix_vector(df, b_form, b_cart)
        b_cart[:] = b_cart/det_df

        # velocity field: 2-form components
        u_form[0] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u2_1, starts2[0], pn)
        u_form[1] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u2_2, starts2[1], pn)
        u_form[2] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u2_3, starts2[2], pn)
        
        linalg.matrix_vector(df, u_form, u_cart)
        u_cart[:] = u_cart/det_df

        # electric field E = B x U
        linalg.cross(b_cart, u_cart, e_cart)

        # update velocities
        markers[ip, 3:6] += dt*e_cart

    #$ omp end parallel

    

def push_bxu_Hcurl(markers: 'float[:,:]', dt: 'float',
                   pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                   starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                   kind_map: 'int', params_map: 'float[:]',
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
        b_form[0] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts2[0], pn)
        b_form[1] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2[1], pn)
        b_form[2] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts2[2], pn)
        
        # magnetic field: Cartesian components
        linalg.matrix_vector(df, b_form, b_cart)
        b_cart[:] = b_cart/det_df

        # velocity field: 1-form components
        u_form[0] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u1_1, starts1[0], pn)
        u_form[1] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u1_2, starts1[1], pn)
        u_form[2] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, u1_3, starts1[2], pn)

        # velocity field: Cartesian components
        linalg.matrix_vector(dfinv_t, u_form, u_cart)
        
        # electric field E = B x U
        linalg.cross(b_cart, u_cart, e_cart)

        # update velocities
        markers[ip, 3:6] += dt*e_cart

    #$ omp end parallel

    

def push_bxu_H1vec(markers: 'float[:,:]', dt: 'float',
                   pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                   starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                   kind_map: 'int', params_map: 'float[:]',
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
        b_form[0] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts2[0], pn)
        b_form[1] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2[1], pn)
        b_form[2] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts2[2], pn)
        
        # magnetic field: Cartesian components
        linalg.matrix_vector(df, b_form, b_cart)
        b_cart[:] = b_cart/det_df

        # velocity field: vector field components
        u_form[0] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, uv_1, starts0, pn)
        u_form[1] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, uv_2, starts0, pn)
        u_form[2] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, uv_3, starts0, pn)
        
        # velocity field: Cartesian components
        linalg.matrix_vector(df, u_form, u_cart)

        # electric field E = B x U
        linalg.cross(b_cart, u_cart, e_cart)

        # update velocities
        markers[ip, 3:6] += dt*e_cart

    #$ omp end parallel

    
    
def push_bxu_Hdiv_pauli(markers: 'float[:,:]', dt: 'float',
                        pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                        starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                        kind_map: 'int', params_map: 'float[:]',
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
        b_form[0] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts2[0], pn)
        b_form[1] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2[1], pn)
        b_form[2] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts2[2], pn)
        
        # magnetic field: Cartesian components
        linalg.matrix_vector(df, b_form, b_cart)
        b_cart[:] = b_cart/det_df
        
        # magnetic field: evaluation of gradient (1-form)
        b_diff[0] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1], pn[2], der1, bn2, bn3, span1, span2, span3, b0, starts0, pn)
        b_diff[1] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1], pn[2], bn1, der2, bn3, span1, span2, span3, b0, starts0, pn)
        b_diff[2] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1], pn[2], bn1, bn2, der3, span1, span2, span3, b0, starts0, pn)
        
        # magnetic field: evaluation of gradient (Cartesian components)
        linalg.matrix_vector(dfinv_t, b_diff, b_grad)

        # velocity field: 2-form components
        u_form[0] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u2_1, starts2[0], pn)
        u_form[1] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u2_2, starts2[1], pn)
        u_form[2] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u2_3, starts2[2], pn)
        
        linalg.matrix_vector(df, u_form, u_cart)
        u_cart[:] = u_cart/det_df

        # electric field E = B x U
        linalg.cross(b_cart, u_cart, e_cart)
        
        # additional artificial electric field of Pauli markers
        e_cart[:] = e_cart - mu[ip]*b_grad

        # update velocities
        markers[ip, 3:6] += dt*e_cart

    #$ omp end parallel
    

    
def push_pc_Xu_full(markers: 'float[:,:]', n_markers: 'int',
                    pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                    starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',
                    kind_map: 'int', params_map: 'float[:]',
                    p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                    ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                    cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                    dt: 'float',
                    grad_Xu_11: 'float[:,:,:]', grad_Xu_12: 'float[:,:,:]', grad_Xu_13: 'float[:,:,:]',
                    grad_Xu_21: 'float[:,:,:]', grad_Xu_22: 'float[:,:,:]', grad_Xu_23: 'float[:,:,:]',
                    grad_Xu_31: 'float[:,:,:]', grad_Xu_32: 'float[:,:,:]', grad_Xu_33: 'float[:,:,:]'):
    r'''Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = - DF^{-\top} \left(  \boldsymbol \Lambda^1 \mathbb G \mathcal X(\mathbf u, \mathbf v)  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\mathbf u` 
    are the coefficients of the mhd velocity field (either 1-form or 2-form) and :math:`\mathcal X`
    is either the MHD operator :meth:`struphy.psydac_api.mhd_ops_pure_psydac.MHDOperators.assemble_X1` (if u is 1-form)
    or :meth:`struphy.psydac_api.mhd_ops_pure_psydac.MHDOperators.assemble_X2` (if u is 2-form).

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

    #$ omp parallel private(ip, eta1, eta2, eta3, v, df, dfinv, dfinv_t, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, e, e_cart)
    #$ omp for
    for ip in range(n_markers):

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]
        v[:] = markers[3:6, ip]

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
        e[0] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_Xu_11 * v[0] + grad_Xu_21 * v[1] + grad_Xu_31 * v[2], starts1, pn)
        e[1] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_Xu_12 * v[0] + grad_Xu_22 * v[1] + grad_Xu_32 * v[2], starts2, pn)
        e[2] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_Xu_13 * v[0] + grad_Xu_23 * v[1] + grad_Xu_33 * v[2], starts3, pn)

        # electric field in Cartesian coordinates
        linalg.matrix_vector(dfinv_t, e, e_cart)

        # update velocities
        markers[3:6, ip] -= dt*e_cart

    #$ omp end parallel


def push_pc_Xu_perp(markers: 'float[:,:]', n_markers: 'int',
                    pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                    starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',
                    kind_map: 'int', params_map: 'float[:]',
                    p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                    ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                    cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                        dt: 'float',
                        grad_Xu_11: 'float[:,:,:]', grad_Xu_12: 'float[:,:,:]', grad_Xu_13: 'float[:,:,:]',
                        grad_Xu_21: 'float[:,:,:]', grad_Xu_22: 'float[:,:,:]', grad_Xu_23: 'float[:,:,:]',
                    grad_Xu_31: 'float[:,:,:]', grad_Xu_32: 'float[:,:,:]', grad_Xu_33: 'float[:,:,:]',
                    b2_1: 'float[:,:,:]', b2_2: 'float[:,:,:]', b2_3: 'float[:,:,:]'):
    r'''Updates

    .. math::

        \frac{\mathbf v^{n+1}_p - \mathbf v^n_p}{\Delta t} = - DF^{-\top} \left(  \boldsymbol \Lambda^1 \mathbb G \mathcal X(\mathbf u, \mathbf v_\perp)  \right)^n_p

    for each marker :math:`p` in markers array, where :math:`\mathbf u` 
    are the coefficients of the mhd velocity field (either 1-form or 2-form) and :math:`\mathcal X`
    is either the MHD operator :meth:`struphy.psydac_api.mhd_ops_pure_psydac.MHDOperators.assemble_X1` (if u is 1-form)
    or :meth:`struphy.psydac_api.mhd_ops_pure_psydac.MHDOperators.assemble_X2` (if u is 2-form).

    Parameters
    ----------
        grad_Xu_ij: array[float]
            3d array of FE coeffs of :math:`\nabla_j(\mathcal X \cdot \mathbf u)_i`. i,j=1,2,3.

        b2_1, b2_2, b2_3: array[float]
            3d array of FE coeffs of B-field as 2-form.
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)

    # allocate for field evaluations
    e = empty(3, dtype=float)
    e_cart = empty(3, dtype=float)
    b = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    b0 = empty(3, dtype=float)

    # particle velocity (cartesian, perpendicular, v x b0, b0 x vperp)
    v = empty(3, dtype=float)
    vperp = empty(3, dtype=float)
    vxb0 = empty(3, dtype=float)
    b0xvperp = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    #$ omp parallel private(ip, eta1, eta2, eta3, v, df, dfinv, dfinv_t, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b, b_cart, b0, vxb0, vperp, e, e_cart)
    #$ omp for
    for ip in range(n_markers):

        eta1 = markers[0, ip]
        eta2 = markers[1, ip]
        eta3 = markers[2, ip]
        v[:] = markers[3:6, ip]

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)
        linalg.matrix_inv(df, dfinv)
        linalg.transpose(dfinv, dfinv_t)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # B-field
        b[0] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts1, pn)
        b[1] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2, pn)
        b[2] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts3, pn)

        # push-forward to physical domain
        linalg.matrix_vector(df, b, b_cart)
        b_cart[:] = b_cart/det_df

        # normalized magnetic field direction
        b_norm = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)
        b0[:] = b_cart/b_norm

        # perpendicular velocity
        linalg.cross(v, b0, vxb0)
        linalg.cross(b0, vxb0, vperp)

        # Evaluate grad(X(u, v_perp)) at the particle positions
        e[0] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_Xu_11 * vperp[0] + grad_Xu_21 * vperp[1] + grad_Xu_31 * vperp[2], starts1, pn)
        e[1] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_Xu_12 * vperp[0] + grad_Xu_22 * vperp[1] + grad_Xu_32 * vperp[2], starts2, pn)
        e[2] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_Xu_13 * vperp[0] + grad_Xu_23 * vperp[1] + grad_Xu_33 * vperp[2], starts3, pn)

        # electric field in Cartesian coordinates
        linalg.matrix_vector(dfinv_t, e, e_cart)

        # update velocities
        markers[3:6, ip] -= dt*e_cart

    #$ omp end parallel


def push_eta_rk4(markers: 'float[:,:]', dt: 'float',
                 pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                 starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                 kind_map: 'int', params_map: 'float[:]',
                 p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                 ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                 cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]'):
    r'''Fourth order Runge-Kutta solve of 

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v

    for each marker :math:`p` in markers array, where :math:`\mathbf v` is constant.
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)

    # marker position and velocity
    eta = empty(3, dtype=float)
    v = empty(3, dtype=float)

    # intermediate steps in RK4
    k1 = empty(3, dtype=float)
    k2 = empty(3, dtype=float)
    k3 = empty(3, dtype=float)
    k4 = empty(3, dtype=float)
    
    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, eta, v, eta1, eta2, eta3, df, dfinv, k1, k2, k3, k4)
    #$ omp for
    for ip in range(n_markers):
        
        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, :3]
        v[:] = markers[ip, 3:6]

        # ----------------- step 1 in Runge-Kutta method -------------------
        eta1 = eta[0]
        eta2 = eta[1]
        eta3 = eta[2]

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # evaluate inverse Jacobian matrix
        linalg.matrix_inv(df, dfinv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k1)

        # ----------------- step 2 in Runge-Kutta method -------------------
        eta1 = (eta[0] + dt*k1[0]/2) % 1.
        eta2 = (eta[1] + dt*k1[1]/2) % 1.
        eta3 = (eta[2] + dt*k1[2]/2) % 1.

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # evaluate inverse Jacobian matrix
        linalg.matrix_inv(df, dfinv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k2)

        # ------------------ step 3 in Runge-Kutta method ------------------
        eta1 = (eta[0] + dt*k2[0]/2) % 1.
        eta2 = (eta[1] + dt*k2[1]/2) % 1.
        eta3 = (eta[2] + dt*k2[2]/2) % 1.

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # evaluate inverse Jacobian matrix
        linalg.matrix_inv(df, dfinv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k3)

        # ------------------ step 4 in Runge-Kutta method ------------------
        eta1 = (eta[0] + dt*k3[0]) % 1.
        eta2 = (eta[1] + dt*k3[1]) % 1.
        eta3 = (eta[2] + dt*k3[2]) % 1.

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # evaluate inverse Jacobian matrix
        linalg.matrix_inv(df, dfinv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k4)

        #  ---------------- update logical coordinates ---------------------
        markers[ip, :3] = (eta + dt*(k1 + 2*k2 + 2*k3 + k4)/6) % 1.0

    #$ omp end parallel


def push_pc_eta_rk4_full(markers: 'float[:,:]', n_markers: 'int',
                         pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                         starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',
                         kind_map: 'int', params_map: 'float[:]',
                         p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                         ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                         cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                         dt: 'float',
                         u_1: 'float[:,:,:]', u_2: 'float[:,:,:]', u_3: 'float[:,:,:]',
                         u_basis: 'int',):
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

    # intermediate steps in RK4
    k1 = empty(3, dtype=float)
    k2 = empty(3, dtype=float)
    k3 = empty(3, dtype=float)
    k4 = empty(3, dtype=float)
    k1_v = empty(3, dtype=float)
    k2_v = empty(3, dtype=float)
    k3_v = empty(3, dtype=float)
    k4_v = empty(3, dtype=float)
    k1_u = empty(3, dtype=float)
    k2_u = empty(3, dtype=float)
    k3_u = empty(3, dtype=float)
    k4_u = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    #$ omp parallel private(ip, eta, v, eta1, eta2, eta3, df, dfinv, dfinv_t, ginv, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, u, k1, k2, k3, k4, k1_v, k2_v, k3_v, k4_v, k1_u, k2_u, k3_u, k4_u)
    #$ omp for
    for ip in range(n_markers):

        eta[:] = markers[:3, ip]
        v[:] = markers[3:6, ip]

        # ----------------- step 1 in Runge-Kutta method -------------------
        eta1 = eta[0]
        eta2 = eta[1]
        eta3 = eta[2]

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
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
        linalg.matrix_vector(dfinv, v, k1_v)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # U-field
        if u_basis == 1:
            u[0] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u_1, starts1, pn)
            u[1] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u_2, starts2, pn)
            u[2] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, u_3, starts3, pn)

            # transform to vector field
            linalg.matrix_vector(ginv, u, k1_u)

        elif u_basis == 2:
            u[0] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u_1, starts1, pn)
            u[1] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u_2, starts2, pn)
            u[2] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u_3, starts3, pn)

            # transform to vector field
            k1_u[:] = u/det_df

        # sum contribs
        k1[:] = k1_v + k1_u

        # ----------------- step 2 in Runge-Kutta method -------------------
        eta1 = (eta[0] + dt*k1[0]/2) % 1.
        eta2 = (eta[1] + dt*k1[1]/2) % 1.
        eta3 = (eta[2] + dt*k1[2]/2) % 1.

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
        linalg.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k2_v)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # U-field
        if u_basis == 1:
            u[0] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u_1, starts1, pn)
            u[1] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u_2, starts2, pn)
            u[2] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, u_3, starts3, pn)

            # transform to vector field
            linalg.matrix_vector(ginv, u, k2_u)

        elif u_basis == 2:
            u[0] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u_1, starts1, pn)
            u[1] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u_2, starts2, pn)
            u[2] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u_3, starts3, pn)

            # transform to vector field
            k2_u[:] = u/det_df

        # sum contribs
        k2[:] = k2_v + k2_u

        # ------------------ step 3 in Runge-Kutta method ------------------
        eta1 = (eta[0] + dt*k2[0]/2) % 1.
        eta2 = (eta[1] + dt*k2[1]/2) % 1.
        eta3 = (eta[2] + dt*k2[2]/2) % 1.

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
        linalg.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k3_v)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # U-field
        if u_basis == 1:
            u[0] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u_1, starts1, pn)
            u[1] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u_2, starts2, pn)
            u[2] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, u_3, starts3, pn)

            # transform to vector field
            linalg.matrix_vector(ginv, u, k3_u)

        elif u_basis == 2:
            u[0] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u_1, starts1, pn)
            u[1] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u_2, starts2, pn)
            u[2] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u_3, starts3, pn)

            # transform to vector field
            k3_u[:] = u/det_df

        # sum contribs
        k3[:] = k3_v + k3_u

        # ------------------ step 4 in Runge-Kutta method ------------------
        eta1 = (eta[0] + dt*k3[0]) % 1.
        eta2 = (eta[1] + dt*k3[1]) % 1.
        eta3 = (eta[2] + dt*k3[2]) % 1.

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
        linalg.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k4_v)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # U-field
        if u_basis == 1:
            u[0] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u_1, starts1, pn)
            u[1] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u_2, starts2, pn)
            u[2] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, u_3, starts3, pn)

            # transform to vector field
            linalg.matrix_vector(ginv, u, k4_u)

        elif u_basis == 2:
            u[0] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u_1, starts1, pn)
            u[1] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u_2, starts2, pn)
            u[2] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u_3, starts3, pn)

            # transform to vector field
            k4_u[:] = u/det_df

        # sum contribs
        k4[:] = k4_v + k4_u

        #  ---------------- update logical coordinates ---------------------
        markers[:3, ip] = (eta + dt*(k1 + 2*k2 + 2*k3 + k4)/6) % 1.0

    #$ omp end parallel


def push_pc_eta_rk4_perp(markers: 'float[:,:]', n_markers: 'int',
                         pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                         starts1: 'int[:]', starts2: 'int[:]', starts3: 'int[:]',
                         kind_map: 'int', params_map: 'float[:]',
                         p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                         ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                         cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                         dt: 'float',
                         u_1: 'float[:,:,:]', u_2: 'float[:,:,:]', u_3: 'float[:,:,:]',
                         u_basis: 'int',
                         b2_1: 'float[:,:,:]', b2_2: 'float[:,:,:]', b2_3: 'float[:,:,:]'):
    r'''Fourth order Runge-Kutta solve of 

    .. math::

        \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \mathbf v + \textnormal{vec}( \hat{\mathbf U}^{1(2)})_\perp

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
    u_cart = empty(3, dtype=float)
    b = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    b0 = empty(3, dtype=float)
    uperp = empty(3, dtype=float)
    uxb0 = empty(3, dtype=float)

    # intermediate steps in RK4
    k1 = empty(3, dtype=float)
    k2 = empty(3, dtype=float)
    k3 = empty(3, dtype=float)
    k4 = empty(3, dtype=float)
    k1_v = empty(3, dtype=float)
    k2_v = empty(3, dtype=float)
    k3_v = empty(3, dtype=float)
    k4_v = empty(3, dtype=float)
    k1_u = empty(3, dtype=float)
    k2_u = empty(3, dtype=float)
    k3_u = empty(3, dtype=float)
    k4_u = empty(3, dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    #$ omp parallel private(ip, eta, v, eta1, eta2, eta3, df, dfinv, dfinv_t, ginv, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, u, k1, k2, k3, k4, k1_v, k2_v, k3_v, k4_v, k1_u, k2_u, k3_u, k4_u)
    #$ omp for
    for ip in range(n_markers):

        eta[:] = markers[:3, ip]
        v[:] = markers[3:6, ip]

        # ----------------- step 1 in Runge-Kutta method -------------------
        eta1 = eta[0]
        eta2 = eta[1]
        eta3 = eta[2]

        # evaluate Jacobian, result in df
        map_eval.df(eta1, eta2, eta3,
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
        linalg.matrix_vector(dfinv, v, k1_v)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # U-field
        if u_basis == 1:
            u[0] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u_1, starts1, pn)
            u[1] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u_2, starts2, pn)
            u[2] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, u_3, starts3, pn)

            # push-forward
            linalg.matrix_vector(dfinv_t, u, u_cart)

        elif u_basis == 2:
            u[0] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u_1, starts1, pn)
            u[1] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u_2, starts2, pn)
            u[2] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u_3, starts3, pn)

            # push-forward
            linalg.matrix_vector(df, u, u_cart)
            u_cart[:] = u_cart/det_df

        # B-field (2-form)
        b[0] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts1, pn)
        b[1] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2, pn)
        b[2] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts3, pn)

        # push-forward
        linalg.matrix_vector(df, b, b_cart)
        b_cart[:] = b_cart/det_df

        # normalized magnetic field direction
        b_norm = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)
        b0[:] = b_cart/b_norm

        # perpendicular momentum
        linalg.cross(u_cart, b0, uxb0)
        linalg.cross(b0, uxb0, uperp)

        # pullback as vector field
        linalg.matrix_vector(dfinv, uperp, k1_u)

        # sum contribs
        k1[:] = k1_v + k1_u

        # ----------------- step 2 in Runge-Kutta method -------------------
        eta1 = (eta[0] + dt*k1[0]/2) % 1.
        eta2 = (eta[1] + dt*k1[1]/2) % 1.
        eta3 = (eta[2] + dt*k1[2]/2) % 1.

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
        linalg.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k2_v)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # U-field
        if u_basis == 1:
            u[0] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u_1, starts1, pn)
            u[1] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u_2, starts2, pn)
            u[2] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, u_3, starts3, pn)

            # push-forward
            linalg.matrix_vector(dfinv_t, u, u_cart)

        elif u_basis == 2:
            u[0] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u_1, starts1, pn)
            u[1] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u_2, starts2, pn)
            u[2] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u_3, starts3, pn)

            # push-forward
            linalg.matrix_vector(df, u, u_cart)
            u_cart[:] = u_cart/det_df

        # B-field (2-form)
        b[0] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts1, pn)
        b[1] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2, pn)
        b[2] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts3, pn)

        # push-forward
        linalg.matrix_vector(df, b, b_cart)
        b_cart[:] = b_cart/det_df

        # normalized magnetic field direction
        b_norm = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)
        b0[:] = b_cart/b_norm

        # perpendicular momentum
        linalg.cross(u_cart, b0, uxb0)
        linalg.cross(b0, uxb0, uperp)

        # pullback as vector field
        linalg.matrix_vector(dfinv, uperp, k2_u)

        # sum contribs
        k2[:] = k2_v + k2_u

        # ------------------ step 3 in Runge-Kutta method ------------------
        eta1 = (eta[0] + dt*k2[0]/2) % 1.
        eta2 = (eta[1] + dt*k2[1]/2) % 1.
        eta3 = (eta[2] + dt*k2[2]/2) % 1.

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
        linalg.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k3_v)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # U-field
        if u_basis == 1:
            u[0] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u_1, starts1, pn)
            u[1] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u_2, starts2, pn)
            u[2] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, u_3, starts3, pn)

            # push-forward
            linalg.matrix_vector(dfinv_t, u, u_cart)

        elif u_basis == 2:
            u[0] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u_1, starts1, pn)
            u[1] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u_2, starts2, pn)
            u[2] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u_3, starts3, pn)

            # push-forward
            linalg.matrix_vector(df, u, u_cart)
            u_cart[:] = u_cart/det_df

        # B-field (2-form)
        b[0] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts1, pn)
        b[1] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2, pn)
        b[2] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts3, pn)

        # push-forward
        linalg.matrix_vector(df, b, b_cart)
        b_cart[:] = b_cart/det_df

        # normalized magnetic field direction
        b_norm = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)
        b0[:] = b_cart/b_norm

        # perpendicular momentum
        linalg.cross(u_cart, b0, uxb0)
        linalg.cross(b0, uxb0, uperp)

        # pullback as vector field
        linalg.matrix_vector(dfinv, uperp, k3_u)

        # sum contribs
        k3[:] = k3_v + k3_u

        # ------------------ step 4 in Runge-Kutta method ------------------
        eta1 = (eta[0] + dt*k3[0]) % 1.
        eta2 = (eta[1] + dt*k3[1]) % 1.
        eta3 = (eta[2] + dt*k3[2]) % 1.

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
        linalg.matrix_matrix(dfinv, dfinv_t, ginv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k4_v)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta1)
        span2 = bsp.find_span(tn2, pn[1], eta2)
        span3 = bsp.find_span(tn3, pn[2], eta3)

        bsp.b_d_splines_slim(tn1, pn[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta3, span3, bn3, bd3)

        # U-field
        if u_basis == 1:
            u[0] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u_1, starts1, pn)
            u[1] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u_2, starts2, pn)
            u[2] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, u_3, starts3, pn)

            # push-forward
            linalg.matrix_vector(dfinv_t, u, u_cart)

        elif u_basis == 2:
            u[0] = eval_3d.eval_spline_mpi_3d(
                pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u_1, starts1, pn)
            u[1] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u_2, starts2, pn)
            u[2] = eval_3d.eval_spline_mpi_3d(
                pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u_3, starts3, pn)

            # push-forward
            linalg.matrix_vector(df, u, u_cart)
            u_cart[:] = u_cart/det_df

        # B-field (2-form)
        b[0] = eval_3d.eval_spline_mpi_3d(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b2_1, starts1, pn)
        b[1] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2_2, starts2, pn)
        b[2] = eval_3d.eval_spline_mpi_3d(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b2_3, starts3, pn)

        # push-forward
        linalg.matrix_vector(df, b, b_cart)
        b_cart[:] = b_cart/det_df

        # normalized magnetic field direction
        b_norm = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)
        b0[:] = b_cart/b_norm

        # perpendicular momentum
        linalg.cross(u_cart, b0, uxb0)
        linalg.cross(b0, uxb0, uperp)

        # pullback as vector field
        linalg.matrix_vector(dfinv, uperp, k4_u)

        # sum contribs
        k4[:] = k4_v + k4_u

        #  ---------------- update logical coordinates ---------------------
        markers[:3, ip] = (eta + dt*(k1 + 2*k2 + 2*k3 + k4)/6) % 1.0

    #$ omp end parallel
