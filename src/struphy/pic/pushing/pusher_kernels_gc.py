'Pusher kernels for gyro-center (5D) dynamics.'


from pyccel.decorators import stack_array

import struphy.linear_algebra.linalg_kernels as linalg_kernels 
import struphy.geometry.evaluation_kernels as evaluation_kernels 
import struphy.bsplines.bsplines_kernels as bsplines_kernels 
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d 

from numpy import zeros, empty, shape, sqrt


def a_documentation():
    r'''
    Explainer for arguments of pusher kernels.

    Function naming conventions:

    * starts with ``push_``
    * add a short description of the pusher, e.g. ``push_bxu_H1vec``.

    These kernels are passed to :class:`struphy.pic.pushing.pusher.Pusher` and called via::

        Pusher()

    The arguments passed to each kernel have a pre-defined order, defined in :class:`struphy.pic.pushing.pusher.Pusher`.
    This order is as follows (you can copy and paste from existing pusher_kernels functions):

    1. Marker info:
        * ``markers: 'float[:,:]'``          # local marker array

    2. Step info:
        * ``dt: 'float'``                    # time step
        * ``stage: 'int'``                   # current stage of the pusher (e.g. 0,1,2,3 for RK4)

    3. Derham spline bases info:
        * ``pn: 'int[:]'``                   # N-spline degree in each direction
        * ``tn1: 'float[:]'``                # N-spline knot vector 
        * ``tn2: 'float[:]'``
        * ``tn3: 'float[:]'``    

    4. mpi.comm start indices of FE coeffs on current process:
        - ``starts: 'int[:]'``               # start indices of current process

    5. Mapping info:
        - ``kind_map: 'int'``                # mapping identifier 
        - ``params_map: 'float[:]'``         # mapping parameters
        - ``p_map: 'int[:]'``                # spline degree
        - ``t1_map: 'float[:]'``             # knot vector 
        - ``t2_map: 'float[:]'``             
        - ``t3_map: 'float[:]'`` 
        - ``ind1_map: int[:,:]``             # Indices of non-vanishing splines in format (number of mapping grid cells, p_map + 1)       
        - ``ind2_map: int[:,:]`` 
        - ``ind3_map: int[:,:]``            
        - ``cx: 'float[:,:,:]'``             # control points for Fx
        - ``cy: 'float[:,:,:]'``             # control points for Fy
        - ``cz: 'float[:,:,:]'``             # control points for Fz                         

    6. Optional: additional parameters, for example
        - ``b2_1: 'float[:,:,:]'``           # spline coefficients of b2_1
        - ``b2_2: 'float[:,:,:]'``           # spline coefficients of b2_2
        - ``b2_3: 'float[:,:,:]'``           # spline coefficients of b2_3
        - ``f0_params: 'float[:]'``          # parameters of equilibrium background
        - ``maxiter: int``                   # maximum number of iterations for implicit pusher
        - ``tol: float``                     # error tolerance for implicit pusher
    '''

    print('This is just the docstring function.')


@stack_array('dfm', 'df_t', 'g', 'g_inv', 'bn1', 'bn2', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'k', 'bb', 'grad_abs_b', 'curl_norm_b', 'norm_b1', 'norm_b2', 'b_star', 'temp1', 'temp2')
def push_gc_bxEstar_explicit_multistage(markers: 'float[:,:]', dt: float, stage: int,
                                        pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                        starts: 'int[:]',
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
    r'''Single stage of a s-stage explicit pushing step for the :math:`\mathbf b_ \times \nabla B_\parallel` guiding center drift,

    Marker update:

    .. math::

        \begin{aligned}
            \dot{\boldsymbol \eta}_p &= \epsilon \mu_p \frac{1}{ B^*_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})}  G^{-1}(\boldsymbol \eta_p) \hat{\mathbf b}^2_0(\boldsymbol \eta_p) \times G^{-1}(\boldsymbol \eta_p) \hat \nabla \hat{B}^0_0 (\boldsymbol \eta_p) \,,
            \\
            \dot v_{\parallel,\,p} &= 0 \,.
        \end{aligned}

    for each marker :math:`p` in markers array.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
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
    bb = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    temp1 = empty(3, dtype=float)
    temp2 = empty(3, dtype=float)

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

    #$ omp parallel private(ip, e, v, mu, k, det_df, dfm, df_t, g, g_inv, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, bb, grad_abs_b, curl_norm_b, norm_b1, norm_b2, b_star, temp1, temp2, abs_b_star_para)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 9] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        v = markers[ip, 3]
        mu = markers[ip, 4]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e[0], e[1], e[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # evaluate inverse of G
        linalg_kernels.transpose(dfm, df_t)
        linalg_kernels.matrix_matrix(df_t, dfm, g)
        linalg_kernels.matrix_inv(g, g_inv)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval fields
        # grad_abs_b; 1form
        grad_abs_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)
        grad_abs_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts)
        grad_abs_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts)

        # norm_b1; 1form
        norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
        norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
        norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

        # norm_b2; 2form
        norm_b2[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts)
        norm_b2[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts)
        norm_b2[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts)

        # b; 2form
        bb[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        bb[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        bb[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # eval Bstar and transform to H1vec
        b_star[:] = bb + epsilon*v*curl_norm_b
        b_star /= det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # calculate norm_b X grad_abs_b and transform to H1vec
        linalg_kernels.matrix_vector(g_inv, grad_abs_b, temp1)
        linalg_kernels.cross(norm_b2, temp1, temp2)
        linalg_kernels.matrix_vector(g_inv, temp2, temp1)

        # calculate k
        k[:] = epsilon*mu/abs_b_star_para*temp1

        # accumulation for last stage
        markers[ip, 13:16] += dt*b[stage]*k

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = markers[ip, 9:12] + \
            dt*a[stage]*k + last*markers[ip, 13:16]

    #$ omp end parallel


@stack_array('dfm', 'bn1', 'bn2', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'k', 'bb', 'grad_abs_b', 'curl_norm_b', 'norm_b1', 'b_star')
def push_gc_Bstar_explicit_multistage(markers: 'float[:,:]', dt: float, stage: int,
                                      pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                      starts: 'int[:]',
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
    r'''Single stage of a s-stage explicit pushing step for the :math:`\mathbf B^*` guiding center drift,

    Marker update:

    .. math::

        \begin{aligned}
            \dot{\boldsymbol \eta}_p &= \frac{1}{B^*_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})} \frac{1}{\sqrt{g(\boldsymbol \eta_p)}} \hat{\mathbf B}^{*2} (\boldsymbol \eta_p, v_{\parallel,\,p}) \, v_{\parallel,p} \,,
            \\
            \dot v_{\parallel,\,p} &= - \frac{1}{B^*_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})}  \frac{1}{\sqrt{g(\boldsymbol \eta_p)}} \hat{\mathbf B}^{*2} (\boldsymbol \eta_p, v_{\parallel,\,p}) \cdot \mu_p \hat \nabla \hat{B}^0_\parallel(\boldsymbol \eta_p) \,.
        \end{aligned}

    for each marker :math:`p` in markers array.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    bb = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)

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

    #$ omp parallel private(ip, e, v, mu, k, k_v, det_df, dfm, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, bb, grad_abs_b, curl_norm_b, norm_b1, b_star, temp, abs_b_star_para)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 9] == -1.:
            continue

        if stage == 0.:
            # save initial parallel velocity
            markers[ip, 12] = markers[ip, 3]

        e[:] = markers[ip, 0:3]
        v = markers[ip, 3]
        mu = markers[ip, 4]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e[0], e[1], e[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval fields
        # grad_abs_b; 1form
        grad_abs_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)
        grad_abs_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts)
        grad_abs_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts)

        # norm_b1; 1form
        norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
        norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
        norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

        # b; 2form
        bb[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        bb[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        bb[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # calculate Bstar and transform to H1vec
        b_star[:] = bb + epsilon*v*curl_norm_b
        b_star /= det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # calculate k for X
        k[:] = b_star/abs_b_star_para*v

        # calculate k_v for v
        temp = linalg_kernels.scalar_dot(b_star, grad_abs_b)
        k_v = -1*mu/abs_b_star_para*temp

        # accumulation for last stage
        markers[ip, 13:16] += dt*b[stage]*k
        markers[ip, 16] += dt*b[stage]*k_v

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = markers[ip, 9:12] + \
            dt*a[stage]*k + last*markers[ip, 13:16]
        markers[ip, 3] = markers[ip, 12] + dt * \
            a[stage]*k_v + last*markers[ip, 16]

    #$ omp end parallel


@stack_array('dfm', 'df_t', 'g', 'g_inv', 'bn1', 'bn2', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'k', 'bb', 'grad_abs_b', 'curl_norm_b', 'norm_b1', 'norm_b2', 'b_star', 'temp1', 'temp2', 'temp3')
def push_gc_all_explicit_multistage(markers: 'float[:,:]', dt: float, stage: int,
                                    pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                    starts: 'int[:]',
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
    r'''Single stage of a s-stage explicit pushing step for the guiding center drift,

    Marker update:

    .. math::

        \begin{aligned}
            \dot{\boldsymbol \eta}_p &= \epsilon \mu_p \frac{1}{ B^*_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})}  G^{-1}(\boldsymbol \eta_p) \hat{\mathbf b}^2_0(\boldsymbol \eta_p) \times G^{-1}(\boldsymbol \eta_p) \hat \nabla \hat{B}^0_0 (\boldsymbol \eta_p) + \frac{1}{B^*_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})} \frac{1}{\sqrt{g(\boldsymbol \eta_p)}} \hat{\mathbf B}^{*2} (\boldsymbol \eta_p, v_{\parallel,\,p}) \, v_{\parallel,p}\,,
            \\
            \dot v_{\parallel,\,p} &=  - \frac{1}{B^*_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})}  \frac{1}{\sqrt{g(\boldsymbol \eta_p)}} \hat{\mathbf B}^{*2} (\boldsymbol \eta_p, v_{\parallel,\,p}) \cdot \mu_p \hat \nabla \hat{B}^0_\parallel(\boldsymbol \eta_p) \,,
        \end{aligned}

    for each marker :math:`p` in markers array.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
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
    bb = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    temp1 = empty(3, dtype=float)
    temp2 = empty(3, dtype=float)
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

        if markers[ip, 9] == -1.:
            continue

        if stage == 0.:
            # save initial parallel velocity
            markers[ip, 12] = markers[ip, 3]

        e[:] = markers[ip, 0:3]
        v = markers[ip, 3]
        mu = markers[ip, 4]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e[0], e[1], e[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # evaluate inverse of G
        linalg_kernels.transpose(dfm, df_t)
        linalg_kernels.matrix_matrix(df_t, dfm, g)
        linalg_kernels.matrix_inv(g, g_inv)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # grad_abs_b; 1form
        grad_abs_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)
        grad_abs_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts)
        grad_abs_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts)

        # norm_b1; 1form
        norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
        norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
        norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

        # norm_b2; 2form
        norm_b2[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts)
        norm_b2[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts)
        norm_b2[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts)

        # b; 2form
        bb[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        bb[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        bb[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # transform to H1vec
        b_star[:] = bb + epsilon*v*curl_norm_b
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # calculate norm_b X grad_abs_b
        linalg_kernels.matrix_vector(g_inv, grad_abs_b, temp1)

        linalg_kernels.cross(norm_b2, temp1, temp2)

        linalg_kernels.matrix_vector(g_inv, temp2, temp3)

        # calculate k
        k[:] = (epsilon*mu*temp3 + b_star*v)/abs_b_star_para

        # calculate k_v for v
        temp = linalg_kernels.scalar_dot(b_star, grad_abs_b)

        k_v = -1*mu/abs_b_star_para*temp

        # accumulation for last stage
        markers[ip, 13:16] += dt*b[stage]*k
        markers[ip, 16] += dt*b[stage]*k_v

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = markers[ip, 9:12] + \
            dt*a[stage]*k + last*markers[ip, 13:16]
        markers[ip, 3] = markers[ip, 12] + dt * \
            a[stage]*k_v + last*markers[ip, 16]


@stack_array('bn1', 'bn2', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'e_diff', 'grad_I', 'S', 'temp', 'tmp2')
def push_gc_bxEstar_discrete_gradient(markers: 'float[:,:]', dt: float, stage: int,
                                      pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                      starts: 'int[:]',
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
                                      grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]',
                                      maxiter: int, tol: float):
    r'''Single step of the fixed-point iteration (:math:`k`-index) for the discrete gradient method with 2nd order(Gonzalez, mid-point)

    .. math::

        {\mathbf X}^k_{n+1} = {\mathbf X}_n + \Delta t \, \mathbb S({\mathbf X}_n, {\mathbf X}^{k-1}_{n+1}) \bar{\nabla} I ({\mathbf X}_n, {\mathbf X}^{k-1}_{n+1})

    where :math:`\mathbf X_n` denotes the gyro-center particle position at time :math:`t = n \Delta t` and

    .. math::

        \mathbb S(\mathbf X_n, \mathbf X_{n+1}) &= \epsilon \frac{1}{ B^*_\parallel (\mathbf X_{n+1/2})}  G^{-1}(\mathbf X_{n+1/2}) \hat{\mathbf b}^2_0(\mathbf X_{n+1/2}) \times G^{-1}(\mathbf X_{n+1/2})\,, 

        \bar{\nabla} I ({\mathbf X}_n, {\mathbf X}_{n+1}) &= \nabla H(\mathbf X_{n+1/2}) + ({\mathbf X}_{n+1} + {\mathbf X}_{n}) \frac{H(\mathbf X_{n+1}) - H(\mathbf X_{n}) - ({\mathbf X}_{n+1} - {\mathbf X}_n)\cdot \nabla H(\mathbf X_{n+1/2})}{||{\mathbf X}_{n+1} - {\mathbf X}_n||^2}\,,

        H({\mathbf X}_{n}) &= \mu \hat B^0_\parallel({\mathbf X}_{n})\,.

    where :math:`\mathbf X_{n+1/2} = \frac{\mathbf X_n + \mathbf X_{n+1}}{2}` and
    the velocity :math:`v_\parallel` and magentic moment :math:`\mu` are constant in this step.
    '''

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)
    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    temp = empty(3, dtype=float)
    tmp2 = empty(3, dtype=float)
    S = zeros((3, 3), dtype=float)
    grad_I = empty(3, dtype=float)

    # marker position e
    e = empty(3, dtype=float)
    e_diff = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 9] == -1.:
            continue

        e[:] = markers[ip, 0:3]

        e_diff[:] = e[:] - markers[ip, 9:12]
        mu = markers[ip, 4]

        if abs(e_diff[0]/e[0]) < tol and abs(e_diff[1]/e[1]) < tol and abs(e_diff[2]/e[2]) < tol:
            markers[ip, 9] = -1.
            markers[ip, 10] = stage

            continue

        # TODO: replace with better idea
        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts)

        # assemble S
        S[0, 1] = markers[ip, 13]
        S[0, 2] = markers[ip, 14]
        S[1, 0] = -markers[ip, 13]
        S[1, 2] = markers[ip, 15]
        S[2, 0] = -markers[ip, 14]
        S[2, 1] = -markers[ip, 15]

        # calculate grad_I
        tmp2[:] = markers[ip, 16:19]
        temp_scalar = linalg_kernels.scalar_dot(e_diff, tmp2)
        temp_scalar2 = e_diff[0]**2 + e_diff[1]**2 + e_diff[2]**2

        grad_I[:] = markers[ip, 16:19] + e_diff[:] * \
            (abs_b0*mu - markers[ip, 19] - temp_scalar)/temp_scalar2

        linalg_kernels.matrix_vector(S, grad_I, temp)

        markers[ip, 0:3] = markers[ip, 9:12] + dt*temp[:]

        markers[ip, 16:19] = markers[ip, 0:3]

        e_diff[:] = markers[ip, 0:3] - e[:]

        # TODO: replace with better idea
        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        diff = sqrt((e_diff[0]/e[0])**2 + (e_diff[1]/e[1])**2 +
                    (e_diff[2]/e[2])**2)

        if diff < tol:
            markers[ip, 9] = -1.
            markers[ip, 10] = stage

            continue

        markers[ip, 0:3] = (markers[ip, 0:3] + markers[ip, 9:12])/2.


@stack_array('bn1', 'bn2', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'e_diff', 'grad_I', 'tmp')
def push_gc_Bstar_discrete_gradient(markers: 'float[:,:]', dt: float, stage: int,
                                    pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                    starts: 'int[:]',
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
                                    grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]',
                                    maxiter: int, tol: float):
    r'''Single step of the fixed-point iteration (:math:`k`-index) for the discrete gradient method with 2nd order(Gonzalez, mid-point)

    .. math::

        {\mathbf Z}^k_{n+1} = {\mathbf Z}_n + \Delta t \, \mathbb S({\mathbf Z}_n, {\mathbf Z}^{k-1}_{n+1}) \bar{\nabla} I ({\mathbf Z}_n, {\mathbf Z}^{k-1}_{n+1})

    where :math:`\mathbf X_n` denotes the gyro-center particle position at time :math:`t = n \Delta t` and :math:`\mathbf Z_n = (\mathbf X_n, v_{\parallel, n})`.

    .. math::

        \mathbb S(\mathbf Z_n, \mathbf Z_{n+1}) &= \frac{1}{B^*_\parallel (\mathbf Z_{n+1/2})} \frac{1}{\sqrt{g(\mathbf X_{n+1/2})}} \hat{\mathbf B}^{*2} (\mathbf Z_{n+1/2}) \,, 

        \bar{\nabla} I ({\mathbf Z}_n, {\mathbf Z}_{n+1}) &= \nabla H(\mathbf Z_{n+1/2}) + ({\mathbf Z}_{n+1} + {\mathbf Z}_{n}) \frac{H(\mathbf Z_{n+1})- H(\mathbf Z_{n}) - ({\mathbf Z}_{n+1} - {\mathbf Z}_n)\cdot \nabla H(\mathbf Z_{n+1/2})}{||{\mathbf Z}_{n+1} - {\mathbf Z}_n||^2}\,,

        H(\mathbf Z_{n}) &= \mu \hat B^0_\parallel({\mathbf X}_{n}) + \frac{1}{2} v^2_{\parallel,n} \,.

    where :math:`\mathbf Z_{n+1/2} = \frac{\mathbf Z_n + \mathbf Z_{n+1}}{2}`
    and magentic moment :math:`\mu` are constant in this step.
    '''

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    grad_I = empty(3, dtype=float)

    # marker position e
    e = empty(3, dtype=float)
    e_diff = empty(3, dtype=float)
    tmp = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 9] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_diff[:] = e[:] - markers[ip, 9:12]

        # TODO: replace with better idea
        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        v = markers[ip, 3]
        v_old = markers[ip, 12]
        v_mid = (markers[ip, 3] + markers[ip, 12])/2.
        mu = markers[ip, 4]

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts)

        # calculate grad_I
        tmp[:] = markers[ip, 17:20]
        temp_scalar = linalg_kernels.scalar_dot(e_diff, tmp)
        temp_scalar2 = e_diff[0]**2 + e_diff[1]**2 + \
            e_diff[2]**2 + (v - v_old)**2

        if temp_scalar2 == 0.:
            grad_I[:] = 0.
            grad_Iv = v_mid

        else:
            grad_I[:] = markers[ip, 17:20] + e_diff * \
                (abs_b0*mu - markers[ip, 16] - temp_scalar)/temp_scalar2
            grad_Iv = v_mid + (v - v_old)*(abs_b0*mu -
                                           markers[ip, 16] - temp_scalar)/temp_scalar2

        tmp[:] = markers[ip, 13:16]
        temp_scalar3 = linalg_kernels.scalar_dot(tmp, grad_I)

        markers[ip, 0:3] = markers[ip, 9:12] + dt*markers[ip, 13:16]*grad_Iv
        markers[ip, 3] = markers[ip, 12] - dt*temp_scalar3

        markers[ip, 17:21] = markers[ip, 0:4]

        e_diff[:] = e[:] - markers[ip, 0:3]

        # TODO: replace with better idea
        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        diff = sqrt((e_diff[0]/e[0])**2 + (e_diff[1]/e[1]) **
                    2 + (e_diff[2]/e[2])**2 + (v - markers[ip, 3])**2)

        if diff < tol:
            markers[ip, 9] = -1.
            markers[ip, 10] = stage
            continue

        markers[ip, 0:4] = (markers[ip, 0:4] + markers[ip, 9:13])/2.


@stack_array('bn1', 'bn2', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'e_diff', 'grad_I', 'S', 'temp', 'tmp2')
def push_gc_bxEstar_discrete_gradient_faster(markers: 'float[:,:]', dt: float, stage: int,
                                             pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                             starts: 'int[:]',
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
                                             grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]',
                                             maxiter: int, tol: float):
    r'''Single step of the fixed-point iteration (:math:`k`-index) for the discrete gradient method with 2nd order faster scheme(fixed :math:`\mathbb S`, Gonzalez, mid-point)

    .. math::

        {\mathbf X}^k_{n+1} = {\mathbf X}_n + \Delta t \, \mathbb S({\mathbf X}_n, {\mathbf X}^{k-1}_{n+1}) \bar{\nabla} I ({\mathbf X}_n, {\mathbf X}^{k-1}_{n+1})

    where :math:`\mathbf X_n` denotes the gyro-center particle position at time :math:`t = n \Delta t` and

    .. math::

        \mathbb S(\mathbf X_n, \mathbf X_{n+1}) &\approx \mathbb S(\mathbf X_n) = \epsilon \frac{1}{ B^*_\parallel (\mathbf X_n)}  G^{-1}(\mathbf X_n) \hat{\mathbf b}^2_0(\mathbf X_n) \times G^{-1}(\mathbf X_n)\,, 

        \bar{\nabla} I ({\mathbf X}_n, {\mathbf X}_{n+1}) &= \nabla H(\mathbf X_{n+1/2}) + ({\mathbf X}_{n+1} + {\mathbf X}_{n}) \frac{H(\mathbf X_{n+1}) - H(\mathbf X_{n}) - ({\mathbf X}_{n+1} - {\mathbf X}_n)\cdot \nabla H(\mathbf X_{n+1/2})}{||{\mathbf X}_{n+1} - {\mathbf X}_n||^2}\,,

        H({\mathbf X}_{n}) &= \mu \hat B^0_\parallel({\mathbf X}_{n})\,.

    where :math:`\mathbf X_{n+1/2} = \frac{\mathbf X_n + \mathbf X_{n+1}}{2}` and the velocity :math:`v_\parallel` and magentic moment :math:`\mu` are constant in this step.
    '''

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    temp = empty(3, dtype=float)
    tmp2 = empty(3, dtype=float)
    S = zeros((3, 3), dtype=float)
    grad_I = empty(3, dtype=float)

    # marker position e
    e = empty(3, dtype=float)
    e_diff = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 9] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_diff[:] = e[:] - markers[ip, 9:12]
        mu = markers[ip, 4]

        if abs(e_diff[0]/e[0]) < tol and abs(e_diff[1]/e[1]) < tol and abs(e_diff[2]/e[2]) < tol:
            markers[ip, 9] = -1.
            markers[ip, 10] = stage

            continue

        # TODO: replace with better idea
        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts)

        # assemble S
        S[0, 1] = markers[ip, 13]
        S[0, 2] = markers[ip, 14]
        S[1, 0] = -markers[ip, 13]
        S[1, 2] = markers[ip, 15]
        S[2, 0] = -markers[ip, 14]
        S[2, 1] = -markers[ip, 15]

        # calculate grad_I
        tmp2[:] = markers[ip, 16:19]
        temp_scalar = linalg_kernels.scalar_dot(e_diff, tmp2)
        temp_scalar2 = e_diff[0]**2 + e_diff[1]**2 + e_diff[2]**2

        grad_I[:] = markers[ip, 16:19] + e_diff[:] * \
            (abs_b0*mu - markers[ip, 19] - temp_scalar)/temp_scalar2

        linalg_kernels.matrix_vector(S, grad_I, temp)

        markers[ip, 0:3] = markers[ip, 9:12] + dt*temp[:]

        markers[ip, 16:19] = markers[ip, 0:3]

        e_diff[:] = markers[ip, 0:3] - e[:]

        # TODO: replace with better idea
        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        diff = sqrt((e_diff[0]/e[0])**2 + (e_diff[1]/e[1])
                    ** 2 + (e_diff[2]/e[2])**2)

        if diff < tol:
            markers[ip, 9] = -1.
            markers[ip, 10] = stage

            continue

        markers[ip, 0:3] = (markers[ip, 0:3] + markers[ip, 9:12])/2.


@stack_array('bn1', 'bn2', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'e_diff', 'grad_I', 'tmp')
def push_gc_Bstar_discrete_gradient_faster(markers: 'float[:,:]', dt: float, stage: int,
                                           pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                           starts: 'int[:]',
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
                                           grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]',
                                           maxiter: int, tol: float):
    r'''Single step of the fixed-point iteration (:math:`k`-index) for the discrete gradient method with 2nd order faster scheme(fixed :math:`\mathbb S`, Gonzalez, mid-point)

    .. math::

        {\mathbf Z}^k_{n+1} = {\mathbf Z}_n + \Delta t \, \mathbb S({\mathbf Z}_n, {\mathbf Z}^{k-1}_{n+1}) \bar{\nabla} I ({\mathbf Z}_n, {\mathbf Z}^{k-1}_{n+1})

    where :math:`\mathbf X_n` denotes the gyro-center particle position at time :math:`t = n \Delta t` and :math:`\mathbf Z_n = (\mathbf X_n, v_{\parallel, n})`.

    .. math::

        \mathbb S(\mathbf Z_n, \mathbf Z_{n+1}) &\approx \mathbb S(\mathbf Z_n) = \frac{1}{B^*_\parallel (\mathbf Z_n)} \frac{1}{\sqrt{g(\mathbf X_n)}} \hat{\mathbf B}^{*2} (\mathbf Z_n) \,, 

        \bar{\nabla} I ({\mathbf Z}_n, {\mathbf Z}_{n+1}) &= \nabla H(\mathbf Z_{n+1/2}) + ({\mathbf Z}_{n+1} + {\mathbf Z}_{n}) \frac{H(\mathbf Z_{n+1})- H(\mathbf Z_{n}) - ({\mathbf Z}_{n+1} - {\mathbf Z}_n)\cdot \nabla H(\mathbf Z_{n+1/2})}{||{\mathbf Z}_{n+1} - {\mathbf Z}_n||^2}\,,

        H(\mathbf Z_{n}) &= \mu \hat B^0_\parallel({\mathbf X}_{n}) + \frac{1}{2} v^2_{\parallel,n} \,.

    where :math:`\mathbf Z_{n+1/2} = \frac{\mathbf Z_n + \mathbf Z_{n+1}}{2}`
    and magentic moment :math:`\mu` are constant in this step.
    '''

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    grad_I = empty(3, dtype=float)

    # marker position e
    e = empty(3, dtype=float)
    e_diff = empty(3, dtype=float)
    tmp = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 9] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_diff[:] = e[:] - markers[ip, 9:12]

        # TODO: replace with better idea
        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        v = markers[ip, 3]
        v_old = markers[ip, 12]
        v_mid = (markers[ip, 3] + markers[ip, 12])/2.
        mu = markers[ip, 4]

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts)

        # calculate grad_I
        tmp[:] = markers[ip, 17:20]
        temp_scalar = linalg_kernels.scalar_dot(e_diff, tmp)
        temp_scalar2 = e_diff[0]**2 + e_diff[1]**2 + \
            e_diff[2]**2 + (v - v_old)**2

        if temp_scalar2 == 0.:
            grad_I[:] = 0.
            grad_Iv = v_mid

        else:
            grad_I[:] = markers[ip, 17:20] + e_diff * \
                (abs_b0*mu - markers[ip, 16] - temp_scalar)/temp_scalar2
            grad_Iv = v_mid + (v - v_old)*(abs_b0*mu -
                                           markers[ip, 16] - temp_scalar)/temp_scalar2

        tmp[:] = markers[ip, 13:16]
        temp_scalar3 = linalg_kernels.scalar_dot(tmp, grad_I)

        markers[ip, 0:3] = markers[ip, 9:12] + dt*markers[ip, 13:16]*grad_Iv
        markers[ip, 3] = markers[ip, 12] - dt*temp_scalar3

        markers[ip, 17:21] = markers[ip, 0:4]

        e_diff[:] = e[:] - markers[ip, 0:3]

        # TODO: replace with better idea
        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        diff = sqrt((e_diff[0]/e[0])**2 + (e_diff[1]/e[1]) **
                    2 + (e_diff[2]/e[2])**2 + (v - markers[ip, 3])**2)

        if diff < tol:
            markers[ip, 9] = -1.
            markers[ip, 10] = stage
            continue

        markers[ip, 0:4] = (markers[ip, 0:4] + markers[ip, 9:13])/2.


@stack_array('bn1', 'bn2', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'e_diff', 'e_old', 'F', 'S', 'temp', 'identity', 'grad_abs_b', 'grad_I', 'Jacobian_grad_I', 'Jacobian', 'Jacobian_inv')
def push_gc_bxEstar_discrete_gradient_Itoh_Newton(markers: 'float[:,:]', dt: float, stage: int,
                                                  pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                                  starts: 'int[:]',
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
                                                  grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]',
                                                  maxiter: int, tol: float):
    r'''Single step of the Newton-Raphson iteration (:math:`k`-index) for the discrete gradient method with 1st order faster scheme(fixed :math:`\mathbb S`, Itoh-Abe).
    Looking for a root (:math:`F=0`) of the function defined as

    .. math::

        F(\mathbf X_{n+1}) = \mathbf X_{n+1} - \mathbf X_n - \Delta t \mathbb S(\mathbf X_n) \bar \nabla I(\mathbf X_n, \mathbf X_{n+1})

    where :math:`\mathbf X_n` denotes the gyro-center particle position at time :math:`t = n \Delta t`.

    From the initial guess

    .. math::

        \mathbf X^0_{n+1} = \mathbf X_n + \Delta t \mathbb S(\mathbf X_n) \nabla H(\mathbf X_n) \,,

    iteratively solve the following equation

    .. math::

        \mathbf X^{k+1}_{n+1} = \mathbf X^k_n - J_F^{-1}(\mathbf X^k_{n+1}) F(\mathbf X^k_{n+1}) \,,

    where the Jacobian of F is given as

    .. math::

        (J_F)_{i,j} = (J_F)_{i,j} = \frac{\partial F_i}{\partial \mathbf X_{n+1,j}}  = 
        \begin{pmatrix}
            1 & 0 & 0 \\
            0 & 1 & 0 \\
            0 & 0 & 1
        \end{pmatrix}
        - \Delta t \mathbb S(\mathbf X_n) \frac{\partial \bar \nabla I_i}{\partial \mathbf X_{n+1,j}} \,.

    .. math::

        \mathbb S(\mathbf X_n, \mathbf X_{n+1}) &\approx \mathbb S(\mathbf X_n) = \epsilon \frac{1}{ B^*_\parallel (\mathbf X_n)}  G^{-1}(\mathbf X_n) \hat{\mathbf b}^2_0(\mathbf X_n) \times G^{-1}(\mathbf X_n)\,, 
        \\
        \bar \nabla I (\mathbf X_n, \mathbf X_{n+1}) &= 
        \begin{pmatrix}
            (H(X_{1,n+1},X_{2,n},X_{3,n})     - H(X_{1,n},X_{2,n},X_{3,n})    ) / (X_{1,n+1} - X_{1,n}) \\
            (H(X_{1,n+1},X_{2,n+1},X_{3,n})   - H(X_{1,n+1},X_{2,n},X_{3,n})  ) / (X_{2,n+1} - X_{2,n}) \\
            (H(X_{1,n+1},X_{2,n+1},X_{3,n+1}) - H(X_{1,n+1},X_{2,n+1},X_{3,n})) / (X_{3,n+1} - X_{3,n})
        \end{pmatrix} \,,
        \\
        H(\mathbf X_{n}) &= \mu \hat B^0_\parallel({\mathbf X}_{n})\,.

    where the velocity :math:`v_\parallel` and magentic moment :math:`\mu` are constant in this step.
    '''
    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    identity = zeros((3, 3), dtype=float)
    temp = empty(3, dtype=float)
    F = empty(3, dtype=float)
    S = zeros((3, 3), dtype=float)
    grad_abs_b = empty(3, dtype=float)
    grad_I = empty(3, dtype=float)
    Jacobian_grad_I = zeros((3, 3), dtype=float)
    Jacobian = zeros((3, 3), dtype=float)
    Jacobian_inv = zeros((3, 3), dtype=float)

    # marker position e
    e = empty(3, dtype=float)
    e_old = empty(3, dtype=float)
    e_diff = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 9] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_old[:] = markers[ip, 9:12]
        mu = markers[ip, 4]

        e_diff[:] = e[:] - e_old[:]

        if abs(e_diff[0]/e[0]) < tol and abs(e_diff[1]/e[1]) < tol and abs(e_diff[2]/e[2]) < tol:
            markers[ip, 9] = -1.
            markers[ip, 10] = stage

            continue

        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        # assemble S
        S[0, 1] = markers[ip, 13]
        S[0, 2] = markers[ip, 14]
        S[1, 0] = -markers[ip, 13]
        S[1, 2] = markers[ip, 15]
        S[2, 0] = -markers[ip, 14]
        S[2, 1] = -markers[ip, 15]

        # identity matrix
        identity[0, 0] = 1.
        identity[1, 1] = 1.
        identity[2, 2] = 1.

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts)

        # grad_abs_b; 1form
        grad_abs_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)
        grad_abs_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts)
        grad_abs_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts)

        # assemble gradI
        grad_I[0] = mu*(markers[ip, 20] - markers[ip, 19])/(e_diff[0])
        grad_I[1] = mu*(markers[ip, 16] - markers[ip, 20])/(e_diff[1])
        grad_I[2] = mu*(abs_b0 - markers[ip, 16])/(e_diff[2])

        # calculate F = eta - eta_old + dt*S*grad_I
        linalg_kernels.matrix_vector(S, grad_I, F)
        F *= -dt
        F += e_diff[:]

        # assemble Jacobian_grad_I
        Jacobian_grad_I[0, 0] = mu*(markers[ip, 21]*(e_diff[0]) -
                                    markers[ip, 20] + markers[ip, 19])/(e_diff[0])**2
        Jacobian_grad_I[1, 0] = mu * \
            (markers[ip, 17] - markers[ip, 21])/(e_diff[1])
        Jacobian_grad_I[2, 0] = mu * \
            (grad_abs_b[0] - markers[ip, 17])/(e_diff[2])
        Jacobian_grad_I[0, 1] = 0.
        Jacobian_grad_I[1, 1] = mu*(markers[ip, 18]*(e_diff[1]) -
                                    markers[ip, 16] + markers[ip, 20])/(e_diff[1])**2
        Jacobian_grad_I[2, 1] = mu * \
            (grad_abs_b[1] - markers[ip, 18])/(e_diff[2])
        Jacobian_grad_I[0, 2] = 0.
        Jacobian_grad_I[1, 2] = 0.
        Jacobian_grad_I[2, 2] = mu*(grad_abs_b[2]*(e_diff[2]) -
                                    abs_b0 + markers[ip, 16])/(e_diff[2])**2

        # assemble Jacobian and its inverse
        linalg_kernels.matrix_matrix(S, Jacobian_grad_I, Jacobian)
        Jacobian *= dt
        Jacobian += identity

        linalg_kernels.matrix_inv(Jacobian, Jacobian_inv)

        # calculate eta_new
        linalg_kernels.matrix_vector(Jacobian_inv, F, temp)
        markers[ip, 16:19] = e[:] - temp

        diff = sqrt((temp[0]/e[0])**2 + (temp[1]/e[1])**2 + (temp[2]/e[2])**2)

        if diff < tol:
            markers[ip, 9] = -1.
            markers[ip, 10] = stage
            markers[ip, 0:3] = markers[ip, 16:19]

            continue

        if stage == maxiter-1:
            markers[ip, 0:3] = markers[ip, 16:19]

            continue

        markers[ip, 0] = markers[ip, 16]
        markers[ip, 1] = e_old[1]
        markers[ip, 2] = e_old[2]


@stack_array('bn1', 'bn2', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'e_diff', 'e_old', 'F', 'S', 'temp', 'identity', 'grad_abs_b', 'grad_I', 'Jacobian_grad_I', 'Jacobian', 'Jacobian_inv', 'Jacobian_temp34', 'Jacobian_temp33', 'tmp')
def push_gc_Bstar_discrete_gradient_Itoh_Newton(markers: 'float[:,:]', dt: float, stage: int,
                                                pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                                starts: 'int[:]',
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
                                                grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]',
                                                maxiter: int, tol: float):
    r'''Single step of the Newton-Raphson iteration (:math:`k`-index) for the discrete gradient method with 1st order faster scheme(fixed :math:`\mathbb S`, Itoh-Abe).
    Looking for a root (:math:`F=0`) of the function defined as

    .. math::

        F(\mathbf Z_{n+1}) = \mathbf Z_{n+1} - \mathbf Z_n - \Delta t \mathbb S(\mathbf Z_n) \bar \nabla I(\mathbf Z_n, \mathbf Z_{n+1})

    where :math:`\mathbf X_n` denotes the gyro-center particle position at time :math:`t = n \Delta t` and :math:`\mathbf Z_n = (\mathbf X_n, v_{\parallel, n})`.

    From the initial guess

    .. math::

        \mathbf Z^0_{n+1} = \mathbf Z_n + \Delta t \mathbb S(\mathbf Z_n) \nabla H(\mathbf Z_n) \,,

    iteratively solve the following equation

    .. math::

        \mathbf Z^{k+1}_{n+1} = \mathbf Z^k_n - J_F^{-1}(\mathbf Z^k_{n+1}) F(\mathbf Z^k_{n+1}) \,,

    where the Jacobian of F is given as

    .. math::

        (J_F)_{i,j} = (J_F)_{i,j} = \frac{\partial F_i}{\partial \mathbf Z_{n+1,j}}  = 
        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 1
        \end{pmatrix}
        - \Delta t \mathbb S(\mathbf Z_n) \frac{\partial \bar \nabla I_i}{\partial \mathbf Z_{n+1,j}} \,.

    .. math::

        \mathbb S(\mathbf Z_n, \mathbf Z_{n+1}) &\approx \mathbb S(\mathbf Z_n) = \frac{1}{B^*_\parallel (\mathbf Z_n)} \frac{1}{\sqrt{g(\mathbf X_n)}} \hat{\mathbf B}^{*2} (\mathbf Z_n) \,, 
        \\
        \bar \nabla I (\mathbf Z_n, \mathbf Z_{n+1}) &= 
        \begin{pmatrix}
            (H(X_{1,n+1},X_{2,n},X_{3,n}, v_{\parallel, n})     - H(X_{1,n},X_{2,n},X_{3,n}, v_{\parallel, n})    ) / (X_{1,n+1} - X_{1,n}) \\
            (H(X_{1,n+1},X_{2,n+1},X_{3,n}, v_{\parallel, n})   - H(X_{1,n+1},X_{2,n},X_{3,n}, v_{\parallel, n})  ) / (X_{2,n+1} - X_{2,n}) \\
            (H(X_{1,n+1},X_{2,n+1},X_{3,n+1}, v_{\parallel, n}) - H(X_{1,n+1},X_{2,n+1},X_{3,n}, v_{\parallel, n})) / (X_{3,n+1} - X_{3,n}) \\
            (H(X_{1,n+1},X_{2,n+1},X_{3,n+1}, v_{\parallel, n+1}) - H(X_{1,n+1},X_{2,n+1},X_{3,n+1}, v_{\parallel, n})) / (v_{\parallel,n+1} - v_{\parallel,n}) \\
        \end{pmatrix} \,,
        \\
        H(\mathbf Z_{n}) &= \mu \hat B^0_\parallel({\mathbf X}_{n}) + \frac{1}{2} v^2_{\parallel,n} \,.

    where magentic moment :math:`\mu` are constant in this step.
    '''

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    identity = zeros((4, 4), dtype=float)
    temp = empty(4, dtype=float)
    F = empty(4, dtype=float)
    S = zeros((4, 4), dtype=float)
    grad_abs_b = empty(3, dtype=float)
    grad_I = empty(4, dtype=float)
    Jacobian_grad_I = zeros((4, 4), dtype=float)
    Jacobian = zeros((4, 4), dtype=float)
    Jacobian_inv = zeros((4, 4), dtype=float)
    Jacobian_temp34 = zeros((3, 4), dtype=float)
    Jacobian_temp33 = zeros((3, 3), dtype=float)
    tmp = zeros((3, 3), dtype=float)

    # marker position e
    e = empty(3, dtype=float)
    e_old = empty(3, dtype=float)
    e_diff = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 9] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_old[:] = markers[ip, 9:12]
        v = markers[ip, 3]
        v_old = markers[ip, 12]
        v_mid = (v + v_old)/2.
        mu = markers[ip, 4]

        e_diff[:] = e[:] - e_old[:]

        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        # assemble S
        S[0:3, 3] = markers[ip, 13:16]
        S[3, 0:3] = -markers[ip, 13:16]

        # identity matrix
        identity[0, 0] = 1.
        identity[1, 1] = 1.
        identity[2, 2] = 1.
        identity[3, 3] = 1.

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts)

        # grad_abs_b; 1form
        grad_abs_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)
        grad_abs_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts)
        grad_abs_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts)

        # assemble gradI and Jacobian_grad_I
        if e_diff[0] == 0.:
            grad_I[0] == 0.
        else:
            grad_I[0] = mu*(markers[ip, 20] - markers[ip, 19])/(e_diff[0])
            Jacobian_grad_I[0, 0] = mu * \
                (markers[ip, 21]*(e_diff[0]) -
                 markers[ip, 20] + markers[ip, 19])/(e_diff[0])**2

        if e_diff[1] == 0.:
            grad_I[1] == 0.
        else:
            grad_I[1] = mu*(markers[ip, 16] - markers[ip, 20])/(e_diff[1])
            Jacobian_grad_I[1, 0] = mu * \
                (markers[ip, 17] - markers[ip, 21])/(e_diff[1])
            Jacobian_grad_I[1, 1] = mu * \
                (markers[ip, 18]*(e_diff[1]) -
                 markers[ip, 16] + markers[ip, 20])/(e_diff[1])**2

        if e_diff[2] == 0.:
            grad_I[2] == 0.
        else:
            grad_I[2] = mu*(abs_b0 - markers[ip, 16])/(e_diff[2])
            Jacobian_grad_I[2, 0] = mu * \
                (grad_abs_b[0] - markers[ip, 17])/(e_diff[2])
            Jacobian_grad_I[2, 1] = mu * \
                (grad_abs_b[1] - markers[ip, 18])/(e_diff[2])
            Jacobian_grad_I[2, 2] = mu * \
                (grad_abs_b[2]*(e_diff[2]) - abs_b0 +
                 markers[ip, 16])/(e_diff[2])**2

        grad_I[3] = v_mid
        Jacobian_grad_I[3, 3] = 0.5

        # calculate F = eta - eta_old + dt*S*grad_I
        linalg_kernels.matrix_vector4(S, grad_I, F)
        F *= -dt
        F[0:3] += e_diff[:]
        F[3] += v - v_old

        # assemble Jacobian and its inverse
        linalg_kernels.matrix_matrix4(S, Jacobian_grad_I, Jacobian)
        Jacobian *= -dt
        Jacobian += identity

        # Inverse of the Jacobian
        det_J = linalg_kernels.det4(Jacobian)

        tmp[:] = Jacobian[1:, 1:]
        Jacobian_inv[0, 0] = linalg_kernels.det(tmp)/det_J
        Jacobian_temp33[:] = Jacobian[(0, 2, 3), 1:]
        Jacobian_inv[0, 1] = -linalg_kernels.det(Jacobian_temp33)/det_J
        Jacobian_temp33[:] = Jacobian[(0, 1, 3), 1:]
        Jacobian_inv[0, 2] = linalg_kernels.det(Jacobian_temp33)/det_J
        tmp[:] = Jacobian[:3, 1:]
        Jacobian_inv[0, 3] = -linalg_kernels.det(tmp)/det_J

        Jacobian_temp33[:] = Jacobian[1:, (0, 2, 3)]
        Jacobian_inv[1, 0] = -linalg_kernels.det(Jacobian_temp33)/det_J
        Jacobian_temp34[:] = Jacobian[(0, 2, 3), :]
        Jacobian_temp33[:] = Jacobian_temp34[:, (0, 2, 3)]
        Jacobian_inv[1, 1] = linalg_kernels.det(Jacobian_temp33)/det_J
        Jacobian_temp34[:] = Jacobian[(0, 1, 3), :]
        Jacobian_temp33[:] = Jacobian_temp34[:, (0, 2, 3)]
        Jacobian_inv[1, 2] = -linalg_kernels.det(Jacobian_temp33)/det_J
        Jacobian_temp33[:] = Jacobian[:3, (0, 2, 3)]
        Jacobian_inv[1, 3] = linalg_kernels.det(Jacobian_temp33)/det_J

        Jacobian_temp33[:] = Jacobian[1:, (0, 1, 3)]
        Jacobian_inv[2, 0] = linalg_kernels.det(Jacobian_temp33)/det_J
        Jacobian_temp34[:] = Jacobian[(0, 2, 3), :]
        Jacobian_temp33[:] = Jacobian_temp34[:, (0, 1, 3)]
        Jacobian_inv[2, 1] = -linalg_kernels.det(Jacobian_temp33)/det_J
        Jacobian_temp34[:] = Jacobian[(0, 1, 3), :]
        Jacobian_temp33[:] = Jacobian_temp34[:, (0, 1, 3)]
        Jacobian_inv[2, 2] = linalg_kernels.det(Jacobian_temp33)/det_J
        Jacobian_temp33[:] = Jacobian[:3, (0, 1, 3)]
        Jacobian_inv[2, 3] = -linalg_kernels.det(Jacobian_temp33)/det_J

        tmp[:] = Jacobian[1:, :3]
        Jacobian_inv[3, 0] = -linalg_kernels.det(tmp)/det_J
        Jacobian_temp33[:] = Jacobian[(0, 2, 3), :3]
        Jacobian_inv[3, 1] = linalg_kernels.det(Jacobian_temp33)/det_J
        Jacobian_temp33[:] = Jacobian[(0, 1, 3), :3]
        Jacobian_inv[3, 2] = -linalg_kernels.det(Jacobian_temp33)/det_J
        tmp[:] = Jacobian[:3, :3]
        Jacobian_inv[3, 3] = linalg_kernels.det(tmp)/det_J

        # calculate eta_new
        linalg_kernels.matrix_vector4(Jacobian_inv, F, temp)
        markers[ip, 16:19] = e[:] - temp[0:3]
        markers[ip, 3] = v - temp[3]

        diff = sqrt((temp[0]/e[0])**2 + (temp[1]/e[1])**2 +
                    (temp[2]/e[2])**2 + (temp[3])**2)

        if diff < tol:
            markers[ip, 9] = -1.
            markers[ip, 10] = stage
            markers[ip, 0:3] = markers[ip, 16:19]

            continue

        if stage == maxiter-1:
            markers[ip, 0:3] = markers[ip, 16:19]

            continue

        markers[ip, 0] = markers[ip, 16]
        markers[ip, 1] = e_old[1]
        markers[ip, 2] = e_old[2]


@stack_array('dfm', 'bn1', 'bn2', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'u', 'b', 'b_star', 'norm_b1', 'curl_norm_b')
def push_gc_cc_J1_H1vec(markers: 'float[:,:]', dt: float, stage: int,
                        pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                        starts: 'int[:]',
                        kind_map: int, params_map: 'float[:]',
                        p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                        ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                        cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                        epsilon: float,
                        b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                        norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                        curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                        u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]'):
    r'''Velocity update step for the `CurrentCoupling5DCurlb <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.propagators.propagators_coupling.CurrentCoupling5DCurlb>`_

    Marker update :

    .. math::

        v_{\parallel,p}^{n+1} =  v_{\parallel,p}^n - \frac{\Delta t}{2} \hat B^{*,-1}_\parallel(\mathbf X_p, v^n_{\parallel,p}) \frac{1}{\sqrt{g(\mathbf X_p)}} v_{\parallel,p}^n \hat{\mathbf B}^2(\mathbf X_p) \times(\hat \nabla \times \hat{\mathbf b}_0)(\mathbf X_p) \Lambda^v (\mathbf u^{n+1} + \mathbf u^n ) (\mathbf X_p) \,,

    for each marker :math:`p` in markers array.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    # marker position eta
    eta = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]
        v = markers[ip, 3]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta[0], eta[1], eta[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # eval all the needed field
        # b; 2form
        b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # u; 0form
        u[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u1, starts)
        u[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u2, starts)
        u[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u3, starts)

        # norm_b1; 1form
        norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
        norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
        norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # b_star; 2form in H1vec
        b_star[:] = (b + curl_norm_b*v*epsilon)/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # electric field E(1) = B(2) X U(0)
        linalg_kernels.cross(b, u, e)

        # curl_norm_b dot electric field
        temp = linalg_kernels.scalar_dot(e, curl_norm_b) / det_df

        markers[ip, 3] += temp/abs_b_star_para*v*dt


@stack_array('dfm', 'df_t', 'g', 'g_inv', 'bn1', 'bn2', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'u', 'u0', 'b', 'b_star', 'norm_b1', 'curl_norm_b')
def push_gc_cc_J1_Hcurl(markers: 'float[:,:]', dt: float, stage: int,
                        pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                        starts: 'int[:]',
                        kind_map: int, params_map: 'float[:]',
                        p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                        ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                        cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                        epsilon: float,
                        b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                        norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                        curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                        u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]'):
    r'''Velocity update step for the `CurrentCoupling5DCurlb <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.propagators.propagators_coupling.CurrentCoupling5DCurlb>`_

    Marker update:

    .. math::

        v_{\parallel,p}^{n+1} =  v_{\parallel,p}^n - \frac{\Delta t}{2} \hat B^{*,-1}_\parallel(\mathbf X_p, v^n_{\parallel,p}) \frac{1}{\sqrt{g(\mathbf X_p)}} v_{\parallel,p}^n G^{-1}(\mathbf X_p) \hat{\mathbf B}^2(\mathbf X_p) \times(\hat \nabla \times \hat{\mathbf b}_0)(\mathbf X_p) \Lambda^1 (\mathbf u^{n+1} + \mathbf u^n ) (\mathbf X_p) \,,

    for each marker :math:`p` in markers array.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
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
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    u0 = empty(3, dtype=float)
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    # marker position eta
    eta = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]
        v = markers[ip, 3]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta[0], eta[1], eta[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # evaluate inverse of G
        linalg_kernels.transpose(dfm, df_t)
        linalg_kernels.matrix_matrix(df_t, dfm, g)
        linalg_kernels.matrix_inv(g, g_inv)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # eval all the needed field
        # b; 2form
        b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # u; 1form
        u[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u1, starts)
        u[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u2, starts)
        u[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, u3, starts)

        # norm_b1; 1form
        norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
        norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
        norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # b_star; 2form in H1vec
        b_star[:] = (b + curl_norm_b*v*epsilon)/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # transform u into H1vec
        linalg_kernels.matrix_vector(g_inv, u, u0)

        # electric field E(1) = B(2) X U(0)
        linalg_kernels.cross(b, u0, e)

        # curl_norm_b dot electric field
        temp = linalg_kernels.scalar_dot(e, curl_norm_b) / det_df

        markers[ip, 3] += temp/abs_b_star_para*v*dt


@stack_array('dfm', 'bn1', 'bn2', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'u', 'b', 'b_star', 'norm_b1', 'curl_norm_b')
def push_gc_cc_J1_Hdiv(markers: 'float[:,:]', dt: float, stage: int,
                       pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                       starts: 'int[:]',
                       kind_map: int, params_map: 'float[:]',
                       p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                       ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                       cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                       epsilon: float,
                       b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                       norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                       curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                       u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]'):
    r'''Velocity update step for the `CurrentCoupling5DCurlb <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.propagators.propagators_coupling.CurrentCoupling5DCurlb>`_

    Marker update:

    .. math::

        v_{\parallel,p}^{n+1} =  v_{\parallel,p}^n - \frac{\Delta t}{2} \hat B^{*,-1}_\parallel(\mathbf X_p, v^n_{\parallel,p}) \frac{1}{\sqrt{g(\mathbf X_p)}} \frac{1}{\sqrt{g(\mathbf X_p)}} v_{\parallel,p}^n \hat{\mathbf B}^2(\mathbf X_p) \times(\hat \nabla \times \hat{\mathbf b}_0)(\mathbf X_p) \Lambda^2 (\mathbf u^{n+1} + \mathbf u^n ) (\mathbf X_p) \,,

    for each marker :math:`p` in markers array.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    # marker position eta
    eta = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]
        v = markers[ip, 3]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta[0], eta[1], eta[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # eval all the needed field
        # b; 2form
        b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # u; 2form
        u[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u1, starts)
        u[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u2, starts)
        u[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u3, starts)

        # norm_b1; 1form
        norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
        norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
        norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # b_star; 2form in H1vec
        b_star[:] = (b + curl_norm_b*v*epsilon)/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # transform u into H1vec
        u = u/det_df

        # electric field E(1) = B(2) X U(0)
        linalg_kernels.cross(b, u, e)

        # curl_norm_b dot electric field
        temp = linalg_kernels.scalar_dot(e, curl_norm_b) / det_df

        markers[ip, 3] += temp/abs_b_star_para*v*dt


@stack_array('dfm', 'df_t', 'df_inv_t', 'g_inv', 'bn1', 'bn2', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'u', 'bb', 'b_star', 'norm_b1', 'norm_b2', 'curl_norm_b', 'tmp1', 'tmp2', 'b_prod', 'norm_b2_prod')
def push_gc_cc_J2_stage_H1vec(markers: 'float[:,:]', dt: float, stage: int,
                              pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                              starts: 'int[:]',
                              kind_map: int, params_map: 'float[:]',
                              p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                              ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                              cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                              epsilon: float,
                              b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                              norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                              norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                              curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                              u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]',
                              a: 'float[:]', b: 'float[:]', c: 'float[:]'):
    r'''Single stage of a s-stage explicit pushing step for the `CurrentCoupling5DGradBxB <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.propagators.propagators_coupling.CurrentCoupling5DGradBxB>`_

    Marker update:

    .. math::

        \mathbf X^{n+1} = \mathbf X^n - \frac{\Delta t}{2} \hat B^{*,-1}_\parallel(\mathbf X_p, v^n_{\parallel,p}) G^{-1}(\mathbf X_p) \hat{\mathbf b}_0^2(\mathbf X_p) \times G^{-1}(\mathbf X_p) \hat{\mathbf B}^2(\mathbf X_p) \times \Lambda^v (\mathbf u^{n+1} + \mathbf u^n ) (\mathbf X_p) \,,

    for each marker :math:`p` in markers array.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    tmp1 = empty((3, 3), dtype=float)
    tmp2 = empty((3, 3), dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    norm_b2_prod = empty((3, 3), dtype=float)
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    bb = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    # marker position eta
    eta = empty(3, dtype=float)

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

        if markers[ip, 9] == -1.:
            continue

        eta[:] = markers[ip, 0:3]
        v = markers[ip, 3]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta[0], eta[1], eta[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # eval all the needed field
        # b; 2form
        bb[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        bb[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        bb[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # u; 0form
        u[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u1, starts)
        u[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u2, starts)
        u[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u3, starts)

        # norm_b1; 1form
        norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
        norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
        norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

        # norm_b; 2form
        norm_b2[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts)
        norm_b2[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts)
        norm_b2[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # operator bx() as matrix
        b_prod[0, 1] = -bb[2]
        b_prod[0, 2] = +bb[1]
        b_prod[1, 0] = +bb[2]
        b_prod[1, 2] = -bb[0]
        b_prod[2, 0] = -bb[1]
        b_prod[2, 1] = +bb[0]

        norm_b2_prod[0, 1] = -norm_b2[2]
        norm_b2_prod[0, 2] = +norm_b2[1]
        norm_b2_prod[1, 0] = +norm_b2[2]
        norm_b2_prod[1, 2] = -norm_b2[0]
        norm_b2_prod[2, 0] = -norm_b2[1]
        norm_b2_prod[2, 1] = +norm_b2[0]

        # b_star; 2form in H1vec
        b_star[:] = (bb + curl_norm_b*v*epsilon)/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        linalg_kernels.matrix_matrix(g_inv, norm_b2_prod, tmp1)
        linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)
        linalg_kernels.matrix_matrix(tmp2, b_prod, tmp1)

        linalg_kernels.matrix_vector(tmp1, u, e)

        e /= abs_b_star_para

        # markers[ip, :3] -= e/abs_b_star_para*dt

        markers[ip, 13:16] -= dt*b[stage]*e
        markers[ip, 0:3] = markers[ip, 9:12] + \
            dt*a[stage]*e + last*markers[ip, 13:16]


@stack_array('dfm', 'df_t', 'df_inv_t', 'g_inv', 'bn1', 'bn2', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'u', 'bb', 'b_star', 'norm_b1', 'norm_b2', 'curl_norm_b', 'tmp1', 'tmp2', 'b_prod', 'norm_b2_prod')
def push_gc_cc_J2_stage_Hdiv(markers: 'float[:,:]', dt: float, stage: int,
                             pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                             starts: 'int[:]',
                             kind_map: int, params_map: 'float[:]',
                             p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                             ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                             cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                             epsilon: float,
                             b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                             norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                             norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                             curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                             u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]',
                             a: 'float[:]', b: 'float[:]', c: 'float[:]'):
    r'''Single stage of a s-stage explicit pushing step for the `CurrentCoupling5DGradBxB <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.propagators.propagators_coupling.CurrentCoupling5DGradBxB>`_

    Marker update:

    .. math::

        \mathbf X^{n+1} = \mathbf X^n - \frac{\Delta t}{2} \hat B^{*,-1}_\parallel(\mathbf X_p, v^n_{\parallel,p}) \frac{1}{\sqrt{g(\mathbf X_p)}} G^{-1}(\mathbf X_p) \hat{\mathbf b}_0^2(\mathbf X_p) \times G^{-1}(\mathbf X_p) \hat{\mathbf B}^2(\mathbf X_p) \times \Lambda^2 (\mathbf u^{n+1} + \mathbf u^n ) (\mathbf X_p) \,,

    for each marker :math:`p` in markers array.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_inv = empty((3, 3), dtype=float)
    df_inv_t = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    tmp1 = zeros((3, 3), dtype=float)
    tmp2 = zeros((3, 3), dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    norm_b2_prod = zeros((3, 3), dtype=float)
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    bb = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    # marker position eta
    eta = empty(3, dtype=float)

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

        if markers[ip, 9] == -1.:
            continue

        eta[:] = markers[ip, 0:3]
        v = markers[ip, 3]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta[0], eta[1], eta[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], eta[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], eta[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], eta[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # eval all the needed field
        # b; 2form
        bb[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        bb[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        bb[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # u; 2form
        u[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u1, starts)
        u[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u2, starts)
        u[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u3, starts)

        # norm_b1; 1form
        norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
        norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
        norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

        # norm_b; 2form
        norm_b2[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts)
        norm_b2[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts)
        norm_b2[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # operator bx() as matrix
        b_prod[0, 1] = -bb[2]
        b_prod[0, 2] = +bb[1]
        b_prod[1, 0] = +bb[2]
        b_prod[1, 2] = -bb[0]
        b_prod[2, 0] = -bb[1]
        b_prod[2, 1] = +bb[0]

        norm_b2_prod[0, 1] = -norm_b2[2]
        norm_b2_prod[0, 2] = +norm_b2[1]
        norm_b2_prod[1, 0] = +norm_b2[2]
        norm_b2_prod[1, 2] = -norm_b2[0]
        norm_b2_prod[2, 0] = -norm_b2[1]
        norm_b2_prod[2, 1] = +norm_b2[0]

        # b_star; 2form in H1vec
        b_star[:] = (bb + curl_norm_b*v*epsilon)/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        linalg_kernels.matrix_matrix(g_inv, norm_b2_prod, tmp1)
        linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)
        linalg_kernels.matrix_matrix(tmp2, b_prod, tmp1)

        linalg_kernels.matrix_vector(tmp1, u, e)

        e /= abs_b_star_para
        e /= det_df

        # markers[ip, :3] -= e/abs_b_star_para*dt

        markers[ip, 13:16] -= dt*b[stage]*e
        markers[ip, 0:3] = markers[ip, 9:12] + \
            dt*a[stage]*e + last*markers[ip, 13:16]


# def push_gc_cc_J2_dg_prepare_H1vec(markers: 'float[:,:]', dt: float, stage: int,
#                                    pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
#                                    starts: 'int[:]',
#                                    kind_map: int, params_map: 'float[:]',
#                                    p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
#                                    ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
#                                    cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
#                                    epsilon: float,
#                                    b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
#                                    norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
#                                    norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
#                                    curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
#                                    u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]'):
#     r'''DRAFT
#     '''

#     # allocate metric coeffs
#     dfm = empty((3, 3), dtype=float)
#     df_inv = empty((3, 3), dtype=float)
#     df_inv_t = empty((3, 3), dtype=float)
#     g_inv = empty((3, 3), dtype=float)

#     # allocate spline values
#     bn1 = empty(pn[0] + 1, dtype=float)
#     bn2 = empty(pn[1] + 1, dtype=float)
#     bn3 = empty(pn[2] + 1, dtype=float)

#     bd1 = empty(pn[0], dtype=float)
#     bd2 = empty(pn[1], dtype=float)
#     bd3 = empty(pn[2], dtype=float)

#     # containers for fields
#     tmp1 = empty((3, 3), dtype=float)
#     tmp2 = empty((3, 3), dtype=float)
#     b_prod = zeros((3, 3), dtype=float)
#     norm_b2_prod = zeros((3, 3), dtype=float)
#     e = empty(3, dtype=float)
#     u = empty(3, dtype=float)
#     b = empty(3, dtype=float)
#     b_star = empty(3, dtype=float)
#     norm_b1 = empty(3, dtype=float)
#     norm_b2 = empty(3, dtype=float)
#     curl_norm_b = empty(3, dtype=float)

#     # marker position eta
#     eta = empty(3, dtype=float)

#     # get number of markers
#     n_markers = shape(markers)[0]

#     for ip in range(n_markers):

#         # only do something if particle is a "true" particle (i.e. not a hole)
#         if markers[ip, 0] == -1.:
#             continue

#         eta[:] = markers[ip, 0:3]
#         v = markers[ip, 3]

#         # evaluate Jacobian, result in dfm
#         map_eval.df(eta[0], eta[1], eta[2],
#                     kind_map, params_map,
#                     t1_map, t2_map, t3_map, p_map,
#                     ind1_map, ind2_map, ind3_map,
#                     cx, cy, cz,
#                     dfm)

#         # metric coeffs
#         det_df = linalg_kernels.det(dfm)
#         linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
#         linalg_kernels.transpose(df_inv, df_inv_t)
#         linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

#         # spline evaluation
#         span1 = bsplines_kernels.find_span(tn1, pn[0], eta[0])
#         span2 = bsplines_kernels.find_span(tn2, pn[1], eta[1])
#         span3 = bsplines_kernels.find_span(tn3, pn[2], eta[2])

#         bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
#         bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
#         bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

#         # eval all the needed field
#         # b; 2form
#         b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
#         b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
#         b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

#         # u; 0form
#         u[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u1, starts)
#         u[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u2, starts)
#         u[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u3, starts)

#         # norm_b1; 1form
#         norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
#         norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
#         norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

#         # norm_b; 2form
#         norm_b2[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts)
#         norm_b2[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts)
#         norm_b2[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts)

#         # curl_norm_b; 2form
#         curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
#         curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
#         curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

#         # operator bx() as matrix
#         b_prod[0, 1] = -b[2]
#         b_prod[0, 2] = +b[1]
#         b_prod[1, 0] = +b[2]
#         b_prod[1, 2] = -b[0]
#         b_prod[2, 0] = -b[1]
#         b_prod[2, 1] = +b[0]

#         norm_b2_prod[0, 1] = -norm_b2[2]
#         norm_b2_prod[0, 2] = +norm_b2[1]
#         norm_b2_prod[1, 0] = +norm_b2[2]
#         norm_b2_prod[1, 2] = -norm_b2[0]
#         norm_b2_prod[2, 0] = -norm_b2[1]
#         norm_b2_prod[2, 1] = +norm_b2[0]

#         # b_star; 2form in H1vec
#         b_star[:] = (b + curl_norm_b*v*epsilon)/det_df

#         # calculate abs_b_star_para
#         abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

#         linalg_kernels.matrix_matrix(g_inv, norm_b2_prod, tmp1)
#         linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)
#         linalg_kernels.matrix_matrix(tmp2, b_prod, tmp1)

#         linalg_kernels.matrix_vector(tmp1, u, e)

#         markers[ip, 0:3] = markers[ip, 9:12] - e/abs_b_star_para*dt


# def push_gc_cc_J2_dg_H1vec(markers: 'float[:,:]', dt: float, stage: int,
#                            pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
#                            starts: 'int[:]',
#                            kind_map: int, params_map: 'float[:]',
#                            p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
#                            ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
#                            cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
#                            epsilon: float,
#                            b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
#                            norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
#                            norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
#                            curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
#                            u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]'):
#     r'''DRAFT
#     '''

#     # allocate metric coeffs
#     dfm = empty((3, 3), dtype=float)
#     df_inv = empty((3, 3), dtype=float)
#     df_inv_t = empty((3, 3), dtype=float)
#     g_inv = empty((3, 3), dtype=float)

#     # allocate spline values
#     bn1 = empty(pn[0] + 1, dtype=float)
#     bn2 = empty(pn[1] + 1, dtype=float)
#     bn3 = empty(pn[2] + 1, dtype=float)

#     bd1 = empty(pn[0], dtype=float)
#     bd2 = empty(pn[1], dtype=float)
#     bd3 = empty(pn[2], dtype=float)

#     # containers for fields
#     tmp1 = empty((3, 3), dtype=float)
#     tmp2 = empty((3, 3), dtype=float)
#     b_prod = zeros((3, 3), dtype=float)
#     norm_b2_prod = zeros((3, 3), dtype=float)
#     e = empty(3, dtype=float)
#     u = empty(3, dtype=float)
#     b = empty(3, dtype=float)
#     b_star = empty(3, dtype=float)
#     norm_b1 = empty(3, dtype=float)
#     norm_b2 = empty(3, dtype=float)
#     curl_norm_b = empty(3, dtype=float)

#     # marker position eta
#     eta = empty(3, dtype=float)

#     # get number of markers
#     n_markers = shape(markers)[0]

#     for ip in range(n_markers):

#         # only do something if particle is a "true" particle (i.e. not a hole)
#         if markers[ip, 0] == -1.:
#             continue

#         eta[:] = markers[ip, 0:3]
#         v = markers[ip, 3]

#         # evaluate Jacobian, result in dfm
#         map_eval.df(eta[0], eta[1], eta[2],
#                     kind_map, params_map,
#                     t1_map, t2_map, t3_map, p_map,
#                     ind1_map, ind2_map, ind3_map,
#                     cx, cy, cz,
#                     dfm)

#         # metric coeffs
#         det_df = linalg_kernels.det(dfm)
#         linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
#         linalg_kernels.transpose(df_inv, df_inv_t)
#         linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

#         # spline evaluation
#         span1 = bsplines_kernels.find_span(tn1, pn[0], eta[0])
#         span2 = bsplines_kernels.find_span(tn2, pn[1], eta[1])
#         span3 = bsplines_kernels.find_span(tn3, pn[2], eta[2])

#         bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
#         bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
#         bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

#         # eval all the needed field
#         # b; 2form
#         b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
#         b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
#         b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

#         # u; 0form
#         u[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u1, starts)
#         u[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u2, starts)
#         u[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u3, starts)

#         # norm_b1; 1form
#         norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
#         norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
#         norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

#         # norm_b; 2form
#         norm_b2[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts)
#         norm_b2[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts)
#         norm_b2[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts)

#         # curl_norm_b; 2form
#         curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
#         curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
#         curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

#         # operator bx() as matrix
#         b_prod[0, 1] = -b[2]
#         b_prod[0, 2] = +b[1]
#         b_prod[1, 0] = +b[2]
#         b_prod[1, 2] = -b[0]
#         b_prod[2, 0] = -b[1]
#         b_prod[2, 1] = +b[0]

#         norm_b2_prod[0, 1] = -norm_b2[2]
#         norm_b2_prod[0, 2] = +norm_b2[1]
#         norm_b2_prod[1, 0] = +norm_b2[2]
#         norm_b2_prod[1, 2] = -norm_b2[0]
#         norm_b2_prod[2, 0] = -norm_b2[1]
#         norm_b2_prod[2, 1] = +norm_b2[0]

#         # b_star; 2form in H1vec
#         b_star[:] = (b + curl_norm_b*v*epsilon)/det_df

#         # calculate abs_b_star_para
#         abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

#         linalg_kernels.matrix_matrix(g_inv, norm_b2_prod, tmp1)
#         linalg_kernels.matrix_matrix(tmp1, g_inv, tmp2)
#         linalg_kernels.matrix_matrix(tmp2, b_prod, tmp1)

#         linalg_kernels.matrix_vector(tmp1, u, e)

#         markers[ip, 0:3] = markers[ip, 9:12] - e/abs_b_star_para*dt


# def push_gc_cc_J2_dg_faster_H1vec(markers: 'float[:,:]', dt: float, stage: int,
#                                   pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
#                                   starts: 'int[:]',
#                                   kind_map: int, params_map: 'float[:]',
#                                   p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
#                                   ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
#                                   cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
#                                   epsilon: float,
#                                   b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
#                                   norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
#                                   norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
#                                   curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
#                                   u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]'):
#     r'''DRAFT
#     '''
#     # allocate spline values
#     bn1 = empty(pn[0] + 1, dtype=float)
#     bn2 = empty(pn[1] + 1, dtype=float)
#     bn3 = empty(pn[2] + 1, dtype=float)

#     bd1 = empty(pn[0], dtype=float)
#     bd2 = empty(pn[1], dtype=float)
#     bd3 = empty(pn[2], dtype=float)

#     # containers for fields
#     tmp = empty((3, 3), dtype=float)
#     e = empty(3, dtype=float)
#     u = empty(3, dtype=float)

#     # marker position eta
#     eta = empty(3, dtype=float)

#     # get number of markers
#     n_markers = shape(markers)[0]

#     for ip in range(n_markers):

#         # only do something if particle is a "true" particle (i.e. not a hole)
#         if markers[ip, 0] == -1.:
#             continue

#         eta[:] = markers[ip, 0:3]

#         # spline evaluation
#         span1 = bsplines_kernels.find_span(tn1, pn[0], eta[0])
#         span2 = bsplines_kernels.find_span(tn2, pn[1], eta[1])
#         span3 = bsplines_kernels.find_span(tn3, pn[2], eta[2])

#         bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
#         bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
#         bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

#         # eval all the needed field
#         # u; 0form
#         u[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u1, starts)
#         u[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u2, starts)
#         u[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u3, starts)

#         tmp[:, :] = ((markers[ip, 18], markers[ip, 19], markers[ip, 20]),
#                      (markers[ip, 19], markers[ip, 21], markers[ip, 22]),
#                      (markers[ip, 20], markers[ip, 22], markers[ip, 23]))

#         linalg_kernels.matrix_vector(tmp, u, e)

#         markers[ip, 0:3] = markers[ip, 9:12] - e*dt


# def push_gc_cc_J2_dg_faster_Hcurl(markers: 'float[:,:]', dt: float, stage: int,
#                                   pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
#                                   starts: 'int[:]',
#                                   kind_map: int, params_map: 'float[:]',
#                                   p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
#                                   ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
#                                   cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
#                                   epsilon: float,
#                                   b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
#                                   norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
#                                   norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
#                                   curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
#                                   u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]'):
#     r'''DRAFT
#     '''
#     # allocate spline values
#     bn1 = empty(pn[0] + 1, dtype=float)
#     bn2 = empty(pn[1] + 1, dtype=float)
#     bn3 = empty(pn[2] + 1, dtype=float)

#     bd1 = empty(pn[0], dtype=float)
#     bd2 = empty(pn[1], dtype=float)
#     bd3 = empty(pn[2], dtype=float)

#     # containers for fields
#     tmp = empty((3, 3), dtype=float)
#     e = empty(3, dtype=float)
#     u = empty(3, dtype=float)

#     # marker position eta
#     eta = empty(3, dtype=float)

#     # get number of markers
#     n_markers = shape(markers)[0]

#     for ip in range(n_markers):

#         # only do something if particle is a "true" particle (i.e. not a hole)
#         if markers[ip, 0] == -1.:
#             continue

#         eta[:] = markers[ip, 0:3]

#         # spline evaluation
#         span1 = bsplines_kernels.find_span(tn1, pn[0], eta[0])
#         span2 = bsplines_kernels.find_span(tn2, pn[1], eta[1])
#         span3 = bsplines_kernels.find_span(tn3, pn[2], eta[2])

#         bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
#         bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
#         bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

#         # eval all the needed field
#         # u; 1form
#         u[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u1, starts)
#         u[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u2, starts)
#         u[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, u3, starts)

#         tmp[:, :] = ((markers[ip, 18], markers[ip, 19], markers[ip, 20]),
#                      (markers[ip, 19], markers[ip, 21], markers[ip, 22]),
#                      (markers[ip, 20], markers[ip, 22], markers[ip, 23]))

#         linalg_kernels.matrix_vector(tmp, u, e)

#         markers[ip, 0:3] = markers[ip, 9:12] - e*dt


# def push_gc_cc_J2_dg_faster_Hdiv(markers: 'float[:,:]', dt: float, stage: int,
#                                  pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
#                                  starts: 'int[:]',
#                                  kind_map: int, params_map: 'float[:]',
#                                  p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
#                                  ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
#                                  cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
#                                  epsilon: float,
#                                  b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
#                                  norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
#                                  norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
#                                  curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
#                                  u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]'):
#     r'''DRAFT
#     '''
#     # allocate spline values
#     bn1 = empty(pn[0] + 1, dtype=float)
#     bn2 = empty(pn[1] + 1, dtype=float)
#     bn3 = empty(pn[2] + 1, dtype=float)

#     bd1 = empty(pn[0], dtype=float)
#     bd2 = empty(pn[1], dtype=float)
#     bd3 = empty(pn[2], dtype=float)

#     # containers for fields
#     tmp = empty((3, 3), dtype=float)
#     e = empty(3, dtype=float)
#     u = empty(3, dtype=float)

#     # marker position eta
#     eta = empty(3, dtype=float)

#     # get number of markers
#     n_markers = shape(markers)[0]

#     for ip in range(n_markers):

#         # only do something if particle is a "true" particle (i.e. not a hole)
#         if markers[ip, 0] == -1.:
#             continue

#         eta[:] = markers[ip, 0:3]

#         # spline evaluation
#         span1 = bsplines_kernels.find_span(tn1, pn[0], eta[0])
#         span2 = bsplines_kernels.find_span(tn2, pn[1], eta[1])
#         span3 = bsplines_kernels.find_span(tn3, pn[2], eta[2])

#         bsplines_kernels.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
#         bsplines_kernels.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
#         bsplines_kernels.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

#         # eval all the needed field
#         # u; 2form
#         u[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u1, starts)
#         u[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u2, starts)
#         u[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
#             pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u3, starts)

#         tmp[:, :] = ((markers[ip, 18], markers[ip, 19], markers[ip, 20]),
#                      (markers[ip, 19], markers[ip, 21], markers[ip, 22]),
#                      (markers[ip, 20], markers[ip, 22], markers[ip, 23]))

#         linalg_kernels.matrix_vector(tmp, u, e)

#         markers[ip, 0:3] = markers[ip, 9:12] - e*dt
