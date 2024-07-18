'Pusher kernels for gyro-center (5D) dynamics.'


from pyccel.decorators import stack_array

import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
# do not remove; needed to identify dependencies
import struphy.pic.pushing.pusher_args_kernels as pusher_args_kernels

from struphy.pic.pushing.pusher_args_kernels import DerhamArguments, DomainArguments
from struphy.bsplines.evaluation_kernels_3d import get_spans, eval_0form_spline_mpi, eval_1form_spline_mpi, eval_2form_spline_mpi, eval_3form_spline_mpi, eval_vectorfield_spline_mpi

from numpy import zeros, empty, shape, sqrt


@stack_array('dfm', 'df_t', 'g', 'g_inv', 'k', 'bb', 'grad_abs_b', 'curl_norm_b', 'norm_b1', 'norm_b2', 'b_star', 'temp1', 'temp2')
def push_gc_bxEstar_explicit_multistage(markers: 'float[:,:]',
                                        dt: float,
                                        stage: int,
                                        args_derham: 'DerhamArguments',
                                        args_domain: 'DomainArguments',
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

    # containers for fields
    bb = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    temp1 = empty(3, dtype=float)
    temp2 = empty(3, dtype=float)

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

    #$ omp parallel private(ip, v, mu, k, det_df, dfm, df_t, g, g_inv, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, bb, grad_abs_b, curl_norm_b, norm_b1, norm_b2, b_star, temp1, temp2, abs_b_star_para)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 11] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]
        mu = markers[ip, 9]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              args_domain,
                              dfm)

        # evaluate inverse of G
        linalg_kernels.transpose(dfm, df_t)
        linalg_kernels.matrix_matrix(df_t, dfm, g)
        linalg_kernels.matrix_inv(g, g_inv)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # grad_abs_b; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              grad_abs_b1,
                              grad_abs_b2,
                              grad_abs_b3,
                              grad_abs_b)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              norm_b11,
                              norm_b12,
                              norm_b13,
                              norm_b1)

        # norm_b2; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              norm_b21,
                              norm_b22,
                              norm_b23,
                              norm_b2)

        # b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              b1,
                              b2,
                              b3,
                              bb)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_norm_b1,
                              curl_norm_b2,
                              curl_norm_b3,
                              curl_norm_b)

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
        markers[ip, 15:18] += dt*b[stage]*k

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = markers[ip, 11:14] + \
            dt*a[stage]*k + last*markers[ip, 15:18]

    #$ omp end parallel


@stack_array('dfm', 'k', 'bb', 'grad_abs_b', 'curl_norm_b', 'norm_b1', 'b_star')
def push_gc_Bstar_explicit_multistage(markers: 'float[:,:]',
                                      dt: float,
                                      stage: int,
                                      args_derham: 'DerhamArguments',
                                      args_domain: 'DomainArguments',
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

    # containers for fields
    bb = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    b_star = empty(3, dtype=float)

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

        if markers[ip, 11] == -1.:
            continue

        if stage == 0.:
            # save initial parallel velocity
            markers[ip, 14] = markers[ip, 3]

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]
        mu = markers[ip, 9]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              args_domain,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # grad_abs_b; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              grad_abs_b1,
                              grad_abs_b2,
                              grad_abs_b3,
                              grad_abs_b)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              norm_b11,
                              norm_b12,
                              norm_b13,
                              norm_b1)

        # b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              b1,
                              b2,
                              b3,
                              bb)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_norm_b1,
                              curl_norm_b2,
                              curl_norm_b3,
                              curl_norm_b)

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
        markers[ip, 15:18] += dt*b[stage]*k
        markers[ip, 18] += dt*b[stage]*k_v

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = markers[ip, 11:14] + \
            dt*a[stage]*k + last*markers[ip, 15:18]
        markers[ip, 3] = markers[ip, 14] + dt * \
            a[stage]*k_v + last*markers[ip, 18]

    #$ omp end parallel


@stack_array('dfm', 'df_t', 'g', 'g_inv', 'k', 'bb', 'grad_abs_b', 'curl_norm_b', 'norm_b1', 'norm_b2', 'b_star', 'temp1', 'temp2')
def push_gc_bxEstarWithPhi_explicit_multistage(markers: 'float[:,:]',
                                               dt: float,
                                               stage: int,
                                               args_derham: 'DerhamArguments',
                                               args_domain: 'DomainArguments',
                                               efield_1: 'float[:,:,:]', efield_2: 'float[:,:,:]', efield_3: 'float[:,:,:]',
                                               gradB1_1: 'float[:,:,:]', gradB1_2: 'float[:,:,:]', gradB1_3: 'float[:,:,:]',
                                               absB0: 'float[:,:,:]',
                                               curl_unit_b1_1: 'float[:,:,:]', curl_unit_b1_2: 'float[:,:,:]', curl_unit_b1_3: 'float[:,:,:]',
                                               unit_b1_1: 'float[:,:,:]', unit_b1_2: 'float[:,:,:]', unit_b1_3: 'float[:,:,:]',
                                               epsilon: float,
                                               Z: int,
                                               a: 'float[:]', b: 'float[:]', c: 'float[:]'):
    r'''Single stage of a s-stage explicit pushing step for the :math:`\mathbf b_ \times E^*`  drift kinetic electrostatic adiabatic,

    Marker update:

    .. math::

        \begin{aligned}
            \dot{\boldsymbol \eta}_p &=  \frac{1}{ B^{*3}_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})} \hat{\mathbf E}^{*1} (\boldsymbol \eta_p, v_{\parallel,\,p})  \times  \hat{\mathbf b}^1_0(\boldsymbol \eta_p) \,,
            \\
            \dot v_{\parallel,\,p} &= 0 \,.
        \end{aligned}

    for each marker :math:`p` in markers array.
    '''
    # metric coefficients
    df_mat = empty((3, 3), dtype=float)

    # containers for fields
    efield = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)
    e_star = empty(3, dtype=float)
    unit_b1 = empty(3, dtype=float)
    curl_unit_b1 = empty(3, dtype=float)

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

    #$ omp parallel private(ip, v, mu, k, det_df, dfm, df_t, g, g_inv, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, bb, grad_abs_b, curl_norm_b, norm_b1, norm_b2, b_star, temp1, temp2, abs_b_star_para)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 9] == -1.:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v = markers[ip, 3]
        mu = markers[ip, 9]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e1, e2, e3,
                              args_domain,
                              df_mat)

        det_df = linalg_kernels.det(df_mat)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # electric field: 1-form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              efield_1,
                              efield_2,
                              efield_3,
                              efield)

        # grad absB0; 1-form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              gradB1_1,
                              gradB1_2,
                              gradB1_3,
                              grad_abs_b)

        # absB0; 0-form
        absB0_at_eta = eval_0form_spline_mpi(span1, span2, span3,
                                             args_derham,
                                             absB0)

        # curl_unit_b1; 2-form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_unit_b1_1,
                              curl_unit_b1_2,
                              curl_unit_b1_3,
                              curl_unit_b1)

        # unit b1; 1-form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              unit_b1_1,
                              unit_b1_2,
                              unit_b1_3,
                              unit_b1)

        # E*
        e_star[:] = efield - epsilon / Z * mu * grad_abs_b

        # curvature times det_df
        curvature_det_df = linalg_kernels.scalar_dot(curl_unit_b1, unit_b1)

        # B^*_parallel times det_df
        b_star_para = absB0_at_eta * det_df + epsilon/Z * v * curvature_det_df

        # calculate E* x b0
        linalg_kernels.cross(e_star, unit_b1, k)

        # calculate k
        k /= b_star_para

        # accumulation for last stage
        markers[ip, 15:18] += dt*b[stage]*k

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = markers[ip, 11:14] + \
            dt*a[stage]*k + last*markers[ip, 15:18]

    #$ omp end parallel


@stack_array('dfm', 'k', 'bb', 'grad_abs_b', 'curl_norm_b', 'norm_b1', 'b_star')
def push_gc_BstarWithPhi_explicit_multistage(markers: 'float[:,:]',
                                             dt: float,
                                             stage: int,
                                             args_derham: 'DerhamArguments',
                                             args_domain: 'DomainArguments',
                                             efield_1: 'float[:,:,:]', efield_2: 'float[:,:,:]', efield_3: 'float[:,:,:]',
                                             beq_1: 'float[:,:,:]', beq_2: 'float[:,:,:]', beq_3: 'float[:,:,:]',
                                             curl_b1: 'float[:,:,:]', curl_b2: 'float[:,:,:]', curl_b3: 'float[:,:,:]',
                                             grad_absB0_1: 'float[:,:,:]', grad_absB0_2: 'float[:,:,:]', grad_absB0_3: 'float[:,:,:]',
                                             unit_b11: 'float[:,:,:]', unit_b12: 'float[:,:,:]', unit_b13: 'float[:,:,:]',
                                             absB0: 'float[:,:,:]',
                                             epsilon: float,
                                             Z: int,
                                             a: 'float[:]', b: 'float[:]', c: 'float[:]'):
    r'''Single stage of a s-stage explicit pushing step for the :math:`\mathbf B^*` drift kinetic electrostatic adiabatic,

    Marker update:

    .. math::

        \begin{aligned}
            \dot{\boldsymbol \eta}_p &= \frac{1}{B^{*3}_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})} \hat{\mathbf B}^{*2} (\boldsymbol \eta_p, v_{\parallel,\,p}) \, v_{\parallel,p} \,,
            \\
            \dot v_{\parallel,\,p} &= \frac{1}{B^{*3}_\parallel (\boldsymbol \eta_p, v_{\parallel,\,p})}  \hat{\mathbf B}^{*2} (\boldsymbol \eta_p, v_{\parallel,\,p}) \cdot \hat{\mathbf E}^{*1} (\boldsymbol \eta_p, v_{\parallel,\,p})  \,.
        \end{aligned}

    for each marker :math:`p` in markers array.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # containers for fields
    bb = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)
    curl_b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    efield = empty(3, dtype=float)
    e_star = empty(3, dtype=float)
    unit_b1 = empty(3, dtype=float)

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

    #$ omp parallel private(ip, v, mu, k, k_v, det_df, dfm, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, bb, grad_abs_b, curl_norm_b, norm_b1, b_star, temp, abs_b_star_para)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 11] == -1.:
            continue

        if stage == 0.:
            # save initial parallel velocity
            markers[ip, 14] = markers[ip, 3]

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v = markers[ip, 3]
        mu = markers[ip, 9]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e1, e2, e3,
                              args_domain,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # grad_abs_b; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              grad_absB0_1,
                              grad_absB0_2,
                              grad_absB0_3,
                              grad_abs_b)

        # b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              beq_1,
                              beq_2,
                              beq_3,
                              bb)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              unit_b11,
                              unit_b12,
                              unit_b13,
                              unit_b1)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_b1,
                              curl_b2,
                              curl_b3,
                              curl_b)

        # absB0; 0-form
        absB0_at_eta = eval_0form_spline_mpi(span1, span2, span3,
                                             args_derham,
                                             absB0)

        # electric field: 1-form components
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              efield_1,
                              efield_2,
                              efield_3,
                              efield)

        # calculate Bstar
        b_star[:] = bb + epsilon/Z*v*curl_b

        # E*
        e_star[:] = efield - epsilon / Z * mu * grad_abs_b

        # curvature times det_df
        curvature_det_df = linalg_kernels.scalar_dot(curl_b, unit_b1)

        # B^*_parallel times det_df
        b_star_para = absB0_at_eta * det_df + epsilon/Z * v * curvature_det_df

        # calculate k for X
        k[:] = b_star*v/b_star_para

        # calculate k_v for v
        temp = linalg_kernels.scalar_dot(e_star, b_star)
        k_v = Z/epsilon * temp / b_star_para

        # accumulation for last stage
        markers[ip, 15:18] += dt*b[stage]*k
        markers[ip, 18] += dt*b[stage]*k_v

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = markers[ip, 11:14] + \
            dt*a[stage]*k + last*markers[ip, 15:18]
        markers[ip, 3] = markers[ip, 14] + dt * \
            a[stage]*k_v + last*markers[ip, 18]

    #$ omp end parallel


@stack_array('dfm', 'df_t', 'g', 'g_inv', 'k', 'bb', 'grad_abs_b', 'curl_norm_b', 'norm_b1', 'norm_b2', 'b_star', 'temp1', 'temp2', 'temp3')
def push_gc_all_explicit_multistage(markers: 'float[:,:]',
                                    dt: float,
                                    stage: int,
                                    args_derham: 'DerhamArguments',
                                    args_domain: 'DomainArguments',
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

        if markers[ip, 11] == -1.:
            continue

        if stage == 0.:
            # save initial parallel velocity
            markers[ip, 14] = markers[ip, 3]

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]
        mu = markers[ip, 9]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              args_domain,
                              dfm)

        # evaluate inverse of G
        linalg_kernels.transpose(dfm, df_t)
        linalg_kernels.matrix_matrix(df_t, dfm, g)
        linalg_kernels.matrix_inv(g, g_inv)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # grad_abs_b; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              grad_abs_b1,
                              grad_abs_b2,
                              grad_abs_b3,
                              grad_abs_b)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              norm_b11,
                              norm_b12,
                              norm_b13,
                              norm_b1)

        # norm_b2; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              norm_b21,
                              norm_b22,
                              norm_b23,
                              norm_b2)

        # b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              b1,
                              b2,
                              b3,
                              bb)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_norm_b1,
                              curl_norm_b2,
                              curl_norm_b3,
                              curl_norm_b)

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
        markers[ip, 15:18] += dt*b[stage]*k
        markers[ip, 18] += dt*b[stage]*k_v

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = markers[ip, 11:14] + \
            dt*a[stage]*k + last*markers[ip, 15:18]
        markers[ip, 3] = markers[ip, 14] + dt * \
            a[stage]*k_v + last*markers[ip, 18]


@stack_array('e', 'e_diff', 'grad_I', 'S', 'temp', 'tmp2')
def push_gc_bxEstar_discrete_gradient(markers: 'float[:,:]',
                                      dt: float,
                                      stage: int,
                                      args_derham: 'DerhamArguments',
                                      args_domain: 'DomainArguments',
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

        if markers[ip, 11] == -1.:
            continue

        e[:] = markers[ip, 0:3]

        e_diff[:] = e[:] - markers[ip, 11:14]
        mu = markers[ip, 9]

        if abs(e_diff[0]/e[0]) < tol and abs(e_diff[1]/e[1]) < tol and abs(e_diff[2]/e[2]) < tol:
            markers[ip, 11] = -1.
            markers[ip, 12] = stage

            continue

        # TODO: replace with better idea
        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        # spline evaluation
        span1, span2, span3 = get_spans(e[0], e[1], e[2], args_derham)

        # abs_b; 0form
        abs_b0 = eval_0form_spline_mpi(span1, span2, span3,
                                       args_derham,
                                       abs_b)

        # assemble S
        S[0, 1] = markers[ip, 15]
        S[0, 2] = markers[ip, 16]
        S[1, 0] = -markers[ip, 15]
        S[1, 2] = markers[ip, 17]
        S[2, 0] = -markers[ip, 16]
        S[2, 1] = -markers[ip, 17]

        # calculate grad_I
        tmp2[:] = markers[ip, 18:21]
        temp_scalar = linalg_kernels.scalar_dot(e_diff, tmp2)
        temp_scalar2 = e_diff[0]**2 + e_diff[1]**2 + e_diff[2]**2

        grad_I[:] = markers[ip, 18:21] + e_diff[:] * \
            (abs_b0*mu - markers[ip, 21] - temp_scalar)/temp_scalar2

        linalg_kernels.matrix_vector(S, grad_I, temp)

        markers[ip, 0:3] = markers[ip, 11:14] + dt*temp[:]

        markers[ip, 18:21] = markers[ip, 0:3]

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
            markers[ip, 11] = -1.
            markers[ip, 12] = stage

            continue

        markers[ip, 0:3] = (markers[ip, 0:3] + markers[ip, 11:14])/2.


@stack_array('e', 'e_diff', 'grad_I', 'S', 'temp', 'tmp2')
def push_gc_bxEstarwithPhi_discrete_gradient(markers: 'float[:,:]',
                                             dt: float,
                                             stage: int,
                                             args_derham: 'DerhamArguments',
                                             args_domain: 'DomainArguments',
                                             efield_1: 'float[:,:,:]', efield_2: 'float[:,:,:]', efield_3: 'float[:,:,:]',
                                             grad_absB0_1: 'float[:,:,:]', grad_absB0_2: 'float[:,:,:]', grad_absB0_3: 'float[:,:,:]',
                                             absB0: 'float[:,:,:]',
                                             curl_b_dot_b0: 'float[:,:,:]',
                                             unit_b1_1: 'float[:,:,:]', unit_b1_2: 'float[:,:,:]', unit_b1_3: 'float[:,:,:]',
                                             epsilon: float,
                                             Z: int,
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

    # containers for fields
    temp = empty(3, dtype=float)
    tmp2 = empty(3, dtype=float)
    tmp3 = empty(3, dtype=float)
    S = zeros((3, 3), dtype=float)
    grad_I = empty(3, dtype=float)
    efield = empty(3, dtype=float)

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
        span1, span2, span3 = get_spans(e[0], e[1], e[2], args_derham)

        # abs_b; 0form
        abs_b0 = eval_0form_spline_mpi(span1, span2, span3,
                                       args_derham,
                                       absB0)

        # electric field: 1-form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              efield_1,
                              efield_2,
                              efield_3,
                              efield)

        # assemble S
        S[0, 1] = markers[ip, 13]
        S[0, 2] = markers[ip, 14]
        S[1, 0] = -markers[ip, 13]
        S[1, 2] = markers[ip, 15]
        S[2, 0] = -markers[ip, 14]
        S[2, 1] = -markers[ip, 15]

        # calculate grad_I
        tmp2[:] = markers[ip, 16:19]
        # temp_scalar = linalg_kernels.scalar_dot(e_diff, tmp2)
        temp_scalar2 = e_diff[0]**2 + e_diff[1]**2 + e_diff[2]**2

        tmp3 = abs_b0*mu + epsilon / Z * efield

        grad_I[:] = (tmp3 - markers[ip, 19])/temp_scalar2

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


@stack_array('e', 'e_diff', 'grad_I', 'tmp')
def push_gc_Bstar_discrete_gradient(markers: 'float[:,:]',
                                    dt: float,
                                    stage: int,
                                    args_derham: 'DerhamArguments',
                                    args_domain: 'DomainArguments',
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

        if markers[ip, 11] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_diff[:] = e[:] - markers[ip, 11:14]

        # TODO: replace with better idea
        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        v = markers[ip, 3]
        v_old = markers[ip, 14]
        v_mid = (markers[ip, 3] + markers[ip, 14])/2.
        mu = markers[ip, 9]

        # spline evaluation
        span1, span2, span3 = get_spans(e[0], e[1], e[2], args_derham)

        # abs_b; 0form
        abs_b0 = eval_0form_spline_mpi(span1, span2, span3,
                                       args_derham,
                                       abs_b)

        # calculate grad_I
        tmp[:] = markers[ip, 19:22]
        temp_scalar = linalg_kernels.scalar_dot(e_diff, tmp)
        temp_scalar2 = e_diff[0]**2 + e_diff[1]**2 + \
            e_diff[2]**2 + (v - v_old)**2

        if temp_scalar2 == 0.:
            grad_I[:] = 0.
            grad_Iv = v_mid

        else:
            grad_I[:] = markers[ip, 19:22] + e_diff * \
                (abs_b0*mu - markers[ip, 18] - temp_scalar)/temp_scalar2
            grad_Iv = v_mid + (v - v_old)*(abs_b0*mu -
                                           markers[ip, 18] - temp_scalar)/temp_scalar2

        tmp[:] = markers[ip, 15:18]
        temp_scalar3 = linalg_kernels.scalar_dot(tmp, grad_I)

        markers[ip, 0:3] = markers[ip, 11:14] + dt*markers[ip, 15:18]*grad_Iv
        markers[ip, 3] = markers[ip, 14] - dt*temp_scalar3

        markers[ip, 19:23] = markers[ip, 0:4]

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
            markers[ip, 11] = -1.
            markers[ip, 12] = stage
            continue

        markers[ip, 0:4] = (markers[ip, 0:4] + markers[ip, 11:15])/2.


@stack_array('e', 'e_diff', 'grad_I', 'S', 'temp', 'tmp2')
def push_gc_bxEstar_discrete_gradient_faster(markers: 'float[:,:]',
                                             dt: float,
                                             stage: int,
                                             args_derham: 'DerhamArguments',
                                             args_domain: 'DomainArguments',
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

        if markers[ip, 11] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_diff[:] = e[:] - markers[ip, 11:14]
        mu = markers[ip, 9]

        if abs(e_diff[0]/e[0]) < tol and abs(e_diff[1]/e[1]) < tol and abs(e_diff[2]/e[2]) < tol:
            markers[ip, 11] = -1.
            markers[ip, 12] = stage

            continue

        # TODO: replace with better idea
        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        # spline evaluation
        span1, span2, span3 = get_spans(e[0], e[1], e[2], args_derham)

        # abs_b; 0form
        abs_b0 = eval_0form_spline_mpi(span1, span2, span3,
                                       args_derham,
                                       abs_b)

        # assemble S
        S[0, 1] = markers[ip, 15]
        S[0, 2] = markers[ip, 16]
        S[1, 0] = -markers[ip, 15]
        S[1, 2] = markers[ip, 17]
        S[2, 0] = -markers[ip, 16]
        S[2, 1] = -markers[ip, 17]

        # calculate grad_I
        tmp2[:] = markers[ip, 18:21]
        temp_scalar = linalg_kernels.scalar_dot(e_diff, tmp2)
        temp_scalar2 = e_diff[0]**2 + e_diff[1]**2 + e_diff[2]**2

        grad_I[:] = markers[ip, 18:21] + e_diff[:] * \
            (abs_b0*mu - markers[ip, 21] - temp_scalar)/temp_scalar2

        linalg_kernels.matrix_vector(S, grad_I, temp)

        markers[ip, 0:3] = markers[ip, 11:14] + dt*temp[:]

        markers[ip, 18:21] = markers[ip, 0:3]

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
            markers[ip, 11] = -1.
            markers[ip, 12] = stage

            continue

        markers[ip, 0:3] = (markers[ip, 0:3] + markers[ip, 11:14])/2.


@stack_array('e', 'e_diff', 'grad_I', 'tmp')
def push_gc_Bstar_discrete_gradient_faster(markers: 'float[:,:]',
                                           dt: float,
                                           stage: int,
                                           args_derham: 'DerhamArguments',
                                           args_domain: 'DomainArguments',
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

        if markers[ip, 11] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_diff[:] = e[:] - markers[ip, 11:14]

        # TODO: replace with better idea
        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        v = markers[ip, 3]
        v_old = markers[ip, 14]
        v_mid = (markers[ip, 3] + markers[ip, 14])/2.
        mu = markers[ip, 9]

        # spline evaluation
        span1, span2, span3 = get_spans(e[0], e[1], e[2], args_derham)

        # abs_b; 0form
        abs_b0 = eval_0form_spline_mpi(span1, span2, span3,
                                       args_derham,
                                       abs_b)

        # calculate grad_I
        tmp[:] = markers[ip, 19:22]
        temp_scalar = linalg_kernels.scalar_dot(e_diff, tmp)
        temp_scalar2 = e_diff[0]**2 + e_diff[1]**2 + \
            e_diff[2]**2 + (v - v_old)**2

        if temp_scalar2 == 0.:
            grad_I[:] = 0.
            grad_Iv = v_mid

        else:
            grad_I[:] = markers[ip, 19:22] + e_diff * \
                (abs_b0*mu - markers[ip, 18] - temp_scalar)/temp_scalar2
            grad_Iv = v_mid + (v - v_old)*(abs_b0*mu -
                                           markers[ip, 18] - temp_scalar)/temp_scalar2

        tmp[:] = markers[ip, 15:18]
        temp_scalar3 = linalg_kernels.scalar_dot(tmp, grad_I)

        markers[ip, 0:3] = markers[ip, 11:14] + dt*markers[ip, 15:18]*grad_Iv
        markers[ip, 3] = markers[ip, 14] - dt*temp_scalar3

        markers[ip, 19:23] = markers[ip, 0:4]

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
            markers[ip, 11] = -1.
            markers[ip, 12] = stage
            continue

        markers[ip, 0:4] = (markers[ip, 0:4] + markers[ip, 11:15])/2.


@stack_array('e', 'e_diff', 'e_old', 'F', 'S', 'temp', 'identity', 'grad_abs_b', 'grad_I', 'Jacobian_grad_I', 'Jacobian', 'Jacobian_inv')
def push_gc_bxEstar_discrete_gradient_Itoh_Newton(markers: 'float[:,:]',
                                                  dt: float,
                                                  stage: int,
                                                  args_derham: 'DerhamArguments',
                                                  args_domain: 'DomainArguments',
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

        if markers[ip, 11] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_old[:] = markers[ip, 11:14]
        mu = markers[ip, 9]

        e_diff[:] = e[:] - e_old[:]

        if abs(e_diff[0]/e[0]) < tol and abs(e_diff[1]/e[1]) < tol and abs(e_diff[2]/e[2]) < tol:
            markers[ip, 11] = -1.
            markers[ip, 12] = stage

            continue

        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        # assemble S
        S[0, 1] = markers[ip, 15]
        S[0, 2] = markers[ip, 16]
        S[1, 0] = -markers[ip, 15]
        S[1, 2] = markers[ip, 17]
        S[2, 0] = -markers[ip, 16]
        S[2, 1] = -markers[ip, 17]

        # identity matrix
        identity[0, 0] = 1.
        identity[1, 1] = 1.
        identity[2, 2] = 1.

        # spline evaluation
        span1, span2, span3 = get_spans(e[0], e[1], e[2], args_derham)

        # abs_b; 0form
        abs_b0 = eval_0form_spline_mpi(span1, span2, span3,
                                       args_derham,
                                       abs_b)

        # grad_abs_b; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              grad_abs_b1,
                              grad_abs_b2,
                              grad_abs_b3,
                              grad_abs_b)

        # assemble gradI
        grad_I[0] = mu*(markers[ip, 22] - markers[ip, 21])/(e_diff[0])
        grad_I[1] = mu*(markers[ip, 18] - markers[ip, 22])/(e_diff[1])
        grad_I[2] = mu*(abs_b0 - markers[ip, 18])/(e_diff[2])

        # calculate F = eta - eta_old + dt*S*grad_I
        linalg_kernels.matrix_vector(S, grad_I, F)
        F *= -dt
        F += e_diff[:]

        # assemble Jacobian_grad_I
        Jacobian_grad_I[0, 0] = mu*(markers[ip, 23]*(e_diff[0]) -
                                    markers[ip, 22] + markers[ip, 21])/(e_diff[0])**2
        Jacobian_grad_I[1, 0] = mu * \
            (markers[ip, 19] - markers[ip, 23])/(e_diff[1])
        Jacobian_grad_I[2, 0] = mu * \
            (grad_abs_b[0] - markers[ip, 19])/(e_diff[2])
        Jacobian_grad_I[0, 1] = 0.
        Jacobian_grad_I[1, 1] = mu*(markers[ip, 20]*(e_diff[1]) -
                                    markers[ip, 18] + markers[ip, 22])/(e_diff[1])**2
        Jacobian_grad_I[2, 1] = mu * \
            (grad_abs_b[1] - markers[ip, 20])/(e_diff[2])
        Jacobian_grad_I[0, 2] = 0.
        Jacobian_grad_I[1, 2] = 0.
        Jacobian_grad_I[2, 2] = mu*(grad_abs_b[2]*(e_diff[2]) -
                                    abs_b0 + markers[ip, 18])/(e_diff[2])**2

        # assemble Jacobian and its inverse
        linalg_kernels.matrix_matrix(S, Jacobian_grad_I, Jacobian)
        Jacobian *= dt
        Jacobian += identity

        linalg_kernels.matrix_inv(Jacobian, Jacobian_inv)

        # calculate eta_new
        linalg_kernels.matrix_vector(Jacobian_inv, F, temp)
        markers[ip, 18:21] = e[:] - temp

        diff = sqrt((temp[0]/e[0])**2 + (temp[1]/e[1])**2 + (temp[2]/e[2])**2)

        if diff < tol:
            markers[ip, 11] = -1.
            markers[ip, 12] = stage
            markers[ip, 0:3] = markers[ip, 18:21]

            continue

        if stage == maxiter-1:
            markers[ip, 0:3] = markers[ip, 18:21]

            continue

        markers[ip, 0] = markers[ip, 18]
        markers[ip, 1] = e_old[1]
        markers[ip, 2] = e_old[2]


@stack_array('e', 'e_diff', 'e_old', 'F', 'S', 'temp', 'identity', 'grad_abs_b', 'grad_I', 'Jacobian_grad_I', 'Jacobian', 'Jacobian_inv', 'Jacobian_temp34', 'Jacobian_temp33', 'tmp')
def push_gc_Bstar_discrete_gradient_Itoh_Newton(markers: 'float[:,:]',
                                                dt: float,
                                                stage: int,
                                                args_derham: 'DerhamArguments',
                                                args_domain: 'DomainArguments',
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

        if markers[ip, 11] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_old[:] = markers[ip, 11:14]
        v = markers[ip, 3]
        v_old = markers[ip, 14]
        v_mid = (v + v_old)/2.
        mu = markers[ip, 9]

        e_diff[:] = e[:] - e_old[:]

        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        # assemble S
        S[0:3, 3] = markers[ip, 15:18]
        S[3, 0:3] = -markers[ip, 15:18]

        # identity matrix
        identity[0, 0] = 1.
        identity[1, 1] = 1.
        identity[2, 2] = 1.
        identity[3, 3] = 1.

        # spline evaluation
        span1, span2, span3 = get_spans(e[0], e[1], e[2], args_derham)

        # abs_b; 0form
        abs_b0 = eval_0form_spline_mpi(span1, span2, span3,
                                       args_derham,
                                       abs_b)

        # grad_abs_b; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              grad_abs_b1,
                              grad_abs_b2,
                              grad_abs_b3,
                              grad_abs_b)

        # assemble gradI and Jacobian_grad_I
        if e_diff[0] == 0.:
            grad_I[0] == 0.
        else:
            grad_I[0] = mu*(markers[ip, 22] - markers[ip, 21])/(e_diff[0])
            Jacobian_grad_I[0, 0] = mu * \
                (markers[ip, 23]*(e_diff[0]) -
                 markers[ip, 22] + markers[ip, 21])/(e_diff[0])**2

        if e_diff[1] == 0.:
            grad_I[1] == 0.
        else:
            grad_I[1] = mu*(markers[ip, 18] - markers[ip, 22])/(e_diff[1])
            Jacobian_grad_I[1, 0] = mu * \
                (markers[ip, 19] - markers[ip, 23])/(e_diff[1])
            Jacobian_grad_I[1, 1] = mu * \
                (markers[ip, 20]*(e_diff[1]) -
                 markers[ip, 18] + markers[ip, 22])/(e_diff[1])**2

        if e_diff[2] == 0.:
            grad_I[2] == 0.
        else:
            grad_I[2] = mu*(abs_b0 - markers[ip, 18])/(e_diff[2])
            Jacobian_grad_I[2, 0] = mu * \
                (grad_abs_b[0] - markers[ip, 19])/(e_diff[2])
            Jacobian_grad_I[2, 1] = mu * \
                (grad_abs_b[1] - markers[ip, 20])/(e_diff[2])
            Jacobian_grad_I[2, 2] = mu * \
                (grad_abs_b[2]*(e_diff[2]) - abs_b0 +
                 markers[ip, 18])/(e_diff[2])**2

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
        markers[ip, 18:21] = e[:] - temp[0:3]
        markers[ip, 3] = v - temp[3]

        diff = sqrt((temp[0]/e[0])**2 + (temp[1]/e[1])**2 +
                    (temp[2]/e[2])**2 + (temp[3])**2)

        if diff < tol:
            markers[ip, 11] = -1.
            markers[ip, 12] = stage
            markers[ip, 0:3] = markers[ip, 18:21]

            continue

        if stage == maxiter-1:
            markers[ip, 0:3] = markers[ip, 18:21]

            continue

        markers[ip, 0] = markers[ip, 18]
        markers[ip, 1] = e_old[1]
        markers[ip, 2] = e_old[2]


@stack_array('dfm', 'e', 'u', 'b', 'b_star', 'norm_b1', 'curl_norm_b')
def push_gc_cc_J1_H1vec(markers: 'float[:,:]',
                        dt: float,
                        stage: int,
                        args_derham: 'DerhamArguments',
                        args_domain: 'DomainArguments',
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

    # containers for fields
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              args_domain,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              b1,
                              b2,
                              b3,
                              b)

        # u; H1vec
        eval_vectorfield_spline_mpi(span1, span2, span3,
                                    args_derham,
                                    u1,
                                    u2,
                                    u3,
                                    u)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              norm_b11,
                              norm_b12,
                              norm_b13,
                              norm_b1)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_norm_b1,
                              curl_norm_b2,
                              curl_norm_b3,
                              curl_norm_b)

        # b_star; in H1vec
        b_star[:] = (b + curl_norm_b*v*epsilon)/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # electric field E(1) = B(2) X U(0)
        linalg_kernels.cross(b, u, e)

        # curl_norm_b dot electric field
        temp = linalg_kernels.scalar_dot(e, curl_norm_b) / det_df

        markers[ip, 3] += temp/abs_b_star_para*v*dt


@stack_array('dfm', 'df_t', 'g', 'g_inv', 'e', 'u', 'u0', 'b', 'b_star', 'norm_b1', 'curl_norm_b')
def push_gc_cc_J1_Hcurl(markers: 'float[:,:]',
                        dt: float,
                        stage: int,
                        args_derham: 'DerhamArguments',
                        args_domain: 'DomainArguments',
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

    # containers for fields
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    u0 = empty(3, dtype=float)
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              args_domain,
                              dfm)

        # evaluate inverse of G
        linalg_kernels.transpose(dfm, df_t)
        linalg_kernels.matrix_matrix(df_t, dfm, g)
        linalg_kernels.matrix_inv(g, g_inv)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              b1,
                              b2,
                              b3,
                              b)

        # u; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              u1,
                              u2,
                              u3,
                              u)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              norm_b11,
                              norm_b12,
                              norm_b13,
                              norm_b1)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_norm_b1,
                              curl_norm_b2,
                              curl_norm_b3,
                              curl_norm_b)

        # b_star; in H1vec
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


@stack_array('dfm', 'e', 'u', 'b', 'b_star', 'norm_b1', 'curl_norm_b')
def push_gc_cc_J1_Hdiv(markers: 'float[:,:]',
                       dt: float,
                       stage: int,
                       args_derham: 'DerhamArguments',
                       args_domain: 'DomainArguments',
                       epsilon: float,
                       b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                       norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                       curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                       u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]',
                       boundary_cut: float):
    r'''Velocity update step for the `CurrentCoupling5DCurlb <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.propagators.propagators_coupling.CurrentCoupling5DCurlb>`_

    Marker update:

    .. math::

        v_{\parallel,p}^{n+1} =  v_{\parallel,p}^n - \frac{\Delta t}{2} \hat B^{*,-1}_\parallel(\mathbf X_p, v^n_{\parallel,p}) \frac{1}{\sqrt{g(\mathbf X_p)}} \frac{1}{\sqrt{g(\mathbf X_p)}} v_{\parallel,p}^n \hat{\mathbf B}^2(\mathbf X_p) \times(\hat \nabla \times \hat{\mathbf b}_0)(\mathbf X_p) \Lambda^2 (\mathbf u^{n+1} + \mathbf u^n ) (\mathbf X_p) \,,

    for each marker :math:`p` in markers array.
    '''

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # containers for fields
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    #$ omp parallel private(ip, boundary_cut, eta, v, det_df, dfm, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, b, u, e, curl_norm_b, norm_b1, b_star, tmp, abs_b_star_para)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]

        if eta1 < boundary_cut or eta1 > 1. - boundary_cut:
            continue

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              args_domain,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              b1,
                              b2,
                              b3,
                              b)

        # u; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              u1,
                              u2,
                              u3,
                              u)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              norm_b11,
                              norm_b12,
                              norm_b13,
                              norm_b1)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_norm_b1,
                              curl_norm_b2,
                              curl_norm_b3,
                              curl_norm_b)

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

    #$ omp end parallel


@stack_array('dfm', 'df_t', 'df_inv_t', 'g_inv', 'e', 'u', 'bb', 'b_star', 'norm_b1', 'norm_b2', 'curl_norm_b', 'tmp1', 'tmp2', 'b_prod', 'norm_b2_prod')
def push_gc_cc_J2_stage_H1vec(markers: 'float[:,:]',
                              dt: float,
                              stage: int,
                              args_derham: 'DerhamArguments',
                              args_domain: 'DomainArguments',
                              epsilon: float,
                              b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                              norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                              norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                              curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                              u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]',
                              a: 'float[:]', b: 'float[:]', c: 'float[:]'):
    r'''Single stage of a s-stage explicit pushing step for the `CurrentCoupling5DGradB <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.propagators.propagators_coupling.CurrentCoupling5DGradB>`_

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

        if markers[ip, 11] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              args_domain,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              b1,
                              b2,
                              b3,
                              bb)

        # u; H1vec
        eval_vectorfield_spline_mpi(span1, span2, span3,
                                    args_derham,
                                    u1,
                                    u2,
                                    u3,
                                    u)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              norm_b11,
                              norm_b12,
                              norm_b13,
                              norm_b1)

        # norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              norm_b21,
                              norm_b22,
                              norm_b23,
                              norm_b2)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_norm_b1,
                              curl_norm_b2,
                              curl_norm_b3,
                              curl_norm_b)

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

        markers[ip, 15:18] -= dt*b[stage]*e
        markers[ip, 0:3] = markers[ip, 11:14] - \
            dt*a[stage]*e + last*markers[ip, 15:18]


@stack_array('dfm', 'df_inv', 'df_inv_t', 'g_inv', 'e', 'u', 'bb', 'b_star', 'norm_b1', 'norm_b2', 'curl_norm_b', 'tmp1', 'tmp2', 'b_prod', 'norm_b2_prod')
def push_gc_cc_J2_stage_Hdiv(markers: 'float[:,:]',
                             dt: float,
                             stage: int,
                             args_derham: 'DerhamArguments',
                             args_domain: 'DomainArguments',
                             epsilon: float,
                             b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                             norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                             norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                             curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                             u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]',
                             a: 'float[:]', b: 'float[:]', c: 'float[:]', boundary_cut: float):
    r'''Single stage of a s-stage explicit pushing step for the `CurrentCoupling5DGradB <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.propagators.propagators_coupling.CurrentCoupling5DGradB>`_

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

    # get number of markers
    n_markers = shape(markers)[0]

    # get number of stages
    n_stages = shape(b)[0]

    if stage == n_stages - 1:
        last = 1.
    else:
        last = 0.

    #$ omp parallel firstprivate(b_prod, norm_b2_prod) private(ip, boundary_cut, eta, v, det_df, dfm, df_inv, df_inv_t, g_inv, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, bb, u, e, curl_norm_b, norm_b1, norm_b2, b_star, temp1, temp2, abs_b_star_para)
    #$ omp for
    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 11] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]
        v = markers[ip, 3]

        if eta1 < boundary_cut or eta2 > 1. - boundary_cut:
            continue

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              args_domain,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)
        linalg_kernels.matrix_inv_with_det(dfm, det_df, df_inv)
        linalg_kernels.transpose(df_inv, df_inv_t)
        linalg_kernels.matrix_matrix(df_inv, df_inv_t, g_inv)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              b1,
                              b2,
                              b3,
                              bb)

        # u; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              u1,
                              u2,
                              u3,
                              u)

        # norm_b1; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              norm_b11,
                              norm_b12,
                              norm_b13,
                              norm_b1)

        # norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              norm_b21,
                              norm_b22,
                              norm_b23,
                              norm_b2)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_norm_b1,
                              curl_norm_b2,
                              curl_norm_b3,
                              curl_norm_b)

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

        markers[ip, 15:18] -= dt*b[stage]*e
        markers[ip, 0:3] = markers[ip, 11:14] - \
            dt*a[stage]*e + last*markers[ip, 15:18]

    #$ omp end parallel
