'Initialization routines (initial guess, evaluations) for 5D gyro-center pusher kernels.'


from pyccel.decorators import stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.geometry.evaluation_kernels as evaluation_kernels
# do not remove; needed to identify dependencies
import struphy.pic.pushing.pusher_args_kernels as pusher_args_kernels

from struphy.pic.pushing.pusher_args_kernels import DerhamArguments, DomainArguments
from struphy.bsplines.evaluation_kernels_3d import get_spans, eval_0form_spline_mpi, eval_1form_spline_mpi, eval_2form_spline_mpi, eval_3form_spline_mpi, eval_vectorfield_spline_mpi

from numpy import empty, shape, zeros, sqrt, log, abs


@stack_array('dfm', 'dfinv', 'g', 'g_inv', 'S', 'b', 'b_star', 'bcross', 'grad_abs_b', 'curl_norm_b', 'norm_b1', 'norm_b2', 'temp', 'temp1', 'temp2')
def init_gc_bxEstar_discrete_gradient(markers: 'float[:,:]',
                                      dt: float,
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
    r"""Initialization kernel for the pusher kernel `push_gc_bxEstar_discrete_gradient <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_gradient>`_ .
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    g = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # containers
    S = zeros((3, 3), dtype=float)
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    bcross = zeros((3, 3), dtype=float)
    grad_abs_b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
    temp = empty(3, dtype=float)
    temp1 = zeros((3, 3), dtype=float)
    temp2 = zeros((3, 3), dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v = markers[ip, 3]
        mu = markers[ip, 9]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e1, e2, e3,
                              args_domain,
                              dfm)

        # evaluate inverse of G
        linalg_kernels.transpose(dfm, df_t)
        linalg_kernels.matrix_matrix(df_t, dfm, g)
        linalg_kernels.matrix_inv(g, g_inv)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # abs_b; 0form
        abs_b0 = eval_0form_spline_mpi(span1, span2, span3,
                                       args_derham,
                                       abs_b)

        # save for later steps
        markers[ip, 21] = abs_b0*mu

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
                              b)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_norm_b1,
                              curl_norm_b2,
                              curl_norm_b3,
                              curl_norm_b)

        # grad_abs_b; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              grad_abs_b1,
                              grad_abs_b2,
                              grad_abs_b3,
                              grad_abs_b)

        # transform to H1vec
        b_star[:] = b + epsilon*v*curl_norm_b
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # assemble b cross (.)
        bcross[0, 1] = -norm_b2[2]
        bcross[0, 2] = norm_b2[1]
        bcross[1, 0] = norm_b2[2]
        bcross[1, 2] = -norm_b2[0]
        bcross[2, 0] = -norm_b2[1]
        bcross[2, 1] = norm_b2[0]

        # calculate G-1 b cross G-1
        linalg_kernels.matrix_matrix(bcross, g_inv, temp1)
        linalg_kernels.matrix_matrix(g_inv, temp1, temp2)

        # calculate S
        S[:, :] = (epsilon*temp2)/abs_b_star_para

        # calculate S1 * grad I1
        linalg_kernels.matrix_vector(S, grad_abs_b, temp)

        # save at the markers
        markers[ip, 0:3] = markers[ip, 0:3] + dt*temp[:]*mu

        markers[ip, 18:21] = markers[ip, 0:3]
        markers[ip, 0:3] = (markers[ip, 0:3] + markers[ip, 11:14])/2.


@stack_array('dfm', 'b', 'grad_abs_b', 'curl_norm_b', 'b_star', 'norm_b1')
def init_gc_Bstar_discrete_gradient(markers: 'float[:,:]',
                                    dt: float,
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
    r"""Initialization kernel for the pusher kernel `push_gc_Bstar_discrete_gradient <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_gradient>`_ .
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # containers
    b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

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

        # abs_b; 0form
        abs_b0 = eval_0form_spline_mpi(span1, span2, span3,
                                       args_derham,
                                       abs_b)

        # save for later steps
        markers[ip, 18] = mu*abs_b0

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
                              b)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_norm_b1,
                              curl_norm_b2,
                              curl_norm_b3,
                              curl_norm_b)

        # grad_abs_b; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              grad_abs_b1,
                              grad_abs_b2,
                              grad_abs_b3,
                              grad_abs_b)

        # transform to H1vec
        b_star[:] = b + epsilon*v*curl_norm_b
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # calculate b_star . grad_abs_b
        b_star_dot_grad_abs_b = linalg_kernels.scalar_dot(
            b_star, grad_abs_b)*mu

        # save at the markers
        markers[ip, 0:3] = markers[ip, 0:3] + dt*b_star[:]/abs_b_star_para*v
        markers[ip, 3] = markers[ip, 14] - dt * \
            b_star_dot_grad_abs_b/abs_b_star_para

        markers[ip, 19:23] = markers[ip, 0:4]
        markers[ip, 0:4] = (markers[ip, 0:4] + markers[ip, 11:15])/2.


@stack_array('dfm', 'dfinv', 'g', 'g_inv', 'S', 'b', 'b_star', 'bcross', 'grad_abs_b', 'curl_norm_b', 'norm_b1', 'norm_b2', 'temp', 'temp1', 'temp2')
def init_gc_bxEstar_discrete_gradient_faster(markers: 'float[:,:]',
                                             dt: float,
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
    r"""Initialization kernel for the pusher kernel `push_gc_bxEstar_discrete_gradient_faster <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_faster>`_ .
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    g = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # containers
    S = zeros((3, 3), dtype=float)
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    bcross = zeros((3, 3), dtype=float)
    grad_abs_b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
    temp = empty(3, dtype=float)
    temp1 = zeros((3, 3), dtype=float)
    temp2 = zeros((3, 3), dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v = markers[ip, 3]
        mu = markers[ip, 9]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e1, e2, e3,
                              args_domain,
                              dfm)

        # evaluate inverse of G
        linalg_kernels.transpose(dfm, df_t)
        linalg_kernels.matrix_matrix(df_t, dfm, g)
        linalg_kernels.matrix_inv(g, g_inv)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # abs_b; 0form
        abs_b0 = eval_0form_spline_mpi(span1, span2, span3,
                                       args_derham,
                                       abs_b)

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
                              b)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_norm_b1,
                              curl_norm_b2,
                              curl_norm_b3,
                              curl_norm_b)

        # grad_abs_b; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              grad_abs_b1,
                              grad_abs_b2,
                              grad_abs_b3,
                              grad_abs_b)

        # transform to H1vec
        b_star[:] = b + epsilon*v*curl_norm_b
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # assemble b cross (.)
        bcross[0, 1] = -norm_b2[2]
        bcross[0, 2] = norm_b2[1]
        bcross[1, 0] = norm_b2[2]
        bcross[1, 2] = -norm_b2[0]
        bcross[2, 0] = -norm_b2[1]
        bcross[2, 1] = norm_b2[0]

        # calculate G-1 b cross G-1
        linalg_kernels.matrix_matrix(bcross, g_inv, temp1)
        linalg_kernels.matrix_matrix(g_inv, temp1, temp2)

        # calculate S
        S[:, :] = (epsilon*temp2)/abs_b_star_para

        # save at the markers
        markers[ip, 15:17] = S[0, 1:3]
        markers[ip, 17] = S[1, 2]
        markers[ip, 21] = abs_b0*mu

        # calculate S1 * grad I1
        linalg_kernels.matrix_vector(S, grad_abs_b, temp)

        # save at the markers
        markers[ip, 0:3] = markers[ip, 0:3] + dt*temp[:]*mu

        markers[ip, 18:21] = markers[ip, 0:3]
        markers[ip, 0:3] = (markers[ip, 0:3] + markers[ip, 11:14])/2.


@stack_array('b', 'grad_abs_b', 'curl_norm_b', 'b_star', 'norm_b1')
def init_gc_Bstar_discrete_gradient_faster(markers: 'float[:,:]',
                                           dt: float,
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
    r"""Initialization kernel for the pusher kernel `push_gc_Bstar_discrete_gradient_faster <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_gradient_faster>`_ .
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # containers
    b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

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

        # abs_b; 0form
        abs_b0 = eval_0form_spline_mpi(span1, span2, span3,
                                       args_derham,
                                       abs_b)

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
                              b)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_norm_b1,
                              curl_norm_b2,
                              curl_norm_b3,
                              curl_norm_b)

        # grad_abs_b; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              grad_abs_b1,
                              grad_abs_b2,
                              grad_abs_b3,
                              grad_abs_b)

        # transform to H1vec
        b_star[:] = b + epsilon*v*curl_norm_b
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # save at the markers
        markers[ip, 15:18] = b_star[:]/abs_b_star_para
        markers[ip, 18] = mu*abs_b0

        # calculate b_star . grad_abs_b
        b_star_dot_grad_abs_b = linalg_kernels.scalar_dot(
            b_star, grad_abs_b)*mu

        # save at the markers
        markers[ip, 0:3] = markers[ip, 0:3] + dt*b_star[:]/abs_b_star_para*v
        markers[ip, 3] = markers[ip, 14] - dt * \
            b_star_dot_grad_abs_b/abs_b_star_para

        markers[ip, 19:23] = markers[ip, 0:4]
        markers[ip, 0:4] = (markers[ip, 0:4] + markers[ip, 11:15])/2.


@stack_array('dfm', 'dfinv', 'g', 'g_inv', 'S', 'b', 'b_star', 'bcross', 'grad_abs_b', 'curl_norm_b', 'norm_b1', 'norm_b2', 'temp', 'temp1', 'temp2')
def init_gc_bxEstar_discrete_gradient_Itoh_Newton(markers: 'float[:,:]',
                                                  dt: float,
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
    r"""Initialization kernel for the pusher kernel `push_gc_bxEstar_discrete_Itoh_Newton <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_Itoh_Newton>`_ .
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    g = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # containers
    S = zeros((3, 3), dtype=float)
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    bcross = zeros((3, 3), dtype=float)
    grad_abs_b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
    temp = empty(3, dtype=float)
    temp1 = zeros((3, 3), dtype=float)
    temp2 = zeros((3, 3), dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v = markers[ip, 3]
        mu = markers[ip, 9]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e1, e2, e3,
                              args_domain,
                              dfm)

        # evaluate inverse of G
        linalg_kernels.transpose(dfm, df_t)
        linalg_kernels.matrix_matrix(df_t, dfm, g)
        linalg_kernels.matrix_inv(g, g_inv)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # abs_b; 0form
        abs_b0 = eval_0form_spline_mpi(span1, span2, span3,
                                       args_derham,
                                       abs_b)

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
                              b)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_norm_b1,
                              curl_norm_b2,
                              curl_norm_b3,
                              curl_norm_b)

        # grad_abs_b; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              grad_abs_b1,
                              grad_abs_b2,
                              grad_abs_b3,
                              grad_abs_b)

        # transform to H1vec
        b_star[:] = b + epsilon*v*curl_norm_b
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # assemble b cross (.) as 3x3 matrix
        bcross[0, 1] = -norm_b2[2]
        bcross[0, 2] = norm_b2[1]
        bcross[1, 0] = norm_b2[2]
        bcross[1, 2] = -norm_b2[0]
        bcross[2, 0] = -norm_b2[1]
        bcross[2, 1] = norm_b2[0]

        # calculate G^-1 b cross G^-1
        linalg_kernels.matrix_matrix(bcross, g_inv, temp1)
        linalg_kernels.matrix_matrix(g_inv, temp1, temp2)

        # calculate S
        S[:, :] = (epsilon*temp2)/abs_b_star_para

        # save at the markers
        markers[ip, 15:17] = S[0, 1:3]
        markers[ip, 17] = S[1, 2]
        markers[ip, 21] = abs_b0

        # calculate S1 * grad I1
        linalg_kernels.matrix_vector(S, grad_abs_b, temp)

        # save at the markers
        markers[ip, 18:21] = markers[ip, 0:3] + dt*temp[:]*mu

        # send particles to the (eta^0_n+1, eta_n, eta_n)
        markers[ip, 0] = markers[ip, 18]


@stack_array('dfm', 'b', 'grad_abs_b', 'curl_norm_b', 'b_star', 'norm_b1')
def init_gc_Bstar_discrete_gradient_Itoh_Newton(markers: 'float[:,:]', dt: float,
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
    r"""Initialization kernel for the pusher kernel `push_gc_Bstar_discrete_Itoh_Newton <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_Itoh_Newton>`_ .
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # containers
    b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

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

        # abs_b; 0form
        abs_b0 = eval_0form_spline_mpi(span1, span2, span3,
                                       args_derham,
                                       abs_b)

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
                              b)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_norm_b1,
                              curl_norm_b2,
                              curl_norm_b3,
                              curl_norm_b)

        # grad_abs_b; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              grad_abs_b1,
                              grad_abs_b2,
                              grad_abs_b3,
                              grad_abs_b)

        # transform to H1vec
        b_star[:] = b + epsilon*v*curl_norm_b
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # save at the markers
        markers[ip, 15:18] = b_star[:]/abs_b_star_para
        markers[ip, 21] = abs_b0

        # calculate b_star . grad_abs_b
        b_star_dot_grad_abs_b = linalg_kernels.scalar_dot(
            b_star, grad_abs_b)*mu

        # save at the markers
        markers[ip, 18:21] = markers[ip, 0:3] + dt*b_star[:]/abs_b_star_para*v
        markers[ip, 3] = markers[ip, 14] - dt * \
            b_star_dot_grad_abs_b/abs_b_star_para

        # send particles to the (eta^0_n+1,eta_n, eta_n)
        markers[ip, 0] = markers[ip, 18]


@stack_array('dfm', 'dfinv', 'g', 'g_inv', 'S', 'b', 'b_star', 'bcross', 'grad_abs_b', 'curl_norm_b', 'norm_b1', 'norm_b2', 'temp1', 'temp2')
def gc_bxEstar_discrete_gradient_eval_gradI(markers: 'float[:,:]',
                                            dt: float,
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
    r"""Evaluation kernel for the pusher kernel `push_gc_bxEstar_discrete_gradient <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_gradient>`_ .
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)
    df_t = empty((3, 3), dtype=float)
    g = empty((3, 3), dtype=float)
    g_inv = empty((3, 3), dtype=float)

    # containers
    S = zeros((3, 3), dtype=float)
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    bcross = zeros((3, 3), dtype=float)
    grad_abs_b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
    temp1 = zeros((3, 3), dtype=float)
    temp2 = zeros((3, 3), dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 11] == -1.:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v = markers[ip, 3]
        markers[ip, 0:3] = markers[ip, 18:21]
        mu = markers[ip, 9]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e1, e2, e3,
                              args_domain,
                              dfm)

        # evaluate inverse of G
        linalg_kernels.transpose(dfm, df_t)
        linalg_kernels.matrix_matrix(df_t, dfm, g)
        linalg_kernels.matrix_inv(g, g_inv)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

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
                              b)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_norm_b1,
                              curl_norm_b2,
                              curl_norm_b3,
                              curl_norm_b)

        # grad_abs_b; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              grad_abs_b1,
                              grad_abs_b2,
                              grad_abs_b3,
                              grad_abs_b)

        # transform to H1vec
        b_star[:] = b + epsilon*v*curl_norm_b
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # assemble b cross (.) as 3x3 matrix
        bcross[0, 1] = -norm_b2[2]
        bcross[0, 2] = norm_b2[1]
        bcross[1, 0] = norm_b2[2]
        bcross[1, 2] = -norm_b2[0]
        bcross[2, 0] = -norm_b2[1]
        bcross[2, 1] = norm_b2[0]

        # calculate G-1 b cross G-1
        linalg_kernels.matrix_matrix(bcross, g_inv, temp1)
        linalg_kernels.matrix_matrix(g_inv, temp1, temp2)

        # calculate S
        S[:, :] = (epsilon*temp2)/abs_b_star_para

        # save at the markers
        markers[ip, 15:17] = S[0, 1:3]
        markers[ip, 17] = S[1, 2]

        markers[ip, 18:21] = mu*grad_abs_b[:]


@stack_array('dfm', 'b', 'grad_abs_b', 'curl_norm_b', 'b_star', 'norm_b1')
def gc_Bstar_discrete_gradient_eval_gradI(markers: 'float[:,:]',
                                          dt: float,
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
    r"""Evaluation kernel for the pusher kernel `push_gc_Bstar_discrete_gradient <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_gradient>`_ .
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # containers
    grad_abs_b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 11] == -1.:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        v_mid = markers[ip, 3]
        markers[ip, 0:4] = markers[ip, 19:23]
        mu = markers[ip, 9]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e1, e2, e3,
                              args_domain,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

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
                              b)

        # curl_norm_b; 2form
        eval_2form_spline_mpi(span1, span2, span3,
                              args_derham,
                              curl_norm_b1,
                              curl_norm_b2,
                              curl_norm_b3,
                              curl_norm_b)

        # grad_abs_b; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              grad_abs_b1,
                              grad_abs_b2,
                              grad_abs_b3,
                              grad_abs_b)

        # transform to H1vec
        b_star[:] = b + epsilon*v_mid*curl_norm_b
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # save at the markers
        markers[ip, 15:18] = b_star[:]/abs_b_star_para
        markers[ip, 19:22] = mu*grad_abs_b[:]


@stack_array('dfm', 'grad_abs_b')
def gc_bxEstar_discrete_gradient_faster_eval_gradI(markers: 'float[:,:]',
                                                   dt: float,
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
    r"""Evaluation kernel for the pusher kernel `push_gc_bxEstar_discrete_gradient_faster <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_faster>`_ .
    """

    # containers
    grad_abs_b = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 11] == -1.:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        markers[ip, 0:3] = markers[ip, 18:21]
        mu = markers[ip, 9]

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # grad_abs_b; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              grad_abs_b1,
                              grad_abs_b2,
                              grad_abs_b3,
                              grad_abs_b)

        markers[ip, 18:21] = mu*grad_abs_b[:]


@stack_array('dfm', 'grad_abs_b')
def gc_Bstar_discrete_gradient_faster_eval_gradI(markers: 'float[:,:]',
                                                 dt: float,
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
    r"""Evaluation kernel for the pusher kernel `push_gc_Bstar_discrete_gradient_faster <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_gradient_faster>`_ .
    """

    # containers
    grad_abs_b = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 11] == -1.:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]
        markers[ip, 0:4] = markers[ip, 19:23]
        mu = markers[ip, 9]

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # grad_abs_b; 1form
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              grad_abs_b1,
                              grad_abs_b2,
                              grad_abs_b3,
                              grad_abs_b)

        markers[ip, 19:22] = mu*grad_abs_b[:]


@stack_array('dfm')
def gc_bxEstar_discrete_gradient_Itoh_Newton_eval1(markers: 'float[:,:]',
                                                   dt: float,
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
    r"""First evaluation kernel for the pusher kernel `push_gc_bxEstar_discrete_Itoh_Newton <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_Itoh_Newton>`_ .
    TODO: better name than eval1
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 11] == -1.:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e1, e2, e3,
                              args_domain,
                              dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # abs_b; 0form
        markers[ip, 22] = eval_0form_spline_mpi(span1, span2, span3,
                                                args_derham,
                                                abs_b)

        # grad_abs_b; 1form
        markers[ip, 23] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            args_derham.pn[0] - 1,
            args_derham.pn[1],
            args_derham.pn[2],
            args_derham.bd1,
            args_derham.bn2,
            args_derham.bn3,
            span1, span2, span3,
            grad_abs_b1,
            args_derham.starts)

        # send particles to the (eta^0_n+1, eta^0_n+1, eta_n)
        markers[ip, 1] = markers[ip, 19]


@stack_array('dfm')
def gc_bxEstar_discrete_gradient_Itoh_Newton_eval2(markers: 'float[:,:]',
                                                   dt: float,
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
    r"""Second evaluation kernel for the pusher kernel `push_gc_bxEstar_discrete_Itoh_Newton <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_Itoh_Newton>`_ .
    TODO: better name than eval2
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 11] == -1.:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]

        # send particles to the (eta^0_n+1, eta^0_n+1, eta^0_n+1)
        markers[ip, 2] = markers[ip, 20]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e1, e2, e3,
                              args_domain,
                              dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # abs_b; 0form
        markers[ip, 18] = eval_0form_spline_mpi(span1, span2, span3,
                                                args_derham,
                                                abs_b)

        # grad_abs_b; 1form
        markers[ip, 19] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            args_derham.pn[0] - 1,
            args_derham.pn[1],
            args_derham.pn[2],
            args_derham.bd1,
            args_derham.bn2,
            args_derham.bn3,
            span1, span2, span3,
            grad_abs_b1,
            args_derham.starts)
        markers[ip, 20] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            args_derham.pn[0],
            args_derham.pn[1] - 1,
            args_derham.pn[2],
            args_derham.bn1,
            args_derham.bd2,
            args_derham.bn3,
            span1, span2, span3,
            grad_abs_b2,
            args_derham.starts)


@stack_array('dfm')
def gc_Bstar_discrete_gradient_Itoh_Newton_eval1(markers: 'float[:,:]',
                                                 dt: float,
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
    r"""First evaluation kernel for the pusher kernel `push_gc_Bstar_discrete_Itoh_Newton <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_Itoh_Newton>`_ .
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 11] == -1.:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e1, e2, e3,
                              args_domain,
                              dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # abs_b; 0form
        markers[ip, 22] = eval_0form_spline_mpi(span1, span2, span3,
                                                args_derham,
                                                abs_b)

        # grad_abs_b; 1form
        markers[ip, 23] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            args_derham.pn[0] - 1,
            args_derham.pn[1],
            args_derham.pn[2],
            args_derham.bd1,
            args_derham.bn2,
            args_derham.bn3,
            span1, span2, span3,
            grad_abs_b1,
            args_derham.starts)

        # send particles to the (eta^0_n+1,eta^0_n+1, eta_n)
        markers[ip, 1] = markers[ip, 19]


@stack_array('dfm')
def gc_Bstar_discrete_gradient_Itoh_Newton_eval2(markers: 'float[:,:]',
                                                 dt: float,
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
    r"""Second evaluation kernel for the pusher kernel `push_gc_Bstar_discrete_Itoh_Newton <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_Itoh_Newton>`_ .
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 11] == -1.:
            continue

        e1 = markers[ip, 0]
        e2 = markers[ip, 1]
        e3 = markers[ip, 2]

        # send particles to the (eta^0_n+1,eta^0_n+1, eta^0_n+1)
        markers[ip, 2] = markers[ip, 20]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e1, e2, e3,
                              args_domain,
                              dfm)

        # spline evaluation
        span1, span2, span3 = get_spans(e1, e2, e3, args_derham)

        # abs_b; 0form
        markers[ip, 18] = eval_0form_spline_mpi(span1, span2, span3,
                                                args_derham,
                                                abs_b)

        # grad_abs_b; 1form
        markers[ip, 19] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            args_derham.pn[0] - 1,
            args_derham.pn[1],
            args_derham.pn[2],
            args_derham.bd1,
            args_derham.bn2,
            args_derham.bn3,
            span1, span2, span3,
            grad_abs_b1,
            args_derham.starts)
        markers[ip, 20] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            args_derham.pn[0],
            args_derham.pn[1] - 1,
            args_derham.pn[2],
            args_derham.bn1,
            args_derham.bd2,
            args_derham.bn3,
            span1, span2, span3,
            grad_abs_b2,
            args_derham.starts)
