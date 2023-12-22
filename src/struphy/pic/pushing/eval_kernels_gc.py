'Initialization routines (initial guess, evaluations) for 5D gyro-center pusher kernels.'


from pyccel.decorators import stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d
import struphy.linear_algebra.linalg_kernels as linalg_kernels
import struphy.geometry.evaluation_kernels as evaluation_kernels

from numpy import empty, shape, zeros, sqrt, log, abs


@stack_array('dfm', 'dfinv', 'g', 'g_inv', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'S', 'b', 'b_star', 'bcross', 'grad_abs_b', 'curl_norm_b', 'norm_b1', 'norm_b2', 'temp', 'temp1', 'temp2')
def init_gc_bxEstar_discrete_gradient(markers: 'float[:,:]', dt: float,
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
    r"""Initialization kernel for the pusher kernel `push_gc_bxEstar_discrete_gradient <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_gradient>`_ .
    """

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

    # marker position e
    e = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
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

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts)

        # save for later steps
        markers[ip, 19] = abs_b0*mu

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
        b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # grad_abs_b; 1form
        grad_abs_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)
        grad_abs_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts)
        grad_abs_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts)

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

        markers[ip, 16:19] = markers[ip, 0:3]
        markers[ip, 0:3] = (markers[ip, 0:3] + markers[ip, 9:12])/2.


@stack_array('dfm', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'b', 'grad_abs_b', 'curl_norm_b', 'b_star', 'norm_b1')
def init_gc_Bstar_discrete_gradient(markers: 'float[:,:]', dt: float,
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
    r"""Initialization kernel for the pusher kernel `push_gc_Bstar_discrete_gradient <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_gradient>`_ .
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers
    b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)

    # marker position e
    e = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

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

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts)

        # save for later steps
        markers[ip, 16] = mu*abs_b0

        # norm_b1; 1form
        norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
        norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
        norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

        # b; 2form
        b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # grad_abs_b; 1form
        grad_abs_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)
        grad_abs_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts)
        grad_abs_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts)

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
        markers[ip, 3] = markers[ip, 12] - dt * \
            b_star_dot_grad_abs_b/abs_b_star_para

        markers[ip, 17:21] = markers[ip, 0:4]
        markers[ip, 0:4] = (markers[ip, 0:4] + markers[ip, 9:13])/2.


@stack_array('dfm', 'dfinv', 'g', 'g_inv', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'S', 'b', 'b_star', 'bcross', 'grad_abs_b', 'curl_norm_b', 'norm_b1', 'norm_b2', 'temp', 'temp1', 'temp2')
def init_gc_bxEstar_discrete_gradient_faster(markers: 'float[:,:]', dt: float,
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
    r"""Initialization kernel for the pusher kernel `push_gc_bxEstar_discrete_gradient_faster <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_faster>`_ .
    """

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

    # marker position e
    e = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
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

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts)

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
        b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # grad_abs_b; 1form
        grad_abs_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)
        grad_abs_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts)
        grad_abs_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts)

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
        markers[ip, 13:15] = S[0, 1:3]
        markers[ip, 15] = S[1, 2]
        markers[ip, 19] = abs_b0*mu

        # calculate S1 * grad I1
        linalg_kernels.matrix_vector(S, grad_abs_b, temp)

        # save at the markers
        markers[ip, 0:3] = markers[ip, 0:3] + dt*temp[:]*mu

        markers[ip, 16:19] = markers[ip, 0:3]
        markers[ip, 0:3] = (markers[ip, 0:3] + markers[ip, 9:12])/2.


@stack_array('dfm', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'b', 'grad_abs_b', 'curl_norm_b', 'b_star', 'norm_b1')
def init_gc_Bstar_discrete_gradient_faster(markers: 'float[:,:]', dt: float,
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
    r"""Initialization kernel for the pusher kernel `push_gc_Bstar_discrete_gradient_faster <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_gradient_faster>`_ .
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers
    b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)

    # marker position e
    e = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

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

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts)

        # norm_b1; 1form
        norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
        norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
        norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

        # b; 2form
        b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # grad_abs_b; 1form
        grad_abs_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)
        grad_abs_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts)
        grad_abs_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts)

        # transform to H1vec
        b_star[:] = b + epsilon*v*curl_norm_b
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # save at the markers
        markers[ip, 13:16] = b_star[:]/abs_b_star_para
        markers[ip, 16] = mu*abs_b0

        # calculate b_star . grad_abs_b
        b_star_dot_grad_abs_b = linalg_kernels.scalar_dot(
            b_star, grad_abs_b)*mu

        # save at the markers
        markers[ip, 0:3] = markers[ip, 0:3] + dt*b_star[:]/abs_b_star_para*v
        markers[ip, 3] = markers[ip, 12] - dt * \
            b_star_dot_grad_abs_b/abs_b_star_para

        markers[ip, 17:21] = markers[ip, 0:4]
        markers[ip, 0:4] = (markers[ip, 0:4] + markers[ip, 9:13])/2.


@stack_array('dfm', 'dfinv', 'g', 'g_inv', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'S', 'b', 'b_star', 'bcross', 'grad_abs_b', 'curl_norm_b', 'norm_b1', 'norm_b2', 'temp', 'temp1', 'temp2')
def init_gc_bxEstar_discrete_gradient_Itoh_Newton(markers: 'float[:,:]', dt: float,
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
    r"""Initialization kernel for the pusher kernel `push_gc_bxEstar_discrete_Itoh_Newton <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_Itoh_Newton>`_ .
    """

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

    # marker position e
    e = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
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

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts)

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
        b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # grad_abs_b; 1form
        grad_abs_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)
        grad_abs_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts)
        grad_abs_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts)

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
        markers[ip, 13:15] = S[0, 1:3]
        markers[ip, 15] = S[1, 2]
        markers[ip, 19] = abs_b0

        # calculate S1 * grad I1
        linalg_kernels.matrix_vector(S, grad_abs_b, temp)

        # save at the markers
        markers[ip, 16:19] = markers[ip, 0:3] + dt*temp[:]*mu

        # send particles to the (eta^0_n+1, eta_n, eta_n)
        markers[ip, 0] = markers[ip, 16]


@stack_array('dfm', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'b', 'grad_abs_b', 'curl_norm_b', 'b_star', 'norm_b1')
def init_gc_Bstar_discrete_gradient_Itoh_Newton(markers: 'float[:,:]', dt: float,
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
    r"""Initialization kernel for the pusher kernel `push_gc_Bstar_discrete_Itoh_Newton <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_Itoh_Newton>`_ .
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers
    b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    grad_abs_b = empty(3, dtype=float)

    # marker position e
    e = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

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

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts)

        # norm_b1; 1form
        norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
        norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
        norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

        # b; 2form
        b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # grad_abs_b; 1form
        grad_abs_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)
        grad_abs_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts)
        grad_abs_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts)

        # transform to H1vec
        b_star[:] = b + epsilon*v*curl_norm_b
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # save at the markers
        markers[ip, 13:16] = b_star[:]/abs_b_star_para
        markers[ip, 19] = abs_b0

        # calculate b_star . grad_abs_b
        b_star_dot_grad_abs_b = linalg_kernels.scalar_dot(
            b_star, grad_abs_b)*mu

        # save at the markers
        markers[ip, 16:19] = markers[ip, 0:3] + dt*b_star[:]/abs_b_star_para*v
        markers[ip, 3] = markers[ip, 12] - dt * \
            b_star_dot_grad_abs_b/abs_b_star_para

        # send particles to the (eta^0_n+1,eta_n, eta_n)
        markers[ip, 0] = markers[ip, 16]


@stack_array('dfm', 'dfinv', 'g', 'g_inv', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e', 'S', 'b', 'b_star', 'bcross', 'grad_abs_b', 'curl_norm_b', 'norm_b1', 'norm_b2', 'temp1', 'temp2')
def gc_bxEstar_discrete_gradient_eval_gradI(markers: 'float[:,:]', dt: float,
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
    r"""Evaluation kernel for the pusher kernel `push_gc_bxEstar_discrete_gradient <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_gradient>`_ .
    """

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

    # marker position e
    e_mid = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 9] == -1.:
            continue

        e_mid[:] = markers[ip, 0:3]
        v = markers[ip, 3]
        markers[ip, 0:3] = markers[ip, 16:19]
        mu = markers[ip, 4]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e_mid[0], e_mid[1], e_mid[2],
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
        span1 = bsplines_kernels.find_span(tn1, pn[0], e_mid[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e_mid[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e_mid[2])

        bsplines_kernels.b_d_splines_slim(
            tn1, pn[0], e_mid[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(
            tn2, pn[1], e_mid[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(
            tn3, pn[2], e_mid[2], span3, bn3, bd3)

        # eval all the needed field
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
        b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # grad_abs_b; 1form
        grad_abs_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)
        grad_abs_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts)
        grad_abs_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts)

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
        markers[ip, 13:15] = S[0, 1:3]
        markers[ip, 15] = S[1, 2]

        markers[ip, 16:19] = mu*grad_abs_b[:]


@stack_array('dfm', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e_mid', 'b', 'grad_abs_b', 'curl_norm_b', 'b_star', 'norm_b1')
def gc_Bstar_discrete_gradient_eval_gradI(markers: 'float[:,:]', dt: float,
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
    r"""Evaluation kernel for the pusher kernel `push_gc_Bstar_discrete_gradient <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_gradient>`_ .
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers
    grad_abs_b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    b = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

    # marker position e
    e_mid = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 9] == -1.:
            continue

        e_mid[:] = markers[ip, 0:3]
        v_mid = markers[ip, 3]
        markers[ip, 0:4] = markers[ip, 17:21]
        mu = markers[ip, 4]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e_mid[0], e_mid[1], e_mid[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e_mid[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e_mid[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e_mid[2])

        bsplines_kernels.b_d_splines_slim(
            tn1, pn[0], e_mid[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(
            tn2, pn[1], e_mid[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(
            tn3, pn[2], e_mid[2], span3, bn3, bd3)

        # eval all the needed field
        # norm_b1; 1form
        norm_b1[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts)
        norm_b1[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts)
        norm_b1[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts)

        # b; 2form
        b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts)
        b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts)
        b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts)

        # curl_norm_b; 2form
        curl_norm_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts)
        curl_norm_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts)
        curl_norm_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts)

        # grad_abs_b; 1form
        grad_abs_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)
        grad_abs_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts)
        grad_abs_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts)

        # transform to H1vec
        b_star[:] = b + epsilon*v_mid*curl_norm_b
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg_kernels.scalar_dot(norm_b1, b_star)

        # save at the markers
        markers[ip, 13:16] = b_star[:]/abs_b_star_para
        markers[ip, 17:20] = mu*grad_abs_b[:]


@stack_array('dfm', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e_mid', 'grad_abs_b')
def gc_bxEstar_discrete_gradient_faster_eval_gradI(markers: 'float[:,:]', dt: float,
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
    r"""Evaluation kernel for the pusher kernel `push_gc_bxEstar_discrete_gradient_faster <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_gradient_faster>`_ .
    """

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers
    grad_abs_b = empty(3, dtype=float)

    # marker position e
    e_mid = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 9] == -1.:
            continue

        e_mid[:] = markers[ip, 0:3]
        markers[ip, 0:3] = markers[ip, 16:19]
        mu = markers[ip, 4]

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e_mid[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e_mid[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e_mid[2])

        bsplines_kernels.b_d_splines_slim(
            tn1, pn[0], e_mid[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(
            tn2, pn[1], e_mid[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(
            tn3, pn[2], e_mid[2], span3, bn3, bd3)

        # eval all the needed field
        # grad_abs_b; 1form
        grad_abs_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)
        grad_abs_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts)
        grad_abs_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts)

        markers[ip, 16:19] = mu*grad_abs_b[:]


@stack_array('dfm', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e_mid', 'grad_abs_b')
def gc_Bstar_discrete_gradient_faster_eval_gradI(markers: 'float[:,:]', dt: float,
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
    r"""Evaluation kernel for the pusher kernel `push_gc_Bstar_discrete_gradient_faster <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_gradient_faster>`_ .
    """

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers
    grad_abs_b = empty(3, dtype=float)

    # marker position e
    e_mid = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 9] == -1.:
            continue

        e_mid[:] = markers[ip, 0:3]
        markers[ip, 0:4] = markers[ip, 17:21]
        mu = markers[ip, 4]

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e_mid[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e_mid[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e_mid[2])

        bsplines_kernels.b_d_splines_slim(
            tn1, pn[0], e_mid[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(
            tn2, pn[1], e_mid[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(
            tn3, pn[2], e_mid[2], span3, bn3, bd3)

        # eval all the needed field
        # grad_abs_b; 1form
        grad_abs_b[0] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)
        grad_abs_b[1] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts)
        grad_abs_b[2] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts)

        markers[ip, 17:20] = mu*grad_abs_b[:]


@stack_array('dfm', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e')
def gc_bxEstar_discrete_gradient_Itoh_Newton_eval1(markers: 'float[:,:]', dt: float,
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
    r"""First evaluation kernel for the pusher kernel `push_gc_bxEstar_discrete_Itoh_Newton <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_Itoh_Newton>`_ .
    TODO: better name than eval1
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # marker position e
    e = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 9] == -1.:
            continue

        e[:] = markers[ip, 0:3]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e[0], e[1], e[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        markers[ip, 20] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts)

        # grad_abs_b; 1form
        markers[ip, 21] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)

        # send particles to the (eta^0_n+1, eta^0_n+1, eta_n)
        markers[ip, 1] = markers[ip, 17]


@stack_array('dfm', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e')
def gc_bxEstar_discrete_gradient_Itoh_Newton_eval2(markers: 'float[:,:]', dt: float,
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
    r"""Second evaluation kernel for the pusher kernel `push_gc_bxEstar_discrete_Itoh_Newton <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_bxEstar_discrete_Itoh_Newton>`_ .
    TODO: better name than eval2
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # marker position e
    e = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 9] == -1.:
            continue

        e[:] = markers[ip, 0:3]

        # send particles to the (eta^0_n+1, eta^0_n+1, eta^0_n+1)
        markers[ip, 2] = markers[ip, 18]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e[0], e[1], e[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        markers[ip, 16] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts)

        # grad_abs_b; 1form
        markers[ip, 17] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)
        markers[ip, 18] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts)


@stack_array('dfm', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e')
def gc_Bstar_discrete_gradient_Itoh_Newton_eval1(markers: 'float[:,:]', dt: float,
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
    r"""First evaluation kernel for the pusher kernel `push_gc_Bstar_discrete_Itoh_Newton <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_Itoh_Newton>`_ .
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # marker position e
    e = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 9] == -1.:
            continue

        e[:] = markers[ip, 0:3]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e[0], e[1], e[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        markers[ip, 20] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts)

        # grad_abs_b; 1form
        markers[ip, 21] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)

        # send particles to the (eta^0_n+1,eta^0_n+1, eta_n)
        markers[ip, 1] = markers[ip, 17]


@stack_array('dfm', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'e')
def gc_Bstar_discrete_gradient_Itoh_Newton_eval2(markers: 'float[:,:]', dt: float,
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
    r"""Second evaluation kernel for the pusher kernel `push_gc_Bstar_discrete_Itoh_Newton <https://struphy.pages.mpcdf.de/struphy/sections/propagators.html#struphy.pic.pushing.pusher_kernels_gc.push_gc_Bstar_discrete_Itoh_Newton>`_ .
    """

    # allocate metric coeffs
    dfm = empty((3, 3), dtype=float)

    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # marker position e
    e = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 9] == -1.:
            continue

        e[:] = markers[ip, 0:3]

        # send particles to the (eta^0_n+1,eta^0_n+1, eta^0_n+1)
        markers[ip, 2] = markers[ip, 18]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(e[0], e[1], e[2],
                              kind_map, params_map,
                              t1_map, t2_map, t3_map, p_map,
                              ind1_map, ind2_map, ind3_map,
                              cx, cy, cz,
                              dfm)

        # spline evaluation
        span1 = bsplines_kernels.find_span(tn1, pn[0], e[0])
        span2 = bsplines_kernels.find_span(tn2, pn[1], e[1])
        span3 = bsplines_kernels.find_span(tn3, pn[2], e[2])

        bsplines_kernels.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsplines_kernels.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsplines_kernels.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        markers[ip, 16] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts)

        # grad_abs_b; 1form
        markers[ip, 17] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts)
        markers[ip, 18] = evaluation_kernels_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts)
