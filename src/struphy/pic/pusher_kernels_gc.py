from pyccel.decorators import stack_array

import struphy.linear_algebra.core as linalg
import struphy.geometry.map_eval as map_eval
import struphy.b_splines.bsplines_kernels as bsp
import struphy.b_splines.bspline_evaluation_3d as eval_3d

from numpy import zeros, empty, shape, sqrt


def push_gc1_explicit_stage(markers: 'float[:,:]', dt: float, stage: int,
                            pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                            starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                            kind_map: int, params_map: 'float[:]',
                            p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                            ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                            cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                            kappa: float,
                            b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                            norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                            norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                            curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                            grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]',
                            a: 'float[:]', b: 'float[:]', c: 'float[:]'):
    r'''Single stage of a s-stage Runge-Kutta solve of 

    .. math::

        \dot{\mathbf X} &= \frac{\mu}{\kappa B^*_\parallel}  G^{-1}(\eta_p(t)) \hat{\mathbb{b}}^2_0 \times G^{-1}(\eta_p(t)) \hat \nabla |\mathcal{P}^B \hat{\mathbb{B}}^2| \,,

        \dot v_\parallel &= 0 \,.

    for each marker :math:`p` in markers array.
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
    bb = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

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
        grad_abs_b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # norm_b1; 1form
        norm_b1[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # norm_b2; 2form
        norm_b2[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts2[0])
        norm_b2[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts2[1])
        norm_b2[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts2[2])

        # b; 2form
        bb[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        bb[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        bb[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

        # transform to H1vec
        b_star[:] = bb + 1/kappa*v*curl_norm_b
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # calculate norm_b X grad_abs_b
        linalg.matrix_vector(g_inv, grad_abs_b, temp1)

        linalg.cross(norm_b2, temp1, temp2)

        linalg.matrix_vector(g_inv, temp2, temp1)

        # calculate k
        k[:] = 1/kappa*mu/abs_b_star_para*temp1

        # accumulation for last stage
        markers[ip, 13:16] += dt*b[stage]*k

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = markers[ip, 9:12] + \
            dt*a[stage]*k + last*markers[ip, 13:16]


def push_gc2_explicit_stage(markers: 'float[:,:]', dt: float, stage: int,
                            pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                            starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                            kind_map: int, params_map: 'float[:]',
                            p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                            ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                            cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                            kappa: float,
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
    bb = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

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
        grad_abs_b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # norm_b1; 1form
        norm_b1[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # b; 2form
        bb[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        bb[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        bb[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

        # transform to H1vec
        b_star[:] = bb + 1/kappa*v*curl_norm_b
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
        markers[ip, 0:3] = markers[ip, 9:12] + \
            dt*a[stage]*k + last*markers[ip, 13:16]
        markers[ip, 3] = markers[ip, 12] + dt * \
            a[stage]*k_v + last*markers[ip, 16]


def push_gc_explicit_stage(markers: 'float[:,:]', dt: float, stage: int,
                           pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                           starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                           kind_map: int, params_map: 'float[:]',
                           p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                           ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                           cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                           kappa: float,
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
    bb = empty(3, dtype=float)
    curl_norm_b = empty(3, dtype=float)

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
        grad_abs_b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # norm_b1; 1form
        norm_b1[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # norm_b2; 2form
        norm_b2[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts2[0])
        norm_b2[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts2[1])
        norm_b2[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts2[2])

        # b; 2form
        bb[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        bb[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        bb[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

        # transform to H1vec
        b_star[:] = bb + 1/kappa*v*curl_norm_b
        b_star[:] = b_star/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # calculate norm_b X grad_abs_b
        linalg.matrix_vector(g_inv, grad_abs_b, temp1)

        linalg.cross(norm_b2, temp1, temp2)

        linalg.matrix_vector(g_inv, temp2, temp3)

        # calculate k
        k[:] = (1/kappa*mu*temp3 + b_star*v)/abs_b_star_para

        # calculate k_v for v
        temp = linalg.scalar_dot(b_star, grad_abs_b)

        k_v = -1*mu/abs_b_star_para*temp

        # accumulation for last stage
        markers[ip, 13:16] += dt*b[stage]*k
        markers[ip, 16] += dt*b[stage]*k_v

        # update positions for intermediate stages or last stage
        markers[ip, 0:3] = markers[ip, 9:12] + \
            dt*a[stage]*k + last*markers[ip, 13:16]
        markers[ip, 3] = markers[ip, 12] + dt * \
            a[stage]*k_v + last*markers[ip, 16]


def push_gc1_discrete_gradients(markers: 'float[:,:]', dt: float, stage: int, tol: float,
                                domain_array: 'float[:]',
                                pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                                kind_map: int, params_map: 'float[:]',
                                p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                kappa: float,
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

        if markers[ip, 21] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_diff[:] = e[:] - markers[ip, 9:12]
        mu = markers[ip, 4]

        if abs(e_diff[0]/e[0]) < tol and abs(e_diff[1]/e[1]) < tol and abs(e_diff[2]/e[2]) < tol:
            markers[ip, 21] = -1.
            markers[ip, 20] = stage

            continue

        # TODO: replace with better idea
        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts0)

        # assemble S
        S[0, 1] = markers[ip, 13]
        S[0, 2] = markers[ip, 14]
        S[1, 0] = markers[ip, 13]
        S[1, 2] = markers[ip, 15]
        S[2, 0] = -markers[ip, 14]
        S[2, 1] = -markers[ip, 15]

        # calculate grad_I
        temp_scalar = linalg.scalar_dot(e_diff[:], markers[ip, 16:19])
        temp_scalar2 = e_diff[0]**2 + e_diff[1]**2 + e_diff[2]**2

        grad_I[:] = markers[ip, 16:19] + e_diff[:] * \
            (abs_b0*mu - markers[ip, 19] - temp_scalar)/temp_scalar2

        linalg.matrix_vector(S, grad_I, temp)

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
            markers[ip, 21] = -1.
            markers[ip, 20] = stage

            continue

        markers[ip, 0:3] = (markers[ip, 0:3] + markers[ip, 9:12])/2.


def push_gc2_discrete_gradients(markers: 'float[:,:]', dt: float, stage: int, tol: float,
                                domain_array: 'float[:]',
                                pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                                kind_map: int, params_map: 'float[:]',
                                p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                kappa: float,
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

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 21] == -1.:
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
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts0)

        # calculate grad_I
        temp_scalar = linalg.scalar_dot(e_diff[:], markers[ip, 17:20])
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

        temp_scalar3 = linalg.scalar_dot(markers[ip, 13:16], grad_I)

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
            markers[ip, 21] = -1.
            markers[ip, 20] = stage
            continue

        markers[ip, 0:4] = (markers[ip, 0:4] + markers[ip, 9:13])/2.


def push_gc1_discrete_gradients_faster(markers: 'float[:,:]', dt: float, stage: int, tol: float,
                                       domain_array: 'float[:]',
                                       pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                       starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                                       kind_map: int, params_map: 'float[:]',
                                       p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                       ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                       cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                       kappa: float,
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
    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    temp = empty(3, dtype=float)
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

        if markers[ip, 21] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_diff[:] = e[:] - markers[ip, 9:12]
        mu = markers[ip, 4]

        if abs(e_diff[0]/e[0]) < tol and abs(e_diff[1]/e[1]) < tol and abs(e_diff[2]/e[2]) < tol:
            markers[ip, 21] = -1.
            markers[ip, 20] = stage

            continue

        # TODO: replace with better idea
        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts0)

        # assemble S
        S[0, 1] = markers[ip, 13]
        S[0, 2] = markers[ip, 14]
        S[1, 0] = markers[ip, 13]
        S[1, 2] = markers[ip, 15]
        S[2, 0] = -markers[ip, 14]
        S[2, 1] = -markers[ip, 15]

        # calculate grad_I
        temp_scalar = linalg.scalar_dot(e_diff[:], markers[ip, 16:19])
        temp_scalar2 = e_diff[0]**2 + e_diff[1]**2 + e_diff[2]**2

        grad_I[:] = markers[ip, 16:19] + e_diff[:] * \
            (abs_b0*mu - markers[ip, 19] - temp_scalar)/temp_scalar2

        linalg.matrix_vector(S, grad_I, temp)

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
            markers[ip, 21] = -1.
            markers[ip, 20] = stage

            continue

        markers[ip, 0:3] = (markers[ip, 0:3] + markers[ip, 9:12])/2.


def push_gc2_discrete_gradients_faster(markers: 'float[:,:]', dt: float, stage: int, tol: float,
                                       domain_array: 'float[:]',
                                       pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                       starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                                       kind_map: int, params_map: 'float[:]',
                                       p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                       ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                       cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                       kappa: float,
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

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        if markers[ip, 21] == -1.:
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
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts0)

        # calculate grad_I
        temp_scalar = linalg.scalar_dot(e_diff[:], markers[ip, 17:20])
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

        temp_scalar3 = linalg.scalar_dot(markers[ip, 13:16], grad_I)

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
            markers[ip, 21] = -1.
            markers[ip, 20] = stage
            continue

        markers[ip, 0:4] = (markers[ip, 0:4] + markers[ip, 9:13])/2.


def push_gc1_discrete_gradients_Itoh_Newton(markers: 'float[:,:]', dt: float, stage: int, max_iter: int, tol: float,
                                            domain_array: 'float[:]',
                                            pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                            starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                                            kind_map: int, params_map: 'float[:]',
                                            p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                            ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                            cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                            kappa: float,
                                            abs_b: 'float[:,:,:]',
                                            b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                                            norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                                            norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                                            curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                                            grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]'):
    r'''
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
    Jacobian_grad_I = empty((3, 3), dtype=float)
    Jacobian = empty((3, 3), dtype=float)
    Jacobian_inv = empty((3, 3), dtype=float)

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

        if markers[ip, 23] == -1.:
            continue

        e[:] = markers[ip, 0:3]
        e_old[:] = markers[ip, 9:12]
        mu = markers[ip, 4]

        e_diff[:] = e[:] - e_old[:]

        if e_diff[0] == 0. and e_diff[1] == 0. and e_diff[2] == 0:
            markers[ip, 23] = -1.
            markers[ip, 14] = stage

            continue

        for axis in range(3):
            if e_diff[axis] > 0.5:
                e_diff[axis] -= 1.
            elif e_diff[axis] < -0.5:
                e_diff[axis] += 1.

        # assemble S
        S[0, 1] = markers[ip, 13]
        S[0, 2] = markers[ip, 14]
        S[1, 0] = markers[ip, 13]
        S[1, 2] = markers[ip, 15]
        S[2, 0] = -markers[ip, 14]
        S[2, 1] = -markers[ip, 15]

        # identity matrix
        identity[0, 0] = 1.
        identity[1, 1] = 1.
        identity[2, 2] = 1.

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts0)

        # grad_abs_b; 1form
        grad_abs_b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # assemble gradI
        grad_I[0] = mu*(markers[ip, 20] - markers[ip, 19])/(e_diff[0])
        grad_I[1] = mu*(markers[ip, 22] - markers[ip, 20])/(e_diff[1])
        grad_I[2] = mu*(abs_b0 - markers[ip, 22])/(e_diff[2])

        # calculate F = eta - eta_old + dt*S*grad_I
        linalg.matrix_vector(S, grad_I, F)
        F *= -dt
        F += e_diff[:]

        # assemble Jacobian_grad_I
        Jacobian_grad_I[0, 0] = mu*(markers[ip, 21]*(e_diff[0]) -
                                    markers[ip, 20] + markers[ip, 19])/(e_diff[0])**2
        Jacobian_grad_I[1, 0] = mu * \
            (markers[ip, 23] - markers[ip, 21])/(e_diff[1])
        Jacobian_grad_I[2, 0] = mu * \
            (grad_abs_b[0] - markers[ip, 23])/(e_diff[2])
        Jacobian_grad_I[0, 1] = 0.
        Jacobian_grad_I[1, 1] = mu*(markers[ip, 24]*(e_diff[1]) -
                                    markers[ip, 22] + markers[ip, 20])/(e_diff[1])**2
        Jacobian_grad_I[2, 1] = mu * \
            (grad_abs_b[1] - markers[ip, 24])/(e_diff[2])
        Jacobian_grad_I[0, 2] = 0.
        Jacobian_grad_I[1, 2] = 0.
        Jacobian_grad_I[2, 2] = mu*(grad_abs_b[2]*(e_diff[2]) -
                                    abs_b0 + markers[ip, 22])/(e_diff[2])**2

        # assemble Jacobian and its inverse
        linalg.matrix_matrix(S, Jacobian_grad_I, Jacobian)
        Jacobian *= dt
        Jacobian += identity

        linalg.matrix_inv(Jacobian, Jacobian_inv)

        # calculate eta_new
        linalg.matrix_vector(Jacobian_inv, F, temp)
        markers[ip, 16:19] = e[:] - temp

        diff = sqrt((temp[0]/e[0])**2 + (temp[1]/e[1])**2 + (temp[2]/e[2])**2)

        if diff < tol:
            markers[ip, 23] = -1.
            markers[ip, 14] = stage
            markers[ip, 0:3] = markers[ip, 16:19]

            continue

        if stage == max_iter-1:
            markers[ip, 0:3] = markers[ip, 16:19]

            continue

        markers[ip, 0] = markers[ip, 16]
        markers[ip, 1] = e_old[1]
        markers[ip, 2] = e_old[2]


def push_gc2_discrete_gradients_Itoh_Newton(markers: 'float[:,:]', dt: float, stage: int, max_iter: int, tol: float,
                                            domain_array: 'float[:]',
                                            pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                            starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                                            kind_map: int, params_map: 'float[:]',
                                            p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                            ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                            cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                            kappa: float,
                                            abs_b: 'float[:,:,:]',
                                            b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                                            norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                                            norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                                            curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                                            grad_abs_b1: 'float[:,:,:]', grad_abs_b2: 'float[:,:,:]', grad_abs_b3: 'float[:,:,:]'):
    r'''
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
    Jacobian = empty((4, 4), dtype=float)
    Jacobian_inv = empty((4, 4), dtype=float)
    Jacobian_temp34 = empty((3, 4), dtype=float)
    Jacobian_temp33 = empty((3, 3), dtype=float)

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

        if markers[ip, 23] == -1.:
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
        span1 = bsp.find_span(tn1, pn[0], e[0])
        span2 = bsp.find_span(tn2, pn[1], e[1])
        span3 = bsp.find_span(tn3, pn[2], e[2])

        bsp.b_d_splines_slim(tn1, pn[0], e[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], e[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], e[2], span3, bn3, bd3)

        # eval all the needed field
        # abs_b; 0form
        abs_b0 = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, abs_b, starts0)

        # grad_abs_b; 1form
        grad_abs_b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, grad_abs_b1, starts1[0])
        grad_abs_b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, grad_abs_b2, starts1[1])
        grad_abs_b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, grad_abs_b3, starts1[2])

        # assemble gradI and Jacobian_grad_I
        if e_diff[0] == 0.:
            grad_I[0] == 0.
        else:
            grad_I[0] = mu*(markers[ip, 20] - markers[ip, 19])/(e_diff[0])
            Jacobian_grad_I[0, 0] = mu*(markers[ip, 21]*(e_diff[0]) -
                                        markers[ip, 20] + markers[ip, 19])/(e_diff[0])**2

        if e_diff[1] == 0.:
            grad_I[1] == 0.
        else:
            grad_I[1] = mu*(markers[ip, 22] - markers[ip, 20])/(e_diff[1])
            Jacobian_grad_I[1, 0] = mu * \
                (markers[ip, 20] - markers[ip, 21])/(e_diff[1])
            Jacobian_grad_I[1, 1] = mu*(markers[ip, 21]*(e_diff[1]) -
                                        markers[ip, 22] + markers[ip, 20])/(e_diff[1])**2

        if e_diff[2] == 0.:
            grad_I[2] == 0.
        else:
            grad_I[2] = mu*(abs_b0 - markers[ip, 22])/(e_diff[2])
            Jacobian_grad_I[2, 0] = mu * \
                (grad_abs_b[0] - markers[ip, 20])/(e_diff[2])
            Jacobian_grad_I[2, 1] = mu * \
                (grad_abs_b[1] - markers[ip, 21])/(e_diff[2])
            Jacobian_grad_I[2, 2] = mu*(grad_abs_b[2]*(e_diff[2]) -
                                        abs_b0 + markers[ip, 22])/(e_diff[2])**2

        grad_I[3] = v_mid
        Jacobian_grad_I[3, 3] = 0.5

        # calculate F = eta - eta_old + dt*S*grad_I
        linalg.matrix_vector4(S, grad_I, F)
        F *= -dt
        F[0:3] += e_diff[:]
        F[3] += v - v_old

        # assemble Jacobian and its inverse
        linalg.matrix_matrix4(S, Jacobian_grad_I, Jacobian)
        Jacobian *= -dt
        Jacobian += identity

        # Inverse of the Jacobian
        det_J = linalg.det4(Jacobian)

        Jacobian_inv[0, 0] = linalg.det(Jacobian[1:, 1:])/det_J
        Jacobian_inv[0, 1] = -linalg.det(Jacobian[(0, 2, 3), 1:])/det_J
        Jacobian_inv[0, 2] = linalg.det(Jacobian[(0, 1, 3), 1:])/det_J
        Jacobian_inv[0, 3] = -linalg.det(Jacobian[:3, 1:])/det_J

        Jacobian_inv[1, 0] = -linalg.det(Jacobian[1:, (0, 2, 3)])/det_J
        Jacobian_temp34 = Jacobian[(0, 2, 3), :]
        Jacobian_temp33 = Jacobian_temp34[:, (0, 2, 3)]
        Jacobian_inv[1, 1] = linalg.det(Jacobian_temp33)/det_J
        Jacobian_temp34 = Jacobian[(0, 1, 3), :]
        Jacobian_temp33 = Jacobian_temp34[:, (0, 2, 3)]
        Jacobian_inv[1, 2] = -linalg.det(Jacobian_temp33)/det_J
        Jacobian_inv[1, 3] = linalg.det(Jacobian[:3, (0, 2, 3)])/det_J

        Jacobian_inv[2, 0] = linalg.det(Jacobian[1:, (0, 1, 3)])/det_J
        Jacobian_temp34 = Jacobian[(0, 2, 3), :]
        Jacobian_temp33 = Jacobian_temp34[:, (0, 1, 3)]
        Jacobian_inv[2, 1] = -linalg.det(Jacobian_temp33)/det_J
        Jacobian_temp34 = Jacobian[(0, 1, 3), :]
        Jacobian_temp33 = Jacobian_temp34[:, (0, 1, 3)]
        Jacobian_inv[2, 2] = linalg.det(Jacobian_temp33)/det_J
        Jacobian_inv[2, 3] = -linalg.det(Jacobian[:3, (0, 1, 3)])/det_J

        Jacobian_inv[3, 0] = -linalg.det(Jacobian[1:, :3])/det_J
        Jacobian_inv[3, 1] = linalg.det(Jacobian[(0, 2, 3), :3])/det_J
        Jacobian_inv[3, 2] = -linalg.det(Jacobian[(0, 1, 3), :3])/det_J
        Jacobian_inv[3, 3] = linalg.det(Jacobian[:3, :3])/det_J

        # calculate eta_new
        linalg.matrix_vector4(Jacobian_inv, F, temp)
        markers[ip, 16:19] = e[:] - temp[0:3]
        markers[ip, 3] = v - temp[3]

        diff = sqrt((temp[0]/e[0])**2 + (temp[1]/e[1])**2 +
                    (temp[2]/e[2])**2 + (temp[3])**2)

        if diff < tol:
            markers[ip, 23] = -1.
            markers[ip, 14] = stage
            markers[ip, 0:3] = markers[ip, 16:19]

            continue

        if stage == max_iter-1:
            markers[ip, 0:3] = markers[ip, 16:19]

            continue

        markers[ip, 0] = markers[ip, 16]
        markers[ip, 1] = e_old[1]
        markers[ip, 2] = e_old[2]


def push_gc_cc_J1_H1vec(markers: 'float[:,:]', dt: float, stage: int,
                        pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                        starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                        kind_map: int, params_map: 'float[:]',
                        p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                        ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                        cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                        kappa: float,
                        b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                        norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                        curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                        u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]'):
    r'''
    TODO
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

        # evaluate Jacobian, result in df
        map_eval.df(eta[0], eta[1], eta[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta[0])
        span2 = bsp.find_span(tn2, pn[1], eta[1])
        span3 = bsp.find_span(tn3, pn[2], eta[2])

        bsp.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # eval all the needed field
        # b; 2form
        b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # u; 0form
        u[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u1, starts0)
        u[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u2, starts0)
        u[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u3, starts0)

        # norm_b1; 1form
        norm_b1[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

        # b_star; 2form in H1vec
        b_star[:] = (b + curl_norm_b*v/kappa)/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # electric field E(1) = B(2) X U(0)
        linalg.cross(b, u, e)

        # curl_norm_b dot electric field
        temp = linalg.scalar_dot(e, curl_norm_b) / det_df

        markers[ip, 3] += temp/abs_b_star_para*v*dt


def push_gc_cc_J1_Hcurl(markers: 'float[:,:]', dt: float, stage: int,
                        pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                        starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                        kind_map: int, params_map: 'float[:]',
                        p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                        ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                        cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                        kappa: float,
                        b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                        norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                        curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                        u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]'):
    r'''
    TODO
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

        # evaluate Jacobian, result in df
        map_eval.df(eta[0], eta[1], eta[2],
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
        span1 = bsp.find_span(tn1, pn[0], eta[0])
        span2 = bsp.find_span(tn2, pn[1], eta[1])
        span3 = bsp.find_span(tn3, pn[2], eta[2])

        bsp.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # eval all the needed field
        # b; 2form
        b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # u; 1form
        u[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u1, starts1[0])
        u[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u2, starts1[1])
        u[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, u3, starts1[2])

        # norm_b1; 1form
        norm_b1[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

        # b_star; 2form in H1vec
        b_star[:] = (b + curl_norm_b*v/kappa)/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # transform u into H1vec
        linalg.matrix_vector(g_inv, u, u0)

        # electric field E(1) = B(2) X U(0)
        linalg.cross(b, u0, e)

        # curl_norm_b dot electric field
        temp = linalg.scalar_dot(e, curl_norm_b) / det_df

        markers[ip, 3] += temp/abs_b_star_para*v*dt


def push_gc_cc_J1_Hdiv(markers: 'float[:,:]', dt: float, stage: int,
                       pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                       starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                       kind_map: int, params_map: 'float[:]',
                       p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                       ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                       cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                       kappa: float,
                       b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                       norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                       curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                       u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]'):
    r'''
    TODO
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

        # evaluate Jacobian, result in df
        map_eval.df(eta[0], eta[1], eta[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta[0])
        span2 = bsp.find_span(tn2, pn[1], eta[1])
        span3 = bsp.find_span(tn3, pn[2], eta[2])

        bsp.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # eval all the needed field
        # b; 2form
        b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # u; 2form
        u[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u1, starts2[0])
        u[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u2, starts2[1])
        u[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u3, starts2[2])

        # norm_b1; 1form
        norm_b1[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

        # b_star; 2form in H1vec
        b_star[:] = (b + curl_norm_b*v/kappa)/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        # transform u into H1vec
        u = u/det_df

        # electric field E(1) = B(2) X U(0)
        linalg.cross(b, u, e)

        # curl_norm_b dot electric field
        temp = linalg.scalar_dot(e, curl_norm_b) / det_df

        markers[ip, 3] += temp/abs_b_star_para*v*dt


def push_gc_cc_J2_dg_prepare_H1vec(markers: 'float[:,:]', dt: float, stage: int,
                                   pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                   starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                                   kind_map: int, params_map: 'float[:]',
                                   p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                   ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                   cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                   kappa: float,
                                   b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                                   norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                                   norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                                   curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                                   u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]'):
    r'''
    TODO
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
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
    norm_b2_prod = zeros((3, 3), dtype=float)
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
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

        # evaluate Jacobian, result in df
        map_eval.df(eta[0], eta[1], eta[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)
        linalg.matrix_inv_with_det(df, det_df, df_inv)
        linalg.transpose(df_inv, df_inv_t)
        linalg.matrix_matrix(df_inv, df_inv_t, g_inv)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta[0])
        span2 = bsp.find_span(tn2, pn[1], eta[1])
        span3 = bsp.find_span(tn3, pn[2], eta[2])

        bsp.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # eval all the needed field
        # b; 2form
        b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # u; 0form
        u[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u1, starts0)
        u[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u2, starts0)
        u[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u3, starts0)

        # norm_b1; 1form
        norm_b1[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # norm_b; 2form
        norm_b2[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts2[0])
        norm_b2[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts2[1])
        norm_b2[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts2[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        norm_b2_prod[0, 1] = -norm_b2[2]
        norm_b2_prod[0, 2] = +norm_b2[1]
        norm_b2_prod[1, 0] = +norm_b2[2]
        norm_b2_prod[1, 2] = -norm_b2[0]
        norm_b2_prod[2, 0] = -norm_b2[1]
        norm_b2_prod[2, 1] = +norm_b2[0]

        # b_star; 2form in H1vec
        b_star[:] = (b + curl_norm_b*v/kappa)/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        linalg.matrix_matrix(g_inv, norm_b2_prod, tmp1)
        linalg.matrix_matrix(tmp1, g_inv, tmp2)
        linalg.matrix_matrix(tmp2, b_prod, tmp1)

        linalg.matrix_vector(tmp1, u, e)

        markers[ip, 0:3] = markers[ip, 9:12] - e/abs_b_star_para*dt


def push_gc_cc_J2_dg_H1vec(markers: 'float[:,:]', dt: float, stage: int,
                           pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                           starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                           kind_map: int, params_map: 'float[:]',
                           p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                           ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                           cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                           kappa: float,
                           b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                           norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                           norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                           curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                           u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]'):
    r'''
    TODO
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
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
    norm_b2_prod = zeros((3, 3), dtype=float)
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)
    b = empty(3, dtype=float)
    b_star = empty(3, dtype=float)
    norm_b1 = empty(3, dtype=float)
    norm_b2 = empty(3, dtype=float)
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

        # evaluate Jacobian, result in df
        map_eval.df(eta[0], eta[1], eta[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)
        linalg.matrix_inv_with_det(df, det_df, df_inv)
        linalg.transpose(df_inv, df_inv_t)
        linalg.matrix_matrix(df_inv, df_inv_t, g_inv)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta[0])
        span2 = bsp.find_span(tn2, pn[1], eta[1])
        span3 = bsp.find_span(tn3, pn[2], eta[2])

        bsp.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # eval all the needed field
        # b; 2form
        b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # u; 0form
        u[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u1, starts0)
        u[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u2, starts0)
        u[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u3, starts0)

        # norm_b1; 1form
        norm_b1[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # norm_b; 2form
        norm_b2[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts2[0])
        norm_b2[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts2[1])
        norm_b2[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts2[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

        # operator bx() as matrix
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] = +b[1]
        b_prod[1, 0] = +b[2]
        b_prod[1, 2] = -b[0]
        b_prod[2, 0] = -b[1]
        b_prod[2, 1] = +b[0]

        norm_b2_prod[0, 1] = -norm_b2[2]
        norm_b2_prod[0, 2] = +norm_b2[1]
        norm_b2_prod[1, 0] = +norm_b2[2]
        norm_b2_prod[1, 2] = -norm_b2[0]
        norm_b2_prod[2, 0] = -norm_b2[1]
        norm_b2_prod[2, 1] = +norm_b2[0]

        # b_star; 2form in H1vec
        b_star[:] = (b + curl_norm_b*v/kappa)/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        linalg.matrix_matrix(g_inv, norm_b2_prod, tmp1)
        linalg.matrix_matrix(tmp1, g_inv, tmp2)
        linalg.matrix_matrix(tmp2, b_prod, tmp1)

        linalg.matrix_vector(tmp1, u, e)

        markers[ip, 0:3] = markers[ip, 9:12] - e/abs_b_star_para*dt


def push_gc_cc_J2_dg_faster_H1vec(markers: 'float[:,:]', dt: float, stage: int,
                                  pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                  starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                                  kind_map: int, params_map: 'float[:]',
                                  p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                  ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                  cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                  kappa: float,
                                  b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                                  norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                                  norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                                  curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                                  u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]'):
    r'''
    TODO
    '''
    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    tmp = empty((3, 3), dtype=float)
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)

    # marker position eta
    eta = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta[0])
        span2 = bsp.find_span(tn2, pn[1], eta[1])
        span3 = bsp.find_span(tn3, pn[2], eta[2])

        bsp.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # eval all the needed field
        # u; 0form
        u[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u1, starts0)
        u[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u2, starts0)
        u[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u3, starts0)

        tmp[:, :] = ((markers[ip, 18], markers[ip, 19], markers[ip, 20]),
                     (markers[ip, 19], markers[ip, 21], markers[ip, 22]),
                     (markers[ip, 20], markers[ip, 22], markers[ip, 23]))

        linalg.matrix_vector(tmp, u, e)

        markers[ip, 0:3] = markers[ip, 9:12] - e*dt


def push_gc_cc_J2_dg_faster_Hcurl(markers: 'float[:,:]', dt: float, stage: int,
                                  pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                  starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                                  kind_map: int, params_map: 'float[:]',
                                  p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                  ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                  cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                  kappa: float,
                                  b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                                  norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                                  norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                                  curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                                  u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]'):
    r'''
    TODO
    '''
    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    tmp = empty((3, 3), dtype=float)
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)

    # marker position eta
    eta = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta[0])
        span2 = bsp.find_span(tn2, pn[1], eta[1])
        span3 = bsp.find_span(tn3, pn[2], eta[2])

        bsp.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # eval all the needed field
        # u; 1form
        u[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, u1, starts1[0])
        u[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, u2, starts1[1])
        u[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, u3, starts1[2])

        tmp[:, :] = ((markers[ip, 18], markers[ip, 19], markers[ip, 20]),
                     (markers[ip, 19], markers[ip, 21], markers[ip, 22]),
                     (markers[ip, 20], markers[ip, 22], markers[ip, 23]))

        linalg.matrix_vector(tmp, u, e)

        markers[ip, 0:3] = markers[ip, 9:12] - e*dt


def push_gc_cc_J2_dg_faster_Hdiv(markers: 'float[:,:]', dt: float, stage: int,
                                 pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                                 starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                                 kind_map: int, params_map: 'float[:]',
                                 p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                                 ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                                 cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                                 kappa: float,
                                 b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                                 norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                                 norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                                 curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                                 u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]'):
    r'''
    TODO
    '''
    # allocate spline values
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    bd1 = empty(pn[0], dtype=float)
    bd2 = empty(pn[1], dtype=float)
    bd3 = empty(pn[2], dtype=float)

    # containers for fields
    tmp = empty((3, 3), dtype=float)
    e = empty(3, dtype=float)
    u = empty(3, dtype=float)

    # marker position eta
    eta = empty(3, dtype=float)

    # get number of markers
    n_markers = shape(markers)[0]

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta[:] = markers[ip, 0:3]

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta[0])
        span2 = bsp.find_span(tn2, pn[1], eta[1])
        span3 = bsp.find_span(tn3, pn[2], eta[2])

        bsp.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # eval all the needed field
        # u; 2form
        u[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u1, starts2[0])
        u[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u2, starts2[1])
        u[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u3, starts2[2])

        tmp[:, :] = ((markers[ip, 18], markers[ip, 19], markers[ip, 20]),
                     (markers[ip, 19], markers[ip, 21], markers[ip, 22]),
                     (markers[ip, 20], markers[ip, 22], markers[ip, 23]))

        linalg.matrix_vector(tmp, u, e)

        markers[ip, 0:3] = markers[ip, 9:12] - e*dt


def push_gc_cc_J2_stage_H1vec(markers: 'float[:,:]', dt: float, stage: int,
                              pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                              starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                              kind_map: int, params_map: 'float[:]',
                              p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                              ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                              cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                              kappa: float,
                              b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                              norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                              norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                              curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                              u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]',
                              a: 'float[:]', b: 'float[:]', c: 'float[:]'):
    r'''
    TODO
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
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

        eta[:] = markers[ip, 0:3]
        v = markers[ip, 3]

        # evaluate Jacobian, result in df
        map_eval.df(eta[0], eta[1], eta[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)
        linalg.matrix_inv_with_det(df, det_df, df_inv)
        linalg.transpose(df_inv, df_inv_t)
        linalg.matrix_matrix(df_inv, df_inv_t, g_inv)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta[0])
        span2 = bsp.find_span(tn2, pn[1], eta[1])
        span3 = bsp.find_span(tn3, pn[2], eta[2])

        bsp.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # eval all the needed field
        # b; 2form
        bb[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        bb[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        bb[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # u; 0form
        u[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u1, starts0)
        u[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u2, starts0)
        u[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2], bn1, bn2, bn3, span1, span2, span3, u3, starts0)

        # norm_b1; 1form
        norm_b1[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # norm_b; 2form
        norm_b2[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts2[0])
        norm_b2[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts2[1])
        norm_b2[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts2[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

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
        b_star[:] = (bb + curl_norm_b*v/kappa)/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        linalg.matrix_matrix(g_inv, norm_b2_prod, tmp1)
        linalg.matrix_matrix(tmp1, g_inv, tmp2)
        linalg.matrix_matrix(tmp2, b_prod, tmp1)

        linalg.matrix_vector(tmp1, u, e)

        e /= abs_b_star_para

        # markers[ip, :3] -= e/abs_b_star_para*dt

        markers[ip, 13:16] -= dt*b[stage]*e
        markers[ip, 0:3] = markers[ip, 9:12] + \
            dt*a[stage]*e + last*markers[ip, 13:16]


def push_gc_cc_J2_stage_Hdiv(markers: 'float[:,:]', dt: float, stage: int,
                             pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]',
                             starts0: 'int[:]', starts1: 'int[:,:]', starts2: 'int[:,:]', starts3: 'int[:]',
                             kind_map: int, params_map: 'float[:]',
                             p_map: 'int[:]', t1_map: 'float[:]', t2_map: 'float[:]', t3_map: 'float[:]',
                             ind1_map: 'int[:,:]', ind2_map: 'int[:,:]', ind3_map: 'int[:,:]',
                             cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]',
                             kappa: float,
                             b1: 'float[:,:,:]', b2: 'float[:,:,:]', b3: 'float[:,:,:]',
                             norm_b11: 'float[:,:,:]', norm_b12: 'float[:,:,:]', norm_b13: 'float[:,:,:]',
                             norm_b21: 'float[:,:,:]', norm_b22: 'float[:,:,:]', norm_b23: 'float[:,:,:]',
                             curl_norm_b1: 'float[:,:,:]', curl_norm_b2: 'float[:,:,:]', curl_norm_b3: 'float[:,:,:]',
                             u1: 'float[:,:,:]', u2: 'float[:,:,:]', u3: 'float[:,:,:]',
                             a: 'float[:]', b: 'float[:]', c: 'float[:]'):
    r'''
    TODO
    '''

    # allocate metric coeffs
    df = empty((3, 3), dtype=float)
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

        eta[:] = markers[ip, 0:3]
        v = markers[ip, 3]

        # evaluate Jacobian, result in df
        map_eval.df(eta[0], eta[1], eta[2],
                    kind_map, params_map,
                    t1_map, t2_map, t3_map, p_map,
                    ind1_map, ind2_map, ind3_map,
                    cx, cy, cz,
                    df)

        # metric coeffs
        det_df = linalg.det(df)
        linalg.matrix_inv_with_det(df, det_df, df_inv)
        linalg.transpose(df_inv, df_inv_t)
        linalg.matrix_matrix(df_inv, df_inv_t, g_inv)

        # spline evaluation
        span1 = bsp.find_span(tn1, pn[0], eta[0])
        span2 = bsp.find_span(tn2, pn[1], eta[1])
        span3 = bsp.find_span(tn3, pn[2], eta[2])

        bsp.b_d_splines_slim(tn1, pn[0], eta[0], span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, pn[1], eta[1], span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, pn[2], eta[2], span3, bn3, bd3)

        # eval all the needed field
        # b; 2form
        bb[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, b1, starts2[0])
        bb[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, b2, starts2[1])
        bb[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, b3, starts2[2])

        # u; 2form
        u[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, u1, starts2[0])
        u[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, u2, starts2[1])
        u[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, u3, starts2[2])

        # norm_b1; 1form
        norm_b1[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2], bd1, bn2, bn3, span1, span2, span3, norm_b11, starts1[0])
        norm_b1[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2], bn1, bd2, bn3, span1, span2, span3, norm_b12, starts1[1])
        norm_b1[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1], pn[2] - 1, bn1, bn2, bd3, span1, span2, span3, norm_b13, starts1[2])

        # norm_b; 2form
        norm_b2[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, norm_b21, starts2[0])
        norm_b2[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, norm_b22, starts2[1])
        norm_b2[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, norm_b23, starts2[2])

        # curl_norm_b; 2form
        curl_norm_b[0] = eval_3d.eval_spline_mpi_kernel(
            pn[0], pn[1] - 1, pn[2] - 1, bn1, bd2, bd3, span1, span2, span3, curl_norm_b1, starts2[0])
        curl_norm_b[1] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1], pn[2] - 1, bd1, bn2, bd3, span1, span2, span3, curl_norm_b2, starts2[1])
        curl_norm_b[2] = eval_3d.eval_spline_mpi_kernel(
            pn[0] - 1, pn[1] - 1, pn[2], bd1, bd2, bn3, span1, span2, span3, curl_norm_b3, starts2[2])

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
        b_star[:] = (bb + curl_norm_b*v/kappa)/det_df

        # calculate abs_b_star_para
        abs_b_star_para = linalg.scalar_dot(norm_b1, b_star)

        linalg.matrix_matrix(g_inv, norm_b2_prod, tmp1)
        linalg.matrix_matrix(tmp1, g_inv, tmp2)
        linalg.matrix_matrix(tmp2, b_prod, tmp1)

        linalg.matrix_vector(tmp1, u, e)

        e /= abs_b_star_para/det_df

        # markers[ip, :3] -= e/abs_b_star_para*dt

        markers[ip, 13:16] -= dt*b[stage]*e
        markers[ip, 0:3] = markers[ip, 9:12] + \
            dt*a[stage]*e + last*markers[ip, 13:16]
