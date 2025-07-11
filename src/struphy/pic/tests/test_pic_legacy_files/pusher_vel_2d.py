# import pyccel decorators


# import modules for B-spline evaluation
import struphy.bsplines.bsplines_kernels as bsp

# import module for matrix-matrix and matrix-vector multiplications
import struphy.linear_algebra.linalg_kernels as linalg

# import modules for mapping evaluation
import struphy.pic.tests.test_pic_legacy_files.mappings_3d_fast as mapping_fast
import struphy.pic.tests.test_pic_legacy_files.spline_evaluation_2d as eva2


# ==========================================================================================================
def pusher_step3(
    particles: "float[:,:]",
    dt: "float",
    t1: "float[:]",
    t2: "float[:]",
    p: "int[:]",
    nel: "int[:]",
    nbase_n: "int[:]",
    nbase_d: "int[:]",
    np: "int",
    b_eq_1: "float[:,:,:]",
    b_eq_2: "float[:,:,:]",
    b_eq_3: "float[:,:,:]",
    b_p_1: "float[:,:,:]",
    b_p_2: "float[:,:,:]",
    b_p_3: "float[:,:,:]",
    b_norm: "float[:,:,:]",
    u1: "float[:,:,:]",
    u2: "float[:,:,:]",
    u3: "float[:,:,:]",
    basis_u: "int",
    kind_map: "int",
    params_map: "float[:]",
    tf1: "float[:]",
    tf2: "float[:]",
    tf3: "float[:]",
    pf: "int[:]",
    nelf: "int[:]",
    nbasef: "int[:]",
    cx: "float[:,:,:]",
    cy: "float[:,:,:]",
    cz: "float[:,:,:]",
    mu: "float[:]",
    power: "float[:]",
    n_tor: "int",
):
    from numpy import cos, empty, pi, sin, zeros

    # ============== for magnetic field evaluation ============
    # spline degrees
    pn1 = p[0]
    pn2 = p[1]

    pd1 = pn1 - 1
    pd2 = pn2 - 1

    # p + 1 non-vanishing basis functions up tp degree p
    b1 = empty((pn1 + 1, pn1 + 1), dtype=float)
    b2 = empty((pn2 + 1, pn2 + 1), dtype=float)

    # left and right values for spline evaluation
    l1 = empty(pn1, dtype=float)
    l2 = empty(pn2, dtype=float)

    r1 = empty(pn1, dtype=float)
    r2 = empty(pn2, dtype=float)

    # scaling arrays for M-splines
    d1 = empty(pn1, dtype=float)
    d2 = empty(pn2, dtype=float)

    # p + 1 non-vanishing derivatives
    der1 = empty(pn1 + 1, dtype=float)
    der2 = empty(pn2 + 1, dtype=float)

    # non-vanishing N-splines at particle position
    bn1 = empty(pn1 + 1, dtype=float)
    bn2 = empty(pn2 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pd1 + 1, dtype=float)
    bd2 = empty(pd2 + 1, dtype=float)

    # cos/sin at particle position
    cs = empty(2, dtype=float)

    # magnetic field, velocity field and electric field at particle position
    u = empty(3, dtype=float)
    b = empty(3, dtype=float)
    b_grad = empty(3, dtype=float)

    u_cart = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    b_grad_cart = empty(3, dtype=float)

    e_cart = empty(3, dtype=float)
    # ==========================================================

    # ================ for mapping evaluation ==================
    # spline degrees
    pf1 = pf[0]
    pf2 = pf[1]
    pf3 = pf[2]

    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f = empty((pf3 + 1, pf3 + 1), dtype=float)

    # left and right values for spline evaluation
    l1f = empty(pf1, dtype=float)
    l2f = empty(pf2, dtype=float)
    l3f = empty(pf3, dtype=float)

    r1f = empty(pf1, dtype=float)
    r2f = empty(pf2, dtype=float)
    r3f = empty(pf3, dtype=float)

    # scaling arrays for M-splines
    d1f = empty(pf1, dtype=float)
    d2f = empty(pf2, dtype=float)
    d3f = empty(pf3, dtype=float)

    # pf + 1 derivatives
    der1f = empty(pf1 + 1, dtype=float)
    der2f = empty(pf2 + 1, dtype=float)
    der3f = empty(pf3 + 1, dtype=float)

    # needed mapping quantities
    fx = empty(3, dtype=float)
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    # ==========================================================

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ip, eta1, eta2, eta3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, det_df, dfinv, dfinv_t, span1, span2, l1, l2, r1, r2, b1, b2, d1, d2, der1, der2, bn1, bn2, bd1, bd2, cs, u, u_cart, b, b_cart, b_grad, b_grad_cart, e_cart)
    for ip in range(np):
        # only do something if particle is inside the logical domain (0 < s < 1)
        if particles[0, ip] < 0.0 or particles[0, ip] > 1.0:
            continue

        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]

        # ========= mapping evaluation =============
        span1f = int(eta1 * nelf[0]) + pf1
        span2f = int(eta2 * nelf[1]) + pf2
        span3f = int(eta3 * nelf[2]) + pf3

        # evaluate Jacobian matrix
        mapping_fast.df_all(
            kind_map,
            params_map,
            tf1,
            tf2,
            tf3,
            pf,
            nbasef,
            span1f,
            span2f,
            span3f,
            cx,
            cy,
            cz,
            l1f,
            l2f,
            l3f,
            r1f,
            r2f,
            r3f,
            b1f,
            b2f,
            b3f,
            d1f,
            d2f,
            d3f,
            der1f,
            der2f,
            der3f,
            eta1,
            eta2,
            eta3,
            df,
            fx,
            0,
        )

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)
        # ==========================================

        # ========== field evaluation ==============
        span1 = int(eta1 * nel[0]) + pn1
        span2 = int(eta2 * nel[1]) + pn2

        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, eta1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, eta2, span2, l2, r2, b2, d2, der2)

        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]

        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]

        # cos/sin at particle position
        cs[0] = cos(2 * pi * n_tor * eta3)
        cs[1] = sin(2 * pi * n_tor * eta3)

        # velocity field (0-form, push-forward with df)
        if basis_u == 0:
            u[:] = 0.0

            for i in range(nbase_n[2]):
                u[0] += (
                    eva2.evaluation_kernel_2d(pn1, pn2, bn1, bn2, span1, span2, nbase_n[0], nbase_n[1], u1[:, :, i])
                    * cs[i]
                )
                u[1] += (
                    eva2.evaluation_kernel_2d(pn1, pn2, bn1, bn2, span1, span2, nbase_n[0], nbase_n[1], u2[:, :, i])
                    * cs[i]
                )
                u[2] += (
                    eva2.evaluation_kernel_2d(pn1, pn2, bn1, bn2, span1, span2, nbase_n[0], nbase_n[1], u3[:, :, i])
                    * cs[i]
                )

            linalg.matrix_vector(df, u, u_cart)

        # velocity field (1-form, push forward with df^(-T))
        elif basis_u == 1:
            u[:] = 0.0

            for i in range(nbase_n[2]):
                u[0] += (
                    eva2.evaluation_kernel_2d(
                        pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], u1[:, :, i]
                    )
                    * cs[i]
                )
                u[1] += (
                    eva2.evaluation_kernel_2d(
                        pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], u2[:, :, i]
                    )
                    * cs[i]
                )
                u[2] += (
                    eva2.evaluation_kernel_2d(
                        pn1, pn2, bn1, bn2, span1 - 0, span2 - 0, nbase_n[0], nbase_n[1], u3[:, :, i]
                    )
                    * cs[i]
                )

            linalg.matrix_vector(dfinv_t, u, u_cart)

        # velocity field (2-form, push forward with df/|det df|)
        elif basis_u == 2:
            u[:] = 0.0

            for i in range(nbase_n[2]):
                u[0] += (
                    eva2.evaluation_kernel_2d(
                        pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], u1[:, :, i]
                    )
                    * cs[i]
                )
                u[1] += (
                    eva2.evaluation_kernel_2d(
                        pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], u2[:, :, i]
                    )
                    * cs[i]
                )
                u[2] += (
                    eva2.evaluation_kernel_2d(
                        pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], u3[:, :, i]
                    )
                    * cs[i]
                )

            linalg.matrix_vector(df, u, u_cart)

            u_cart[0] = u_cart[0] / det_df
            u_cart[1] = u_cart[1] / det_df
            u_cart[2] = u_cart[2] / det_df

        # equilibrium magnetic field (2-form)
        b[0] = eva2.evaluation_kernel_2d(
            pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_eq_1[:, :, 0]
        )
        b[1] = eva2.evaluation_kernel_2d(
            pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_eq_2[:, :, 0]
        )
        b[2] = eva2.evaluation_kernel_2d(
            pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_eq_3[:, :, 0]
        )

        # perturbed magnetic field (2-form)
        for i in range(nbase_n[2]):
            b[0] += (
                eva2.evaluation_kernel_2d(
                    pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_p_1[:, :, i]
                )
                * cs[i]
            )
            b[1] += (
                eva2.evaluation_kernel_2d(
                    pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_p_2[:, :, i]
                )
                * cs[i]
            )
            b[2] += (
                eva2.evaluation_kernel_2d(
                    pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_p_3[:, :, i]
                )
                * cs[i]
            )

        # push-forward to physical domain
        linalg.matrix_vector(df, b, b_cart)

        b_cart[0] = b_cart[0] / det_df
        b_cart[1] = b_cart[1] / det_df
        b_cart[2] = b_cart[2] / det_df

        # gradient of absolute value of magnetic field (1-form)
        b_grad[0] = eva2.evaluation_kernel_2d(
            pn1, pn2, der1, bn2, span1, span2, nbase_n[0], nbase_n[1], b_norm[:, :, 0]
        )
        b_grad[1] = eva2.evaluation_kernel_2d(
            pn1, pn2, bn1, der2, span1, span2, nbase_n[0], nbase_n[1], b_norm[:, :, 0]
        )
        b_grad[2] = 0.0

        # push-forward to physical domain
        linalg.matrix_vector(dfinv_t, b_grad, b_grad_cart)

        # electric field B x U
        linalg.cross(b_cart, u_cart, e_cart)

        # additional artificial electric field if Pauli particles are used
        e_cart[:] = e_cart - mu[ip] * b_grad_cart

        # power transfer (v.E)
        power[ip] = particles[3, ip] * e_cart[0] + particles[4, ip] * e_cart[1] + particles[5, ip] * e_cart[2]
        # ==========================================

        # ======== particle pushing ================
        particles[3, ip] += dt * e_cart[0]
        particles[4, ip] += dt * e_cart[1]
        particles[5, ip] += dt * e_cart[2]
        # ==========================================

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    ierr = 0


# ==========================================================================================================
def pusher_step5(
    particles: "float[:,:]",
    dt: "float",
    t1: "float[:]",
    t2: "float[:]",
    p: "int[:]",
    nel: "int[:]",
    nbase_n: "int[:]",
    nbase_d: "int[:]",
    np: "int",
    b_eq_1: "float[:,:,:]",
    b_eq_2: "float[:,:,:]",
    b_eq_3: "float[:,:,:]",
    b_p_1: "float[:,:,:]",
    b_p_2: "float[:,:,:]",
    b_p_3: "float[:,:,:]",
    kind_map: "int",
    params_map: "float[:]",
    tf1: "float[:]",
    tf2: "float[:]",
    tf3: "float[:]",
    pf: "int[:]",
    nelf: "int[:]",
    nbasef: "int[:]",
    cx: "float[:,:,:]",
    cy: "float[:,:,:]",
    cz: "float[:,:,:]",
    n_tor: "int",
):
    from numpy import cos, empty, pi, sin, sqrt, zeros

    # ============== for magnetic field evaluation ============
    # spline degrees
    pn1 = p[0]
    pn2 = p[1]

    pd1 = pn1 - 1
    pd2 = pn2 - 1

    # p + 1 non-vanishing basis functions up tp degree p
    b1 = empty((pn1 + 1, pn1 + 1), dtype=float)
    b2 = empty((pn2 + 1, pn2 + 1), dtype=float)

    # left and right values for spline evaluation
    l1 = empty(pn1, dtype=float)
    l2 = empty(pn2, dtype=float)

    r1 = empty(pn1, dtype=float)
    r2 = empty(pn2, dtype=float)

    # scaling arrays for M-splines
    d1 = empty(pn1, dtype=float)
    d2 = empty(pn2, dtype=float)

    # non-vanishing N-splines at particle position
    bn1 = empty(pn1 + 1, dtype=float)
    bn2 = empty(pn2 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pd1 + 1, dtype=float)
    bd2 = empty(pd2 + 1, dtype=float)

    # cos/sin at particle position
    cs = empty(2, dtype=float)

    # magnetic field at particle position (2-form, cartesian, normalized cartesian)
    b = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    b0 = empty(3, dtype=float)

    # particle velocity (cartesian, perpendicular, v x b0, b0 x vperp)
    v = empty(3, dtype=float)
    vperp = empty(3, dtype=float)
    vxb0 = empty(3, dtype=float)
    b0xvperp = empty(3, dtype=float)
    # ==========================================================

    # ================ for mapping evaluation ==================
    # spline degrees
    pf1 = pf[0]
    pf2 = pf[1]
    pf3 = pf[2]

    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f = empty((pf3 + 1, pf3 + 1), dtype=float)

    # left and right values for spline evaluation
    l1f = empty(pf1, dtype=float)
    l2f = empty(pf2, dtype=float)
    l3f = empty(pf3, dtype=float)

    r1f = empty(pf1, dtype=float)
    r2f = empty(pf2, dtype=float)
    r3f = empty(pf3, dtype=float)

    # scaling arrays for M-splines
    d1f = empty(pf1, dtype=float)
    d2f = empty(pf2, dtype=float)
    d3f = empty(pf3, dtype=float)

    # pf + 1 derivatives
    der1f = empty(pf1 + 1, dtype=float)
    der2f = empty(pf2 + 1, dtype=float)
    der3f = empty(pf3 + 1, dtype=float)

    # needed mapping quantities
    fx = empty(3, dtype=float)
    df = empty((3, 3), dtype=float)
    # ==========================================================

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ip, eta1, eta2, eta3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, det_df, span1, span2, l1, l2, r1, r2, b1, b2, d1, d2, bn1, bn2, bd1, bd2, cs, b, b_cart, b_norm, b0, v, vpar, vxb0, vperp, b0xvperp)
    for ip in range(np):
        # only do something if particle is inside the logical domain (0 < s < 1)
        if particles[0, ip] < 0.0 or particles[0, ip] > 1.0:
            continue

        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]

        # ========= mapping evaluation =============
        span1f = int(eta1 * nelf[0]) + pf1
        span2f = int(eta2 * nelf[1]) + pf2
        span3f = int(eta3 * nelf[2]) + pf3

        # evaluate Jacobian matrix
        mapping_fast.df_all(
            kind_map,
            params_map,
            tf1,
            tf2,
            tf3,
            pf,
            nbasef,
            span1f,
            span2f,
            span3f,
            cx,
            cy,
            cz,
            l1f,
            l2f,
            l3f,
            r1f,
            r2f,
            r3f,
            b1f,
            b2f,
            b3f,
            d1f,
            d2f,
            d3f,
            der1f,
            der2f,
            der3f,
            eta1,
            eta2,
            eta3,
            df,
            fx,
            0,
        )

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))
        # ==========================================

        # ========== field evaluation ==============
        span1 = int(eta1 * nel[0]) + pn1
        span2 = int(eta2 * nel[1]) + pn2

        # evaluation of basis functions
        bsp.basis_funs_all(t1, pn1, eta1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, pn2, eta2, span2, l2, r2, b2, d2)

        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]

        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]

        # cos/sin at particle position
        cs[0] = cos(2 * pi * n_tor * eta3)
        cs[1] = sin(2 * pi * n_tor * eta3)

        # equilibrium magnetic field (2-form)
        b[0] = eva2.evaluation_kernel_2d(
            pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_eq_1[:, :, 0]
        )
        b[1] = eva2.evaluation_kernel_2d(
            pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_eq_2[:, :, 0]
        )
        b[2] = eva2.evaluation_kernel_2d(
            pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_eq_3[:, :, 0]
        )

        # perturbed magnetic field (2-form)
        for i in range(nbase_n[2]):
            b[0] += (
                eva2.evaluation_kernel_2d(
                    pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_p_1[:, :, i]
                )
                * cs[i]
            )
            b[1] += (
                eva2.evaluation_kernel_2d(
                    pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_p_2[:, :, i]
                )
                * cs[i]
            )
            b[2] += (
                eva2.evaluation_kernel_2d(
                    pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_p_3[:, :, i]
                )
                * cs[i]
            )

        # push-forward to physical domain
        linalg.matrix_vector(df, b, b_cart)

        b_cart[0] = b_cart[0] / det_df
        b_cart[1] = b_cart[1] / det_df
        b_cart[2] = b_cart[2] / det_df

        # absolute value of magnetic field
        b_norm = sqrt(b_cart[0] ** 2 + b_cart[1] ** 2 + b_cart[2] ** 2)

        # normalized magnetic field direction
        b0[0] = b_cart[0] / b_norm
        b0[1] = b_cart[1] / b_norm
        b0[2] = b_cart[2] / b_norm
        # ==========================================

        # ======== particle pushing ================
        # particle velocity
        v[:] = particles[3:6, ip]

        # parallel velocity v . b0
        vpar = v[0] * b0[0] + v[1] * b0[1] + v[2] * b0[2]

        # perpendicular velocity b0 x (v x b0)
        linalg.cross(v, b0, vxb0)
        linalg.cross(b0, vxb0, vperp)

        # analytical rotation
        linalg.cross(b0, vperp, b0xvperp)

        particles[3:6, ip] = vpar * b0 + cos(b_norm * dt) * vperp - sin(b_norm * dt) * b0xvperp
        # ==========================================

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    ierr = 0
