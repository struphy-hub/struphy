# import pyccel decorators


# import modules for B-spline evaluation
import struphy.bsplines.bsplines_kernels as bsp

# import module for matrix-matrix and matrix-vector multiplications
import struphy.linear_algebra.linalg_kernels as linalg

# import modules for mapping evaluation
import struphy.pic.tests.test_pic_legacy_files.mappings_3d as mapping
import struphy.pic.tests.test_pic_legacy_files.mappings_3d_fast as mapping_fast
import struphy.pic.tests.test_pic_legacy_files.spline_evaluation_3d as eva3


# ==========================================================================================================
def pusher_step4(
    particles: "float[:,:]",
    dt: "float",
    np: "int",
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
    bc: "int",
):
    from numpy import arctan2, cos, empty, pi, sin, sqrt

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
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    fx = empty(3, dtype=float)
    # ========================================================

    # ======= particle position and velocity =================
    e = empty(3, dtype=float)
    v = empty(3, dtype=float)

    e_new = empty(3, dtype=float)
    # ========================================================

    # ===== intermediate stps in 4th order Runge-Kutta =======
    k1 = empty(3, dtype=float)
    k2 = empty(3, dtype=float)
    k3 = empty(3, dtype=float)
    k4 = empty(3, dtype=float)
    # ========================================================

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ip, e, v, e_new, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, dfinv, k1, k2, k3, k4)
    for ip in range(np):
        # only do something if particle is inside the logical domain (0 < s < 1)
        if particles[0, ip] < 0.0 or particles[0, ip] > 1.0:
            continue

        # current position and velocity
        e[:] = particles[0:3, ip]
        v[:] = particles[3:6, ip]

        # ----------- step 1 in Runge-Kutta method -----------------------
        e_new[0] = e[0]
        e_new[1] = e[1]
        e_new[2] = e[2]

        span1f = int(e_new[0] * nelf[0]) + pf1
        span2f = int(e_new[1] * nelf[1]) + pf2
        span3f = int(e_new[2] * nelf[2]) + pf3

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
            e_new[0],
            e_new[1],
            e_new[2],
            df,
            fx,
            0,
        )

        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k1)
        # ------------------------------------------------------------------

        # ----------------- step 2 in Runge-Kutta method -------------------
        e_new[0] = e[0] + dt * k1[0] / 2

        # check boundary condition in eta_1 direction

        # periodic
        if bc == 0:
            e_new[0] = e_new[0] % 1.0

        # lost
        elif bc == 1:
            if e_new[0] > 1.0:
                particles[6, ip] = 0.0
                particles[0, ip] = 1.5
                continue

            elif e_new[0] < 0.0:
                particles[6, ip] = 0.0
                particles[0, ip] = -0.5
                continue

        e_new[1] = (e[1] + dt * k1[1] / 2) % 1.0
        e_new[2] = (e[2] + dt * k1[2] / 2) % 1.0

        span1f = int(e_new[0] * nelf[0]) + pf1
        span2f = int(e_new[1] * nelf[1]) + pf2
        span3f = int(e_new[2] * nelf[2]) + pf3

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
            e_new[0],
            e_new[1],
            e_new[2],
            df,
            fx,
            0,
        )

        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k2)
        # ------------------------------------------------------------------

        # ------------------ step 3 in Runge-Kutta method ------------------
        e_new[0] = e[0] + dt * k2[0] / 2

        # check boundary condition in eta_1 direction

        # periodic
        if bc == 0:
            e_new[0] = e_new[0] % 1.0

        # lost
        elif bc == 1:
            if e_new[0] > 1.0:
                particles[6, ip] = 0.0
                particles[0, ip] = 1.5
                continue

            elif e_new[0] < 0.0:
                particles[6, ip] = 0.0
                particles[0, ip] = -0.5
                continue

        e_new[1] = (e[1] + dt * k2[1] / 2) % 1.0
        e_new[2] = (e[2] + dt * k2[2] / 2) % 1.0

        span1f = int(e_new[0] * nelf[0]) + pf1
        span2f = int(e_new[1] * nelf[1]) + pf2
        span3f = int(e_new[2] * nelf[2]) + pf3

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
            e_new[0],
            e_new[1],
            e_new[2],
            df,
            fx,
            0,
        )

        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k3)
        # ------------------------------------------------------------------

        # ------------------ step 4 in Runge-Kutta method ------------------
        e_new[0] = e[0] + dt * k3[0]

        # check boundary condition in eta_1 direction

        # periodic
        if bc == 0:
            e_new[0] = e_new[0] % 1.0

        # lost
        elif bc == 1:
            if e_new[0] > 1.0:
                particles[6, ip] = 0.0
                particles[0, ip] = 1.5
                continue

            elif e_new[0] < 0.0:
                particles[6, ip] = 0.0
                particles[0, ip] = -0.5
                continue

        e_new[1] = (e[1] + dt * k3[1]) % 1.0
        e_new[2] = (e[2] + dt * k3[2]) % 1.0

        span1f = int(e_new[0] * nelf[0]) + pf1
        span2f = int(e_new[1] * nelf[1]) + pf2
        span3f = int(e_new[2] * nelf[2]) + pf3

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
            e_new[0],
            e_new[1],
            e_new[2],
            df,
            fx,
            0,
        )

        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k4)
        # ------------------------------------------------------------------

        #  ---------------- update logical coordinates ---------------------
        e_new[0] = e[0] + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6

        # check boundary condition in eta_1 direction

        # periodic
        if bc == 0:
            e_new[0] = e_new[0] % 1.0

        # lost
        elif bc == 1:
            if e_new[0] > 1.0:
                particles[6, ip] = 0.0
                particles[0, ip] = 1.5
                continue

            elif e_new[0] < 0.0:
                particles[6, ip] = 0.0
                particles[0, ip] = -0.5
                continue

        e_new[1] = (e[1] + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6) % 1
        e_new[2] = (e[2] + dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6) % 1

        particles[0, ip] = e_new[0]
        particles[1, ip] = e_new[1]
        particles[2, ip] = e_new[2]
        # ------------------------------------------------------------------

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    ierr = 0


# ========================================================================================================
def reflect(
    df,
):
    from numpy import empty, sqrt

    vg = empty(3, dtype=float)

    basis = empty((3, 3), dtype=float)
    basis_inv = empty((3, 3), dtype=float)

    # calculate normalized basis vectors
    norm1 = sqrt(df_inv[0, 0] ** 2 + df_inv[0, 1] ** 2 + df_inv[0, 2] ** 2)

    norm2 = sqrt(df[0, 1] ** 2 + df[1, 1] ** 2 + df[2, 1] ** 2)
    norm3 = sqrt(df[0, 2] ** 2 + df[1, 2] ** 2 + df[2, 2] ** 2)

    basis[:, 0] = df_inv[0, :] / norm1

    basis[:, 1] = df[:, 1] / norm2
    basis[:, 2] = df[:, 2] / norm3

    linalg.matrix_inv(basis, basis_inv)

    linalg.matrix_vector(basis_inv, v, vg)

    vg[0] = -vg[0]

    linalg.matrix_vector(basis, vg, v)


# ==========================================================================================================
def pusher_step4_pcart(
    particles: "float[:,:]",
    dt: "float",
    np: "int",
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
    map_pseudo: "int",
    r0_pseudo: "float",
):
    from numpy import empty, zeros

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
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)

    df_old = empty((3, 3), dtype=float)
    dfinv_old = empty((3, 3), dtype=float)

    fx = empty(3, dtype=float)

    # needed mapping quantities for pseudo-cartesian coordinates
    df_pseudo = empty((3, 3), dtype=float)

    df_pseudo_old = empty((3, 3), dtype=float)
    fx_pseudo = empty(3, dtype=float)

    params_pseudo = empty(3, dtype=float)

    params_pseudo[0] = 0.0
    params_pseudo[1] = 1.0
    params_pseudo[2] = r0_pseudo
    # ========================================================

    # ======= particle position and velocity =================
    eta = empty(3, dtype=float)
    v = empty(3, dtype=float)
    v_temp = empty(3, dtype=float)
    # ========================================================

    # ===== intermediate stps in 4th order Runge-Kutta =======
    k1 = empty(3, dtype=float)
    k2 = empty(3, dtype=float)
    k3 = empty(3, dtype=float)
    k4 = empty(3, dtype=float)
    # ========================================================

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ip, eta, v, fx_pseudo, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df_old, fx, dfinv_old, df_pseudo_old, df, dfinv, df_pseudo, v_temp, k1, k2, k3, k4)
    for ip in range(np):
        # only do something if particle is inside the logical domain (s < 1)
        if particles[0, ip] > 1.0:
            continue

        # old logical coordinates and velocities
        eta[:] = particles[0:3, ip]
        v[:] = particles[3:6, ip]

        # compute old pseudo-cartesian coordinates
        fx_pseudo[0] = mapping.f(
            eta[0],
            eta[1],
            eta[2],
            1,
            map_pseudo,
            params_pseudo,
            tf1,
            tf2,
            tf3,
            pf,
            nbasef,
            cx,
            cy,
            cz,
        )
        fx_pseudo[1] = mapping.f(
            eta[0],
            eta[1],
            eta[2],
            2,
            map_pseudo,
            params_pseudo,
            tf1,
            tf2,
            tf3,
            pf,
            nbasef,
            cx,
            cy,
            cz,
        )
        fx_pseudo[2] = mapping.f(
            eta[0],
            eta[1],
            eta[2],
            3,
            map_pseudo,
            params_pseudo,
            tf1,
            tf2,
            tf3,
            pf,
            nbasef,
            cx,
            cy,
            cz,
        )

        # evaluate old Jacobian matrix of mapping F
        span1f = int(eta[0] * nelf[0]) + pf1
        span2f = int(eta[1] * nelf[1]) + pf2
        span3f = int(eta[2] * nelf[2]) + pf3

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
            eta[0],
            eta[1],
            eta[2],
            df_old,
            fx,
            0,
        )

        # evaluate old inverse Jacobian matrix of mapping F
        mapping_fast.df_inv_all(df_old, dfinv_old)

        # evaluate old Jacobian matrix of mapping F_pseudo
        df_pseudo_old[0, 0] = mapping.df(
            eta[0],
            eta[1],
            eta[2],
            11,
            map_pseudo,
            params_pseudo,
            tf1,
            tf2,
            tf3,
            pf,
            nbasef,
            cx,
            cy,
            cz,
        )
        df_pseudo_old[0, 1] = mapping.df(
            eta[0],
            eta[1],
            eta[2],
            12,
            map_pseudo,
            params_pseudo,
            tf1,
            tf2,
            tf3,
            pf,
            nbasef,
            cx,
            cy,
            cz,
        )
        df_pseudo_old[0, 2] = mapping.df(
            eta[0],
            eta[1],
            eta[2],
            13,
            map_pseudo,
            params_pseudo,
            tf1,
            tf2,
            tf3,
            pf,
            nbasef,
            cx,
            cy,
            cz,
        )

        df_pseudo_old[1, 0] = mapping.df(
            eta[0],
            eta[1],
            eta[2],
            21,
            map_pseudo,
            params_pseudo,
            tf1,
            tf2,
            tf3,
            pf,
            nbasef,
            cx,
            cy,
            cz,
        )
        df_pseudo_old[1, 1] = mapping.df(
            eta[0],
            eta[1],
            eta[2],
            22,
            map_pseudo,
            params_pseudo,
            tf1,
            tf2,
            tf3,
            pf,
            nbasef,
            cx,
            cy,
            cz,
        )
        df_pseudo_old[1, 2] = mapping.df(
            eta[0],
            eta[1],
            eta[2],
            23,
            map_pseudo,
            params_pseudo,
            tf1,
            tf2,
            tf3,
            pf,
            nbasef,
            cx,
            cy,
            cz,
        )

        df_pseudo_old[2, 0] = mapping.df(
            eta[0],
            eta[1],
            eta[2],
            31,
            map_pseudo,
            params_pseudo,
            tf1,
            tf2,
            tf3,
            pf,
            nbasef,
            cx,
            cy,
            cz,
        )
        df_pseudo_old[2, 1] = mapping.df(
            eta[0],
            eta[1],
            eta[2],
            32,
            map_pseudo,
            params_pseudo,
            tf1,
            tf2,
            tf3,
            pf,
            nbasef,
            cx,
            cy,
            cz,
        )
        df_pseudo_old[2, 2] = mapping.df(
            eta[0],
            eta[1],
            eta[2],
            33,
            map_pseudo,
            params_pseudo,
            tf1,
            tf2,
            tf3,
            pf,
            nbasef,
            cx,
            cy,
            cz,
        )

        while True:
            # ----------- step 1 in Runge-Kutta method -----------------------
            # compute df_pseudo*df_inv*v
            linalg.matrix_vector(dfinv_old, v, v_temp)
            linalg.matrix_vector(df_pseudo_old, v_temp, k1)
            # ------------------------------------------------------------------

            # ----------------- step 2 in Runge-Kutta method -------------------
            # eta[0] = mapping.f_inv(fx_pseudo[0] + dt*k1[0]/2, fx_pseudo[1] + dt*k1[1]/2, fx_pseudo[2] + dt*k1[2]/2, 1, map_pseudo, params_pseudo)
            # eta[1] = mapping.f_inv(fx_pseudo[0] + dt*k1[0]/2, fx_pseudo[1] + dt*k1[1]/2, fx_pseudo[2] + dt*k1[2]/2, 2, map_pseudo, params_pseudo)
            # eta[2] = mapping.f_inv(fx_pseudo[0] + dt*k1[0]/2, fx_pseudo[1] + dt*k1[1]/2, fx_pseudo[2] + dt*k1[2]/2, 3, map_pseudo, params_pseudo)

            eta[0] = 0.5
            eta[1] = 0.5
            eta[2] = 0.5

            # check if particle has left the domain at s = 1: if yes, stop iteration and set weight to zero
            if eta[0] > 1.0:
                particles[6, ip] = 0.0
                particles[0, ip] = 1.5

                break

            # evaluate Jacobian matrix of mapping F
            span1f = int(eta[0] * nelf[0]) + pf1
            span2f = int(eta[1] * nelf[1]) + pf2
            span3f = int(eta[2] * nelf[2]) + pf3

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
                eta[0],
                eta[1],
                eta[2],
                df,
                fx,
                0,
            )

            # evaluate inverse Jacobian matrix of mapping F
            mapping_fast.df_inv_all(df, dfinv)

            # evaluate Jacobian matrix of mapping F_pseudo
            df_pseudo[0, 0] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                11,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[0, 1] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                12,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[0, 2] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                13,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )

            df_pseudo[1, 0] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                21,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[1, 1] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                22,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[1, 2] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                23,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )

            df_pseudo[2, 0] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                31,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[2, 1] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                32,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[2, 2] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                33,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )

            # compute df_pseudo*df_inv*v
            linalg.matrix_vector(dfinv, v, v_temp)
            linalg.matrix_vector(df_pseudo, v_temp, k2)
            # ------------------------------------------------------------------

            # ------------------ step 3 in Runge-Kutta method ------------------
            # eta[0] = mapping.f_inv(fx_pseudo[0] + dt*k2[0]/2, fx_pseudo[1] + dt*k2[1]/2, fx_pseudo[2] + dt*k2[2]/2, 1, map_pseudo, params_pseudo)
            # eta[1] = mapping.f_inv(fx_pseudo[0] + dt*k2[0]/2, fx_pseudo[1] + dt*k2[1]/2, fx_pseudo[2] + dt*k2[2]/2, 2, map_pseudo, params_pseudo)
            # eta[2] = mapping.f_inv(fx_pseudo[0] + dt*k2[0]/2, fx_pseudo[1] + dt*k2[1]/2, fx_pseudo[2] + dt*k2[2]/2, 3, map_pseudo, params_pseudo)

            eta[0] = 0.5
            eta[1] = 0.5
            eta[2] = 0.5

            # check if particle has left the domain at s = 1: if yes, stop iteration and set weight to zero
            if eta[0] > 1.0:
                particles[6, ip] = 0.0
                particles[0, ip] = 1.5

                break

            # evaluate Jacobian matrix of mapping F
            span1f = int(eta[0] * nelf[0]) + pf1
            span2f = int(eta[1] * nelf[1]) + pf2
            span3f = int(eta[2] * nelf[2]) + pf3

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
                eta[0],
                eta[1],
                eta[2],
                df,
                fx,
                0,
            )

            # evaluate inverse Jacobian matrix of mapping F
            mapping_fast.df_inv_all(df, dfinv)

            # evaluate Jacobian matrix of mapping F_pseudo
            df_pseudo[0, 0] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                11,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[0, 1] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                12,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[0, 2] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                13,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )

            df_pseudo[1, 0] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                21,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[1, 1] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                22,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[1, 2] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                23,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )

            df_pseudo[2, 0] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                31,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[2, 1] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                32,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[2, 2] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                33,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )

            # compute df_pseudo*df_inv*v
            linalg.matrix_vector(dfinv, v, v_temp)
            linalg.matrix_vector(df_pseudo, v_temp, k3)
            # ------------------------------------------------------------------

            # ------------------ step 4 in Runge-Kutta method ------------------
            # eta[0] = mapping.f_inv(fx_pseudo[0] + dt*k3[0], fx_pseudo[1] + dt*k3[1], fx_pseudo[2] + dt*k3[2], 1, map_pseudo, params_pseudo)
            # eta[1] = mapping.f_inv(fx_pseudo[0] + dt*k3[0], fx_pseudo[1] + dt*k3[1], fx_pseudo[2] + dt*k3[2], 2, map_pseudo, params_pseudo)
            # eta[2] = mapping.f_inv(fx_pseudo[0] + dt*k3[0], fx_pseudo[1] + dt*k3[1], fx_pseudo[2] + dt*k3[2], 3, map_pseudo, params_pseudo)

            eta[0] = 0.5
            eta[1] = 0.5
            eta[2] = 0.5

            # check if particle has left the domain at s = 1: if yes, stop iteration and set weight to zero
            if eta[0] > 1.0:
                particles[6, ip] = 0.0
                particles[0, ip] = 1.5

                break

            # evaluate Jacobian matrix of mapping F
            span1f = int(eta[0] * nelf[0]) + pf1
            span2f = int(eta[1] * nelf[1]) + pf2
            span3f = int(eta[2] * nelf[2]) + pf3

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
                eta[0],
                eta[1],
                eta[2],
                df,
                fx,
                0,
            )

            # evaluate inverse Jacobian matrix of mapping F
            mapping_fast.df_inv_all(df, dfinv)

            # evaluate Jacobian matrix of mapping F_pseudo
            df_pseudo[0, 0] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                11,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[0, 1] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                12,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[0, 2] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                13,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )

            df_pseudo[1, 0] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                21,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[1, 1] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                22,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[1, 2] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                23,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )

            df_pseudo[2, 0] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                31,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[2, 1] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                32,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )
            df_pseudo[2, 2] = mapping.df(
                eta[0],
                eta[1],
                eta[2],
                33,
                map_pseudo,
                params_pseudo,
                tf1,
                tf2,
                tf3,
                pf,
                nbasef,
                cx,
                cy,
                cz,
            )

            # compute df_pseudo*df_inv*v
            linalg.matrix_vector(dfinv, v, v_temp)
            linalg.matrix_vector(df_pseudo, v_temp, k4)
            # ------------------------------------------------------------------

            #  ---------------- update pseudo-cartesian coordinates ------------
            fx_pseudo[0] = fx_pseudo[0] + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
            fx_pseudo[1] = fx_pseudo[1] + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
            fx_pseudo[2] = fx_pseudo[2] + dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
            # ------------------------------------------------------------------

            # compute logical coordinates
            # eta[0] = mapping.f_inv(fx_pseudo[0], fx_pseudo[1], fx_pseudo[2], 1, map_pseudo, params_pseudo)
            # eta[1] = mapping.f_inv(fx_pseudo[0], fx_pseudo[1], fx_pseudo[2], 2, map_pseudo, params_pseudo)
            # eta[2] = mapping.f_inv(fx_pseudo[0], fx_pseudo[1], fx_pseudo[2], 3, map_pseudo, params_pseudo)

            eta[0] = 0.5
            eta[1] = 0.5
            eta[2] = 0.5

            # check if particle has left the domain at s = 1: if yes, stop iteration and set weight to zero
            if eta[0] > 1.0:
                particles[6, ip] = 0.0
                particles[0, ip] = 1.5

                break

            particles[0, ip] = eta[0]
            particles[1, ip] = eta[1]
            particles[2, ip] = eta[2]

            # set particle velocity (will only change if particle was reflected)
            particles[3, ip] = v[0]
            particles[4, ip] = v[1]
            particles[5, ip] = v[2]

            break

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    ierr = 0


# ==========================================================================================================
def pusher_step4_cart(
    particles: "float[:,:]",
    dt: "float",
    np: "int",
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
    tol: "float",
):
    from numpy import empty

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
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)

    x_old = empty(3, dtype=float)
    x_new = empty(3, dtype=float)

    temp = empty(3, dtype=float)
    # ========================================================

    # ======= particle position and velocity =================
    e = empty(3, dtype=float)
    v = empty(3, dtype=float)
    # ========================================================

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ip, e, v, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, x_old, x_new, dfinv, temp)
    for ip in range(np):
        e[:] = particles[0:3, ip]
        v[:] = particles[3:6, ip]

        span1f = int(e[0] * nelf[0]) + pf1
        span2f = int(e[1] * nelf[1]) + pf2
        span3f = int(e[2] * nelf[2]) + pf3

        # evaluate Jacobian matrix and current Cartesian coordinates
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
            e[0],
            e[1],
            e[2],
            df,
            x_old,
            2,
        )

        # update cartesian coordinates exactly
        x_new[0] = x_old[0] + dt * v[0]
        x_new[1] = x_old[1] + dt * v[1]
        x_new[2] = x_old[2] + dt * v[2]

        # calculate new logical coordinates by solving inverse mapping with Newton-method

        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        while True:
            x_old[:] = x_old - x_new
            linalg.matrix_vector(dfinv, x_old, temp)

            e[0] = e[0] - temp[0]
            e[1] = (e[1] - temp[1]) % 1.0
            e[2] = (e[2] - temp[2]) % 1.0

            span1f = int(e[0] * nelf[0]) + pf1
            span2f = int(e[1] * nelf[1]) + pf2
            span3f = int(e[2] * nelf[2]) + pf3

            # evaluate Jacobian matrix and mapping
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
                e[0],
                e[1],
                e[2],
                df,
                x_old,
                2,
            )

            if abs(x_old[0] - x_new[0]) < tol and abs(x_old[1] - x_new[1]) < tol and abs(x_old[2] - x_new[2]) < tol:
                particles[0:3, ip] = e
                break

            # evaluate inverse Jacobian matrix
            mapping_fast.df_inv_all(df, dfinv)

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    ierr = 0


# ==========================================================================================================
def pusher_rk4_pc_full(
    particles,
    dt,
    t1,
    t2,
    t3,
    p,
    nel,
    nbase_n,
    nbase_d,
    np,
    u1,
    u2,
    u3,
    basis_u,
    kind_map,
    params_map,
    tf1,
    tf2,
    tf3,
    pf,
    nelf,
    nbasef,
    cx,
    cy,
    cz,
    bc,
):
    from numpy import empty

    # ============== for velocity evaluation ============
    # spline degrees
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # p + 1 non-vanishing basis functions up tp degree p
    b1 = empty((pn1 + 1, pn1 + 1), dtype=float)
    b2 = empty((pn2 + 1, pn2 + 1), dtype=float)
    b3 = empty((pn3 + 1, pn3 + 1), dtype=float)

    # left and right values for spline evaluation
    l1 = empty(pn1, dtype=float)
    l2 = empty(pn2, dtype=float)
    l3 = empty(pn3, dtype=float)

    r1 = empty(pn1, dtype=float)
    r2 = empty(pn2, dtype=float)
    r3 = empty(pn3, dtype=float)

    # scaling arrays for M-splines
    d1 = empty(pn1, dtype=float)
    d2 = empty(pn2, dtype=float)
    d3 = empty(pn3, dtype=float)

    # p + 1 non-vanishing derivatives
    der1 = empty(pn1 + 1, dtype=float)
    der2 = empty(pn2 + 1, dtype=float)
    der3 = empty(pn3 + 1, dtype=float)

    # non-vanishing N-splines at particle position
    bn1 = empty(pn1 + 1, dtype=float)
    bn2 = empty(pn2 + 1, dtype=float)
    bn3 = empty(pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pd1 + 1, dtype=float)
    bd2 = empty(pd2 + 1, dtype=float)
    bd3 = empty(pd3 + 1, dtype=float)

    # # velocity field at particle position
    u = empty(3, dtype=float)
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
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    Ginv = empty((3, 3), dtype=float)
    fx = empty(3, dtype=float)
    # ========================================================

    # ======= particle position and velocity =================
    e = empty(3, dtype=float)
    v = empty(3, dtype=float)

    e_new = empty(3, dtype=float)
    # ========================================================

    # ===== intermediate stps in 4th order Runge-Kutta =======
    k1 = empty(3, dtype=float)
    k2 = empty(3, dtype=float)
    k3 = empty(3, dtype=float)
    k4 = empty(3, dtype=float)
    k1_u = empty(3, dtype=float)
    k2_u = empty(3, dtype=float)
    k3_u = empty(3, dtype=float)
    k4_u = empty(3, dtype=float)
    k1_v = empty(3, dtype=float)
    k2_v = empty(3, dtype=float)
    k3_v = empty(3, dtype=float)
    k4_v = empty(3, dtype=float)
    # ========================================================

    for ip in range(np):
        # only do something if particle is inside the logical domain (0 < s < 1)
        if particles[0, ip] < 0.0 or particles[0, ip] > 1.0:
            particles[0:3, ip] = -1.0
            continue

        # current position and velocity
        e[:] = particles[0:3, ip]
        v[:] = particles[3:6, ip]

        # ----------- step 1 in Runge-Kutta method -----------------------
        e_new[0] = e[0]
        e_new[1] = e[1]
        e_new[2] = e[2]
        # ========= mapping evaluation =============
        span1f = int(e_new[0] * nelf[0]) + pf1
        span2f = int(e_new[1] * nelf[1]) + pf2
        span3f = int(e_new[2] * nelf[2]) + pf3

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
            e_new[0],
            e_new[1],
            e_new[2],
            df,
            fx,
            0,
        )

        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)

        # evaluate Ginv matrix
        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
        # ============================================

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k1_v)

        # ========== field evaluation ==============
        span1 = int(e_new[0] * nel[0]) + pn1
        span2 = int(e_new[1] * nel[1]) + pn2
        span3 = int(e_new[2] * nel[2]) + pn3

        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, e_new[0], span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, e_new[1], span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, e_new[2], span3, l3, r3, b3, d3, der3)

        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]

        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # velocity field
        if basis_u == 1:
            u[0] = eva3.evaluation_kernel_3d(
                pd1,
                pn2,
                pn3,
                bd1,
                bn2,
                bn3,
                span1 - 1,
                span2,
                span3,
                nbase_d[0],
                nbase_n[1],
                nbase_n[2],
                u1,
            )
            u[1] = eva3.evaluation_kernel_3d(
                pn1,
                pd2,
                pn3,
                bn1,
                bd2,
                bn3,
                span1,
                span2 - 1,
                span3,
                nbase_n[0],
                nbase_d[1],
                nbase_n[2],
                u2,
            )
            u[2] = eva3.evaluation_kernel_3d(
                pn1,
                pn2,
                pd3,
                bn1,
                bn2,
                bd3,
                span1,
                span2,
                span3 - 1,
                nbase_n[0],
                nbase_n[1],
                nbase_d[2],
                u3,
            )

            linalg.matrix_vector(Ginv, u, k1_u)

        elif basis_u == 2:
            u[0] = eva3.evaluation_kernel_3d(
                pn1,
                pd2,
                pd3,
                bn1,
                bd2,
                bd3,
                span1,
                span2 - 1,
                span3 - 1,
                nbase_n[0],
                nbase_d[1],
                nbase_d[2],
                u1,
            )
            u[1] = eva3.evaluation_kernel_3d(
                pd1,
                pn2,
                pd3,
                bd1,
                bn2,
                bd3,
                span1 - 1,
                span2,
                span3 - 1,
                nbase_d[0],
                nbase_n[1],
                nbase_d[2],
                u2,
            )
            u[2] = eva3.evaluation_kernel_3d(
                pd1,
                pd2,
                pn3,
                bd1,
                bd2,
                bn3,
                span1 - 1,
                span2 - 1,
                span3,
                nbase_d[0],
                nbase_d[1],
                nbase_n[2],
                u3,
            )

            k1_u[:] = u / det_df

        k1[:] = k1_v + k1_u
        # ------------------------------------------------------------------

        # ----------------- step 2 in Runge-Kutta method -------------------
        e_new[0] = e[0] + dt * k1[0] / 2
        e_new[1] = e[1] + dt * k1[1] / 2
        e_new[2] = e[2] + dt * k1[2] / 2

        if e_new[0] < 0.0 or e_new[0] > 1.0 or e_new[1] < 0.0 or e_new[1] > 1.0 or e_new[2] < 0.0 or e_new[2] > 1.0:
            particles[0:3, ip] = -1.0
            continue

        # ========= mapping evaluation =============
        span1f = int(e_new[0] * nelf[0]) + pf1
        span2f = int(e_new[1] * nelf[1]) + pf2
        span3f = int(e_new[2] * nelf[2]) + pf3

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
            e_new[0],
            e_new[1],
            e_new[2],
            df,
            fx,
            0,
        )

        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)

        # evaluate Ginv matrix
        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
        # ============================================

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k2_v)

        # ========== field evaluation ==============
        span1 = int(e_new[0] * nel[0]) + pn1
        span2 = int(e_new[1] * nel[1]) + pn2
        span3 = int(e_new[2] * nel[2]) + pn3

        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, e_new[0], span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, e_new[1], span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, e_new[2], span3, l3, r3, b3, d3, der3)

        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]

        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # velocity field
        if basis_u == 1:
            u[0] = eva3.evaluation_kernel_3d(
                pd1,
                pn2,
                pn3,
                bd1,
                bn2,
                bn3,
                span1 - 1,
                span2,
                span3,
                nbase_d[0],
                nbase_n[1],
                nbase_n[2],
                u1,
            )
            u[1] = eva3.evaluation_kernel_3d(
                pn1,
                pd2,
                pn3,
                bn1,
                bd2,
                bn3,
                span1,
                span2 - 1,
                span3,
                nbase_n[0],
                nbase_d[1],
                nbase_n[2],
                u2,
            )
            u[2] = eva3.evaluation_kernel_3d(
                pn1,
                pn2,
                pd3,
                bn1,
                bn2,
                bd3,
                span1,
                span2,
                span3 - 1,
                nbase_n[0],
                nbase_n[1],
                nbase_d[2],
                u3,
            )

            linalg.matrix_vector(Ginv, u, k2_u)

        elif basis_u == 2:
            u[0] = eva3.evaluation_kernel_3d(
                pn1,
                pd2,
                pd3,
                bn1,
                bd2,
                bd3,
                span1,
                span2 - 1,
                span3 - 1,
                nbase_n[0],
                nbase_d[1],
                nbase_d[2],
                u1,
            )
            u[1] = eva3.evaluation_kernel_3d(
                pd1,
                pn2,
                pd3,
                bd1,
                bn2,
                bd3,
                span1 - 1,
                span2,
                span3 - 1,
                nbase_d[0],
                nbase_n[1],
                nbase_d[2],
                u2,
            )
            u[2] = eva3.evaluation_kernel_3d(
                pd1,
                pd2,
                pn3,
                bd1,
                bd2,
                bn3,
                span1 - 1,
                span2 - 1,
                span3,
                nbase_d[0],
                nbase_d[1],
                nbase_n[2],
                u3,
            )

            k2_u[:] = u / det_df

        k2[:] = k2_v + k2_u
        # ------------------------------------------------------------------

        # ------------------ step 3 in Runge-Kutta method ------------------
        e_new[0] = e[0] + dt * k2[0] / 2
        e_new[1] = e[1] + dt * k2[1] / 2
        e_new[2] = e[2] + dt * k2[2] / 2

        if e_new[0] < 0.0 or e_new[0] > 1.0 or e_new[1] < 0.0 or e_new[1] > 1.0 or e_new[2] < 0.0 or e_new[2] > 1.0:
            particles[0:3, ip] = -1.0
            continue

        # ========= mapping evaluation =============
        span1f = int(e_new[0] * nelf[0]) + pf1
        span2f = int(e_new[1] * nelf[1]) + pf2
        span3f = int(e_new[2] * nelf[2]) + pf3

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
            e_new[0],
            e_new[1],
            e_new[2],
            df,
            fx,
            0,
        )

        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)

        # evaluate Ginv matrix
        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
        # ============================================

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k3_v)

        # ========== field evaluation ==============
        span1 = int(e_new[0] * nel[0]) + pn1
        span2 = int(e_new[1] * nel[1]) + pn2
        span3 = int(e_new[2] * nel[2]) + pn3

        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, e_new[0], span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, e_new[1], span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, e_new[2], span3, l3, r3, b3, d3, der3)

        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]

        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # velocity field
        if basis_u == 1:
            u[0] = eva3.evaluation_kernel_3d(
                pd1,
                pn2,
                pn3,
                bd1,
                bn2,
                bn3,
                span1 - 1,
                span2,
                span3,
                nbase_d[0],
                nbase_n[1],
                nbase_n[2],
                u1,
            )
            u[1] = eva3.evaluation_kernel_3d(
                pn1,
                pd2,
                pn3,
                bn1,
                bd2,
                bn3,
                span1,
                span2 - 1,
                span3,
                nbase_n[0],
                nbase_d[1],
                nbase_n[2],
                u2,
            )
            u[2] = eva3.evaluation_kernel_3d(
                pn1,
                pn2,
                pd3,
                bn1,
                bn2,
                bd3,
                span1,
                span2,
                span3 - 1,
                nbase_n[0],
                nbase_n[1],
                nbase_d[2],
                u3,
            )

            linalg.matrix_vector(Ginv, u, k3_u)

        elif basis_u == 2:
            u[0] = eva3.evaluation_kernel_3d(
                pn1,
                pd2,
                pd3,
                bn1,
                bd2,
                bd3,
                span1,
                span2 - 1,
                span3 - 1,
                nbase_n[0],
                nbase_d[1],
                nbase_d[2],
                u1,
            )
            u[1] = eva3.evaluation_kernel_3d(
                pd1,
                pn2,
                pd3,
                bd1,
                bn2,
                bd3,
                span1 - 1,
                span2,
                span3 - 1,
                nbase_d[0],
                nbase_n[1],
                nbase_d[2],
                u2,
            )
            u[2] = eva3.evaluation_kernel_3d(
                pd1,
                pd2,
                pn3,
                bd1,
                bd2,
                bn3,
                span1 - 1,
                span2 - 1,
                span3,
                nbase_d[0],
                nbase_d[1],
                nbase_n[2],
                u3,
            )

            k3_u[:] = u / det_df

        k3[:] = k3_v + k3_u
        # ------------------------------------------------------------------

        # ------------------ step 4 in Runge-Kutta method ------------------
        e_new[0] = e[0] + dt * k3[0]
        e_new[1] = e[1] + dt * k3[1]
        e_new[2] = e[2] + dt * k3[2]

        if e_new[0] < 0.0 or e_new[0] > 1.0 or e_new[1] < 0.0 or e_new[1] > 1.0 or e_new[2] < 0.0 or e_new[2] > 1.0:
            particles[0:3, ip] = -1.0
            continue

        # ========= mapping evaluation =============
        span1f = int(e_new[0] * nelf[0]) + pf1
        span2f = int(e_new[1] * nelf[1]) + pf2
        span3f = int(e_new[2] * nelf[2]) + pf3

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
            e_new[0],
            e_new[1],
            e_new[2],
            df,
            fx,
            0,
        )

        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)

        # evaluate Ginv matrix
        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
        # ============================================

        # ========== field evaluation ==============
        span1 = int(e_new[0] * nel[0]) + pn1
        span2 = int(e_new[1] * nel[1]) + pn2
        span3 = int(e_new[2] * nel[2]) + pn3

        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, e_new[0], span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, e_new[1], span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, e_new[2], span3, l3, r3, b3, d3, der3)

        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]

        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # velocity field
        if basis_u == 1:
            u[0] = eva3.evaluation_kernel_3d(
                pd1,
                pn2,
                pn3,
                bd1,
                bn2,
                bn3,
                span1 - 1,
                span2,
                span3,
                nbase_d[0],
                nbase_n[1],
                nbase_n[2],
                u1,
            )
            u[1] = eva3.evaluation_kernel_3d(
                pn1,
                pd2,
                pn3,
                bn1,
                bd2,
                bn3,
                span1,
                span2 - 1,
                span3,
                nbase_n[0],
                nbase_d[1],
                nbase_n[2],
                u2,
            )
            u[2] = eva3.evaluation_kernel_3d(
                pn1,
                pn2,
                pd3,
                bn1,
                bn2,
                bd3,
                span1,
                span2,
                span3 - 1,
                nbase_n[0],
                nbase_n[1],
                nbase_d[2],
                u3,
            )

            linalg.matrix_vector(Ginv, u, k4_u)

        elif basis_u == 2:
            u[0] = eva3.evaluation_kernel_3d(
                pn1,
                pd2,
                pd3,
                bn1,
                bd2,
                bd3,
                span1,
                span2 - 1,
                span3 - 1,
                nbase_n[0],
                nbase_d[1],
                nbase_d[2],
                u1,
            )
            u[1] = eva3.evaluation_kernel_3d(
                pd1,
                pn2,
                pd3,
                bd1,
                bn2,
                bd3,
                span1 - 1,
                span2,
                span3 - 1,
                nbase_d[0],
                nbase_n[1],
                nbase_d[2],
                u2,
            )
            u[2] = eva3.evaluation_kernel_3d(
                pd1,
                pd2,
                pn3,
                bd1,
                bd2,
                bn3,
                span1 - 1,
                span2 - 1,
                span3,
                nbase_d[0],
                nbase_d[1],
                nbase_n[2],
                u3,
            )

            k4_u[:] = u / det_df

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k4_v)

        k4[:] = k4_v[:] + k4_u[:]
        # ------------------------------------------------------------------

        #  ---------------- update logical coordinates ---------------------
        e_new[0] = e[0] + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        e_new[1] = e[1] + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        e_new[2] = e[2] + dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6

        if e_new[0] < 0.0 or e_new[0] > 1.0 or e_new[1] < 0.0 or e_new[1] > 1.0 or e_new[2] < 0.0 or e_new[2] > 1.0:
            particles[0:3, ip] = -1.0
            continue

        particles[0, ip] = e_new[0]
        particles[1, ip] = e_new[1]
        particles[2, ip] = e_new[2]
        # ------------------------------------------------------------------

    ierr = 0


# ==========================================================================================================
def pusher_rk4_pc_perp(
    particles: "float[:,:]",
    dt: "float",
    t1: "float[:]",
    t2: "float[:]",
    t3: "float[:]",
    p: "int[:]",
    nel: "int[:]",
    nbase_n: "int[:]",
    nbase_d: "int[:]",
    np: "int",
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
):
    from numpy import empty

    # ============== for velocity evaluation ============
    # spline degrees
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # p + 1 non-vanishing basis functions up tp degree p
    b1 = empty((pn1 + 1, pn1 + 1), dtype=float)
    b2 = empty((pn2 + 1, pn2 + 1), dtype=float)
    b3 = empty((pn3 + 1, pn3 + 1), dtype=float)

    # left and right values for spline evaluation
    l1 = empty(pn1, dtype=float)
    l2 = empty(pn2, dtype=float)
    l3 = empty(pn3, dtype=float)

    r1 = empty(pn1, dtype=float)
    r2 = empty(pn2, dtype=float)
    r3 = empty(pn3, dtype=float)

    # scaling arrays for M-splines
    d1 = empty(pn1, dtype=float)
    d2 = empty(pn2, dtype=float)
    d3 = empty(pn3, dtype=float)

    # p + 1 non-vanishing derivatives
    der1 = empty(pn1 + 1, dtype=float)
    der2 = empty(pn2 + 1, dtype=float)
    der3 = empty(pn3 + 1, dtype=float)

    # non-vanishing N-splines at particle position
    bn1 = empty(pn1 + 1, dtype=float)
    bn2 = empty(pn2 + 1, dtype=float)
    bn3 = empty(pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(pd1 + 1, dtype=float)
    bd2 = empty(pd2 + 1, dtype=float)
    bd3 = empty(pd3 + 1, dtype=float)

    # # velocity field at particle position
    u = empty(3, dtype=float)
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
    df = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    Ginv = empty((3, 3), dtype=float)
    fx = empty(3, dtype=float)
    # ========================================================

    # ======= particle position and velocity =================
    eta = empty(3, dtype=float)
    v = empty(3, dtype=float)
    # ========================================================

    # ===== intermediate stps in 4th order Runge-Kutta =======
    k1 = empty(3, dtype=float)
    k2 = empty(3, dtype=float)
    k3 = empty(3, dtype=float)
    k4 = empty(3, dtype=float)
    k1_u = empty(3, dtype=float)
    k2_u = empty(3, dtype=float)
    k3_u = empty(3, dtype=float)
    k4_u = empty(3, dtype=float)
    k1_v = empty(3, dtype=float)
    k2_v = empty(3, dtype=float)
    k3_v = empty(3, dtype=float)
    k4_v = empty(3, dtype=float)
    # ========================================================

    for ip in range(np):
        eta[:] = particles[0:3, ip]
        v[:] = particles[3:6, ip]

        # ----------- step 1 in Runge-Kutta method -----------------------
        # ========= mapping evaluation =============
        eta1 = eta[0]
        eta2 = eta[1]
        eta3 = eta[2]

        span1f = int(eta[0] * nelf[0]) + pf1
        span2f = int(eta[1] * nelf[1]) + pf2
        span3f = int(eta[2] * nelf[2]) + pf3

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

        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)

        # evaluate Ginv matrix
        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)  ###########
        # ============================================

        # ========== field evaluation ==============
        span1 = int(eta1 * nel[0]) + pn1
        span2 = int(eta2 * nel[1]) + pn2
        span3 = int(eta3 * nel[2]) + pn3

        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, eta1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, eta2, span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, eta3, span3, l3, r3, b3, d3, der3)

        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]

        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # velocity field
        if basis_u == 1:
            u[0] = eva3.evaluation_kernel_3d(
                pd1,
                pn2,
                pn3,
                bd1,
                bn2,
                bn3,
                span1 - 1,
                span2,
                span3,
                nbase_d[0],
                nbase_n[1],
                nbase_n[2],
                u1,
            )
            u[1] = eva3.evaluation_kernel_3d(
                pn1,
                pd2,
                pn3,
                bn1,
                bd2,
                bn3,
                span1,
                span2 - 1,
                span3,
                nbase_n[0],
                nbase_d[1],
                nbase_n[2],
                u2,
            )
            u[2] = eva3.evaluation_kernel_3d(
                pn1,
                pn2,
                pd3,
                bn1,
                bn2,
                bd3,
                span1,
                span2,
                span3 - 1,
                nbase_n[0],
                nbase_n[1],
                nbase_d[2],
                u3,
            )

            linalg.matrix_vector(Ginv, u, k1_u)

        elif basis_u == 2:
            u[0] = eva3.evaluation_kernel_3d(
                pn1,
                pd2,
                pd3,
                bn1,
                bd2,
                bd3,
                span1,
                span2 - 1,
                span3 - 1,
                nbase_n[0],
                nbase_d[1],
                nbase_d[2],
                u1,
            )
            u[1] = eva3.evaluation_kernel_3d(
                pd1,
                pn2,
                pd3,
                bd1,
                bn2,
                bd3,
                span1 - 1,
                span2,
                span3 - 1,
                nbase_d[0],
                nbase_n[1],
                nbase_d[2],
                u2,
            )
            u[2] = eva3.evaluation_kernel_3d(
                pd1,
                pd2,
                pn3,
                bd1,
                bd2,
                bn3,
                span1 - 1,
                span2 - 1,
                span3,
                nbase_d[0],
                nbase_d[1],
                nbase_n[2],
                u3,
            )

            k1_u[:] = u / det_df

        k1_u[0] = 0.0

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k1_v)

        k1[:] = k1_v[:] + k1_u[:]

        # ------------------------------------------------------------------

        # ----------------- step 2 in Runge-Kutta method -------------------
        eta1 = (eta[0] + dt * k1[0] / 2) % 1.0
        eta2 = (eta[1] + dt * k1[1] / 2) % 1.0
        eta3 = (eta[2] + dt * k1[2] / 2) % 1.0

        # ========= mapping evaluation =============
        span1f = int(eta[0] * nelf[0]) + pf1
        span2f = int(eta[1] * nelf[1]) + pf2
        span3f = int(eta[2] * nelf[2]) + pf3

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

        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)

        # evaluate Ginv matrix
        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
        # ============================================

        # ========== field evaluation ==============
        span1 = int(eta1 * nel[0]) + pn1
        span2 = int(eta2 * nel[1]) + pn2
        span3 = int(eta3 * nel[2]) + pn3

        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, eta1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, eta2, span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, eta3, span3, l3, r3, b3, d3, der3)

        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]

        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # velocity field
        if basis_u == 1:
            u[0] = eva3.evaluation_kernel_3d(
                pd1,
                pn2,
                pn3,
                bd1,
                bn2,
                bn3,
                span1 - 1,
                span2,
                span3,
                nbase_d[0],
                nbase_n[1],
                nbase_n[2],
                u1,
            )
            u[1] = eva3.evaluation_kernel_3d(
                pn1,
                pd2,
                pn3,
                bn1,
                bd2,
                bn3,
                span1,
                span2 - 1,
                span3,
                nbase_n[0],
                nbase_d[1],
                nbase_n[2],
                u2,
            )
            u[2] = eva3.evaluation_kernel_3d(
                pn1,
                pn2,
                pd3,
                bn1,
                bn2,
                bd3,
                span1,
                span2,
                span3 - 1,
                nbase_n[0],
                nbase_n[1],
                nbase_d[2],
                u3,
            )

            linalg.matrix_vector(Ginv, u, k2_u)

        elif basis_u == 2:
            u[0] = eva3.evaluation_kernel_3d(
                pn1,
                pd2,
                pd3,
                bn1,
                bd2,
                bd3,
                span1,
                span2 - 1,
                span3 - 1,
                nbase_n[0],
                nbase_d[1],
                nbase_d[2],
                u1,
            )
            u[1] = eva3.evaluation_kernel_3d(
                pd1,
                pn2,
                pd3,
                bd1,
                bn2,
                bd3,
                span1 - 1,
                span2,
                span3 - 1,
                nbase_d[0],
                nbase_n[1],
                nbase_d[2],
                u2,
            )
            u[2] = eva3.evaluation_kernel_3d(
                pd1,
                pd2,
                pn3,
                bd1,
                bd2,
                bn3,
                span1 - 1,
                span2 - 1,
                span3,
                nbase_d[0],
                nbase_d[1],
                nbase_n[2],
                u3,
            )

            k2_u[:] = u / det_df

        k2_u[0] = 0.0

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k2_v)

        k2[:] = k2_v[:] + k2_u[:]
        # ------------------------------------------------------------------

        # ------------------ step 3 in Runge-Kutta method ------------------
        eta1 = (eta[0] + dt * k2[0] / 2) % 1.0
        eta2 = (eta[1] + dt * k2[1] / 2) % 1.0
        eta3 = (eta[2] + dt * k2[2] / 2) % 1.0

        # ========= mapping evaluation =============
        span1f = int(eta[0] * nelf[0]) + pf1
        span2f = int(eta[1] * nelf[1]) + pf2
        span3f = int(eta[2] * nelf[2]) + pf3

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

        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)

        # evaluate Ginv matrix
        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
        # ============================================

        # ========== field evaluation ==============
        span1 = int(eta1 * nel[0]) + pn1
        span2 = int(eta2 * nel[1]) + pn2
        span3 = int(eta3 * nel[2]) + pn3

        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, eta1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, eta2, span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, eta3, span3, l3, r3, b3, d3, der3)

        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]

        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # velocity field
        if basis_u == 1:
            u[0] = eva3.evaluation_kernel_3d(
                pd1,
                pn2,
                pn3,
                bd1,
                bn2,
                bn3,
                span1 - 1,
                span2,
                span3,
                nbase_d[0],
                nbase_n[1],
                nbase_n[2],
                u1,
            )
            u[1] = eva3.evaluation_kernel_3d(
                pn1,
                pd2,
                pn3,
                bn1,
                bd2,
                bn3,
                span1,
                span2 - 1,
                span3,
                nbase_n[0],
                nbase_d[1],
                nbase_n[2],
                u2,
            )
            u[2] = eva3.evaluation_kernel_3d(
                pn1,
                pn2,
                pd3,
                bn1,
                bn2,
                bd3,
                span1,
                span2,
                span3 - 1,
                nbase_n[0],
                nbase_n[1],
                nbase_d[2],
                u3,
            )

            linalg.matrix_vector(Ginv, u, k3_u)

        elif basis_u == 2:
            u[0] = eva3.evaluation_kernel_3d(
                pn1,
                pd2,
                pd3,
                bn1,
                bd2,
                bd3,
                span1,
                span2 - 1,
                span3 - 1,
                nbase_n[0],
                nbase_d[1],
                nbase_d[2],
                u1,
            )
            u[1] = eva3.evaluation_kernel_3d(
                pd1,
                pn2,
                pd3,
                bd1,
                bn2,
                bd3,
                span1 - 1,
                span2,
                span3 - 1,
                nbase_d[0],
                nbase_n[1],
                nbase_d[2],
                u2,
            )
            u[2] = eva3.evaluation_kernel_3d(
                pd1,
                pd2,
                pn3,
                bd1,
                bd2,
                bn3,
                span1 - 1,
                span2 - 1,
                span3,
                nbase_d[0],
                nbase_d[1],
                nbase_n[2],
                u3,
            )

            k3_u[:] = u / det_df

        k3_u[0] = 0.0

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k3_v)

        k3[:] = k3_v[:] + k3_u[:]
        # ------------------------------------------------------------------

        # ------------------ step 4 in Runge-Kutta method ------------------
        eta1 = (eta[0] + dt * k3[0]) % 1.0
        eta2 = (eta[1] + dt * k3[1]) % 1.0
        eta3 = (eta[2] + dt * k3[2]) % 1.0

        # ========= mapping evaluation =============
        span1f = int(eta[0] * nelf[0]) + pf1
        span2f = int(eta[1] * nelf[1]) + pf2
        span3f = int(eta[2] * nelf[2]) + pf3

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

        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)

        # evaluate Ginv matrix
        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
        # ============================================

        # ========== field evaluation ==============
        span1 = int(eta1 * nel[0]) + pn1
        span2 = int(eta2 * nel[1]) + pn2
        span3 = int(eta3 * nel[2]) + pn3

        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, eta1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, eta2, span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, eta3, span3, l3, r3, b3, d3, der3)

        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]

        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # velocity field
        if basis_u == 1:
            u[0] = eva3.evaluation_kernel_3d(
                pd1,
                pn2,
                pn3,
                bd1,
                bn2,
                bn3,
                span1 - 1,
                span2,
                span3,
                nbase_d[0],
                nbase_n[1],
                nbase_n[2],
                u1,
            )
            u[1] = eva3.evaluation_kernel_3d(
                pn1,
                pd2,
                pn3,
                bn1,
                bd2,
                bn3,
                span1,
                span2 - 1,
                span3,
                nbase_n[0],
                nbase_d[1],
                nbase_n[2],
                u2,
            )
            u[2] = eva3.evaluation_kernel_3d(
                pn1,
                pn2,
                pd3,
                bn1,
                bn2,
                bd3,
                span1,
                span2,
                span3 - 1,
                nbase_n[0],
                nbase_n[1],
                nbase_d[2],
                u3,
            )

            linalg.matrix_vector(Ginv, u, k4_u)

        elif basis_u == 2:
            u[0] = eva3.evaluation_kernel_3d(
                pn1,
                pd2,
                pd3,
                bn1,
                bd2,
                bd3,
                span1,
                span2 - 1,
                span3 - 1,
                nbase_n[0],
                nbase_d[1],
                nbase_d[2],
                u1,
            )
            u[1] = eva3.evaluation_kernel_3d(
                pd1,
                pn2,
                pd3,
                bd1,
                bn2,
                bd3,
                span1 - 1,
                span2,
                span3 - 1,
                nbase_d[0],
                nbase_n[1],
                nbase_d[2],
                u2,
            )
            u[2] = eva3.evaluation_kernel_3d(
                pd1,
                pd2,
                pn3,
                bd1,
                bd2,
                bn3,
                span1 - 1,
                span2 - 1,
                span3,
                nbase_d[0],
                nbase_d[1],
                nbase_n[2],
                u3,
            )

            k4_u[:] = u / det_df

        k4_u[0] = 0.0

        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k4_v)

        k4[:] = k4_v[:] + k4_u[:]
        # ------------------------------------------------------------------

        #  ---------------- update logical coordinates ---------------------
        particles[0, ip] = (eta[0] + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6) % 1.0
        particles[1, ip] = (eta[1] + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6) % 1.0
        particles[2, ip] = (eta[2] + dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6) % 1.0

        # ------------------------------------------------------------------

    ierr = 0
