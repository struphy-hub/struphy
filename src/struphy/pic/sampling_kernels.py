from pyccel.decorators import stack_array

# import module for matrix-matrix and matrix-vector multiplications
import struphy.linear_algebra.linalg_kernels as linalg_kernels

# import modules for B-spline evaluation
import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.bsplines.evaluation_kernels_2d as evaluation_kernels_2d
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d

# import module for mapping evaluation
import struphy.geometry.evaluation_kernels as evaluation_kernels


@stack_array('e', 'v')
def set_particles_symmetric_3d_3v(numbers: 'float[:,:]', markers: 'float[:,:]'):

    from numpy import shape, zeros

    e = zeros(3, dtype=float)
    v = zeros(3, dtype=float)

    np = 64*shape(numbers)[0]

    for i_part in range(np):
        ip = i_part % 64

        if ip == 0:
            e[:] = numbers[int(i_part/64), 0:3]
            v[:] = numbers[int(i_part/64), 3:6]

        elif ip % 32 == 0:
            v[2] = 1 - v[2]

        elif ip % 16 == 0:
            v[1] = 1 - v[1]

        elif ip % 8 == 0:
            v[0] = 1 - v[0]

        elif ip % 4 == 0:
            e[2] = 1 - e[2]

        elif ip % 2 == 0:
            e[1] = 1 - e[1]

        else:
            e[0] = 1 - e[0]

        markers[i_part, 0:3] = e
        markers[i_part, 3:6] = v


@stack_array('e', 'v')
def set_particles_symmetric_2d_3v(numbers: 'float[:,:]', markers: 'float[:,:]'):

    from numpy import shape, zeros

    e = zeros(2, dtype=float)
    v = zeros(3, dtype=float)

    np = 32*shape(numbers)[0]

    for i_part in range(np):
        ip = i_part % 32

        if ip == 0:
            e[:] = numbers[int(i_part/32), 0:2]
            v[:] = numbers[int(i_part/32), 2:5]

        elif ip % 16 == 0:
            v[2] = 1 - v[2]

        elif ip % 8 == 0:
            v[1] = 1 - v[1]

        elif ip % 4 == 0:
            v[0] = 1 - v[0]

        elif ip % 2 == 0:
            e[1] = 1 - e[1]

        else:
            e[0] = 1 - e[0]

        markers[i_part, 1:3] = e
        markers[i_part, 3:6] = v


@stack_array('b1', 'b2', 'b3', 'l1', 'l2', 'l3', 'r1', 'r2', 'r3', 'd1', 'd2', 'd3', 'bn1', 'bn2', 'bn3', 'bd1', 'bd2', 'bd3', 'b', 'b_cart', 'b0', 'v', 'vperp', 'vxb0', 'b0xvperp', 'nel1f', 'nel2f', 'nel3f', 'pf1', 'pf2', 'pf3', 'fx', 'df_out', 'dfinv', 'e1', 'e2')
def convert(particles: 'float[:,:]', t1: 'float[:]', t2: 'float[:]', t3: 'float[:]', p: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', ind_d1: 'int[:,:]', ind_d2: 'int[:,:]', ind_d3: 'int[:,:]', b_eq_1: 'float[:,:,:]', b_eq_2: 'float[:,:,:]', b_eq_3: 'float[:,:,:]', kind_map: int, params_map: 'float[:]', tf1: 'float[:]', tf2: 'float[:]', tf3: 'float[:]', pf: 'int[:]', ind1f: 'int[:,:]', ind2f: 'int[:,:]', ind3f: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]'):

    from numpy import shape, empty
    from numpy import sqrt, cos, sin

    # ============== for magnetic field evaluation ============
    # number of elements
    nel1 = shape(ind_n1)[0]
    nel2 = shape(ind_n2)[0]
    nel3 = shape(ind_n3)[0]

    # spline degrees
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # p + 1 non-vanishing basis functions up tp degree p
    b1 = empty((p[0] + 1, p[0] + 1), dtype=float)
    b2 = empty((p[1] + 1, p[1] + 1), dtype=float)
    b3 = empty((p[2] + 1, p[2] + 1), dtype=float)

    # left and right values for spline evaluation
    l1 = empty(p[0], dtype=float)
    l2 = empty(p[1], dtype=float)
    l3 = empty(p[2], dtype=float)

    r1 = empty(p[0], dtype=float)
    r2 = empty(p[1], dtype=float)
    r3 = empty(p[2], dtype=float)

    # scaling arrays for M-splines
    d1 = empty(p[0], dtype=float)
    d2 = empty(p[1], dtype=float)
    d3 = empty(p[2], dtype=float)

    # non-vanishing N-splines at particle position
    bn1 = empty(p[0] + 1, dtype=float)
    bn2 = empty(p[1] + 1, dtype=float)
    bn3 = empty(p[2] + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty(p[0], dtype=float)
    bd2 = empty(p[1], dtype=float)
    bd3 = empty(p[2], dtype=float)

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
    # number of elements
    nel1f = shape(ind1f)[0]
    nel2f = shape(ind2f)[0]
    nel3f = shape(ind3f)[0]

    # spline degrees
    pf1 = pf[0]
    pf2 = pf[1]
    pf3 = pf[2]

    # needed mapping quantities
    fx = empty(3, dtype=float)
    df_out = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    # ==========================================================

    # local basis vectors perpendicular to magnetic field
    e1 = empty(3, dtype=float)
    e2 = empty(3, dtype=float)

    np = shape(particles)[1]

    for ip in range(np):

        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]

        # ========== field evaluation ==============
        span_n1 = int(eta1*nel1) + pn1
        span_n2 = int(eta2*nel2) + pn2

        span_d1 = span_n1 - 1
        span_d2 = span_n2 - 1

        # evaluation of basis functions
        bsplines_kernels.basis_funs_all(t1, pn1, eta1, span_n1, l1, r1, b1, d1)
        bsplines_kernels.basis_funs_all(t2, pn2, eta2, span_n2, l2, r2, b2, d2)

        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]

        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]

        if nel3 > 0:

            span_n3 = int(eta3*nel3) + pn3
            span_d3 = span_n3 - 1

            bsplines_kernels.basis_funs_all(
                t3, pn3, eta3, span_n3, l3, r3, b3, d3)

            bn3[:] = b3[pn3, :]
            bd3[:] = b3[pd3, :pn3] * d3[:]

        # magnetic field (2-form)
        if nel3 > 0:

            b[0] = evaluation_kernels_3d.evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, ind_n1[span_n1 - pn1, :],
                                                              ind_d2[span_d2 - pd2, :], ind_d3[span_d3 - pd3, :], b_eq_1)
            b[1] = evaluation_kernels_3d.evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, ind_d1[span_d1 - pd1, :],
                                                              ind_n2[span_n2 - pn2, :], ind_d3[span_d3 - pd3, :], b_eq_2)
            b[2] = evaluation_kernels_3d.evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, ind_d1[span_d1 - pd1, :],
                                                              ind_d2[span_d2 - pd2, :], ind_n3[span_n3 - pn3, :], b_eq_3)

        else:

            b[0] = evaluation_kernels_2d.evaluation_kernel_2d(
                pn1, pd2, bn1, bd2, ind_n1[span_n1 - pn1, :], ind_d2[span_d2 - pd2, :], b_eq_1[:, :, 0])
            b[1] = evaluation_kernels_2d.evaluation_kernel_2d(
                pd1, pn2, bd1, bn2, ind_d1[span_d1 - pd1, :], ind_n2[span_n2 - pn2, :], b_eq_2[:, :, 0])
            b[2] = evaluation_kernels_2d.evaluation_kernel_2d(
                pd1, pd2, bd1, bd2, ind_d1[span_d1 - pd1, :], ind_d2[span_d2 - pd2, :], b_eq_3[:, :, 0])
        # ==========================================

        # ========= mapping evaluation =============
        # evaluate Jacobian matrix
        evaluation_kernels.df(eta1, eta2, eta3, kind_map, params_map, tf1, tf2,
                              tf3, pf, ind1f, ind2f, ind3f, cx, cy, cz, df_out)

        # evaluate Jacobian determinant
        det_df = abs(linalg_kernels.det(df_out))

        # evaluate inverse Jacobian matrix
        linalg_kernels.matrix_inv_with_det(df_out, det_df, dfinv)

        # extract basis vector perpendicular to b
        e1[0] = dfinv[0, 0]
        e1[1] = dfinv[0, 1]
        e1[2] = dfinv[0, 2]

        e1_norm = sqrt(e1[0]**2 + e1[1]**2 + e1[2]**2)

        e1[:] = e1/e1_norm
        # ==========================================

        # push-forward of magnetic field
        linalg_kernels.matrix_vector(df_out, b, b_cart)
        b_cart[:] = b_cart/det_df

        # absolute value of magnetic field
        b_norm = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)

        # normalized magnetic field direction
        b0[:] = b_cart/b_norm

        # calculate e2 = b0 x e1
        linalg_kernels.cross(b0, e1, e2)

        # calculate Cartesian velocity components
        particles[ip, 3:6] = particles[3, ip]*cos(particles[4, ip])*b0 + particles[3, ip]*sin(
            particles[4, ip])*(cos(particles[5, ip])*e1 + sin(particles[5, ip])*e2)
