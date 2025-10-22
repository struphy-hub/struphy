import cunumpy as xp
from numpy import empty, exp, floor, zeros

import struphy.bsplines.bsplines_kernels as bsp
import struphy.geometry.mappings_kernels as mapping_fast
import struphy.linear_algebra.linalg_kernels as linalg


# ==========================================================================================
def right_hand(
    idnx: "int[:,:]",
    idny: "int[:,:]",
    idnz: "int[:,:]",
    iddx: "int[:,:]",
    iddy: "int[:,:]",
    iddz: "int[:,:]",
    nel1: "int",
    nel2: "int",
    nel3: "int",
    nq1: "int",
    nq2: "int",
    nq3: "int",
    p1: "int",
    p2: "int",
    p3: "int",
    d1: "int",
    d2: "int",
    d3: "int",
    bn1: "float[:,:,:,:]",
    bn2: "float[:,:,:,:]",
    bn3: "float[:,:,:,:]",
    bd1: "float[:,:,:,:]",
    bd2: "float[:,:,:,:]",
    bd3: "float[:,:,:,:]",
    right_1: "float[:,:,:,:,:,:]",
    right_2: "float[:,:,:,:,:,:]",
    right_3: "float[:,:,:,:,:,:]",
    temp_vector_1: "float[:,:,:]",
    temp_vector_2: "float[:,:,:]",
    temp_vector_3: "float[:,:,:]",
):
    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            value = 0.0
                            for il1 in range(d1 + 1):
                                for il2 in range(p2 + 1):
                                    for il3 in range(p3 + 1):
                                        value += (
                                            bd1[ie1, il1, 0, q1]
                                            * bn2[ie2, il2, 0, q2]
                                            * bn3[ie3, il3, 0, q3]
                                            * temp_vector_1[iddx[ie1, il1], idny[ie2, il2], idnz[ie3, il3]]
                                        )

                            right_1[ie1, ie2, ie3, q1, q2, q3] = value

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            value = 0.0
                            for il1 in range(p1 + 1):
                                for il2 in range(d2 + 1):
                                    for il3 in range(p3 + 1):
                                        value += (
                                            bn1[ie1, il1, 0, q1]
                                            * bd2[ie2, il2, 0, q2]
                                            * bn3[ie3, il3, 0, q3]
                                            * temp_vector_2[idnx[ie1, il1], iddy[ie2, il2], idnz[ie3, il3]]
                                        )

                            right_2[ie1, ie2, ie3, q1, q2, q3] = value
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)

    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            value = 0.0
                            for il1 in range(p1 + 1):
                                for il2 in range(p2 + 1):
                                    for il3 in range(d3 + 1):
                                        value += (
                                            bn1[ie1, il1, 0, q1]
                                            * bn2[ie2, il2, 0, q2]
                                            * bd3[ie3, il3, 0, q3]
                                            * temp_vector_3[idnx[ie1, il1], idny[ie2, il2], iddz[ie3, il3]]
                                        )

                            right_3[ie1, ie2, ie3, q1, q2, q3] = value
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel
    ierr = 0


# ==========================================================================================
def weight(
    nel1: "int",
    nel2: "int",
    nel3: "int",
    nq1: "int",
    nq2: "int",
    nq3: "int",
    b1value: "float[:,:,:,:,:,:]",
    b2value: "float[:,:,:,:,:,:]",
    b3value: "float[:,:,:,:,:,:]",
    right_1: "float[:,:,:,:,:,:]",
    right_2: "float[:,:,:,:,:,:]",
    right_3: "float[:,:,:,:,:,:]",
    weight1: "float[:,:,:,:,:,:]",
    weight2: "float[:,:,:,:,:,:]",
    weight3: "float[:,:,:,:,:,:]",
):
    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ie1, ie2, ie3, q1, q2, q3)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            weight1[ie1, ie2, ie3, q1, q2, q3] = (
                                b2value[ie1, ie2, ie3, q1, q2, q3] * right_3[ie1, ie2, ie3, q1, q2, q3]
                                - b3value[ie1, ie2, ie3, q1, q2, q3] * right_2[ie1, ie2, ie3, q1, q2, q3]
                            )
                            weight2[ie1, ie2, ie3, q1, q2, q3] = (
                                b3value[ie1, ie2, ie3, q1, q2, q3] * right_1[ie1, ie2, ie3, q1, q2, q3]
                                - b1value[ie1, ie2, ie3, q1, q2, q3] * right_3[ie1, ie2, ie3, q1, q2, q3]
                            )
                            weight3[ie1, ie2, ie3, q1, q2, q3] = (
                                b1value[ie1, ie2, ie3, q1, q2, q3] * right_2[ie1, ie2, ie3, q1, q2, q3]
                                - b2value[ie1, ie2, ie3, q1, q2, q3] * right_1[ie1, ie2, ie3, q1, q2, q3]
                            )
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    ierr = 0


# ==========================================================================================
def final(
    idnx: "int[:,:]",
    idny: "int[:,:]",
    idnz: "int[:,:]",
    iddx: "int[:,:]",
    iddy: "int[:,:]",
    iddz: "int[:,:]",
    nel1: "int",
    nel2: "int",
    nel3: "int",
    nq1: "int",
    nq2: "int",
    nq3: "int",
    p1: "int",
    p2: "int",
    p3: "int",
    d1: "int",
    d2: "int",
    d3: "int",
    weight1: "float[:,:,:,:,:,:]",
    weight2: "float[:,:,:,:,:,:]",
    weight3: "float[:,:,:,:,:,:]",
    temp_final_1: "float[:,:,:]",
    temp_final_2: "float[:,:,:]",
    temp_final_3: "float[:,:,:]",
    bn1: "float[:,:,:,:]",
    bn2: "float[:,:,:,:]",
    bn3: "float[:,:,:,:]",
    bd1: "float[:,:,:,:]",
    bd2: "float[:,:,:,:]",
    bd3: "float[:,:,:,:]",
):
    temp_final_1[:, :, :] = 0.0
    temp_final_2[:, :, :] = 0.0
    temp_final_3[:, :, :] = 0.0
    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do reduction ( + : temp_final_1) private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for il1 in range(d1 + 1):
                    for il2 in range(p2 + 1):
                        for il3 in range(p3 + 1):
                            value = 0.0
                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):
                                        value += (
                                            weight1[ie1, ie2, ie3, q1, q2, q3]
                                            * bd1[ie1, il1, 0, q1]
                                            * bn2[ie2, il2, 0, q2]
                                            * bn3[ie3, il3, 0, q3]
                                        )

                            temp_final_1[iddx[ie1, il1], idny[ie2, il2], idnz[ie3, il3]] += value

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do reduction ( + : temp_final_2) private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)

    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for il1 in range(p1 + 1):
                    for il2 in range(d2 + 1):
                        for il3 in range(p3 + 1):
                            value = 0.0
                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):
                                        value += (
                                            weight2[ie1, ie2, ie3, q1, q2, q3]
                                            * bn1[ie1, il1, 0, q1]
                                            * bd2[ie2, il2, 0, q2]
                                            * bn3[ie3, il3, 0, q3]
                                        )

                            temp_final_2[idnx[ie1, il1], iddy[ie2, il2], idnz[ie3, il3]] += value

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do reduction ( + : temp_final_3) private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for il1 in range(p1 + 1):
                    for il2 in range(p2 + 1):
                        for il3 in range(d3 + 1):
                            value = 0.0
                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):
                                        value += (
                                            weight3[ie1, ie2, ie3, q1, q2, q3]
                                            * bn1[ie1, il1, 0, q1]
                                            * bn2[ie2, il2, 0, q2]
                                            * bd3[ie3, il3, 0, q3]
                                        )

                            temp_final_3[idnx[ie1, il1], idny[ie2, il2], iddz[ie3, il3]] += value

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    ierr = 0


# ==========================================================================================
def prepre(
    idnx: "int[:,:]",
    idny: "int[:,:]",
    idnz: "int[:,:]",
    iddx: "int[:,:]",
    iddy: "int[:,:]",
    iddz: "int[:,:]",
    det_df: "float[:,:,:,:,:,:]",
    DFIT_11: "float[:,:,:,:,:,:]",
    DFIT_12: "float[:,:,:,:,:,:]",
    DFIT_13: "float[:,:,:,:,:,:]",
    DFIT_21: "float[:,:,:,:,:,:]",
    DFIT_22: "float[:,:,:,:,:,:]",
    DFIT_23: "float[:,:,:,:,:,:]",
    DFIT_31: "float[:,:,:,:,:,:]",
    DFIT_32: "float[:,:,:,:,:,:]",
    DFIT_33: "float[:,:,:,:,:,:]",
    nel1: "int",
    nel2: "int",
    nel3: "int",
    nq1: "int",
    nq2: "int",
    nq3: "int",
    p1: "int",
    p2: "int",
    p3: "int",
    d1: "int",
    d2: "int",
    d3: "int",
    b1value: "float[:,:,:,:,:,:]",
    b2value: "float[:,:,:,:,:,:]",
    b3value: "float[:,:,:,:,:,:]",
    uvalue: "float[:,:,:,:,:,:]",
    b1: "float[:,:,:]",
    b2: "float[:,:,:]",
    b3: "float[:,:,:]",
    dft: "float[:,:]",
    generate_weight1: "float[:]",
    generate_weight3: "float[:]",
    bn1: "float[:,:,:,:]",
    bn2: "float[:,:,:,:]",
    bn3: "float[:,:,:,:]",
    bd1: "float[:,:,:,:]",
    bd2: "float[:,:,:,:]",
    bd3: "float[:,:,:,:]",
    pts1: "float[:,:]",
    pts2: "float[:,:]",
    pts3: "float[:,:]",
    wts1: "float[:,:]",
    wts2: "float[:,:]",
    wts3: "float[:,:]",
):
    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)

    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            value = 0.0
                            for il1 in range(d1 + 1):
                                for il2 in range(p2 + 1):
                                    for il3 in range(p3 + 1):
                                        value += (
                                            bd1[ie1, il1, 0, q1]
                                            * bn2[ie2, il2, 0, q2]
                                            * bn3[ie3, il3, 0, q3]
                                            * b1[iddx[ie1, il1], idny[ie2, il2], idnz[ie3, il3]]
                                        )

                            b1value[ie1, ie2, ie3, q1, q2, q3] = value

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)

    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            value = 0.0
                            for il1 in range(p1 + 1):
                                for il2 in range(d2 + 1):
                                    for il3 in range(p3 + 1):
                                        value += (
                                            bn1[ie1, il1, 0, q1]
                                            * bd2[ie2, il2, 0, q2]
                                            * bn3[ie3, il3, 0, q3]
                                            * b2[idnx[ie1, il1], iddy[ie2, il2], idnz[ie3, il3]]
                                        )

                            b2value[ie1, ie2, ie3, q1, q2, q3] = value
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)

    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            value = 0.0
                            for il1 in range(p1 + 1):
                                for il2 in range(p2 + 1):
                                    for il3 in range(d3 + 1):
                                        value += (
                                            bn1[ie1, il1, 0, q1]
                                            * bn2[ie2, il2, 0, q2]
                                            * bd3[ie3, il3, 0, q3]
                                            * b3[idnx[ie1, il1], idny[ie2, il2], iddz[ie3, il3]]
                                        )

                            b3value[ie1, ie2, ie3, q1, q2, q3] = value
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ie1, ie2, ie3, q1, q2, q3, detdet)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            detdet = wts1[ie1, q1] * wts2[ie2, q2] * wts3[ie3, q3] * uvalue[ie1, ie2, ie3, q1, q2, q3]

                            b1value[ie1, ie2, ie3, q1, q2, q3] = b1value[ie1, ie2, ie3, q1, q2, q3] * detdet
                            b2value[ie1, ie2, ie3, q1, q2, q3] = b2value[ie1, ie2, ie3, q1, q2, q3] * detdet
                            b3value[ie1, ie2, ie3, q1, q2, q3] = b3value[ie1, ie2, ie3, q1, q2, q3] * detdet

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    ierr = 0


# ===================================================================================================================
def piecewise_gather(
    ddt: "float",
    index_shapex: "int[:]",
    index_shapey: "int[:]",
    index_shapez: "int[:]",
    index_diffx: "int",
    index_diffy: "int",
    index_diffz: "int",
    p_shape: "int[:]",
    p_size: "float[:]",
    n_quad: "int[:]",
    pts1: "float[:,:]",
    pts2: "float[:,:]",
    pts3: "float[:,:]",
    Nel: "int[:]",
    particles: "float[:,:]",
    Np_loc: "int",
    Np: "int",
    gather_1: "float[:,:,:,:,:,:]",
    gather_2: "float[:,:,:,:,:,:]",
    gather_3: "float[:,:,:,:,:,:]",
    mid_particles: "float[:,:]",
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
    gather_1[:, :, :, :, :, :] = 0.0
    gather_2[:, :, :, :, :, :] = 0.0
    gather_3[:, :, :, :, :, :] = 0.0
    vel = zeros(3, dtype=float)
    # ==========================
    cell_left = empty(3, dtype=int)
    point_left = zeros(3, dtype=float)
    point_right = zeros(3, dtype=float)

    temp1 = zeros(3, dtype=float)
    temp4 = zeros(3, dtype=float)

    compact = zeros(3, dtype=float)
    compact[0] = (p_shape[0] + 1.0) * p_size[0]
    compact[1] = (p_shape[1] + 1.0) * p_size[1]
    compact[2] = (p_shape[2] + 1.0) * p_size[2]

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
    fx = empty(3, dtype=float)
    # ==========================================================

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do reduction ( + : gather_1, gather_2, gather_3) private (ip, eta1, eta2, eta3, ie1, ie2, ie3, vel, weight_p, point_left, point_right, cell_left, cell_number1, cell_number2, cell_number3, il1, il2 ,il3, q1, q2, q3, temp1, temp4, value_x, value_y, value_z, ww, index1, index2, index3, preindex1, preindex2, preindex3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, det_df)

    for ip in range(Np_loc):
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]

        ie1 = int(eta1 * Nel[0])
        ie2 = int(eta2 * Nel[1])
        ie3 = int(eta3 * Nel[2])

        vel[0] = particles[3, ip] + ddt * mid_particles[0, ip]
        vel[1] = particles[4, ip] + ddt * mid_particles[1, ip]
        vel[2] = particles[5, ip] + ddt * mid_particles[2, ip]
        weight_p = particles[6, ip] / (p_size[0] * p_size[1] * p_size[2]) / Np  # note we need to multiply cell size

        # the points here are still not put in the periodic box [0, 1] x [0, 1] x [0, 1]
        point_left[0] = eta1 - 0.5 * compact[0]
        point_right[0] = eta1 + 0.5 * compact[0]
        point_left[1] = eta2 - 0.5 * compact[1]
        point_right[1] = eta2 + 0.5 * compact[1]
        point_left[2] = eta3 - 0.5 * compact[2]
        point_right[2] = eta3 + 0.5 * compact[2]

        cell_left[0] = int(floor(point_left[0] * Nel[0]))
        cell_left[1] = int(floor(point_left[1] * Nel[1]))
        cell_left[2] = int(floor(point_left[2] * Nel[2]))

        cell_number1 = int(int(floor(point_right[0] * Nel[0])) - cell_left[0] + 1)
        cell_number2 = int(int(floor(point_right[1] * Nel[1])) - cell_left[1] + 1)
        cell_number3 = int(int(floor(point_right[2] * Nel[2])) - cell_left[2] + 1)

        # ======================================
        for il1 in range(cell_number1):
            for il2 in range(cell_number2):
                for il3 in range(cell_number3):
                    for q1 in range(n_quad[0]):
                        for q2 in range(n_quad[1]):
                            for q3 in range(n_quad[2]):
                                temp1[0] = (cell_left[0] + il1) / Nel[0] + pts1[
                                    0, q1
                                ]  # quadrature points in the cell x direction
                                temp4[0] = abs(temp1[0] - eta1) - compact[0] / 2.0  # if > 0, result is 0

                                temp1[1] = (cell_left[1] + il2) / Nel[1] + pts2[0, q2]
                                temp4[1] = abs(temp1[1] - eta2) - compact[1] / 2.0  # if > 0, result is 0

                                temp1[2] = (cell_left[2] + il3) / Nel[2] + pts3[0, q3]
                                temp4[2] = abs(temp1[2] - eta3) - compact[2] / 2.0  # if > 0, result is 0

                                if temp4[0] < 0.0 and temp4[1] < 0.0 and temp4[2] < 0.0:
                                    value_x = bsp.piecewise(p_shape[0], p_size[0], temp1[0] - eta1)
                                    value_y = bsp.piecewise(p_shape[1], p_size[1], temp1[1] - eta2)
                                    value_z = bsp.piecewise(p_shape[2], p_size[2], temp1[2] - eta3)

                                    # ========= mapping evaluation =============
                                    span1f = int(temp1[0] % 1.0 * nelf[0]) + pf[0]
                                    span2f = int(temp1[1] % 1.0 * nelf[1]) + pf[1]
                                    span3f = int(temp1[2] % 1.0 * nelf[2]) + pf[2]
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
                                        temp1[0] % 1.0,
                                        temp1[1] % 1.0,
                                        temp1[2] % 1.0,
                                        df,
                                        fx,
                                        0,
                                    )
                                    # evaluate Jacobian determinant
                                    det_df = abs(linalg.det(df))

                                    ww = weight_p * value_x * value_y * value_z / det_df

                                    preindex1 = int(cell_left[0] + il1 + index_diffx)
                                    preindex2 = int(cell_left[1] + il2 + index_diffy)
                                    preindex3 = int(cell_left[2] + il3 + index_diffz)
                                    index1 = index_shapex[preindex1]
                                    index2 = index_shapey[preindex2]
                                    index3 = index_shapez[preindex3]

                                    gather_1[index1, index2, index3, q1, q2, q3] += vel[0] * ww
                                    gather_2[index1, index2, index3, q1, q2, q3] += vel[1] * ww
                                    gather_3[index1, index2, index3, q1, q2, q3] += vel[2] * ww

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel
    ierr = 0


# ==================================================================================================================
def piecewise_scatter(
    index_shapex: "int[:]",
    index_shapey: "int[:]",
    index_shapez: "int[:]",
    index_diffx: "int",
    index_diffy: "int",
    index_diffz: "int",
    p_shape: "int[:]",
    p_size: "float[:]",
    RK_vector: "float[:,:]",
    Nel: "int[:]",
    Np_loc: "int",
    Np: "int",
    weight_1: "float[:,:,:,:,:,:]",
    weight_2: "float[:,:,:,:,:,:]",
    weight_3: "float[:,:,:,:,:,:]",
    pts1: "float[:,:]",
    pts2: "float[:,:]",
    pts3: "float[:,:]",
    n_quad: "int[:]",
    particles: "float[:,:]",
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
    vel = zeros(3, dtype=float)

    dfinv = zeros((3, 3), dtype=float)
    dfinv_t = zeros((3, 3), dtype=float)
    cell_left = empty(3, dtype=int)
    point_left = zeros(3, dtype=float)
    point_right = zeros(3, dtype=float)

    temp1 = zeros(3, dtype=float)
    temp2 = zeros(3, dtype=float)
    temp3 = empty(3, dtype=int)
    temp4 = zeros(3, dtype=float)

    compact = zeros(3, dtype=float)
    compact[0] = (p_shape[0] + 1.0) * p_size[0]
    compact[1] = (p_shape[1] + 1.0) * p_size[1]
    compact[2] = (p_shape[2] + 1.0) * p_size[2]

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
    fx = empty(3, dtype=float)
    # ==========================================================

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ip, eta1, eta2, eta3, weight_p, point_left, point_right, cell_left, cell_number1, cell_number2, cell_number3, il1, il2, il3, q1, q2, q3, temp1, temp4, value_x, value_y, value_z, ww, index1, index2, index3, preindex1, preindex2, preindex3, vel, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, det_df)

    for ip in range(Np_loc):
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]

        vel[:] = 0.0

        weight_p = particles[6, ip] / (p_size[0] * p_size[1] * p_size[2]) / Np
        # the points here are still not put in the periodic box [0, 1] x [0, 1] x [0, 1]
        point_left[0] = eta1 - 0.5 * compact[0]
        point_right[0] = eta1 + 0.5 * compact[0]
        point_left[1] = eta2 - 0.5 * compact[1]
        point_right[1] = eta2 + 0.5 * compact[1]
        point_left[2] = eta3 - 0.5 * compact[2]
        point_right[2] = eta3 + 0.5 * compact[2]

        cell_left[0] = int(floor(point_left[0] * Nel[0]))
        cell_left[1] = int(floor(point_left[1] * Nel[1]))
        cell_left[2] = int(floor(point_left[2] * Nel[2]))

        cell_number1 = int(int(floor(point_right[0] * Nel[0])) - cell_left[0] + 1)
        cell_number2 = int(int(floor(point_right[1] * Nel[1])) - cell_left[1] + 1)
        cell_number3 = int(int(floor(point_right[2] * Nel[2])) - cell_left[2] + 1)

        # ======================================
        for il1 in range(cell_number1):
            for il2 in range(cell_number2):
                for il3 in range(cell_number3):
                    for q1 in range(n_quad[0]):
                        for q2 in range(n_quad[1]):
                            for q3 in range(n_quad[2]):
                                temp1[0] = (cell_left[0] + il1) / Nel[0] + pts1[
                                    0, q1
                                ]  # quadrature points in the cell x direction
                                temp4[0] = abs(temp1[0] - eta1) - compact[0] / 2  # if > 0, result is 0

                                temp1[1] = (cell_left[1] + il2) / Nel[1] + pts2[0, q2]
                                temp4[1] = abs(temp1[1] - eta2) - compact[1] / 2  # if > 0, result is 0

                                temp1[2] = (cell_left[2] + il3) / Nel[2] + pts3[0, q3]
                                temp4[2] = abs(temp1[2] - eta3) - compact[2] / 2  # if > 0, result is 0

                                if temp4[0] < 0 and temp4[1] < 0 and temp4[2] < 0:
                                    value_x = bsp.piecewise(p_shape[0], p_size[0], temp1[0] - eta1)
                                    value_y = bsp.piecewise(p_shape[1], p_size[1], temp1[1] - eta2)
                                    value_z = bsp.piecewise(p_shape[2], p_size[2], temp1[2] - eta3)

                                    # ========= mapping evaluation =============
                                    span1f = int(temp1[0] % 1.0 * nelf[0]) + pf[0]
                                    span2f = int(temp1[1] % 1.0 * nelf[1]) + pf[1]
                                    span3f = int(temp1[2] % 1.0 * nelf[2]) + pf[2]
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
                                        temp1[0] % 1.0,
                                        temp1[1] % 1.0,
                                        temp1[2] % 1.0,
                                        df,
                                        fx,
                                        0,
                                    )
                                    # evaluate Jacobian determinant
                                    det_df = abs(linalg.det(df))

                                    ww = value_x * value_y * value_z / det_df / (p_size[0] * p_size[1] * p_size[2])

                                    preindex1 = int(cell_left[0] + il1 + index_diffx)
                                    preindex2 = int(cell_left[1] + il2 + index_diffy)
                                    preindex3 = int(cell_left[2] + il3 + index_diffz)
                                    index1 = index_shapex[preindex1]
                                    index2 = index_shapey[preindex2]
                                    index3 = index_shapez[preindex3]

                                    vel[0] += ww * weight_1[index1, index2, index3, q1, q2, q3]
                                    vel[1] += ww * weight_2[index1, index2, index3, q1, q2, q3]
                                    vel[2] += ww * weight_3[index1, index2, index3, q1, q2, q3]

        RK_vector[0, ip] = vel[0]
        RK_vector[1, ip] = vel[1]
        RK_vector[2, ip] = vel[2]

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel
    ierr = 0
