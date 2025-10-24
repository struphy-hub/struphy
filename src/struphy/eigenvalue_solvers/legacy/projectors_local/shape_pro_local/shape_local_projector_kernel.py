import struphy.feec.bsplines_kernels as bsp
import struphy.geometry.mappings_3d_fast as mapping_fast
import struphy.linear_algebra.linalg_kernels as linalg


# ==============================================================================================
def kernel_0_form(
    Np: "int",
    p: "int[:]",
    Nel: "int[:]",
    p_shape: "int[:]",
    p_size: "float[:]",
    particle: "float[:,:]",
    lambdas: "float[:,:,:]",
    kernel_0: "float[:,:,:,:,:,:]",
    num_cell: "int[:]",
    coeff_x: "float[:]",
    coeff_y: "float[:]",
    coeff_z: "float[:]",
    NbaseN: "int[:]",
    related: "int[:]",
    Np_loc: "int",
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
    from numpy import empty, floor, zeros

    cell_left = zeros(3, dtype=int)
    point_left = zeros(3, dtype=float)
    point_right = zeros(3, dtype=float)
    cell_number = zeros(3, dtype=int)
    compact = zeros(3, dtype=float)
    width = zeros(3, dtype=int)
    width2 = zeros(3, dtype=int)

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

    lambdas[:, :, :] = 0.0
    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do reduction ( + : kernel_0, lambdas) private (ip, w, width2, cell_left, point_left, point_right, cell_number, compact, width, mat_f, i1, i2, i3, il1, il2, il3, index1, index2, index3, value_x, value_y, value_z, final_1, final_2, final_3, lambda_index1, lambda_index2, lambda_index3, global_i1, global_i2, global_i3, global_il1, global_il2, global_il3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, det_df, eta1, eta2, eta3)
    for ip in range(Np_loc):
        # lambdas[:,:,:] = 0.0
        w = particle[6, ip] / Np
        compact[0] = (p_shape[0] + 1.0) * p_size[0]
        compact[1] = (p_shape[1] + 1.0) * p_size[1]
        compact[2] = (p_shape[2] + 1.0) * p_size[2]

        point_left[0] = particle[0, ip] - 0.5 * compact[0]
        point_right[0] = particle[0, ip] + 0.5 * compact[0]
        point_left[1] = particle[1, ip] - 0.5 * compact[1]
        point_right[1] = particle[1, ip] + 0.5 * compact[1]
        point_left[2] = particle[2, ip] - 0.5 * compact[2]
        point_right[2] = particle[2, ip] + 0.5 * compact[2]

        cell_left[0] = int(floor(point_left[0] * Nel[0]))
        cell_left[1] = int(floor(point_left[1] * Nel[1]))
        cell_left[2] = int(floor(point_left[2] * Nel[2]))

        cell_number[0] = int(floor(point_right[0] * Nel[0])) - cell_left[0] + 1
        cell_number[1] = int(floor(point_right[1] * Nel[1])) - cell_left[1] + 1
        cell_number[2] = int(floor(point_right[2] * Nel[2])) - cell_left[2] + 1

        for il1 in range(3):
            if (p[il1] + cell_number[il1] - 1) > NbaseN[il1]:
                width[il1] = NbaseN[il1]
            else:
                width[il1] = p[il1] + cell_number[il1] - 1

        mat_f = empty(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], num_cell[1], num_cell[2]),
            dtype=float,
        )
        mat_f[:, :, :, :, :, :] = 0.0

        # evaluation of function at interpolation points
        for i1 in range(cell_number[0]):
            index1 = cell_left[0] + i1
            for i2 in range(cell_number[1]):
                index2 = cell_left[1] + i2
                for i3 in range(cell_number[2]):  # num_cell = 1, p = 1; num_cell = 2, p >= 2
                    index3 = cell_left[2] + i3
                    for il1 in range(num_cell[0]):
                        eta1 = 1.0 / Nel[0] * index1 + 1.0 / Nel[0] / num_cell[0] * il1
                        value_x = bsp.piecewise(p_shape[0], p_size[0], abs(eta1 - particle[0, ip]))
                        for il2 in range(num_cell[1]):
                            eta2 = 1.0 / Nel[1] * index2 + 1 / Nel[1] / num_cell[1] * il2
                            value_y = bsp.piecewise(p_shape[1], p_size[1], abs(eta2 - particle[1, ip]))
                            for il3 in range(num_cell[2]):
                                eta3 = 1.0 / Nel[2] * index3 + 1 / Nel[2] / num_cell[2] * il3
                                value_z = bsp.piecewise(p_shape[2], p_size[2], abs(eta3 - particle[2, ip]))
                                # ========= mapping evaluation =============
                                span1f = int((eta1 % 1.0) * nelf[0]) + pf1
                                span2f = int((eta2 % 1.0) * nelf[1]) + pf2
                                span3f = int((eta3 % 1.0) * nelf[2]) + pf3
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
                                    eta1 % 1.0,
                                    eta2 % 1.0,
                                    eta3 % 1.0,
                                    df,
                                    fx,
                                    0,
                                )
                                det_df = abs(linalg.det(df))
                                mat_f[i1, i2, i3, il1, il2, il3] = (
                                    value_x * value_y * value_z / (p_size[0] * p_size[1] * p_size[2]) / det_df
                                )  # here should devided by det_df = abs(linalg.det(df))

        # coefficients
        for i1 in range(cell_number[0]):
            index1 = cell_left[0] + i1
            for i2 in range(cell_number[1]):
                index2 = cell_left[1] + i2
                for i3 in range(cell_number[2]):
                    index3 = cell_left[2] + i3
                    for lambda_index1 in range(p[0]):
                        final_1 = (lambda_index1 + index1) % NbaseN[0]
                        for lambda_index2 in range(p[1]):
                            final_2 = (lambda_index2 + index2) % NbaseN[1]
                            for lambda_index3 in range(p[2]):
                                final_3 = (lambda_index3 + index3) % NbaseN[2]
                                for il1 in range(num_cell[0]):
                                    for il2 in range(num_cell[1]):
                                        for il3 in range(num_cell[2]):
                                            lambdas[final_1, final_2, final_3] += (
                                                coeff_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * mat_f[i1, i2, i3, il1, il2, il3]
                                            )

        for i1 in range(width[0]):
            global_i1 = (cell_left[0] + i1) % NbaseN[0]
            for i2 in range(width[1]):
                global_i2 = (cell_left[1] + i2) % NbaseN[1]
                for i3 in range(width[2]):
                    global_i3 = (cell_left[2] + i3) % NbaseN[2]
                    for il1 in range(3):
                        width2[il1] = 2 * related[il1] + 1
                    if width2[0] > NbaseN[0]:
                        width2[0] = NbaseN[0]
                    if width2[1] > NbaseN[1]:
                        width2[1] = NbaseN[1]
                    if width2[2] > NbaseN[2]:
                        width2[2] = NbaseN[2]
                    for il1 in range(width2[0]):
                        global_il1 = (global_i1 + il1 - int(floor(width2[0] / 2.0))) % NbaseN[0]
                        for il2 in range(width2[1]):
                            global_il2 = (global_i2 + il2 - int(floor(width2[1] / 2.0))) % NbaseN[1]
                            for il3 in range(width2[2]):
                                global_il3 = (global_i3 + il3 - int(floor(width2[2] / 2.0))) % NbaseN[2]
                                kernel_0[global_i1, global_i2, global_i3, il1, il2, il3] += w * (
                                    lambdas[global_i1, global_i2, global_i3]
                                    * lambdas[global_il1, global_il2, global_il3]
                                )

        del mat_f
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    ierr = 0


# ==============================================================================================
def potential_kernel_0_form(
    Np: "int",
    p: "int[:]",
    Nel: "int[:]",
    p_shape: "int[:]",
    p_size: "float[:]",
    particle: "float[:,:]",
    lambdas: "float[:,:,:]",
    kernel_0: "float[:,:,:,:,:,:]",
    num_cell: "int[:]",
    coeff_x: "float[:]",
    coeff_y: "float[:]",
    coeff_z: "float[:]",
    NbaseN: "int[:]",
    related: "int[:]",
    Np_loc: "int",
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
    from numpy import empty, floor, zeros

    cell_left = zeros(3, dtype=int)
    point_left = zeros(3, dtype=float)
    point_right = zeros(3, dtype=float)
    cell_number = zeros(3, dtype=int)
    compact = zeros(3, dtype=float)
    width = zeros(3, dtype=int)
    width2 = zeros(3, dtype=int)

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

    lambdas[:, :, :] = 0.0
    det_df = params_map[0] * params_map[1] * params_map[2]
    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do reduction ( + : lambdas) private (ip, w, width, cell_left, point_left, point_right, cell_number, compact, mat_f, i1, i2, i3, il1, il2, il3, index1, index2, index3, value_x, value_y, value_z, final_1, final_2, final_3, eta1, eta2, eta3, lambda_index1, lambda_index2, lambda_index3)
    for ip in range(Np_loc):
        # lambdas[:,:,:] = 0.0
        w = particle[6, ip] / Np
        compact[0] = (p_shape[0] + 1.0) * p_size[0]
        compact[1] = (p_shape[1] + 1.0) * p_size[1]
        compact[2] = (p_shape[2] + 1.0) * p_size[2]

        point_left[0] = particle[0, ip] - 0.5 * compact[0]
        point_right[0] = particle[0, ip] + 0.5 * compact[0]
        point_left[1] = particle[1, ip] - 0.5 * compact[1]
        point_right[1] = particle[1, ip] + 0.5 * compact[1]
        point_left[2] = particle[2, ip] - 0.5 * compact[2]
        point_right[2] = particle[2, ip] + 0.5 * compact[2]

        cell_left[0] = int(floor(point_left[0] * Nel[0]))
        cell_left[1] = int(floor(point_left[1] * Nel[1]))
        cell_left[2] = int(floor(point_left[2] * Nel[2]))

        cell_number[0] = int(floor(point_right[0] * Nel[0])) - cell_left[0] + 1
        cell_number[1] = int(floor(point_right[1] * Nel[1])) - cell_left[1] + 1
        cell_number[2] = int(floor(point_right[2] * Nel[2])) - cell_left[2] + 1

        for il1 in range(3):
            if (p[il1] + cell_number[il1] - 1) > NbaseN[il1]:
                width[il1] = NbaseN[il1]
            else:
                width[il1] = p[il1] + cell_number[il1] - 1

        mat_f = empty(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], num_cell[1], num_cell[2]),
            dtype=float,
        )
        mat_f[:, :, :, :, :, :] = 0.0

        # evaluation of function at interpolation points
        for i1 in range(cell_number[0]):
            index1 = cell_left[0] + i1
            for i2 in range(cell_number[1]):
                index2 = cell_left[1] + i2
                for i3 in range(cell_number[2]):  # num_cell = 1, p = 1; num_cell = 2, p >= 2
                    index3 = cell_left[2] + i3
                    for il1 in range(num_cell[0]):
                        eta1 = 1.0 / Nel[0] * index1 + 1.0 / Nel[0] / num_cell[0] * il1
                        value_x = bsp.piecewise(p_shape[0], p_size[0], abs(eta1 - particle[0, ip]))
                        for il2 in range(num_cell[1]):
                            eta2 = 1.0 / Nel[1] * index2 + 1 / Nel[1] / num_cell[1] * il2
                            value_y = bsp.piecewise(p_shape[1], p_size[1], abs(eta2 - particle[1, ip]))
                            for il3 in range(num_cell[2]):
                                eta3 = 1.0 / Nel[2] * index3 + 1 / Nel[2] / num_cell[2] * il3
                                value_z = bsp.piecewise(p_shape[2], p_size[2], abs(eta3 - particle[2, ip]))
                                mat_f[i1, i2, i3, il1, il2, il3] = (
                                    w * value_x * value_y * value_z / (p_size[0] * p_size[1] * p_size[2]) / det_df
                                )  # here should devided by det_df = abs(linalg.det(df))

        # coefficients
        for i1 in range(cell_number[0]):
            index1 = cell_left[0] + i1
            for i2 in range(cell_number[1]):
                index2 = cell_left[1] + i2
                for i3 in range(cell_number[2]):
                    index3 = cell_left[2] + i3
                    for lambda_index1 in range(p[0]):
                        final_1 = (lambda_index1 + index1) % NbaseN[0]
                        for lambda_index2 in range(p[1]):
                            final_2 = (lambda_index2 + index2) % NbaseN[1]
                            for lambda_index3 in range(p[2]):
                                final_3 = (lambda_index3 + index3) % NbaseN[2]
                                for il1 in range(num_cell[0]):
                                    for il2 in range(num_cell[1]):
                                        for il3 in range(num_cell[2]):
                                            lambdas[final_1, final_2, final_3] += (
                                                coeff_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * mat_f[i1, i2, i3, il1, il2, il3]
                                            )

        del mat_f
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    ierr = 0


# ==============================================================================================
def kernel_1_form(
    right1: "float[:,:,:]",
    right2: "float[:,:,:]",
    right3: "float[:,:,:]",
    pts1: "float[:]",
    pts2: "float[:]",
    pts3: "float[:]",
    wts1: "float[:]",
    wts2: "float[:]",
    wts3: "float[:]",
    Np: "int",
    quad: "int[:]",
    p: "int[:]",
    Nel: "int[:]",
    p_shape: "int[:]",
    p_size: "float[:]",
    particle: "float[:,:]",
    lambdas_11: "float[:,:,:]",
    lambdas_12: "float[:,:,:]",
    lambdas_13: "float[:,:,:]",
    lambdas_21: "float[:,:,:]",
    lambdas_22: "float[:,:,:]",
    lambdas_23: "float[:,:,:]",
    lambdas_31: "float[:,:,:]",
    lambdas_32: "float[:,:,:]",
    lambdas_33: "float[:,:,:]",
    kernel_11: "float[:,:,:,:,:,:]",
    kernel_12: "float[:,:,:,:,:,:]",
    kernel_13: "float[:,:,:,:,:,:]",
    kernel_22: "float[:,:,:,:,:,:]",
    kernel_23: "float[:,:,:,:,:,:]",
    kernel_33: "float[:,:,:,:,:,:]",
    num_cell: "int[:]",
    coeff_i_x: "float[:]",
    coeff_i_y: "float[:]",
    coeff_i_z: "float[:]",
    coeff_h_x: "float[:]",
    coeff_h_y: "float[:]",
    coeff_h_z: "float[:]",
    NbaseN: "int[:]",
    NbaseD: "int[:]",
    related: "int[:]",
    Np_loc: "int",
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
    from numpy import empty, floor, zeros

    cell_left = zeros(3, dtype=int)
    point_left = zeros(3, dtype=float)
    point_right = zeros(3, dtype=float)
    cell_number = zeros(3, dtype=int)
    compact = zeros(3, dtype=float)

    width = zeros(3, dtype=int)
    width[0] = p[0] + cell_number[0] - 1  # the number of coefficients obtained from this smoothed delta function
    width[1] = p[1] + cell_number[1] - 1  # the number of coefficients obtained from this smoothed delta function
    width[2] = p[2] + cell_number[2] - 1  # the number of coefficients obtained from this smoothed delta function

    width2 = zeros(3, dtype=int)
    width2[0] = 2 * related[0] + 1
    width2[1] = 2 * related[1] + 1
    width2[2] = 2 * related[2] + 1

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
    df = zeros((3, 3), dtype=float)
    dft = zeros((3, 3), dtype=float)
    # ====================================
    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do reduction ( + : kernel_11, kernel_12, kernel_13, kernel_22, kernel_23, kernel_33, right1, right2, right3) private (mid1, mid2, mid3, ip, w, det_df, vol, width2, lambdas_11, lambdas_22, lambdas_33, lambdas_12, lambdas_13, lambdas_21, lambdas_23, lambdas_31, lambdas_32, cell_left, point_left, point_right, cell_number, compact, width, mat_11, mat_12, mat_13, mat_21, mat_22, mat_23, mat_31, mat_32, mat_33, i1, i2, i3, il1, il2, il3, index1, index2, index3, value_x, value_y, value_z, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, dft, lambda_index1, lambda_index2, lambda_index3, global_i1, global_i2, global_i3, global_il1, global_il2, global_il3, f_int, jl1, eta1, eta2, eta3, final_index1, final_index2, final_index3)
    for ip in range(Np_loc):
        w = particle[6, ip] / Np

        lambdas_11[:, :, :] = 0.0
        lambdas_12[:, :, :] = 0.0
        lambdas_13[:, :, :] = 0.0

        lambdas_21[:, :, :] = 0.0
        lambdas_22[:, :, :] = 0.0
        lambdas_23[:, :, :] = 0.0

        lambdas_31[:, :, :] = 0.0
        lambdas_32[:, :, :] = 0.0
        lambdas_33[:, :, :] = 0.0

        # ==================================
        compact[0] = (p_shape[0] + 1.0) * p_size[0]
        compact[1] = (p_shape[1] + 1.0) * p_size[1]
        compact[2] = (p_shape[2] + 1.0) * p_size[2]

        point_left[0] = particle[0, ip] - 0.5 * compact[0]
        point_right[0] = particle[0, ip] + 0.5 * compact[0]
        point_left[1] = particle[1, ip] - 0.5 * compact[1]
        point_right[1] = particle[1, ip] + 0.5 * compact[1]
        point_left[2] = particle[2, ip] - 0.5 * compact[2]
        point_right[2] = particle[2, ip] + 0.5 * compact[2]

        cell_left[0] = int(floor(point_left[0] * Nel[0]))
        cell_left[1] = int(floor(point_left[1] * Nel[1]))
        cell_left[2] = int(floor(point_left[2] * Nel[2]))

        cell_number[0] = int(floor(point_right[0] * Nel[0])) - cell_left[0] + 1
        cell_number[1] = int(floor(point_right[1] * Nel[1])) - cell_left[1] + 1
        cell_number[2] = int(floor(point_right[2] * Nel[2])) - cell_left[2] + 1

        vol = 1.0 / (p_size[0] * p_size[1] * p_size[2])

        # evaluation of function at interpolation/quadrature points
        mat_11 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], 2, num_cell[1], num_cell[2], quad[0]),
            dtype=float,
        )
        mat_21 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], 2, num_cell[1], num_cell[2], quad[0]),
            dtype=float,
        )
        mat_31 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], 2, num_cell[1], num_cell[2], quad[0]),
            dtype=float,
        )

        mat_12 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], 2, num_cell[2], quad[1]),
            dtype=float,
        )
        mat_22 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], 2, num_cell[2], quad[1]),
            dtype=float,
        )
        mat_32 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], 2, num_cell[2], quad[1]),
            dtype=float,
        )

        mat_13 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], num_cell[1], 2, quad[2]),
            dtype=float,
        )
        mat_23 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], num_cell[1], 2, quad[2]),
            dtype=float,
        )
        mat_33 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], num_cell[1], 2, quad[2]),
            dtype=float,
        )

        for i1 in range(cell_number[0]):
            index1 = cell_left[0] + i1
            for i2 in range(cell_number[1]):
                index2 = cell_left[1] + i2
                for i3 in range(cell_number[2]):  # num_cell = 1, p = 1; num_cell = 2, p >= 2
                    index3 = cell_left[2] + i3
                    for il1 in range(2):
                        for il2 in range(num_cell[1]):
                            eta2 = 1.0 / Nel[1] * index2 + 1 / Nel[1] / num_cell[1] * il2
                            span2f = int((eta2 % 1.0) * nelf[1]) + pf2
                            value_y = bsp.piecewise(p_shape[1], p_size[1], abs(eta2 - particle[1, ip]))
                            for il3 in range(num_cell[2]):
                                eta3 = 1.0 / Nel[2] * index3 + 1 / Nel[2] / num_cell[2] * il3
                                span3f = int((eta3 % 1.0) * nelf[2]) + pf3
                                value_z = bsp.piecewise(p_shape[2], p_size[2], abs(eta3 - particle[2, ip]))
                                for jl1 in range(quad[0]):
                                    eta1 = 1.0 / Nel[0] * index1 + 1 / Nel[0] / 2.0 * il1 + pts1[jl1]
                                    value_x = bsp.piecewise(p_shape[0], p_size[0], abs(eta1 - particle[0, ip]))
                                    # ========= mapping evaluation =============
                                    span1f = int((eta1 % 1.0) * nelf[0]) + pf1
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
                                        eta1 % 1.0,
                                        eta2 % 1.0,
                                        eta3 % 1.0,
                                        df,
                                        fx,
                                        0,
                                    )
                                    # evaluate transpose of Jacobian matrix
                                    linalg.transpose(df, dft)
                                    det_df = abs(linalg.det(df))

                                    mat_11[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[0, 0] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_21[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[0, 1] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_31[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[0, 2] * value_x * value_y * value_z / det_df * vol
                                    )

                    for il1 in range(num_cell[0]):
                        eta1 = 1.0 / Nel[0] * index1 + 1.0 / Nel[0] / num_cell[0] * il1
                        span1f = int((eta1 % 1.0) * nelf[0]) + pf1
                        value_x = bsp.piecewise(p_shape[0], p_size[0], abs(eta1 - particle[0, ip]))
                        for il2 in range(2):
                            for il3 in range(num_cell[2]):
                                eta3 = 1.0 / Nel[2] * index3 + 1.0 / Nel[2] / num_cell[2] * il3
                                span3f = int((eta3 % 1.0) * nelf[2]) + pf3
                                value_z = bsp.piecewise(p_shape[2], p_size[2], abs(eta3 - particle[2, ip]))
                                for jl1 in range(quad[1]):
                                    eta2 = 1.0 / Nel[1] * index2 + 1.0 / Nel[1] / 2.0 * il2 + pts2[jl1]
                                    value_y = bsp.piecewise(p_shape[1], p_size[1], abs(eta2 - particle[1, ip]))
                                    # ========= mapping evaluation =============
                                    span2f = int((eta2 % 1.0) * nelf[1]) + pf2
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
                                        eta1 % 1.0,
                                        eta2 % 1.0,
                                        eta3 % 1.0,
                                        df,
                                        fx,
                                        0,
                                    )
                                    # evaluate transpose of Jacobian matrix
                                    linalg.transpose(df, dft)
                                    det_df = abs(linalg.det(df))

                                    mat_12[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[1, 0] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_22[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[1, 1] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_32[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[1, 2] * value_x * value_y * value_z / det_df * vol
                                    )
                    for il1 in range(num_cell[0]):
                        eta1 = 1.0 / Nel[0] * index1 + 1.0 / Nel[0] / num_cell[0] * il1
                        span1f = int((eta1 % 1.0) * nelf[0]) + pf1
                        value_x = bsp.piecewise(p_shape[0], p_size[0], abs(eta1 - particle[0, ip]))
                        for il2 in range(num_cell[1]):
                            eta2 = 1.0 / Nel[1] * index2 + 1.0 / Nel[1] / num_cell[1] * il2
                            span2f = int((eta2 % 1.0) * nelf[1]) + pf2
                            value_y = bsp.piecewise(p_shape[1], p_size[1], abs(eta2 - particle[1, ip]))
                            for il3 in range(2):
                                for jl1 in range(quad[2]):
                                    eta3 = 1.0 / Nel[2] * index3 + 1.0 / Nel[2] / 2.0 * il3 + pts3[jl1]
                                    value_z = bsp.piecewise(p_shape[2], p_size[2], abs(eta3 - particle[2, ip]))
                                    # ========= mapping evaluation =============
                                    span3f = int((eta3 % 1.0) * nelf[2]) + pf3
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
                                        eta1 % 1.0,
                                        eta2 % 1.0,
                                        eta3 % 1.0,
                                        df,
                                        fx,
                                        0,
                                    )
                                    # evaluate transpose of Jacobian matrix
                                    linalg.transpose(df, dft)
                                    det_df = abs(linalg.det(df))

                                    mat_13[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[2, 0] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_23[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[2, 1] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_33[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[2, 2] * value_x * value_y * value_z / det_df * vol
                                    )

        for i1 in range(cell_number[0]):
            index1 = cell_left[0] + i1
            for i2 in range(cell_number[1]):
                index2 = cell_left[1] + i2
                for i3 in range(cell_number[2]):
                    index3 = cell_left[2] + i3

                    for lambda_index1 in range(p[0]):
                        final_index1 = (lambda_index1 + index1) % NbaseD[0]
                        for lambda_index2 in range(p[1]):
                            final_index2 = (lambda_index2 + index2) % NbaseN[1]
                            for lambda_index3 in range(p[2]):
                                final_index3 = (lambda_index3 + index3) % NbaseN[2]
                                for il1 in range(2):
                                    for il2 in range(num_cell[1]):
                                        for il3 in range(num_cell[2]):
                                            f_int = 0.0
                                            for jl1 in range(quad[0]):
                                                f_int += wts1[jl1] * mat_11[i1, i2, i3, il1, il2, il3, jl1]
                                            mid1 = (
                                                coeff_h_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_11[final_index1, final_index2, final_index3] += mid1
                                            f_int = 0.0
                                            for jl1 in range(quad[0]):
                                                f_int += wts1[jl1] * mat_21[i1, i2, i3, il1, il2, il3, jl1]
                                            mid2 = (
                                                coeff_h_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_21[final_index1, final_index2, final_index3] += mid2
                                            f_int = 0.0
                                            for jl1 in range(quad[0]):
                                                f_int += wts1[jl1] * mat_31[i1, i2, i3, il1, il2, il3, jl1]
                                            mid3 = (
                                                coeff_h_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_31[final_index1, final_index2, final_index3] += mid3
                                            right1[final_index1, final_index2, final_index3] += w * (
                                                particle[3, ip] * mid1 + particle[4, ip] * mid2 + particle[5, ip] * mid3
                                            )

                    for lambda_index1 in range(p[0]):
                        final_index1 = (lambda_index1 + index1) % NbaseN[0]
                        for lambda_index2 in range(p[1]):
                            final_index2 = (lambda_index2 + index2) % NbaseD[1]
                            for lambda_index3 in range(p[2]):
                                final_index3 = (lambda_index3 + index3) % NbaseN[2]
                                for il1 in range(num_cell[0]):
                                    for il2 in range(2):
                                        for il3 in range(num_cell[2]):
                                            f_int = 0.0
                                            for jl1 in range(quad[1]):
                                                f_int += wts2[jl1] * mat_12[i1, i2, i3, il1, il2, il3, jl1]
                                            mid1 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_h_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_12[final_index1, final_index2, final_index3] += mid1
                                            f_int = 0.0
                                            for jl1 in range(quad[1]):
                                                f_int += wts2[jl1] * mat_22[i1, i2, i3, il1, il2, il3, jl1]
                                            mid2 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_h_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_22[final_index1, final_index2, final_index3] += mid2
                                            f_int = 0.0
                                            for jl1 in range(quad[1]):
                                                f_int += wts2[jl1] * mat_32[i1, i2, i3, il1, il2, il3, jl1]
                                            mid3 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_h_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_32[final_index1, final_index2, final_index3] += mid3
                                            right2[final_index1, final_index2, final_index3] += w * (
                                                particle[3, ip] * mid1 + particle[4, ip] * mid2 + particle[5, ip] * mid3
                                            )

                    for lambda_index1 in range(p[0]):
                        final_index1 = (lambda_index1 + index1) % NbaseN[0]
                        for lambda_index2 in range(p[1]):
                            final_index2 = (lambda_index2 + index2) % NbaseN[1]
                            for lambda_index3 in range(p[2]):
                                final_index3 = (lambda_index3 + index3) % NbaseD[2]
                                for il1 in range(num_cell[0]):
                                    for il2 in range(num_cell[1]):
                                        for il3 in range(2):
                                            f_int = 0.0
                                            for jl1 in range(quad[2]):
                                                f_int += wts3[jl1] * mat_13[i1, i2, i3, il1, il2, il3, jl1]
                                            mid1 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_h_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_13[final_index1, final_index2, final_index3] += mid1
                                            f_int = 0.0
                                            for jl1 in range(quad[2]):
                                                f_int += wts3[jl1] * mat_23[i1, i2, i3, il1, il2, il3, jl1]
                                            mid2 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_h_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_23[final_index1, final_index2, final_index3] += mid2
                                            f_int = 0.0
                                            for jl1 in range(quad[2]):
                                                f_int += wts3[jl1] * mat_33[i1, i2, i3, il1, il2, il3, jl1]
                                            mid3 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_h_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_33[final_index1, final_index2, final_index3] += mid3
                                            right3[final_index1, final_index2, final_index3] += w * (
                                                particle[3, ip] * mid1 + particle[4, ip] * mid2 + particle[5, ip] * mid3
                                            )

        # print('check_inside_lambda', lambdas_11)
        for il1 in range(3):
            width[il1] = p[il1] + cell_number[il1] - 1

        if width[0] > NbaseD[0]:
            width[0] = NbaseD[0]
        if width[1] > NbaseN[1]:
            width[1] = NbaseN[1]
        if width[2] > NbaseN[2]:
            width[2] = NbaseN[2]
        for i1 in range(width[0]):
            global_i1 = (cell_left[0] + i1) % NbaseD[0]
            for i2 in range(width[1]):
                global_i2 = (cell_left[1] + i2) % NbaseN[1]
                for i3 in range(width[2]):
                    global_i3 = (cell_left[2] + i3) % NbaseN[2]
                    # ===== 11 compoponent ==========
                    for il1 in range(3):
                        width2[il1] = 2 * related[il1] + 1
                    if width2[0] > NbaseD[0]:
                        width2[0] = NbaseD[0]
                    if width2[1] > NbaseN[1]:
                        width2[1] = NbaseN[1]
                    if width2[2] > NbaseN[2]:
                        width2[2] = NbaseN[2]
                    for il1 in range(width2[0]):
                        global_il1 = (global_i1 + il1 - int(floor(width2[0] / 2.0))) % NbaseD[0]
                        for il2 in range(width2[1]):
                            global_il2 = (global_i2 + il2 - int(floor(width2[1] / 2.0))) % NbaseN[1]
                            for il3 in range(width2[2]):
                                global_il3 = (global_i3 + il3 - int(floor(width2[2] / 2.0))) % NbaseN[2]
                                kernel_11[global_i1, global_i2, global_i3, il1, il2, il3] += w * (
                                    lambdas_11[global_i1, global_i2, global_i3]
                                    * lambdas_11[global_il1, global_il2, global_il3]
                                    + lambdas_21[global_i1, global_i2, global_i3]
                                    * lambdas_21[global_il1, global_il2, global_il3]
                                    + lambdas_31[global_i1, global_i2, global_i3]
                                    * lambdas_31[global_il1, global_il2, global_il3]
                                )
                    # ===== 12 compoponent ==========
                    for il1 in range(3):
                        width2[il1] = 2 * related[il1] + 1
                    if width2[0] > NbaseN[0]:
                        width2[0] = NbaseN[0]
                    if width2[1] > NbaseD[1]:
                        width2[1] = NbaseD[1]
                    if width2[2] > NbaseN[2]:
                        width2[2] = NbaseN[2]
                    for il1 in range(width2[0]):
                        global_il1 = (global_i1 + il1 - int(floor(width2[0] / 2.0))) % NbaseN[0]
                        for il2 in range(width2[1]):
                            global_il2 = (global_i2 + il2 - int(floor(width2[1] / 2.0))) % NbaseD[1]
                            for il3 in range(width2[2]):
                                global_il3 = (global_i3 + il3 - int(floor(width2[2] / 2.0))) % NbaseN[2]
                                kernel_12[global_i1, global_i2, global_i3, il1, il2, il3] += w * (
                                    lambdas_11[global_i1, global_i2, global_i3]
                                    * lambdas_12[global_il1, global_il2, global_il3]
                                    + lambdas_21[global_i1, global_i2, global_i3]
                                    * lambdas_22[global_il1, global_il2, global_il3]
                                    + lambdas_31[global_i1, global_i2, global_i3]
                                    * lambdas_32[global_il1, global_il2, global_il3]
                                )

                    # ===== 13 compoponent ==========
                    for il1 in range(3):
                        width2[il1] = 2 * related[il1] + 1
                    if width2[0] > NbaseN[0]:
                        width2[0] = NbaseN[0]
                    if width2[1] > NbaseN[1]:
                        width2[1] = NbaseN[1]
                    if width2[2] > NbaseD[2]:
                        width2[2] = NbaseD[2]
                    for il1 in range(width2[0]):
                        global_il1 = (global_i1 + il1 - int(floor(width2[0] / 2.0))) % NbaseN[0]
                        for il2 in range(width2[1]):
                            global_il2 = (global_i2 + il2 - int(floor(width2[1] / 2.0))) % NbaseN[1]
                            for il3 in range(width2[2]):
                                global_il3 = (global_i3 + il3 - int(floor(width2[2] / 2.0))) % NbaseD[2]
                                kernel_13[global_i1, global_i2, global_i3, il1, il2, il3] += w * (
                                    lambdas_11[global_i1, global_i2, global_i3]
                                    * lambdas_13[global_il1, global_il2, global_il3]
                                    + lambdas_21[global_i1, global_i2, global_i3]
                                    * lambdas_23[global_il1, global_il2, global_il3]
                                    + lambdas_31[global_i1, global_i2, global_i3]
                                    * lambdas_33[global_il1, global_il2, global_il3]
                                )

        for il1 in range(3):
            width[il1] = p[il1] + cell_number[il1] - 1
        if width[0] > NbaseN[0]:
            width[0] = NbaseN[0]
        if width[1] > NbaseD[1]:
            width[1] = NbaseD[1]
        if width[2] > NbaseN[2]:
            width[2] = NbaseN[2]
        for i1 in range(width[0]):
            global_i1 = (cell_left[0] + i1) % NbaseN[0]
            for i2 in range(width[1]):
                global_i2 = (cell_left[1] + i2) % NbaseD[1]
                for i3 in range(width[2]):
                    global_i3 = (cell_left[2] + i3) % NbaseN[2]
                    # ===== 22 compoponent ==========
                    for il1 in range(3):
                        width2[il1] = 2 * related[il1] + 1
                    if width2[0] > NbaseN[0]:
                        width2[0] = NbaseN[0]
                    if width2[1] > NbaseD[1]:
                        width2[1] = NbaseD[1]
                    if width2[2] > NbaseN[2]:
                        width2[2] = NbaseN[2]
                    for il1 in range(width2[0]):
                        global_il1 = (global_i1 + il1 - int(floor(width2[0] / 2.0))) % NbaseN[0]
                        for il2 in range(width2[1]):
                            global_il2 = (global_i2 + il2 - int(floor(width2[1] / 2.0))) % NbaseD[1]
                            for il3 in range(width2[2]):
                                global_il3 = (global_i3 + il3 - int(floor(width2[2] / 2.0))) % NbaseN[2]
                                kernel_22[global_i1, global_i2, global_i3, il1, il2, il3] += w * (
                                    lambdas_12[global_i1, global_i2, global_i3]
                                    * lambdas_12[global_il1, global_il2, global_il3]
                                    + lambdas_22[global_i1, global_i2, global_i3]
                                    * lambdas_22[global_il1, global_il2, global_il3]
                                    + lambdas_32[global_i1, global_i2, global_i3]
                                    * lambdas_32[global_il1, global_il2, global_il3]
                                )
                    # ===== 23 compoponent ==========
                    for il1 in range(3):
                        width2[il1] = 2 * related[il1] + 1
                    if width2[0] > NbaseN[0]:
                        width2[0] = NbaseN[0]
                    if width2[1] > NbaseN[1]:
                        width2[1] = NbaseN[1]
                    if width2[2] > NbaseD[2]:
                        width2[2] = NbaseD[2]
                    for il1 in range(width2[0]):
                        global_il1 = (global_i1 + il1 - int(floor(width2[0] / 2.0))) % NbaseN[0]
                        for il2 in range(width2[1]):
                            global_il2 = (global_i2 + il2 - int(floor(width2[1] / 2.0))) % NbaseN[1]
                            for il3 in range(width2[2]):
                                global_il3 = (global_i3 + il3 - int(floor(width2[2] / 2.0))) % NbaseD[2]
                                kernel_23[global_i1, global_i2, global_i3, il1, il2, il3] += w * (
                                    lambdas_12[global_i1, global_i2, global_i3]
                                    * lambdas_13[global_il1, global_il2, global_il3]
                                    + lambdas_22[global_i1, global_i2, global_i3]
                                    * lambdas_23[global_il1, global_il2, global_il3]
                                    + lambdas_32[global_i1, global_i2, global_i3]
                                    * lambdas_33[global_il1, global_il2, global_il3]
                                )

        for il1 in range(3):
            width[il1] = p[il1] + cell_number[il1] - 1
        if width[0] > NbaseN[0]:
            width[0] = NbaseN[0]
        if width[1] > NbaseN[1]:
            width[1] = NbaseN[1]
        if width[2] > NbaseD[2]:
            width[2] = NbaseD[2]

        for i1 in range(width[0]):
            global_i1 = (cell_left[0] + i1) % NbaseN[0]
            for i2 in range(width[1]):
                global_i2 = (cell_left[1] + i2) % NbaseN[1]
                for i3 in range(width[2]):
                    global_i3 = (cell_left[2] + i3) % NbaseD[2]
                    # ===== 33 compoponent ==========
                    for il1 in range(3):
                        width2[il1] = 2 * related[il1] + 1
                    if width2[0] > NbaseN[0]:
                        width2[0] = NbaseN[0]
                    if width2[1] > NbaseN[1]:
                        width2[1] = NbaseN[1]
                    if width2[2] > NbaseD[2]:
                        width2[2] = NbaseD[2]
                    for il1 in range(width2[0]):
                        global_il1 = (global_i1 + il1 - int(floor(width2[0] / 2.0))) % NbaseN[0]
                        for il2 in range(width2[1]):
                            global_il2 = (global_i2 + il2 - int(floor(width2[1] / 2.0))) % NbaseN[1]
                            for il3 in range(width2[2]):
                                global_il3 = (global_i3 + il3 - int(floor(width2[2] / 2.0))) % NbaseD[2]
                                kernel_33[global_i1, global_i2, global_i3, il1, il2, il3] += w * (
                                    lambdas_13[global_i1, global_i2, global_i3]
                                    * lambdas_13[global_il1, global_il2, global_il3]
                                    + lambdas_23[global_i1, global_i2, global_i3]
                                    * lambdas_23[global_il1, global_il2, global_il3]
                                    + lambdas_33[global_i1, global_i2, global_i3]
                                    * lambdas_33[global_il1, global_il2, global_il3]
                                )

        del mat_11
        del mat_12
        del mat_13
        del mat_21
        del mat_22
        del mat_23
        del mat_31
        del mat_32
        del mat_33
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel
    ierr = 0


# ==============================================================================================
def bv_localproj_push(
    dt: "float",
    bb1: "float[:,:,:]",
    bb2: "float[:,:,:]",
    bb3: "float[:,:,:]",
    pts1: "float[:]",
    pts2: "float[:]",
    pts3: "float[:]",
    wts1: "float[:]",
    wts2: "float[:]",
    wts3: "float[:]",
    Np: "int",
    quad: "int[:]",
    p: "int[:]",
    Nel: "int[:]",
    p_shape: "int[:]",
    p_size: "float[:]",
    particle: "float[:,:]",
    lambdas_11: "float[:,:,:]",
    lambdas_12: "float[:,:,:]",
    lambdas_13: "float[:,:,:]",
    lambdas_21: "float[:,:,:]",
    lambdas_22: "float[:,:,:]",
    lambdas_23: "float[:,:,:]",
    lambdas_31: "float[:,:,:]",
    lambdas_32: "float[:,:,:]",
    lambdas_33: "float[:,:,:]",
    num_cell: "int[:]",
    coeff_i_x: "float[:]",
    coeff_i_y: "float[:]",
    coeff_i_z: "float[:]",
    coeff_h_x: "float[:]",
    coeff_h_y: "float[:]",
    coeff_h_z: "float[:]",
    NbaseN: "int[:]",
    NbaseD: "int[:]",
    related: "int[:]",
    Np_loc: "int",
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
    from numpy import empty, floor, zeros

    cell_left = zeros(3, dtype=int)
    point_left = zeros(3, dtype=float)
    point_right = zeros(3, dtype=float)
    cell_number = zeros(3, dtype=int)
    compact = zeros(3, dtype=float)

    vel = zeros(3, dtype=float)
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
    df = zeros((3, 3), dtype=float)
    dft = zeros((3, 3), dtype=float)

    grids_shapex = zeros(p_shape[0] + 2, dtype=float)
    grids_shapey = zeros(p_shape[1] + 2, dtype=float)
    grids_shapez = zeros(p_shape[2] + 2, dtype=float)
    # ====================================
    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (i, grids_shapex, grids_shapey, grids_shapez, vel, mid1, mid2, mid3, ip, w, det_df, vol, lambdas_11, lambdas_12, lambdas_13, lambdas_21, lambdas_22, lambdas_23, lambdas_31, lambdas_32, lambdas_33, cell_left, point_left, point_right, cell_number, compact, mat_11, mat_12, mat_13, mat_21, mat_22, mat_23, mat_31, mat_32, mat_33, i1, i2, i3, il1, il2, il3, index1, index2, index3, value_x, value_y, value_z, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, dft, f_int, jl1, eta1, eta2, eta3, final_index1, final_index2, final_index3, lambda_index1, lambda_index2, lambda_index3)
    for ip in range(Np_loc):
        vel[:] = 0.0

        w = particle[6, ip] / Np

        lambdas_11[:, :, :] = 0.0
        lambdas_12[:, :, :] = 0.0
        lambdas_13[:, :, :] = 0.0

        lambdas_21[:, :, :] = 0.0
        lambdas_22[:, :, :] = 0.0
        lambdas_23[:, :, :] = 0.0

        lambdas_31[:, :, :] = 0.0
        lambdas_32[:, :, :] = 0.0
        lambdas_33[:, :, :] = 0.0

        # ==================================
        compact[0] = (p_shape[0] + 1.0) * p_size[0]
        compact[1] = (p_shape[1] + 1.0) * p_size[1]
        compact[2] = (p_shape[2] + 1.0) * p_size[2]

        point_left[0] = particle[0, ip] - 0.5 * compact[0]
        point_right[0] = particle[0, ip] + 0.5 * compact[0]
        point_left[1] = particle[1, ip] - 0.5 * compact[1]
        point_right[1] = particle[1, ip] + 0.5 * compact[1]
        point_left[2] = particle[2, ip] - 0.5 * compact[2]
        point_right[2] = particle[2, ip] + 0.5 * compact[2]

        cell_left[0] = int(floor(point_left[0] * Nel[0]))
        cell_left[1] = int(floor(point_left[1] * Nel[1]))
        cell_left[2] = int(floor(point_left[2] * Nel[2]))

        cell_number[0] = int(floor(point_right[0] * Nel[0])) - cell_left[0] + 1
        cell_number[1] = int(floor(point_right[1] * Nel[1])) - cell_left[1] + 1
        cell_number[2] = int(floor(point_right[2] * Nel[2])) - cell_left[2] + 1

        for i in range(p_shape[0] + 1):
            grids_shapex[i] = point_left[0] + i * p_size[0]
        grids_shapex[p_shape[0] + 1] = point_right[0]

        for i in range(p_shape[1] + 1):
            grids_shapey[i] = point_left[1] + i * p_size[1]
        grids_shapey[p_shape[1] + 1] = point_right[1]

        for i in range(p_shape[2] + 1):
            grids_shapez[i] = point_left[2] + i * p_size[2]
        grids_shapez[p_shape[2] + 1] = point_right[2]

        vol = 1.0 / (p_size[0] * p_size[1] * p_size[2])

        # evaluation of function at interpolation/quadrature points
        mat_11 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], 2, num_cell[1], num_cell[2], quad[0]),
            dtype=float,
        )
        mat_21 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], 2, num_cell[1], num_cell[2], quad[0]),
            dtype=float,
        )
        mat_31 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], 2, num_cell[1], num_cell[2], quad[0]),
            dtype=float,
        )

        mat_12 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], 2, num_cell[2], quad[1]),
            dtype=float,
        )
        mat_22 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], 2, num_cell[2], quad[1]),
            dtype=float,
        )
        mat_32 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], 2, num_cell[2], quad[1]),
            dtype=float,
        )

        mat_13 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], num_cell[1], 2, quad[2]),
            dtype=float,
        )
        mat_23 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], num_cell[1], 2, quad[2]),
            dtype=float,
        )
        mat_33 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], num_cell[1], 2, quad[2]),
            dtype=float,
        )

        for i1 in range(cell_number[0]):
            index1 = cell_left[0] + i1
            for i2 in range(cell_number[1]):
                index2 = cell_left[1] + i2
                for i3 in range(cell_number[2]):  # num_cell = 1, p = 1; num_cell = 2, p >= 2
                    index3 = cell_left[2] + i3
                    for il1 in range(2):
                        for il2 in range(num_cell[1]):
                            eta2 = 1.0 / Nel[1] * index2 + 1.0 / Nel[1] / num_cell[1] * il2
                            # value_y = bsp.convolution(p_shape[1], grids_shapey, eta2)
                            value_y = bsp.piecewise(p_shape[1], p_size[1], abs(eta2 - particle[1, ip]))
                            for il3 in range(num_cell[2]):
                                eta3 = 1.0 / Nel[2] * index3 + 1.0 / Nel[2] / num_cell[2] * il3
                                # value_z = bsp.convolution(p_shape[2], grids_shapez, eta3)
                                value_z = bsp.piecewise(p_shape[2], p_size[2], abs(eta3 - particle[2, ip]))
                                for jl1 in range(quad[0]):
                                    eta1 = 1.0 / Nel[0] * index1 + 1.0 / Nel[0] / 2.0 * il1 + pts1[jl1]
                                    value_x = bsp.convolution(p_shape[0], grids_shapex, eta1)
                                    # value_x = bsp.piecewise(p_shape[0], p_size[0], abs(eta1 - particle[0, ip]))
                                    # ========= mapping evaluation =============
                                    span1f = int((eta1 % 1.0) * nelf[0]) + pf1
                                    span2f = int((eta2 % 1.0) * nelf[1]) + pf2
                                    span3f = int((eta3 % 1.0) * nelf[2]) + pf3
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
                                        eta1 % 1.0,
                                        eta2 % 1.0,
                                        eta3 % 1.0,
                                        df,
                                        fx,
                                        0,
                                    )
                                    # evaluate inverse Jacobian matrix
                                    linalg.transpose(df, dft)
                                    det_df = abs(linalg.det(df))

                                    mat_11[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[0, 0] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_21[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[0, 1] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_31[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[0, 2] * value_x * value_y * value_z / det_df * vol
                                    )

                    for il1 in range(num_cell[0]):
                        eta1 = 1.0 / Nel[0] * index1 + 1 / Nel[0] / num_cell[0] * il1
                        value_x = bsp.convolution(p_shape[0], grids_shapex, eta1)
                        # value_x = bsp.piecewise(p_shape[0], p_size[0], abs(eta1 - particle[0, ip]))
                        for il2 in range(2):
                            for il3 in range(num_cell[2]):
                                eta3 = 1.0 / Nel[2] * index3 + 1 / Nel[2] / num_cell[2] * il3
                                # value_z = bsp.convolution(p_shape[2], grids_shapez, eta3)
                                value_z = bsp.piecewise(p_shape[2], p_size[2], abs(eta3 - particle[2, ip]))
                                for jl1 in range(quad[1]):
                                    eta2 = 1.0 / Nel[1] * index2 + 1 / Nel[1] / 2.0 * il2 + pts2[jl1]
                                    # value_y = bsp.convolution(p_shape[1], grids_shapey, eta2)
                                    value_y = bsp.piecewise(p_shape[1], p_size[1], abs(eta2 - particle[1, ip]))
                                    # ========= mapping evaluation =============
                                    span1f = int((eta1 % 1.0) * nelf[0]) + pf1
                                    span2f = int((eta2 % 1.0) * nelf[1]) + pf2
                                    span3f = int((eta3 % 1.0) * nelf[2]) + pf3
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
                                        eta1 % 1.0,
                                        eta2 % 1.0,
                                        eta3 % 1.0,
                                        df,
                                        fx,
                                        0,
                                    )
                                    # evaluate inverse Jacobian matrix
                                    linalg.transpose(df, dft)
                                    det_df = abs(linalg.det(df))

                                    mat_12[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[1, 0] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_22[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[1, 1] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_32[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[1, 2] * value_x * value_y * value_z / det_df * vol
                                    )

                    for il1 in range(num_cell[0]):
                        eta1 = 1.0 / Nel[0] * index1 + 1.0 / Nel[0] / num_cell[0] * il1
                        value_x = bsp.convolution(p_shape[0], grids_shapex, eta1)
                        # value_x = bsp.piecewise(p_shape[0], p_size[0], abs(eta1 - particle[0, ip]))
                        for il2 in range(num_cell[1]):
                            eta2 = 1.0 / Nel[1] * index2 + 1.0 / Nel[1] / num_cell[1] * il2
                            # value_y = bsp.convolution(p_shape[1], grids_shapey, eta2)
                            value_y = bsp.piecewise(p_shape[1], p_size[1], abs(eta2 - particle[1, ip]))
                            for il3 in range(2):
                                for jl1 in range(quad[2]):
                                    eta3 = 1.0 / Nel[2] * index3 + 1.0 / Nel[2] / 2.0 * il3 + pts3[jl1]
                                    # value_z = bsp.convolution(p_shape[2], grids_shapez, eta3)
                                    value_z = bsp.piecewise(p_shape[2], p_size[2], abs(eta3 - particle[2, ip]))
                                    # ========= mapping evaluation =============
                                    span1f = int((eta1 % 1.0) * nelf[0]) + pf1
                                    span2f = int((eta2 % 1.0) * nelf[1]) + pf2
                                    span3f = int((eta3 % 1.0) * nelf[2]) + pf3
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
                                        eta1 % 1.0,
                                        eta2 % 1.0,
                                        eta3 % 1.0,
                                        df,
                                        fx,
                                        0,
                                    )
                                    # evaluate inverse Jacobian matrix
                                    linalg.transpose(df, dft)
                                    det_df = abs(linalg.det(df))

                                    mat_13[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[2, 0] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_23[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[2, 1] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_33[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[2, 2] * value_x * value_y * value_z / det_df * vol
                                    )

        for i1 in range(cell_number[0]):
            index1 = cell_left[0] + i1
            for i2 in range(cell_number[1]):
                index2 = cell_left[1] + i2
                for i3 in range(cell_number[2]):
                    index3 = cell_left[2] + i3

                    for lambda_index1 in range(p[0]):
                        final_index1 = (lambda_index1 + index1) % NbaseD[0]
                        for lambda_index2 in range(p[1]):
                            final_index2 = (lambda_index2 + index2) % NbaseN[1]
                            for lambda_index3 in range(p[2]):
                                final_index3 = (lambda_index3 + index3) % NbaseN[2]
                                for il1 in range(2):
                                    for il2 in range(num_cell[1]):
                                        for il3 in range(num_cell[2]):
                                            f_int = 0.0
                                            for jl1 in range(quad[0]):
                                                f_int += wts1[jl1] * mat_11[i1, i2, i3, il1, il2, il3, jl1]
                                            mid1 = (
                                                coeff_h_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[0] += mid1 * bb1[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[0]):
                                                f_int += wts1[jl1] * mat_21[i1, i2, i3, il1, il2, il3, jl1]
                                            mid2 = (
                                                coeff_h_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[1] += mid2 * bb1[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[0]):
                                                f_int += wts1[jl1] * mat_31[i1, i2, i3, il1, il2, il3, jl1]
                                            mid3 = (
                                                coeff_h_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[2] += mid3 * bb1[final_index1, final_index2, final_index3]

                    for lambda_index1 in range(p[0]):
                        final_index1 = (lambda_index1 + index1) % NbaseN[0]
                        for lambda_index2 in range(p[1]):
                            final_index2 = (lambda_index2 + index2) % NbaseD[1]
                            for lambda_index3 in range(p[2]):
                                final_index3 = (lambda_index3 + index3) % NbaseN[2]
                                for il1 in range(num_cell[0]):
                                    for il2 in range(2):
                                        for il3 in range(num_cell[2]):
                                            f_int = 0.0
                                            for jl1 in range(quad[1]):
                                                f_int += wts2[jl1] * mat_12[i1, i2, i3, il1, il2, il3, jl1]
                                            mid1 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_h_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[0] += mid1 * bb2[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[1]):
                                                f_int += wts2[jl1] * mat_22[i1, i2, i3, il1, il2, il3, jl1]
                                            mid2 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_h_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[1] += mid2 * bb2[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[1]):
                                                f_int += wts2[jl1] * mat_32[i1, i2, i3, il1, il2, il3, jl1]
                                            mid3 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_h_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[2] += mid3 * bb2[final_index1, final_index2, final_index3]

                    for lambda_index1 in range(p[0]):
                        final_index1 = (lambda_index1 + index1) % NbaseN[0]
                        for lambda_index2 in range(p[1]):
                            final_index2 = (lambda_index2 + index2) % NbaseN[1]
                            for lambda_index3 in range(p[2]):
                                final_index3 = (lambda_index3 + index3) % NbaseD[2]
                                for il1 in range(num_cell[0]):
                                    for il2 in range(num_cell[1]):
                                        for il3 in range(2):
                                            f_int = 0.0
                                            for jl1 in range(quad[2]):
                                                f_int += wts3[jl1] * mat_13[i1, i2, i3, il1, il2, il3, jl1]
                                            mid1 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_h_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[0] += mid1 * bb3[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[2]):
                                                f_int += wts3[jl1] * mat_23[i1, i2, i3, il1, il2, il3, jl1]
                                            mid2 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_h_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[1] += mid2 * bb3[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[2]):
                                                f_int += wts3[jl1] * mat_33[i1, i2, i3, il1, il2, il3, jl1]
                                            mid3 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_h_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[2] += mid3 * bb3[final_index1, final_index2, final_index3]

        particle[3, ip] += dt * vel[0]
        particle[4, ip] += dt * vel[1]
        particle[5, ip] += dt * vel[2]

        del mat_11
        del mat_12
        del mat_13
        del mat_21
        del mat_22
        del mat_23
        del mat_31
        del mat_32
        del mat_33
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel
    ierr = 0


# ==============================================================================================
def kernel_1_heavy(
    pts1: "float[:]",
    pts2: "float[:]",
    pts3: "float[:]",
    wts1: "float[:]",
    wts2: "float[:]",
    wts3: "float[:]",
    out1: "float[:,:,:]",
    out2: "float[:,:,:]",
    out3: "float[:,:,:]",
    in1: "float[:,:,:]",
    in2: "float[:,:,:]",
    in3: "float[:,:,:]",
    Np: "int",
    quad: "int[:]",
    p: "int[:]",
    Nel: "int[:]",
    p_shape: "int[:]",
    p_size: "float[:]",
    particle: "float[:,:]",
    lambdas_11: "float[:,:,:]",
    lambdas_12: "float[:,:,:]",
    lambdas_13: "float[:,:,:]",
    lambdas_21: "float[:,:,:]",
    lambdas_22: "float[:,:,:]",
    lambdas_23: "float[:,:,:]",
    lambdas_31: "float[:,:,:]",
    lambdas_32: "float[:,:,:]",
    lambdas_33: "float[:,:,:]",
    num_cell: "int[:]",
    coeff_i_x: "float[:]",
    coeff_i_y: "float[:]",
    coeff_i_z: "float[:]",
    coeff_h_x: "float[:]",
    coeff_h_y: "float[:]",
    coeff_h_z: "float[:]",
    NbaseN: "int[:]",
    NbaseD: "int[:]",
    Np_loc: "int",
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
    from numpy import empty, floor, zeros

    cell_left = zeros(3, dtype=int)
    point_left = zeros(3, dtype=float)
    point_right = zeros(3, dtype=float)
    cell_number = zeros(3, dtype=int)
    compact = zeros(3, dtype=float)

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
    df = zeros((3, 3), dtype=float)
    dft = zeros((3, 3), dtype=float)

    grids_shapex = zeros(p_shape[0] + 2, dtype=float)
    grids_shapey = zeros(p_shape[1] + 2, dtype=float)
    grids_shapez = zeros(p_shape[2] + 2, dtype=float)
    # ====================================

    out1[:, :, :] = 0.0
    out2[:, :, :] = 0.0
    out3[:, :, :] = 0.0

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do reduction ( + : out1, out2, out3) private (value1, value2, value3, i, grids_shapex, grids_shapey, grids_shapez, mid1, mid2, mid3, ip, w, det_df, vol, lambdas_11, lambdas_12, lambdas_13, lambdas_21, lambdas_22, lambdas_23, lambdas_31, lambdas_32, lambdas_33, cell_left, point_left, point_right, cell_number, compact, mat_11, mat_12, mat_13, mat_21, mat_22, mat_23, mat_31, mat_32, mat_33, i1, i2, i3, il1, il2, il3, index1, index2, index3, value_x, value_y, value_z, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, dft, lambda_index1, lambda_index2, lambda_index3, eta1, eta2, eta3, final_index1, final_index2, final_index3, f_int)
    for ip in range(Np_loc):
        w = particle[6, ip] / Np

        lambdas_11[:, :, :] = 0.0
        lambdas_22[:, :, :] = 0.0
        lambdas_33[:, :, :] = 0.0

        lambdas_12[:, :, :] = 0.0
        lambdas_13[:, :, :] = 0.0

        lambdas_21[:, :, :] = 0.0
        lambdas_23[:, :, :] = 0.0

        lambdas_31[:, :, :] = 0.0
        lambdas_32[:, :, :] = 0.0

        value1 = 0.0
        value2 = 0.0
        value3 = 0.0

        # ==================================
        compact[0] = (p_shape[0] + 1.0) * p_size[0]
        compact[1] = (p_shape[1] + 1.0) * p_size[1]
        compact[2] = (p_shape[2] + 1.0) * p_size[2]

        point_left[0] = particle[0, ip] - 0.5 * compact[0]
        point_right[0] = particle[0, ip] + 0.5 * compact[0]
        point_left[1] = particle[1, ip] - 0.5 * compact[1]
        point_right[1] = particle[1, ip] + 0.5 * compact[1]
        point_left[2] = particle[2, ip] - 0.5 * compact[2]
        point_right[2] = particle[2, ip] + 0.5 * compact[2]

        cell_left[0] = int(floor(point_left[0] * Nel[0]))
        cell_left[1] = int(floor(point_left[1] * Nel[1]))
        cell_left[2] = int(floor(point_left[2] * Nel[2]))

        cell_number[0] = int(floor(point_right[0] * Nel[0])) - cell_left[0] + 1
        cell_number[1] = int(floor(point_right[1] * Nel[1])) - cell_left[1] + 1
        cell_number[2] = int(floor(point_right[2] * Nel[2])) - cell_left[2] + 1

        for i in range(p_shape[0] + 1):
            grids_shapex[i] = point_left[0] + i * p_size[0]
        grids_shapex[p_shape[0] + 1] = point_right[0]

        for i in range(p_shape[1] + 1):
            grids_shapey[i] = point_left[1] + i * p_size[1]
        grids_shapey[p_shape[1] + 1] = point_right[1]

        for i in range(p_shape[2] + 1):
            grids_shapez[i] = point_left[2] + i * p_size[2]
        grids_shapez[p_shape[2] + 1] = point_right[2]

        vol = 1.0 / (p_size[0] * p_size[1] * p_size[2])

        # evaluation of function at interpolation/quadrature points
        mat_11 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], 2, num_cell[1], num_cell[2], quad[0]),
            dtype=float,
        )
        mat_21 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], 2, num_cell[1], num_cell[2], quad[0]),
            dtype=float,
        )
        mat_31 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], 2, num_cell[1], num_cell[2], quad[0]),
            dtype=float,
        )

        mat_12 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], 2, num_cell[2], quad[1]),
            dtype=float,
        )
        mat_22 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], 2, num_cell[2], quad[1]),
            dtype=float,
        )
        mat_32 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], 2, num_cell[2], quad[1]),
            dtype=float,
        )

        mat_13 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], num_cell[1], 2, quad[2]),
            dtype=float,
        )
        mat_23 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], num_cell[1], 2, quad[2]),
            dtype=float,
        )
        mat_33 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], num_cell[1], 2, quad[2]),
            dtype=float,
        )

        for i1 in range(cell_number[0]):
            index1 = cell_left[0] + i1
            for i2 in range(cell_number[1]):
                index2 = cell_left[1] + i2
                for i3 in range(cell_number[2]):  # num_cell = 1, p = 1; num_cell = 2, p >= 2
                    index3 = cell_left[2] + i3
                    for il1 in range(2):
                        for il2 in range(num_cell[1]):
                            eta2 = 1.0 / Nel[1] * index2 + 1.0 / Nel[1] / num_cell[1] * il2
                            # value_y = bsp.convolution(p_shape[1], grids_shapey, eta2)
                            value_y = bsp.piecewise(p_shape[1], p_size[1], abs(eta2 - particle[1, ip]))
                            for il3 in range(num_cell[2]):
                                eta3 = 1.0 / Nel[2] * index3 + 1.0 / Nel[2] / num_cell[2] * il3
                                # value_z = bsp.convolution(p_shape[2], grids_shapez, eta3)
                                value_z = bsp.piecewise(p_shape[2], p_size[2], abs(eta3 - particle[2, ip]))
                                for jl1 in range(quad[0]):
                                    eta1 = 1.0 / Nel[0] * index1 + 1.0 / Nel[0] / 2.0 * il1 + pts1[jl1]
                                    value_x = bsp.convolution(p_shape[0], grids_shapex, eta1)
                                    # value_x = bsp.piecewise(p_shape[0], p_size[0], abs(eta1 - particle[0, ip]))
                                    # ========= mapping evaluation =============
                                    span1f = int((eta1 % 1.0) * nelf[0]) + pf1
                                    span2f = int((eta2 % 1.0) * nelf[1]) + pf2
                                    span3f = int((eta3 % 1.0) * nelf[2]) + pf3
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
                                        eta1 % 1.0,
                                        eta2 % 1.0,
                                        eta3 % 1.0,
                                        df,
                                        fx,
                                        0,
                                    )
                                    # evaluate inverse Jacobian matrix
                                    linalg.transpose(df, dft)
                                    det_df = abs(linalg.det(df))

                                    mat_11[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[0, 0] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_21[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[0, 1] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_31[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[0, 2] * value_x * value_y * value_z / det_df * vol
                                    )

                    for il1 in range(num_cell[0]):
                        eta1 = 1.0 / Nel[0] * index1 + 1 / Nel[0] / num_cell[0] * il1
                        value_x = bsp.convolution(p_shape[0], grids_shapex, eta1)
                        # value_x = bsp.piecewise(p_shape[0], p_size[0], abs(eta1 - particle[0, ip]))
                        for il2 in range(2):
                            for il3 in range(num_cell[2]):
                                eta3 = 1.0 / Nel[2] * index3 + 1 / Nel[2] / num_cell[2] * il3
                                # value_z = bsp.convolution(p_shape[2], grids_shapez, eta3)
                                value_z = bsp.piecewise(p_shape[2], p_size[2], abs(eta3 - particle[2, ip]))
                                for jl1 in range(quad[1]):
                                    eta2 = 1.0 / Nel[1] * index2 + 1 / Nel[1] / 2.0 * il2 + pts2[jl1]
                                    # value_y = bsp.convolution(p_shape[1], grids_shapey, eta2)
                                    value_y = bsp.piecewise(p_shape[1], p_size[1], abs(eta2 - particle[1, ip]))
                                    # ========= mapping evaluation =============
                                    span1f = int((eta1 % 1.0) * nelf[0]) + pf1
                                    span2f = int((eta2 % 1.0) * nelf[1]) + pf2
                                    span3f = int((eta3 % 1.0) * nelf[2]) + pf3
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
                                        eta1 % 1.0,
                                        eta2 % 1.0,
                                        eta3 % 1.0,
                                        df,
                                        fx,
                                        0,
                                    )
                                    # evaluate inverse Jacobian matrix
                                    linalg.transpose(df, dft)
                                    det_df = abs(linalg.det(df))

                                    mat_12[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[1, 0] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_22[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[1, 1] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_32[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[1, 2] * value_x * value_y * value_z / det_df * vol
                                    )

                    for il1 in range(num_cell[0]):
                        eta1 = 1.0 / Nel[0] * index1 + 1.0 / Nel[0] / num_cell[0] * il1
                        value_x = bsp.convolution(p_shape[0], grids_shapex, eta1)
                        # value_x = bsp.piecewise(p_shape[0], p_size[0], abs(eta1 - particle[0, ip]))
                        for il2 in range(num_cell[1]):
                            eta2 = 1.0 / Nel[1] * index2 + 1.0 / Nel[1] / num_cell[1] * il2
                            # value_y = bsp.convolution(p_shape[1], grids_shapey, eta2)
                            value_y = bsp.piecewise(p_shape[1], p_size[1], abs(eta2 - particle[1, ip]))
                            for il3 in range(2):
                                for jl1 in range(quad[2]):
                                    eta3 = 1.0 / Nel[2] * index3 + 1.0 / Nel[2] / 2.0 * il3 + pts3[jl1]
                                    # value_z = bsp.convolution(p_shape[2], grids_shapez, eta3)
                                    value_z = bsp.piecewise(p_shape[2], p_size[2], abs(eta3 - particle[2, ip]))
                                    # ========= mapping evaluation =============
                                    span1f = int((eta1 % 1.0) * nelf[0]) + pf1
                                    span2f = int((eta2 % 1.0) * nelf[1]) + pf2
                                    span3f = int((eta3 % 1.0) * nelf[2]) + pf3
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
                                        eta1 % 1.0,
                                        eta2 % 1.0,
                                        eta3 % 1.0,
                                        df,
                                        fx,
                                        0,
                                    )
                                    # evaluate inverse Jacobian matrix
                                    linalg.transpose(df, dft)
                                    det_df = abs(linalg.det(df))

                                    mat_13[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[2, 0] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_23[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[2, 1] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_33[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[2, 2] * value_x * value_y * value_z / det_df * vol
                                    )

        for i1 in range(cell_number[0]):
            index1 = cell_left[0] + i1
            for i2 in range(cell_number[1]):
                index2 = cell_left[1] + i2
                for i3 in range(cell_number[2]):
                    index3 = cell_left[2] + i3

                    for lambda_index1 in range(p[0]):
                        final_index1 = (lambda_index1 + index1) % NbaseD[0]
                        for lambda_index2 in range(p[1]):
                            final_index2 = (lambda_index2 + index2) % NbaseN[1]
                            for lambda_index3 in range(p[2]):
                                final_index3 = (lambda_index3 + index3) % NbaseN[2]
                                for il1 in range(2):
                                    for il2 in range(num_cell[1]):
                                        for il3 in range(num_cell[2]):
                                            f_int = 0.0
                                            for jl1 in range(quad[0]):
                                                f_int += wts1[jl1] * mat_11[i1, i2, i3, il1, il2, il3, jl1]
                                            mid1 = (
                                                coeff_h_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_11[final_index1, final_index2, final_index3] += mid1
                                            value1 += mid1 * in1[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[0]):
                                                f_int += wts1[jl1] * mat_21[i1, i2, i3, il1, il2, il3, jl1]
                                            mid2 = (
                                                coeff_h_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_21[final_index1, final_index2, final_index3] += mid2
                                            value2 += mid2 * in1[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[0]):
                                                f_int += wts1[jl1] * mat_31[i1, i2, i3, il1, il2, il3, jl1]
                                            mid3 = (
                                                coeff_h_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_31[final_index1, final_index2, final_index3] += mid3
                                            value3 += mid3 * in1[final_index1, final_index2, final_index3]

                    for lambda_index1 in range(p[0]):
                        final_index1 = (lambda_index1 + index1) % NbaseN[0]
                        for lambda_index2 in range(p[1]):
                            final_index2 = (lambda_index2 + index2) % NbaseD[1]
                            for lambda_index3 in range(p[2]):
                                final_index3 = (lambda_index3 + index3) % NbaseN[2]
                                for il1 in range(num_cell[0]):
                                    for il2 in range(2):
                                        for il3 in range(num_cell[2]):
                                            f_int = 0.0
                                            for jl1 in range(quad[1]):
                                                f_int += wts2[jl1] * mat_12[i1, i2, i3, il1, il2, il3, jl1]
                                            mid1 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_h_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_12[final_index1, final_index2, final_index3] += mid1
                                            value1 += mid1 * in2[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[1]):
                                                f_int += wts2[jl1] * mat_22[i1, i2, i3, il1, il2, il3, jl1]
                                            mid2 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_h_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_22[final_index1, final_index2, final_index3] += mid2
                                            value2 += mid2 * in2[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[1]):
                                                f_int += wts2[jl1] * mat_32[i1, i2, i3, il1, il2, il3, jl1]
                                            mid3 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_h_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_32[final_index1, final_index2, final_index3] += mid3
                                            value3 += mid3 * in2[final_index1, final_index2, final_index3]

                    for lambda_index1 in range(p[0]):
                        final_index1 = (lambda_index1 + index1) % NbaseN[0]
                        for lambda_index2 in range(p[1]):
                            final_index2 = (lambda_index2 + index2) % NbaseN[1]
                            for lambda_index3 in range(p[2]):
                                final_index3 = (lambda_index3 + index3) % NbaseD[2]
                                for il1 in range(num_cell[0]):
                                    for il2 in range(num_cell[1]):
                                        for il3 in range(2):
                                            f_int = 0.0
                                            for jl1 in range(quad[2]):
                                                f_int += wts3[jl1] * mat_13[i1, i2, i3, il1, il2, il3, jl1]
                                            mid1 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_h_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_13[final_index1, final_index2, final_index3] += mid1
                                            value1 += mid1 * in3[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[2]):
                                                f_int += wts3[jl1] * mat_23[i1, i2, i3, il1, il2, il3, jl1]
                                            mid2 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_h_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_23[final_index1, final_index2, final_index3] += mid2
                                            value2 += mid2 * in3[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[2]):
                                                f_int += wts3[jl1] * mat_33[i1, i2, i3, il1, il2, il3, jl1]
                                            mid3 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_h_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            lambdas_33[final_index1, final_index2, final_index3] += mid3
                                            value3 += mid3 * in3[final_index1, final_index2, final_index3]

        for il1 in range(NbaseN[0]):
            for il2 in range(NbaseN[1]):
                for il3 in range(NbaseN[2]):
                    out1[il1, il2, il3] += (
                        w * value1 * (lambdas_11[il1, il2, il3] + lambdas_21[il1, il2, il3] + lambdas_31[il1, il2, il3])
                    )
                    out2[il1, il2, il3] += (
                        w * value2 * (lambdas_12[il1, il2, il3] + lambdas_22[il1, il2, il3] + lambdas_32[il1, il2, il3])
                    )
                    out3[il1, il2, il3] += (
                        w * value3 * (lambdas_13[il1, il2, il3] + lambdas_23[il1, il2, il3] + lambdas_33[il1, il2, il3])
                    )

        del mat_11
        del mat_12
        del mat_13
        del mat_21
        del mat_22
        del mat_23
        del mat_31
        del mat_32
        del mat_33
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel
    ierr = 0


# ==============================================================================================
def vv_1_form(
    wts1: "float[:]",
    wts2: "float[:]",
    wts3: "float[:]",
    pts1: "float[:]",
    pts2: "float[:]",
    pts3: "float[:]",
    ddt: "float",
    right1: "float[:,:,:]",
    right2: "float[:,:,:]",
    right3: "float[:,:,:]",
    Np: "int",
    quad: "int[:]",
    p: "int[:]",
    Nel: "int[:]",
    p_shape: "int[:]",
    p_size: "float[:]",
    particle: "float[:,:]",
    mid_particle: "float[:,:]",
    lambdas_11: "float[:,:,:]",
    lambdas_12: "float[:,:,:]",
    lambdas_13: "float[:,:,:]",
    lambdas_21: "float[:,:,:]",
    lambdas_22: "float[:,:,:]",
    lambdas_23: "float[:,:,:]",
    lambdas_31: "float[:,:,:]",
    lambdas_32: "float[:,:,:]",
    lambdas_33: "float[:,:,:]",
    num_cell: "int[:]",
    coeff_i_x: "float[:]",
    coeff_i_y: "float[:]",
    coeff_i_z: "float[:]",
    coeff_h_x: "float[:]",
    coeff_h_y: "float[:]",
    coeff_h_z: "float[:]",
    NbaseN: "int[:]",
    NbaseD: "int[:]",
    related: "int[:]",
    Np_loc: "int",
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
    from numpy import empty, floor, zeros

    cell_left = zeros(3, dtype=int)
    point_left = zeros(3, dtype=float)
    point_right = zeros(3, dtype=float)
    cell_number = zeros(3, dtype=int)
    compact = zeros(3, dtype=float)

    width = zeros(3, dtype=int)
    width[0] = p[0] + cell_number[0] - 1  # the number of coefficients obtained from this smoothed delta function
    width[1] = p[1] + cell_number[1] - 1  # the number of coefficients obtained from this smoothed delta function
    width[2] = p[2] + cell_number[2] - 1  # the number of coefficients obtained from this smoothed delta function

    width2 = zeros(3, dtype=int)
    width2[0] = 2 * related[0] + 1
    width2[1] = 2 * related[1] + 1
    width2[2] = 2 * related[2] + 1

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
    df = zeros((3, 3), dtype=float)
    dft = zeros((3, 3), dtype=float)

    grids_shapex = zeros(p_shape[0] + 2, dtype=float)
    grids_shapey = zeros(p_shape[1] + 2, dtype=float)
    grids_shapez = zeros(p_shape[2] + 2, dtype=float)

    right1[:, :, :] = 0.0
    right2[:, :, :] = 0.0
    right3[:, :, :] = 0.0
    # ====================================
    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do reduction ( + : right1, right2, right3) private (i, grids_shapex, grids_shapey, grids_shapez, mid1, mid2, mid3, ip, w, det_df, vol, lambdas_11, lambdas_22, lambdas_33, lambdas_12, lambdas_13, lambdas_21, lambdas_23, lambdas_31, lambdas_32, cell_left, point_left, point_right, cell_number, compact, mat_11, mat_12, mat_13, mat_21, mat_22, mat_23, mat_31, mat_32, mat_33, i1, i2, i3, il1, il2, il3, jl1, index1, index2, index3, value_x, value_y, value_z, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, dft, lambda_index1, lambda_index2, lambda_index3, eta1, eta2, eta3, final_index1, final_index2, final_index3, f_int)
    for ip in range(Np_loc):
        w = particle[6, ip] / Np

        lambdas_11[:, :, :] = 0.0
        lambdas_12[:, :, :] = 0.0
        lambdas_13[:, :, :] = 0.0

        lambdas_21[:, :, :] = 0.0
        lambdas_22[:, :, :] = 0.0
        lambdas_23[:, :, :] = 0.0

        lambdas_31[:, :, :] = 0.0
        lambdas_32[:, :, :] = 0.0
        lambdas_33[:, :, :] = 0.0

        # ==================================
        compact[0] = (p_shape[0] + 1.0) * p_size[0]
        compact[1] = (p_shape[1] + 1.0) * p_size[1]
        compact[2] = (p_shape[2] + 1.0) * p_size[2]

        point_left[0] = particle[0, ip] - 0.5 * compact[0]
        point_right[0] = particle[0, ip] + 0.5 * compact[0]
        point_left[1] = particle[1, ip] - 0.5 * compact[1]
        point_right[1] = particle[1, ip] + 0.5 * compact[1]
        point_left[2] = particle[2, ip] - 0.5 * compact[2]
        point_right[2] = particle[2, ip] + 0.5 * compact[2]

        cell_left[0] = int(floor(point_left[0] * Nel[0]))
        cell_left[1] = int(floor(point_left[1] * Nel[1]))
        cell_left[2] = int(floor(point_left[2] * Nel[2]))

        cell_number[0] = int(floor(point_right[0] * Nel[0])) - cell_left[0] + 1
        cell_number[1] = int(floor(point_right[1] * Nel[1])) - cell_left[1] + 1
        cell_number[2] = int(floor(point_right[2] * Nel[2])) - cell_left[2] + 1

        for i in range(p_shape[0] + 1):
            grids_shapex[i] = point_left[0] + i * p_size[0]
        grids_shapex[p_shape[0] + 1] = point_right[0]

        for i in range(p_shape[1] + 1):
            grids_shapey[i] = point_left[1] + i * p_size[1]
        grids_shapey[p_shape[1] + 1] = point_right[1]

        for i in range(p_shape[2] + 1):
            grids_shapez[i] = point_left[2] + i * p_size[2]
        grids_shapez[p_shape[2] + 1] = point_right[2]

        vol = 1.0 / (p_size[0] * p_size[1] * p_size[2])

        # evaluation of function at interpolation/quadrature points
        mat_11 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], 2, num_cell[1], num_cell[2], quad[0]),
            dtype=float,
        )
        mat_21 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], 2, num_cell[1], num_cell[2], quad[0]),
            dtype=float,
        )
        mat_31 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], 2, num_cell[1], num_cell[2], quad[0]),
            dtype=float,
        )

        mat_12 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], 2, num_cell[2], quad[1]),
            dtype=float,
        )
        mat_22 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], 2, num_cell[2], quad[1]),
            dtype=float,
        )
        mat_32 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], 2, num_cell[2], quad[1]),
            dtype=float,
        )

        mat_13 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], num_cell[1], 2, quad[2]),
            dtype=float,
        )
        mat_23 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], num_cell[1], 2, quad[2]),
            dtype=float,
        )
        mat_33 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], num_cell[1], 2, quad[2]),
            dtype=float,
        )

        for i1 in range(cell_number[0]):
            index1 = cell_left[0] + i1
            for i2 in range(cell_number[1]):
                index2 = cell_left[1] + i2
                for i3 in range(cell_number[2]):  # num_cell = 1, p = 1; num_cell = 2, p >= 2
                    index3 = cell_left[2] + i3
                    for il1 in range(2):
                        for il2 in range(num_cell[1]):
                            eta2 = 1.0 / Nel[1] * index2 + 1 / Nel[1] / num_cell[1] * il2
                            span2f = int((eta2 % 1.0) * nelf[1]) + pf2
                            value_y = bsp.piecewise(p_shape[1], p_size[1], abs(eta2 - particle[1, ip]))
                            for il3 in range(num_cell[2]):
                                eta3 = 1.0 / Nel[2] * index3 + 1 / Nel[2] / num_cell[2] * il3
                                span3f = int((eta3 % 1.0) * nelf[2]) + pf3
                                value_z = bsp.piecewise(p_shape[2], p_size[2], abs(eta3 - particle[2, ip]))
                                for jl1 in range(quad[0]):
                                    eta1 = 1.0 / Nel[0] * index1 + 1 / Nel[0] / 2.0 * il1 + pts1[jl1]
                                    value_x = bsp.piecewise(p_shape[0], p_size[0], abs(eta1 - particle[0, ip]))
                                    # ========= mapping evaluation =============
                                    span1f = int((eta1 % 1.0) * nelf[0]) + pf1
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
                                        eta1 % 1.0,
                                        eta2 % 1.0,
                                        eta3 % 1.0,
                                        df,
                                        fx,
                                        0,
                                    )
                                    # evaluate transpose of Jacobian matrix
                                    linalg.transpose(df, dft)
                                    det_df = abs(linalg.det(df))

                                    mat_11[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[0, 0] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_21[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[0, 1] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_31[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[0, 2] * value_x * value_y * value_z / det_df * vol
                                    )

                    for il1 in range(num_cell[0]):
                        eta1 = 1.0 / Nel[0] * index1 + 1.0 / Nel[0] / num_cell[0] * il1
                        span1f = int((eta1 % 1.0) * nelf[0]) + pf1
                        value_x = bsp.piecewise(p_shape[0], p_size[0], abs(eta1 - particle[0, ip]))
                        for il2 in range(2):
                            for il3 in range(num_cell[2]):
                                eta3 = 1.0 / Nel[2] * index3 + 1.0 / Nel[2] / num_cell[2] * il3
                                span3f = int((eta3 % 1.0) * nelf[2]) + pf3
                                value_z = bsp.piecewise(p_shape[2], p_size[2], abs(eta3 - particle[2, ip]))
                                for jl1 in range(quad[1]):
                                    eta2 = 1.0 / Nel[1] * index2 + 1.0 / Nel[1] / 2.0 * il2 + pts2[jl1]
                                    value_y = bsp.piecewise(p_shape[1], p_size[1], abs(eta2 - particle[1, ip]))
                                    # ========= mapping evaluation =============
                                    span2f = int((eta2 % 1.0) * nelf[1]) + pf2
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
                                        eta1 % 1.0,
                                        eta2 % 1.0,
                                        eta3 % 1.0,
                                        df,
                                        fx,
                                        0,
                                    )
                                    # evaluate transpose of Jacobian matrix
                                    linalg.transpose(df, dft)
                                    det_df = abs(linalg.det(df))

                                    mat_12[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[1, 0] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_22[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[1, 1] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_32[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[1, 2] * value_x * value_y * value_z / det_df * vol
                                    )
                    for il1 in range(num_cell[0]):
                        eta1 = 1.0 / Nel[0] * index1 + 1.0 / Nel[0] / num_cell[0] * il1
                        span1f = int((eta1 % 1.0) * nelf[0]) + pf1
                        value_x = bsp.piecewise(p_shape[0], p_size[0], abs(eta1 - particle[0, ip]))
                        for il2 in range(num_cell[1]):
                            eta2 = 1.0 / Nel[1] * index2 + 1.0 / Nel[1] / num_cell[1] * il2
                            span2f = int((eta2 % 1.0) * nelf[1]) + pf2
                            value_y = bsp.piecewise(p_shape[1], p_size[1], abs(eta2 - particle[1, ip]))
                            for il3 in range(2):
                                for jl1 in range(quad[2]):
                                    eta3 = 1.0 / Nel[2] * index3 + 1.0 / Nel[2] / 2.0 * il3 + pts3[jl1]
                                    value_z = bsp.piecewise(p_shape[2], p_size[2], abs(eta3 - particle[2, ip]))
                                    # ========= mapping evaluation =============
                                    span3f = int((eta3 % 1.0) * nelf[2]) + pf3
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
                                        eta1 % 1.0,
                                        eta2 % 1.0,
                                        eta3 % 1.0,
                                        df,
                                        fx,
                                        0,
                                    )
                                    # evaluate transpose of Jacobian matrix
                                    linalg.transpose(df, dft)
                                    det_df = abs(linalg.det(df))

                                    mat_13[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[2, 0] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_23[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[2, 1] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_33[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[2, 2] * value_x * value_y * value_z / det_df * vol
                                    )

        for i1 in range(cell_number[0]):
            index1 = cell_left[0] + i1
            for i2 in range(cell_number[1]):
                index2 = cell_left[1] + i2
                for i3 in range(cell_number[2]):
                    index3 = cell_left[2] + i3

                    for lambda_index1 in range(p[0]):
                        final_index1 = (lambda_index1 + index1) % NbaseD[0]
                        for lambda_index2 in range(p[1]):
                            final_index2 = (lambda_index2 + index2) % NbaseN[1]
                            for lambda_index3 in range(p[2]):
                                final_index3 = (lambda_index3 + index3) % NbaseN[2]
                                for il1 in range(2):
                                    for il2 in range(num_cell[1]):
                                        for il3 in range(num_cell[2]):
                                            f_int = 0.0
                                            for jl1 in range(quad[0]):
                                                f_int += wts1[jl1] * mat_11[i1, i2, i3, il1, il2, il3, jl1]
                                            mid1 = (
                                                coeff_h_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            f_int = 0.0
                                            for jl1 in range(quad[0]):
                                                f_int += wts1[jl1] * mat_21[i1, i2, i3, il1, il2, il3, jl1]
                                            mid2 = (
                                                coeff_h_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            f_int = 0.0
                                            for jl1 in range(quad[0]):
                                                f_int += wts1[jl1] * mat_31[i1, i2, i3, il1, il2, il3, jl1]
                                            mid3 = (
                                                coeff_h_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            right1[final_index1, final_index2, final_index3] += w * (
                                                (particle[3, ip] + ddt * mid_particle[0, ip]) * mid1
                                                + (particle[4, ip] + ddt * mid_particle[1, ip]) * mid2
                                                + (particle[5, ip] + ddt * mid_particle[2, ip]) * mid3
                                            )

                    for lambda_index1 in range(p[0]):
                        final_index1 = (lambda_index1 + index1) % NbaseN[0]
                        for lambda_index2 in range(p[1]):
                            final_index2 = (lambda_index2 + index2) % NbaseD[1]
                            for lambda_index3 in range(p[2]):
                                final_index3 = (lambda_index3 + index3) % NbaseN[2]
                                for il1 in range(num_cell[0]):
                                    for il2 in range(2):
                                        for il3 in range(num_cell[2]):
                                            f_int = 0.0
                                            for jl1 in range(quad[1]):
                                                f_int += wts2[jl1] * mat_12[i1, i2, i3, il1, il2, il3, jl1]
                                            mid1 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_h_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            f_int = 0.0
                                            for jl1 in range(quad[1]):
                                                f_int += wts2[jl1] * mat_22[i1, i2, i3, il1, il2, il3, jl1]
                                            mid2 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_h_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            f_int = 0.0
                                            for jl1 in range(quad[1]):
                                                f_int += wts2[jl1] * mat_32[i1, i2, i3, il1, il2, il3, jl1]
                                            mid3 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_h_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            right2[final_index1, final_index2, final_index3] += w * (
                                                (particle[3, ip] + ddt * mid_particle[0, ip]) * mid1
                                                + (particle[4, ip] + ddt * mid_particle[1, ip]) * mid2
                                                + (particle[5, ip] + ddt * mid_particle[2, ip]) * mid3
                                            )

                    for lambda_index1 in range(p[0]):
                        final_index1 = (lambda_index1 + index1) % NbaseN[0]
                        for lambda_index2 in range(p[1]):
                            final_index2 = (lambda_index2 + index2) % NbaseN[1]
                            for lambda_index3 in range(p[2]):
                                final_index3 = (lambda_index3 + index3) % NbaseD[2]
                                for il1 in range(num_cell[0]):
                                    for il2 in range(num_cell[1]):
                                        for il3 in range(2):
                                            f_int = 0.0
                                            for jl1 in range(quad[2]):
                                                f_int += wts3[jl1] * mat_13[i1, i2, i3, il1, il2, il3, jl1]
                                            mid1 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_h_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            f_int = 0.0
                                            for jl1 in range(quad[2]):
                                                f_int += wts3[jl1] * mat_23[i1, i2, i3, il1, il2, il3, jl1]
                                            mid2 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_h_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            f_int = 0.0
                                            for jl1 in range(quad[2]):
                                                f_int += wts3[jl1] * mat_33[i1, i2, i3, il1, il2, il3, jl1]
                                            mid3 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_h_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            right3[final_index1, final_index2, final_index3] += w * (
                                                (particle[3, ip] + ddt * mid_particle[0, ip]) * mid1
                                                + (particle[4, ip] + ddt * mid_particle[1, ip]) * mid2
                                                + (particle[5, ip] + ddt * mid_particle[2, ip]) * mid3
                                            )

        del mat_11
        del mat_12
        del mat_13
        del mat_21
        del mat_22
        del mat_23
        del mat_31
        del mat_32
        del mat_33

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel
    ierr = 0


# ==============================================================================================
def vv_push(
    out: "float[:,:]",
    dt: "float",
    bb1: "float[:,:,:]",
    bb2: "float[:,:,:]",
    bb3: "float[:,:,:]",
    pts1: "float[:]",
    pts2: "float[:]",
    pts3: "float[:]",
    wts1: "float[:]",
    wts2: "float[:]",
    wts3: "float[:]",
    Np: "int",
    quad: "int[:]",
    p: "int[:]",
    Nel: "int[:]",
    p_shape: "int[:]",
    p_size: "float[:]",
    particle: "float[:,:]",
    lambdas_11: "float[:,:,:]",
    lambdas_12: "float[:,:,:]",
    lambdas_13: "float[:,:,:]",
    lambdas_21: "float[:,:,:]",
    lambdas_22: "float[:,:,:]",
    lambdas_23: "float[:,:,:]",
    lambdas_31: "float[:,:,:]",
    lambdas_32: "float[:,:,:]",
    lambdas_33: "float[:,:,:]",
    num_cell: "int[:]",
    coeff_i_x: "float[:]",
    coeff_i_y: "float[:]",
    coeff_i_z: "float[:]",
    coeff_h_x: "float[:]",
    coeff_h_y: "float[:]",
    coeff_h_z: "float[:]",
    NbaseN: "int[:]",
    NbaseD: "int[:]",
    related: "int[:]",
    Np_loc: "int",
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
    from numpy import empty, floor, zeros

    cell_left = zeros(3, dtype=int)
    point_left = zeros(3, dtype=float)
    point_right = zeros(3, dtype=float)
    cell_number = zeros(3, dtype=int)
    compact = zeros(3, dtype=float)

    vel = zeros(3, dtype=float)
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
    df = zeros((3, 3), dtype=float)
    dft = zeros((3, 3), dtype=float)

    grids_shapex = zeros(p_shape[0] + 2, dtype=float)
    grids_shapey = zeros(p_shape[1] + 2, dtype=float)
    grids_shapez = zeros(p_shape[2] + 2, dtype=float)
    # ====================================
    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (i, grids_shapex, grids_shapey, grids_shapez, vel, mid1, mid2, mid3, ip, w, det_df, vol, lambdas_11, lambdas_12, lambdas_13, lambdas_21, lambdas_22, lambdas_23, lambdas_31, lambdas_32, lambdas_33, cell_left, point_left, point_right, cell_number, compact, mat_11, mat_12, mat_13, mat_21, mat_22, mat_23, mat_31, mat_32, mat_33, i1, i2, i3, il1, il2, il3, jl1, index1, index2, index3, value_x, value_y, value_z, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, dft, eta1, eta2, eta3, final_index1, final_index2, final_index3, lambda_index1, lambda_index2, lambda_index3, f_int)
    for ip in range(Np_loc):
        vel[:] = 0.0

        w = particle[6, ip] / Np

        lambdas_11[:, :, :] = 0.0
        lambdas_12[:, :, :] = 0.0
        lambdas_13[:, :, :] = 0.0

        lambdas_21[:, :, :] = 0.0
        lambdas_22[:, :, :] = 0.0
        lambdas_23[:, :, :] = 0.0

        lambdas_31[:, :, :] = 0.0
        lambdas_32[:, :, :] = 0.0
        lambdas_33[:, :, :] = 0.0

        # ==================================
        compact[0] = (p_shape[0] + 1.0) * p_size[0]
        compact[1] = (p_shape[1] + 1.0) * p_size[1]
        compact[2] = (p_shape[2] + 1.0) * p_size[2]

        point_left[0] = particle[0, ip] - 0.5 * compact[0]
        point_right[0] = particle[0, ip] + 0.5 * compact[0]
        point_left[1] = particle[1, ip] - 0.5 * compact[1]
        point_right[1] = particle[1, ip] + 0.5 * compact[1]
        point_left[2] = particle[2, ip] - 0.5 * compact[2]
        point_right[2] = particle[2, ip] + 0.5 * compact[2]

        cell_left[0] = int(floor(point_left[0] * Nel[0]))
        cell_left[1] = int(floor(point_left[1] * Nel[1]))
        cell_left[2] = int(floor(point_left[2] * Nel[2]))

        cell_number[0] = int(floor(point_right[0] * Nel[0])) - cell_left[0] + 1
        cell_number[1] = int(floor(point_right[1] * Nel[1])) - cell_left[1] + 1
        cell_number[2] = int(floor(point_right[2] * Nel[2])) - cell_left[2] + 1

        for i in range(p_shape[0] + 1):
            grids_shapex[i] = point_left[0] + i * p_size[0]
        grids_shapex[p_shape[0] + 1] = point_right[0]

        for i in range(p_shape[1] + 1):
            grids_shapey[i] = point_left[1] + i * p_size[1]
        grids_shapey[p_shape[1] + 1] = point_right[1]

        for i in range(p_shape[2] + 1):
            grids_shapez[i] = point_left[2] + i * p_size[2]
        grids_shapez[p_shape[2] + 1] = point_right[2]

        vol = 1.0 / (p_size[0] * p_size[1] * p_size[2])

        # evaluation of function at interpolation/quadrature points
        mat_11 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], 2, num_cell[1], num_cell[2], quad[0]),
            dtype=float,
        )
        mat_21 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], 2, num_cell[1], num_cell[2], quad[0]),
            dtype=float,
        )
        mat_31 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], 2, num_cell[1], num_cell[2], quad[0]),
            dtype=float,
        )

        mat_12 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], 2, num_cell[2], quad[1]),
            dtype=float,
        )
        mat_22 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], 2, num_cell[2], quad[1]),
            dtype=float,
        )
        mat_32 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], 2, num_cell[2], quad[1]),
            dtype=float,
        )

        mat_13 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], num_cell[1], 2, quad[2]),
            dtype=float,
        )
        mat_23 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], num_cell[1], 2, quad[2]),
            dtype=float,
        )
        mat_33 = zeros(
            (cell_number[0], cell_number[1], cell_number[2], num_cell[0], num_cell[1], 2, quad[2]),
            dtype=float,
        )

        for i1 in range(cell_number[0]):
            index1 = cell_left[0] + i1
            for i2 in range(cell_number[1]):
                index2 = cell_left[1] + i2
                for i3 in range(cell_number[2]):  # num_cell = 1, p = 1; num_cell = 2, p >= 2
                    index3 = cell_left[2] + i3
                    for il1 in range(2):
                        for il2 in range(num_cell[1]):
                            eta2 = 1.0 / Nel[1] * index2 + 1 / Nel[1] / num_cell[1] * il2
                            span2f = int((eta2 % 1.0) * nelf[1]) + pf2
                            value_y = bsp.piecewise(p_shape[1], p_size[1], abs(eta2 - particle[1, ip]))
                            for il3 in range(num_cell[2]):
                                eta3 = 1.0 / Nel[2] * index3 + 1 / Nel[2] / num_cell[2] * il3
                                span3f = int((eta3 % 1.0) * nelf[2]) + pf3
                                value_z = bsp.piecewise(p_shape[2], p_size[2], abs(eta3 - particle[2, ip]))
                                for jl1 in range(quad[0]):
                                    eta1 = 1.0 / Nel[0] * index1 + 1 / Nel[0] / 2.0 * il1 + pts1[jl1]
                                    value_x = bsp.piecewise(p_shape[0], p_size[0], abs(eta1 - particle[0, ip]))
                                    # ========= mapping evaluation =============
                                    span1f = int((eta1 % 1.0) * nelf[0]) + pf1
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
                                        eta1 % 1.0,
                                        eta2 % 1.0,
                                        eta3 % 1.0,
                                        df,
                                        fx,
                                        0,
                                    )
                                    # evaluate transpose of Jacobian matrix
                                    linalg.transpose(df, dft)
                                    det_df = abs(linalg.det(df))

                                    mat_11[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[0, 0] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_21[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[0, 1] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_31[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[0, 2] * value_x * value_y * value_z / det_df * vol
                                    )

                    for il1 in range(num_cell[0]):
                        eta1 = 1.0 / Nel[0] * index1 + 1.0 / Nel[0] / num_cell[0] * il1
                        span1f = int((eta1 % 1.0) * nelf[0]) + pf1
                        value_x = bsp.piecewise(p_shape[0], p_size[0], abs(eta1 - particle[0, ip]))
                        for il2 in range(2):
                            for il3 in range(num_cell[2]):
                                eta3 = 1.0 / Nel[2] * index3 + 1.0 / Nel[2] / num_cell[2] * il3
                                span3f = int((eta3 % 1.0) * nelf[2]) + pf3
                                value_z = bsp.piecewise(p_shape[2], p_size[2], abs(eta3 - particle[2, ip]))
                                for jl1 in range(quad[1]):
                                    eta2 = 1.0 / Nel[1] * index2 + 1.0 / Nel[1] / 2.0 * il2 + pts2[jl1]
                                    value_y = bsp.piecewise(p_shape[1], p_size[1], abs(eta2 - particle[1, ip]))
                                    # ========= mapping evaluation =============
                                    span2f = int((eta2 % 1.0) * nelf[1]) + pf2
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
                                        eta1 % 1.0,
                                        eta2 % 1.0,
                                        eta3 % 1.0,
                                        df,
                                        fx,
                                        0,
                                    )
                                    # evaluate transpose of Jacobian matrix
                                    linalg.transpose(df, dft)
                                    det_df = abs(linalg.det(df))

                                    mat_12[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[1, 0] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_22[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[1, 1] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_32[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[1, 2] * value_x * value_y * value_z / det_df * vol
                                    )

                    for il1 in range(num_cell[0]):
                        eta1 = 1.0 / Nel[0] * index1 + 1.0 / Nel[0] / num_cell[0] * il1
                        span1f = int((eta1 % 1.0) * nelf[0]) + pf1
                        value_x = bsp.piecewise(p_shape[0], p_size[0], abs(eta1 - particle[0, ip]))
                        for il2 in range(num_cell[1]):
                            eta2 = 1.0 / Nel[1] * index2 + 1.0 / Nel[1] / num_cell[1] * il2
                            span2f = int((eta2 % 1.0) * nelf[1]) + pf2
                            value_y = bsp.piecewise(p_shape[1], p_size[1], abs(eta2 - particle[1, ip]))
                            for il3 in range(2):
                                for jl1 in range(quad[2]):
                                    eta3 = 1.0 / Nel[2] * index3 + 1.0 / Nel[2] / 2.0 * il3 + pts3[jl1]
                                    value_z = bsp.piecewise(p_shape[2], p_size[2], abs(eta3 - particle[2, ip]))
                                    # ========= mapping evaluation =============
                                    span3f = int((eta3 % 1.0) * nelf[2]) + pf3
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
                                        eta1 % 1.0,
                                        eta2 % 1.0,
                                        eta3 % 1.0,
                                        df,
                                        fx,
                                        0,
                                    )
                                    # evaluate transpose of Jacobian matrix
                                    linalg.transpose(df, dft)
                                    det_df = abs(linalg.det(df))

                                    mat_13[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[2, 0] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_23[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[2, 1] * value_x * value_y * value_z / det_df * vol
                                    )
                                    mat_33[i1, i2, i3, il1, il2, il3, jl1] = (
                                        dft[2, 2] * value_x * value_y * value_z / det_df * vol
                                    )

        for i1 in range(cell_number[0]):
            index1 = cell_left[0] + i1
            for i2 in range(cell_number[1]):
                index2 = cell_left[1] + i2
                for i3 in range(cell_number[2]):
                    index3 = cell_left[2] + i3

                    for lambda_index1 in range(p[0]):
                        final_index1 = (lambda_index1 + index1) % NbaseD[0]
                        for lambda_index2 in range(p[1]):
                            final_index2 = (lambda_index2 + index2) % NbaseN[1]
                            for lambda_index3 in range(p[2]):
                                final_index3 = (lambda_index3 + index3) % NbaseN[2]
                                for il1 in range(2):
                                    for il2 in range(num_cell[1]):
                                        for il3 in range(num_cell[2]):
                                            f_int = 0.0
                                            for jl1 in range(quad[0]):
                                                f_int += wts1[jl1] * mat_11[i1, i2, i3, il1, il2, il3, jl1]
                                            mid1 = (
                                                coeff_h_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[0] += mid1 * bb1[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[0]):
                                                f_int += wts1[jl1] * mat_21[i1, i2, i3, il1, il2, il3, jl1]
                                            mid2 = (
                                                coeff_h_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[1] += mid2 * bb1[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[0]):
                                                f_int += wts1[jl1] * mat_31[i1, i2, i3, il1, il2, il3, jl1]
                                            mid3 = (
                                                coeff_h_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[2] += mid3 * bb1[final_index1, final_index2, final_index3]

                    for lambda_index1 in range(p[0]):
                        final_index1 = (lambda_index1 + index1) % NbaseN[0]
                        for lambda_index2 in range(p[1]):
                            final_index2 = (lambda_index2 + index2) % NbaseD[1]
                            for lambda_index3 in range(p[2]):
                                final_index3 = (lambda_index3 + index3) % NbaseN[2]
                                for il1 in range(num_cell[0]):
                                    for il2 in range(2):
                                        for il3 in range(num_cell[2]):
                                            f_int = 0.0
                                            for jl1 in range(quad[1]):
                                                f_int += wts2[jl1] * mat_12[i1, i2, i3, il1, il2, il3, jl1]
                                            mid1 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_h_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[0] += mid1 * bb2[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[1]):
                                                f_int += wts2[jl1] * mat_22[i1, i2, i3, il1, il2, il3, jl1]
                                            mid2 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_h_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[1] += mid2 * bb2[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[1]):
                                                f_int += wts2[jl1] * mat_32[i1, i2, i3, il1, il2, il3, jl1]
                                            mid3 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_h_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_i_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[2] += mid3 * bb2[final_index1, final_index2, final_index3]

                    for lambda_index1 in range(p[0]):
                        final_index1 = (lambda_index1 + index1) % NbaseN[0]
                        for lambda_index2 in range(p[1]):
                            final_index2 = (lambda_index2 + index2) % NbaseN[1]
                            for lambda_index3 in range(p[2]):
                                final_index3 = (lambda_index3 + index3) % NbaseD[2]
                                for il1 in range(num_cell[0]):
                                    for il2 in range(num_cell[1]):
                                        for il3 in range(2):
                                            f_int = 0.0
                                            for jl1 in range(quad[2]):
                                                f_int += wts3[jl1] * mat_13[i1, i2, i3, il1, il2, il3, jl1]
                                            mid1 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_h_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[0] += mid1 * bb3[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[2]):
                                                f_int += wts3[jl1] * mat_23[i1, i2, i3, il1, il2, il3, jl1]
                                            mid2 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_h_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[1] += mid2 * bb3[final_index1, final_index2, final_index3]
                                            f_int = 0.0
                                            for jl1 in range(quad[2]):
                                                f_int += wts3[jl1] * mat_33[i1, i2, i3, il1, il2, il3, jl1]
                                            mid3 = (
                                                coeff_i_x[2 * (p[0] - 1) - 2 * lambda_index1 + il1]
                                                * coeff_i_y[2 * (p[1] - 1) - 2 * lambda_index2 + il2]
                                                * coeff_h_z[2 * (p[2] - 1) - 2 * lambda_index3 + il3]
                                                * f_int
                                            )
                                            vel[2] += mid3 * bb3[final_index1, final_index2, final_index3]

        out[0, ip] = vel[0]
        out[1, ip] = vel[1]
        out[2, ip] = vel[2]

        del mat_11
        del mat_12
        del mat_13
        del mat_21
        del mat_22
        del mat_23
        del mat_31
        del mat_32
        del mat_33
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel
    ierr = 0
