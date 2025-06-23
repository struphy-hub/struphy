import struphy.linear_algebra.linalg_kernels as linalg


# ==========================================================================================
def prepre(
    df_det: "float[:,:,:,:,:,:]",
    G_inv_11: "float[:,:,:,:,:,:]",
    G_inv_12: "float[:,:,:,:,:,:]",
    G_inv_13: "float[:,:,:,:,:,:]",
    G_inv_22: "float[:,:,:,:,:,:]",
    G_inv_23: "float[:,:,:,:,:,:]",
    G_inv_33: "float[:,:,:,:,:,:]",
    N_index_x: "int[:,:]",
    N_index_y: "int[:,:]",
    N_index_z: "int[:,:]",
    D_index_x: "int[:,:]",
    D_index_y: "int[:,:]",
    D_index_z: "int[:,:]",
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
    # ====uvalue is given from other function===========

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
                                            * b1[D_index_x[ie1, il1], N_index_y[ie2, il2], N_index_z[ie3, il3]]
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
                                            * b2[N_index_x[ie1, il1], D_index_y[ie2, il2], N_index_z[ie3, il3]]
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
                                            * b3[N_index_x[ie1, il1], N_index_y[ie2, il2], D_index_z[ie3, il3]]
                                        )

                            b3value[ie1, ie2, ie3, q1, q2, q3] = value
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do private (ie1, ie2, ie3, q1, q2, q3, dft, detdet, generate_weight1, generate_weight3)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            dft[0, 0] = G_inv_11[ie1, ie2, ie3, q1, q2, q3]
                            dft[0, 1] = G_inv_12[ie1, ie2, ie3, q1, q2, q3]
                            dft[0, 2] = G_inv_13[ie1, ie2, ie3, q1, q2, q3]
                            dft[1, 0] = G_inv_12[ie1, ie2, ie3, q1, q2, q3]
                            dft[1, 1] = G_inv_22[ie1, ie2, ie3, q1, q2, q3]
                            dft[1, 2] = G_inv_23[ie1, ie2, ie3, q1, q2, q3]
                            dft[2, 0] = G_inv_13[ie1, ie2, ie3, q1, q2, q3]
                            dft[2, 1] = G_inv_23[ie1, ie2, ie3, q1, q2, q3]
                            dft[2, 2] = G_inv_33[ie1, ie2, ie3, q1, q2, q3]
                            detdet = df_det[ie1, ie2, ie3, q1, q2, q3]
                            generate_weight1[0] = (
                                b1value[ie1, ie2, ie3, q1, q2, q3]
                                * wts1[ie1, q1]
                                * wts2[ie2, q2]
                                * wts3[ie3, q3]
                                * detdet
                            )
                            generate_weight1[1] = (
                                b2value[ie1, ie2, ie3, q1, q2, q3]
                                * wts1[ie1, q1]
                                * wts2[ie2, q2]
                                * wts3[ie3, q3]
                                * detdet
                            )
                            generate_weight1[2] = (
                                b3value[ie1, ie2, ie3, q1, q2, q3]
                                * wts1[ie1, q1]
                                * wts2[ie2, q2]
                                * wts3[ie3, q3]
                                * detdet
                            )
                            linalg.matrix_vector(dft, generate_weight1, generate_weight3)
                            b1value[ie1, ie2, ie3, q1, q2, q3] = generate_weight3[0] * uvalue[ie1, ie2, ie3, q1, q2, q3]
                            b2value[ie1, ie2, ie3, q1, q2, q3] = generate_weight3[1] * uvalue[ie1, ie2, ie3, q1, q2, q3]
                            b3value[ie1, ie2, ie3, q1, q2, q3] = generate_weight3[2] * uvalue[ie1, ie2, ie3, q1, q2, q3]

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    ierr = 0


# ==========================================================================================
def right_hand2(
    N_index_x: "int[:,:]",
    N_index_y: "int[:,:]",
    N_index_z: "int[:,:]",
    D_index_x: "int[:,:]",
    D_index_y: "int[:,:]",
    D_index_z: "int[:,:]",
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
                            for il1 in range(p1 + 1):
                                for il2 in range(d2 + 1):
                                    for il3 in range(d3 + 1):
                                        value += (
                                            bn1[ie1, il1, 0, q1]
                                            * bd2[ie2, il2, 0, q2]
                                            * bd3[ie3, il3, 0, q3]
                                            * temp_vector_1[
                                                N_index_x[ie1, il1], D_index_y[ie2, il2], D_index_z[ie3, il3]
                                            ]
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
                            for il1 in range(d1 + 1):
                                for il2 in range(p2 + 1):
                                    for il3 in range(d3 + 1):
                                        value += (
                                            bd1[ie1, il1, 0, q1]
                                            * bn2[ie2, il2, 0, q2]
                                            * bd3[ie3, il3, 0, q3]
                                            * temp_vector_2[
                                                D_index_x[ie1, il1], N_index_y[ie2, il2], D_index_z[ie3, il3]
                                            ]
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
                            for il1 in range(d1 + 1):
                                for il2 in range(d2 + 1):
                                    for il3 in range(p3 + 1):
                                        value += (
                                            bd1[ie1, il1, 0, q1]
                                            * bd2[ie2, il2, 0, q2]
                                            * bn3[ie3, il3, 0, q3]
                                            * temp_vector_3[
                                                D_index_x[ie1, il1], D_index_y[ie2, il2], N_index_z[ie3, il3]
                                            ]
                                        )

                            right_3[ie1, ie2, ie3, q1, q2, q3] = value
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel
    ierr = 0


# ==========================================================================================
def right_hand1(
    N_index_x: "int[:,:]",
    N_index_y: "int[:,:]",
    N_index_z: "int[:,:]",
    D_index_x: "int[:,:]",
    D_index_y: "int[:,:]",
    D_index_z: "int[:,:]",
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
                                            * temp_vector_1[
                                                D_index_x[ie1, il1], N_index_y[ie2, il2], N_index_z[ie3, il3]
                                            ]
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
                                            * temp_vector_2[
                                                N_index_x[ie1, il1], D_index_y[ie2, il2], N_index_z[ie3, il3]
                                            ]
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
                                            * temp_vector_3[
                                                N_index_x[ie1, il1], N_index_y[ie2, il2], D_index_z[ie3, il3]
                                            ]
                                        )

                            right_3[ie1, ie2, ie3, q1, q2, q3] = value
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel
    ierr = 0


# ==========================================================================================
def weight_2(
    DF_inv_11: "float[:,:,:,:,:,:]",
    DF_inv_12: "float[:,:,:,:,:,:]",
    DF_inv_13: "float[:,:,:,:,:,:]",
    DF_inv_21: "float[:,:,:,:,:,:]",
    DF_inv_22: "float[:,:,:,:,:,:]",
    DF_inv_23: "float[:,:,:,:,:,:]",
    DF_inv_31: "float[:,:,:,:,:,:]",
    DF_inv_32: "float[:,:,:,:,:,:]",
    DF_inv_33: "float[:,:,:,:,:,:]",
    pts1: "float[:,:]",
    pts2: "float[:,:]",
    pts3: "float[:,:]",
    dft: "float[:,:]",
    generate_weight1: "float[:]",
    generate_weight3: "float[:]",
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
    # -- removed omp: #$ omp do private (ie1, ie2, ie3, q1, q2, q3, dft, generate_weight1, generate_weight3)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            dft[0, 0] = DF_inv_11[ie1, ie2, ie3, q1, q2, q3]
                            dft[0, 1] = DF_inv_21[ie1, ie2, ie3, q1, q2, q3]
                            dft[0, 2] = DF_inv_31[ie1, ie2, ie3, q1, q2, q3]
                            dft[1, 0] = DF_inv_12[ie1, ie2, ie3, q1, q2, q3]
                            dft[1, 1] = DF_inv_22[ie1, ie2, ie3, q1, q2, q3]
                            dft[1, 2] = DF_inv_32[ie1, ie2, ie3, q1, q2, q3]
                            dft[2, 0] = DF_inv_13[ie1, ie2, ie3, q1, q2, q3]
                            dft[2, 1] = DF_inv_23[ie1, ie2, ie3, q1, q2, q3]
                            dft[2, 2] = DF_inv_33[ie1, ie2, ie3, q1, q2, q3]

                            generate_weight1[0] = -(
                                b2value[ie1, ie2, ie3, q1, q2, q3] * right_3[ie1, ie2, ie3, q1, q2, q3]
                                - b3value[ie1, ie2, ie3, q1, q2, q3] * right_2[ie1, ie2, ie3, q1, q2, q3]
                            )
                            generate_weight1[1] = -(
                                b3value[ie1, ie2, ie3, q1, q2, q3] * right_1[ie1, ie2, ie3, q1, q2, q3]
                                - b1value[ie1, ie2, ie3, q1, q2, q3] * right_3[ie1, ie2, ie3, q1, q2, q3]
                            )
                            generate_weight1[2] = -(
                                b1value[ie1, ie2, ie3, q1, q2, q3] * right_2[ie1, ie2, ie3, q1, q2, q3]
                                - b2value[ie1, ie2, ie3, q1, q2, q3] * right_1[ie1, ie2, ie3, q1, q2, q3]
                            )
                            linalg.matrix_vector(dft, generate_weight1, generate_weight3)
                            weight1[ie1, ie2, ie3, q1, q2, q3] = generate_weight3[
                                0
                            ]  # -(b2value[ie1,ie2,ie3,q1,q2,q3] * right_3[ie1,ie2,ie3,q1,q2,q3] - b3value[ie1,ie2,ie3,q1,q2,q3] * right_2[ie1,ie2,ie3,q1,q2,q3])
                            weight2[ie1, ie2, ie3, q1, q2, q3] = generate_weight3[
                                1
                            ]  # -(b3value[ie1,ie2,ie3,q1,q2,q3] * right_1[ie1,ie2,ie3,q1,q2,q3] - b1value[ie1,ie2,ie3,q1,q2,q3] * right_3[ie1,ie2,ie3,q1,q2,q3])
                            weight3[ie1, ie2, ie3, q1, q2, q3] = generate_weight3[
                                2
                            ]  # -(b1value[ie1,ie2,ie3,q1,q2,q3] * right_2[ie1,ie2,ie3,q1,q2,q3] - b2value[ie1,ie2,ie3,q1,q2,q3] * right_1[ie1,ie2,ie3,q1,q2,q3])
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    ierr = 0


# ==========================================================================================
def weight_1(
    DF_inv_11: "float[:,:,:,:,:,:]",
    DF_inv_12: "float[:,:,:,:,:,:]",
    DF_inv_13: "float[:,:,:,:,:,:]",
    DF_inv_21: "float[:,:,:,:,:,:]",
    DF_inv_22: "float[:,:,:,:,:,:]",
    DF_inv_23: "float[:,:,:,:,:,:]",
    DF_inv_31: "float[:,:,:,:,:,:]",
    DF_inv_32: "float[:,:,:,:,:,:]",
    DF_inv_33: "float[:,:,:,:,:,:]",
    pts1: "float[:,:]",
    pts2: "float[:,:]",
    pts3: "float[:,:]",
    dft: "float[:,:]",
    generate_weight1: "float[:]",
    generate_weight3: "float[:]",
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
    # -- removed omp: #$ omp do private (ie1, ie2, ie3, q1, q2, q3, dft, generate_weight1, generate_weight3)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            dft[0, 0] = DF_inv_11[ie1, ie2, ie3, q1, q2, q3]
                            dft[0, 1] = DF_inv_12[ie1, ie2, ie3, q1, q2, q3]
                            dft[0, 2] = DF_inv_13[ie1, ie2, ie3, q1, q2, q3]
                            dft[1, 0] = DF_inv_21[ie1, ie2, ie3, q1, q2, q3]
                            dft[1, 1] = DF_inv_22[ie1, ie2, ie3, q1, q2, q3]
                            dft[1, 2] = DF_inv_23[ie1, ie2, ie3, q1, q2, q3]
                            dft[2, 0] = DF_inv_31[ie1, ie2, ie3, q1, q2, q3]
                            dft[2, 1] = DF_inv_32[ie1, ie2, ie3, q1, q2, q3]
                            dft[2, 2] = DF_inv_33[ie1, ie2, ie3, q1, q2, q3]

                            generate_weight1[0] = right_1[ie1, ie2, ie3, q1, q2, q3]
                            generate_weight1[1] = right_2[ie1, ie2, ie3, q1, q2, q3]
                            generate_weight1[2] = right_3[ie1, ie2, ie3, q1, q2, q3]
                            linalg.matrix_vector(dft, generate_weight1, generate_weight3)
                            weight1[ie1, ie2, ie3, q1, q2, q3] = (
                                b2value[ie1, ie2, ie3, q1, q2, q3] * generate_weight3[2]
                                - b3value[ie1, ie2, ie3, q1, q2, q3] * generate_weight3[1]
                            )
                            weight2[ie1, ie2, ie3, q1, q2, q3] = (
                                b3value[ie1, ie2, ie3, q1, q2, q3] * generate_weight3[0]
                                - b1value[ie1, ie2, ie3, q1, q2, q3] * generate_weight3[2]
                            )
                            weight3[ie1, ie2, ie3, q1, q2, q3] = (
                                b1value[ie1, ie2, ie3, q1, q2, q3] * generate_weight3[1]
                                - b2value[ie1, ie2, ie3, q1, q2, q3] * generate_weight3[0]
                            )
    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    ierr = 0


# ==========================================================================================
def final_left(
    N_index_x: "int[:,:]",
    N_index_y: "int[:,:]",
    N_index_z: "int[:,:]",
    D_index_x: "int[:,:]",
    D_index_y: "int[:,:]",
    D_index_z: "int[:,:]",
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
                for il1 in range(p1 + 1):
                    for il2 in range(d2 + 1):
                        for il3 in range(d3 + 1):
                            value = 0.0
                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):
                                        value += (
                                            weight1[ie1, ie2, ie3, q1, q2, q3]
                                            * bn1[ie1, il1, 0, q1]
                                            * bd2[ie2, il2, 0, q2]
                                            * bd3[ie3, il3, 0, q3]
                                        )

                            temp_final_1[N_index_x[ie1, il1], D_index_y[ie2, il2], D_index_z[ie3, il3]] += value

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do reduction ( + : temp_final_2) private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)

    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for il1 in range(d1 + 1):
                    for il2 in range(p2 + 1):
                        for il3 in range(d3 + 1):
                            value = 0.0
                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):
                                        value += (
                                            weight2[ie1, ie2, ie3, q1, q2, q3]
                                            * bd1[ie1, il1, 0, q1]
                                            * bn2[ie2, il2, 0, q2]
                                            * bd3[ie3, il3, 0, q3]
                                        )

                            temp_final_2[D_index_x[ie1, il1], N_index_y[ie2, il2], D_index_z[ie3, il3]] += value

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    # -- removed omp: #$ omp parallel
    # -- removed omp: #$ omp do reduction ( + : temp_final_3) private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3, value)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):
                for il1 in range(d1 + 1):
                    for il2 in range(d2 + 1):
                        for il3 in range(p3 + 1):
                            value = 0.0
                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):
                                        value += (
                                            weight3[ie1, ie2, ie3, q1, q2, q3]
                                            * bd1[ie1, il1, 0, q1]
                                            * bd2[ie2, il2, 0, q2]
                                            * bn3[ie3, il3, 0, q3]
                                        )

                            temp_final_3[D_index_x[ie1, il1], D_index_y[ie2, il2], N_index_z[ie3, il3]] += value

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    ierr = 0


# ==========================================================================================
def final_right(
    N_index_x: "int[:,:]",
    N_index_y: "int[:,:]",
    N_index_z: "int[:,:]",
    D_index_x: "int[:,:]",
    D_index_y: "int[:,:]",
    D_index_z: "int[:,:]",
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

                            temp_final_1[D_index_x[ie1, il1], N_index_y[ie2, il2], N_index_z[ie3, il3]] += value

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

                            temp_final_2[N_index_x[ie1, il1], D_index_y[ie2, il2], N_index_z[ie3, il3]] += value

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

                            temp_final_3[N_index_x[ie1, il1], N_index_y[ie2, il2], D_index_z[ie3, il3]] += value

    # -- removed omp: #$ omp end do
    # -- removed omp: #$ omp end parallel

    ierr = 0
