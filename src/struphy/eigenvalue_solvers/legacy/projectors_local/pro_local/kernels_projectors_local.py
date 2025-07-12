# ==========================================================================================
def kernel_pi0_3d(
    n: "int[:]",
    p: "int[:]",
    coeff_i1: "float[:,:]",
    coeff_i2: "float[:,:]",
    coeff_i3: "float[:,:]",
    coeffi_ind1: "int[:]",
    coeffi_ind2: "int[:]",
    coeffi_ind3: "int[:]",
    x_int_ind1: "int[:,:]",
    x_int_ind2: "int[:,:]",
    x_int_ind3: "int[:,:]",
    mat_f: "float[:,:,:]",
    lambdas: "float[:,:,:]",
):
    n_pts1 = 2 * (p[0] - 1) + 1
    n_pts2 = 2 * (p[1] - 1) + 1
    n_pts3 = 2 * (p[2] - 1) + 1

    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                for il1 in range(n_pts1):
                    for il2 in range(n_pts2):
                        for il3 in range(n_pts3):
                            lambdas[i1, i2, i3] += (
                                coeff_i1[coeffi_ind1[i1], il1]
                                * coeff_i2[coeffi_ind2[i2], il2]
                                * coeff_i3[coeffi_ind3[i3], il3]
                                * mat_f[x_int_ind1[i1, il1], x_int_ind2[i2, il2], x_int_ind3[i3, il3]]
                            )


# ==========================================================================================
def kernel_pi11_3d(
    n: "int[:]",
    p: "int[:]",
    nq: "int[:]",
    coeff_h1: "float[:,:]",
    coeff_i2: "float[:,:]",
    coeff_i3: "float[:,:]",
    coeffh_ind1: "int[:]",
    coeffi_ind2: "int[:]",
    coeffi_ind3: "int[:]",
    x_his_ind1: "int[:,:]",
    x_int_ind2: "int[:,:]",
    x_int_ind3: "int[:,:]",
    wts1: "float[:,:]",
    mat_f: "float[:,:,:,:]",
    lambdas: "float[:,:,:]",
):
    n_pts2 = 2 * (p[1] - 1) + 1
    n_pts3 = 2 * (p[2] - 1) + 1

    n_his1 = 2 * p[0]

    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                for il1 in range(n_his1):
                    for il2 in range(n_pts2):
                        for il3 in range(n_pts3):
                            f_int = 0.0

                            for q1 in range(nq[0]):
                                f_int += (
                                    wts1[x_his_ind1[i1, il1], q1]
                                    * mat_f[x_his_ind1[i1, il1], q1, x_int_ind2[i2, il2], x_int_ind3[i3, il3]]
                                )

                            lambdas[i1, i2, i3] += (
                                coeff_h1[coeffh_ind1[i1], il1]
                                * coeff_i2[coeffi_ind2[i2], il2]
                                * coeff_i3[coeffi_ind3[i3], il3]
                                * f_int
                            )


# ==========================================================================================
def kernel_pi12_3d(
    n: "int[:]",
    p: "int[:]",
    nq: "int[:]",
    coeff_i1: "float[:,:]",
    coeff_h2: "float[:,:]",
    coeff_i3: "float[:,:]",
    coeffi_ind1: "int[:]",
    coeffh_ind2: "int[:]",
    coeffi_ind3: "int[:]",
    x_int_ind1: "int[:,:]",
    x_his_ind2: "int[:,:]",
    x_int_ind3: "int[:,:]",
    wts2: "float[:,:]",
    mat_f: "float[:,:,:,:]",
    lambdas: "float[:,:,:]",
):
    n_pts1 = 2 * (p[0] - 1) + 1
    n_pts3 = 2 * (p[2] - 1) + 1

    n_his2 = 2 * p[1]

    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                for il1 in range(n_pts1):
                    for il2 in range(n_his2):
                        for il3 in range(n_pts3):
                            f_int = 0.0

                            for q2 in range(nq[1]):
                                f_int += (
                                    wts2[x_his_ind2[i2, il2], q2]
                                    * mat_f[x_int_ind1[i1, il1], x_his_ind2[i2, il2], q2, x_int_ind3[i3, il3]]
                                )

                            lambdas[i1, i2, i3] += (
                                coeff_i1[coeffi_ind1[i1], il1]
                                * coeff_h2[coeffh_ind2[i2], il2]
                                * coeff_i3[coeffi_ind3[i3], il3]
                                * f_int
                            )


# ==========================================================================================
def kernel_pi13_3d(
    n: "int[:]",
    p: "int[:]",
    nq: "int[:]",
    coeff_i1: "float[:,:]",
    coeff_i2: "float[:,:]",
    coeff_h3: "float[:,:]",
    coeffi_ind1: "int[:]",
    coeffi_ind2: "int[:]",
    coeffh_ind3: "int[:]",
    x_int_ind1: "int[:,:]",
    x_int_ind2: "int[:,:]",
    x_his_ind3: "int[:,:]",
    wts3: "float[:,:]",
    mat_f: "float[:,:,:,:]",
    lambdas: "float[:,:,:]",
):
    n_pts1 = 2 * (p[0] - 1) + 1
    n_pts2 = 2 * (p[1] - 1) + 1

    n_his3 = 2 * p[2]

    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                for il1 in range(n_pts1):
                    for il2 in range(n_pts2):
                        for il3 in range(n_his3):
                            f_int = 0.0

                            for q3 in range(nq[2]):
                                f_int += (
                                    wts3[x_his_ind3[i3, il3], q3]
                                    * mat_f[x_int_ind1[i1, il1], x_int_ind2[i2, il2], x_his_ind3[i3, il3], q3]
                                )

                            lambdas[i1, i2, i3] += (
                                coeff_i1[coeffi_ind1[i1], il1]
                                * coeff_i2[coeffi_ind2[i2], il2]
                                * coeff_h3[coeffh_ind3[i3], il3]
                                * f_int
                            )


# ==========================================================================================
def kernel_pi21_3d(
    n: "int[:]",
    p: "int[:]",
    nq: "int[:]",
    coeff_i1: "float[:,:]",
    coeff_h2: "float[:,:]",
    coeff_h3: "float[:,:]",
    coeffi_ind1: "int[:]",
    coeffh_ind2: "int[:]",
    coeffh_ind3: "int[:]",
    x_int_ind1: "int[:,:]",
    x_his_ind2: "int[:,:]",
    x_his_ind3: "int[:,:]",
    wts2: "float[:,:]",
    wts3: "float[:,:]",
    mat_f: "float[:,:,:,:,:]",
    lambdas: "float[:,:,:]",
):
    n_pts1 = 2 * (p[0] - 1) + 1

    n_his2 = 2 * p[1]
    n_his3 = 2 * p[2]

    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                for il1 in range(n_pts1):
                    for il2 in range(n_his2):
                        for il3 in range(n_his3):
                            f_int = 0.0

                            for q2 in range(nq[1]):
                                for q3 in range(nq[2]):
                                    wvol = wts2[x_his_ind2[i2, il2], q2] * wts3[x_his_ind3[i3, il3], q3]
                                    f_int += (
                                        wvol
                                        * mat_f[x_int_ind1[i1, il1], x_his_ind2[i2, il2], q2, x_his_ind3[i3, il3], q3]
                                    )

                            lambdas[i1, i2, i3] += (
                                coeff_i1[coeffi_ind1[i1], il1]
                                * coeff_h2[coeffh_ind2[i2], il2]
                                * coeff_h3[coeffh_ind3[i3], il3]
                                * f_int
                            )


# ==========================================================================================
def kernel_pi22_3d(
    n: "int[:]",
    p: "int[:]",
    nq: "int[:]",
    coeff_h1: "float[:,:]",
    coeff_i2: "float[:,:]",
    coeff_h3: "float[:,:]",
    coeffh_ind1: "int[:]",
    coeffi_ind2: "int[:]",
    coeffh_ind3: "int[:]",
    x_his_ind1: "int[:,:]",
    x_int_ind2: "int[:,:]",
    x_his_ind3: "int[:,:]",
    wts1: "float[:,:]",
    wts3: "float[:,:]",
    mat_f: "float[:,:,:,:,:]",
    lambdas: "float[:,:,:]",
):
    n_pts2 = 2 * (p[1] - 1) + 1

    n_his1 = 2 * p[0]
    n_his3 = 2 * p[2]

    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                for il1 in range(n_his1):
                    for il2 in range(n_pts2):
                        for il3 in range(n_his3):
                            f_int = 0.0

                            for q1 in range(nq[0]):
                                for q3 in range(nq[2]):
                                    wvol = wts1[x_his_ind1[i1, il1], q1] * wts3[x_his_ind3[i3, il3], q3]
                                    f_int += (
                                        wvol
                                        * mat_f[x_his_ind1[i1, il1], q1, x_int_ind2[i2, il2], x_his_ind3[i3, il3], q3]
                                    )

                            lambdas[i1, i2, i3] += (
                                coeff_h1[coeffh_ind1[i1], il1]
                                * coeff_i2[coeffi_ind2[i2], il2]
                                * coeff_h3[coeffh_ind3[i3], il3]
                                * f_int
                            )


# ==========================================================================================
def kernel_pi23_3d(
    n: "int[:]",
    p: "int[:]",
    nq: "int[:]",
    coeff_h1: "float[:,:]",
    coeff_h2: "float[:,:]",
    coeff_i3: "float[:,:]",
    coeffh_ind1: "int[:]",
    coeffh_ind2: "int[:]",
    coeffi_ind3: "int[:]",
    x_his_ind1: "int[:,:]",
    x_his_ind2: "int[:,:]",
    x_int_ind3: "int[:,:]",
    wts1: "float[:,:]",
    wts2: "float[:,:]",
    mat_f: "float[:,:,:,:,:]",
    lambdas: "float[:,:,:]",
):
    n_pts3 = 2 * (p[2] - 1) + 1

    n_his1 = 2 * p[0]
    n_his2 = 2 * p[1]

    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                for il1 in range(n_his1):
                    for il2 in range(n_his2):
                        for il3 in range(n_pts3):
                            f_int = 0.0

                            for q1 in range(nq[0]):
                                for q2 in range(nq[1]):
                                    wvol = wts1[x_his_ind1[i1, il1], q1] * wts2[x_his_ind2[i2, il2], q2]
                                    f_int += (
                                        wvol
                                        * mat_f[x_his_ind1[i1, il1], q1, x_his_ind2[i2, il2], q2, x_int_ind3[i3, il3]]
                                    )

                            lambdas[i1, i2, i3] += (
                                coeff_h1[coeffh_ind1[i1], il1]
                                * coeff_h2[coeffh_ind2[i2], il2]
                                * coeff_i3[coeffi_ind3[i3], il3]
                                * f_int
                            )


# ==========================================================================================
def kernel_pi3_3d(
    n: "int[:]",
    p: "int[:]",
    nq: "int[:]",
    coeff_h1: "float[:,:]",
    coeff_h2: "float[:,:]",
    coeff_h3: "float[:,:]",
    coeffh_ind1: "int[:]",
    coeffh_ind2: "int[:]",
    coeffh_ind3: "int[:]",
    x_his_ind1: "int[:,:]",
    x_his_ind2: "int[:,:]",
    x_his_ind3: "int[:,:]",
    wts1: "float[:,:]",
    wts2: "float[:,:]",
    wts3: "float[:,:]",
    mat_f: "float[:,:,:,:,:,:]",
    lambdas: "float[:,:,:]",
):
    n_his1 = 2 * p[0]
    n_his2 = 2 * p[1]
    n_his3 = 2 * p[2]

    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                for il1 in range(n_his1):
                    for il2 in range(n_his2):
                        for il3 in range(n_his3):
                            f_int = 0.0

                            for q1 in range(nq[0]):
                                for q2 in range(nq[1]):
                                    for q3 in range(nq[2]):
                                        wvol = (
                                            wts1[x_his_ind1[i1, il1], q1]
                                            * wts2[x_his_ind2[i2, il2], q2]
                                            * wts3[x_his_ind3[i3, il3], q3]
                                        )
                                        f_int += (
                                            wvol
                                            * mat_f[
                                                x_his_ind1[i1, il1],
                                                q1,
                                                x_his_ind2[i2, il2],
                                                q2,
                                                x_his_ind3[i3, il3],
                                                q3,
                                            ]
                                        )

                            lambdas[i1, i2, i3] += (
                                coeff_h1[coeffh_ind1[i1], il1]
                                * coeff_h2[coeffh_ind2[i2], il2]
                                * coeff_h3[coeffh_ind3[i3], il3]
                                * f_int
                            )
