"""
Integral kernels for mass matrices and L2-projections.
"""

import numpy as np
from numpy import shape

# ================= 1d =================================


def kernel_1d_mat(
    spans1: "int[:]",
    pi1: int,
    pj1: int,
    starts1: int,
    pads1: int,
    w1: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bj1: "float[:,:,:,:]",
    mat_fun: "float[:]",
    data: "float[:,:]",
):
    """
    Performs the integration of Lambda_i * mat_fun(eta1) * Lambda_l for the basis functions (i, l) available on the calling process.

    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).
    """

    ne1 = spans1.size

    nq1 = shape(w1)[1]

    for iel1 in range(ne1):
        for il1 in range(pi1 + 1):
            # global spline indices
            i_global1 = spans1[iel1] - pi1 + il1

            # local spline indices (- starts --> can be negative, will therefore be written to ghost regions)
            i_local1 = i_global1 - starts1

            for jl1 in range(pj1 + 1):
                value = 0.0

                for q1 in range(nq1):
                    value += w1[iel1, q1] * bi1[iel1, il1, 0, q1] * bj1[iel1, jl1, 0, q1] * mat_fun[iel1 * nq1 + q1]

                data[pads1 + i_local1, pads1 + jl1 - il1] += value


def kernel_1d_vec(
    spans1: "int[:]",
    pi1: int,
    starts1: int,
    pads1: int,
    w1: "float[:,:]",
    bi1: "float[:,:,:,:]",
    mat_fun: "float[:]",
    data: "float[:]",
):
    """
    Performs the integration of Lambda_i * mat_fun(eta1) for the basis functions (i) available on the calling process.

    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).
    """

    ne1 = spans1.size

    nq1 = shape(w1)[1]

    for iel1 in range(ne1):
        for il1 in range(pi1 + 1):
            # global spline indices
            i_global1 = spans1[iel1] - pi1 + il1

            # local spline indices (- starts --> can be negative, will therefore be written to ghost regions)
            i_local1 = i_global1 - starts1

            value = 0.0

            for q1 in range(nq1):
                value += w1[iel1, q1] * bi1[iel1, il1, 0, q1] * mat_fun[iel1 * nq1 + q1]

            data[pads1 + i_local1] += value


def kernel_1d_eval(
    spans1: "int[:]",
    pi1: int,
    starts1: int,
    pads1: int,
    bi1: "float[:,:,:,:]",
    coeffs_data: "float[:]",
    values: "float[:]",
):
    """
    Evaluates sum_i [ coeffs_i * Lambda_i(quad_eta1) ] for all quadrature points on the calling process.

    The results are written into values.
    """

    values[:] = 0.0

    ne1 = spans1.size

    nq1 = shape(bi1)[3]

    for iel1 in range(ne1):
        for il1 in range(pi1 + 1):
            # global spline indices
            i_global1 = spans1[iel1] - pi1 + il1

            # local spline indices (- starts --> can be negative, will therefore be written to ghost regions)
            i_local1 = i_global1 - starts1

            for q1 in range(nq1):
                values[iel1 * nq1 + q1] += coeffs_data[pads1 + i_local1] * bi1[iel1, il1, 0, q1]


# ================= 2d =================================


def kernel_2d_mat(
    spans1: "int[:]",
    spans2: "int[:]",
    pi1: int,
    pi2: int,
    pj1: int,
    pj2: int,
    starts1: int,
    starts2: int,
    pads1: int,
    pads2: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    bj1: "float[:,:,:,:]",
    bj2: "float[:,:,:,:]",
    mat_fun: "float[:,:]",
    data: "float[:,:,:,:]",
):
    """
    Performs the integration of Lambda_ij * mat_fun(eta1, eta2) * Lambda_lm for the basis functions (ij, lm) available on the calling process.

    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).
    """

    ne1 = spans1.size
    ne2 = spans2.size

    nq1 = shape(w1)[1]
    nq2 = shape(w2)[1]

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for il1 in range(pi1 + 1):
                for il2 in range(pi2 + 1):
                    # global spline indices
                    i_global1 = spans1[iel1] - pi1 + il1
                    i_global2 = spans2[iel2] - pi2 + il2

                    # local spline indices (- starts --> can be negative, will therefore be written to ghost regions)
                    i_local1 = i_global1 - starts1
                    i_local2 = i_global2 - starts2

                    for jl1 in range(pj1 + 1):
                        for jl2 in range(pj2 + 1):
                            value = 0.0

                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    wvol = w1[iel1, q1] * w2[iel2, q2] * mat_fun[iel1 * nq1 + q1, iel2 * nq2 + q2]
                                    bi = bi1[iel1, il1, 0, q1] * bi2[iel2, il2, 0, q2]
                                    bj = bj1[iel1, jl1, 0, q1] * bj2[iel2, jl2, 0, q2]

                                    value += wvol * bi * bj

                            data[pads1 + i_local1, pads2 + i_local2, pads1 + jl1 - il1, pads2 + jl2 - il2] += value


def kernel_2d_vec(
    spans1: "int[:]",
    spans2: "int[:]",
    pi1: int,
    pi2: int,
    starts1: int,
    starts2: int,
    pads1: int,
    pads2: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    mat_fun: "float[:,:]",
    data: "float[:,:]",
):
    """
    Performs the integration of Lambda_ij * mat_fun(eta1, eta2) for the basis functions (ij) available on the calling process.

    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).
    """

    ne1 = spans1.size
    ne2 = spans2.size

    nq1 = shape(w1)[1]
    nq2 = shape(w2)[1]

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for il1 in range(pi1 + 1):
                for il2 in range(pi2 + 1):
                    # global spline indices
                    i_global1 = spans1[iel1] - pi1 + il1
                    i_global2 = spans2[iel2] - pi2 + il2

                    # local spline indices (- starts --> can be negative, will therefore be written to ghost regions)
                    i_local1 = i_global1 - starts1
                    i_local2 = i_global2 - starts2

                    value = 0.0

                    for q1 in range(nq1):
                        for q2 in range(nq2):
                            wvol = w1[iel1, q1] * w2[iel2, q2] * mat_fun[iel1 * nq1 + q1, iel2 * nq2 + q2]

                            value += wvol * bi1[iel1, il1, 0, q1] * bi2[iel2, il2, 0, q2]

                    data[pads1 + i_local1, pads2 + i_local2] += value


def kernel_2d_eval(
    spans1: "int[:]",
    spans2: "int[:]",
    pi1: int,
    pi2: int,
    starts1: int,
    starts2: int,
    pads1: int,
    pads2: int,
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    coeffs_data: "float[:,:]",
    values: "float[:,:]",
):
    """
    Evaluates sum_ij [ coeffs_ij * Lambda_ij(quad_eta1, quad_eta2) ] for all quadrature points on the calling process.

    The results are written into values.
    """

    values[:, :] = 0.0

    ne1 = spans1.size
    ne2 = spans2.size

    nq1 = shape(bi1)[3]
    nq2 = shape(bi2)[3]

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for il1 in range(pi1 + 1):
                for il2 in range(pi2 + 1):
                    # global spline indices
                    i_global1 = spans1[iel1] - pi1 + il1
                    i_global2 = spans2[iel2] - pi2 + il2

                    # local spline indices (- starts --> can be negative, will therefore be written to ghost regions)
                    i_local1 = i_global1 - starts1
                    i_local2 = i_global2 - starts2

                    for q1 in range(nq1):
                        for q2 in range(nq2):
                            values[iel1 * nq1 + q1, iel2 * nq2 + q2] += (
                                coeffs_data[pads1 + i_local1, pads2 + i_local2]
                                * bi1[iel1, il1, 0, q1]
                                * bi2[iel2, il2, 0, q2]
                            )


# ================= 3d =================================


def kernel_3d_mat(
    spans1: "int[:]",
    spans2: "int[:]",
    spans3: "int[:]",
    pi1: int,
    pi2: int,
    pi3: int,
    pj1: int,
    pj2: int,
    pj3: int,
    starts1: int,
    starts2: int,
    starts3: int,
    pads1: int,
    pads2: int,
    pads3: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    w3: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    bi3: "float[:,:,:,:]",
    bj1: "float[:,:,:,:]",
    bj2: "float[:,:,:,:]",
    bj3: "float[:,:,:,:]",
    mat_fun: "float[:,:,:]",
    data: "float[:,:,:,:,:,:]",
):
    """
    Performs the integration of Lambda_ijk * mat_fun(eta1, eta2, eta3) * Lambda_lmn for the basis functions (ijk, lmn) available on the calling process.

    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).
    """

    ne1 = spans1.size
    ne2 = spans2.size
    ne3 = spans3.size

    nq1 = shape(w1)[1]
    nq2 = shape(w2)[1]
    nq3 = shape(w3)[1]

    tmp_bi1 = np.zeros(nq1)
    tmp_bi2 = np.zeros(nq2)
    tmp_bi3 = np.zeros(nq3)

    tmp_bj1 = np.zeros(nq1)
    tmp_bj2 = np.zeros(nq2)
    tmp_bj3 = np.zeros(nq3)

    tmp_w1 = np.zeros(nq1)
    tmp_w2 = np.zeros(nq2)
    tmp_w3 = np.zeros(nq3)

    tmp_mat_fun = np.zeros((nq1, nq2, nq3))

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for iel3 in range(ne3):
                tmp_mat_fun[:, :, :] = mat_fun[
                    iel1 * nq1 : (iel1 + 1) * nq1, iel2 * nq2 : (iel2 + 1) * nq2, iel3 * nq3 : (iel3 + 1) * nq3
                ]

                tmp_w1[:] = w1[iel1, :]
                tmp_w2[:] = w2[iel2, :]
                tmp_w3[:] = w3[iel3, :]

                for il1 in range(pi1 + 1):
                    for il2 in range(pi2 + 1):
                        for il3 in range(pi3 + 1):
                            tmp_bi1[:] = bi1[iel1, il1, 0, :]
                            tmp_bi2[:] = bi2[iel2, il2, 0, :]
                            tmp_bi3[:] = bi3[iel3, il3, 0, :]

                            # global spline indices
                            i_global1 = spans1[iel1] - pi1 + il1
                            i_global2 = spans2[iel2] - pi2 + il2
                            i_global3 = spans3[iel3] - pi3 + il3

                            # local spline indices (- starts --> can be negative, will therefore be written to ghost regions)
                            i_local1 = i_global1 - starts1
                            i_local2 = i_global2 - starts2
                            i_local3 = i_global3 - starts3

                            for jl1 in range(pj1 + 1):
                                for jl2 in range(pj2 + 1):
                                    for jl3 in range(pj3 + 1):
                                        tmp_bj1[:] = bj1[iel1, jl1, 0, :]
                                        tmp_bj2[:] = bj2[iel2, jl2, 0, :]
                                        tmp_bj3[:] = bj3[iel3, jl3, 0, :]

                                        value = 0.0

                                        for q1 in range(nq1):
                                            for q2 in range(nq2):
                                                for q3 in range(nq3):
                                                    wvol = (
                                                        tmp_w1[q1] * tmp_w2[q2] * tmp_w3[q3] * tmp_mat_fun[q1, q2, q3]
                                                    )

                                                    bi = tmp_bi1[q1] * tmp_bi2[q2] * tmp_bi3[q3]
                                                    bj = tmp_bj1[q1] * tmp_bj2[q2] * tmp_bj3[q3]

                                                    value += wvol * bi * bj

                                        data[
                                            pads1 + i_local1,
                                            pads2 + i_local2,
                                            pads3 + i_local3,
                                            pads1 + jl1 - il1,
                                            pads2 + jl2 - il2,
                                            pads3 + jl3 - il3,
                                        ] += value


def kernel_3d_vec(
    spans1: "int[:]",
    spans2: "int[:]",
    spans3: "int[:]",
    pi1: int,
    pi2: int,
    pi3: int,
    starts1: int,
    starts2: int,
    starts3: int,
    pads1: int,
    pads2: int,
    pads3: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    w3: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    bi3: "float[:,:,:,:]",
    mat_fun: "float[:,:,:]",
    data: "float[:,:,:]",
):
    """
    Performs the integration of Lambda_ijk * mat_fun(eta1, eta2, eta3) for the basis functions (ijk) available on the calling process.

    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).
    """

    ne1 = spans1.size
    ne2 = spans2.size
    ne3 = spans3.size

    nq1 = shape(w1)[1]
    nq2 = shape(w2)[1]
    nq3 = shape(w3)[1]

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for iel3 in range(ne3):
                for il1 in range(pi1 + 1):
                    for il2 in range(pi2 + 1):
                        for il3 in range(pi3 + 1):
                            # global spline indices
                            i_global1 = spans1[iel1] - pi1 + il1
                            i_global2 = spans2[iel2] - pi2 + il2
                            i_global3 = spans3[iel3] - pi3 + il3

                            # local spline indices (- starts --> can be negative, will therefore be written to ghost regions)
                            i_local1 = i_global1 - starts1
                            i_local2 = i_global2 - starts2
                            i_local3 = i_global3 - starts3

                            value = 0.0

                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):
                                        wvol = (
                                            w1[iel1, q1]
                                            * w2[iel2, q2]
                                            * w3[iel3, q3]
                                            * mat_fun[iel1 * nq1 + q1, iel2 * nq2 + q2, iel3 * nq3 + q3]
                                        )

                                        value += (
                                            wvol * bi1[iel1, il1, 0, q1] * bi2[iel2, il2, 0, q2] * bi3[iel3, il3, 0, q3]
                                        )

                            data[pads1 + i_local1, pads2 + i_local2, pads3 + i_local3] += value


def kernel_3d_eval(
    spans1: "int[:]",
    spans2: "int[:]",
    spans3: "int[:]",
    pi1: int,
    pi2: int,
    pi3: int,
    starts1: int,
    starts2: int,
    starts3: int,
    pads1: int,
    pads2: int,
    pads3: int,
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    bi3: "float[:,:,:,:]",
    coeffs_data: "float[:,:,:]",
    values: "float[:,:,:]",
):
    """
    Evaluates sum_ijk [ coeffs_ijk * Lambda_ijk(quad_eta1, quad_eta2, quad_eta3) ] for all quadrature points on the calling process.

    The results are written into values.
    """

    values[:, :, :] = 0.0

    ne1 = spans1.size
    ne2 = spans2.size
    ne3 = spans3.size

    nq1 = shape(bi1)[3]
    nq2 = shape(bi2)[3]
    nq3 = shape(bi3)[3]

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for iel3 in range(ne3):
                for il1 in range(pi1 + 1):
                    for il2 in range(pi2 + 1):
                        for il3 in range(pi3 + 1):
                            # global spline indices
                            i_global1 = spans1[iel1] - pi1 + il1
                            i_global2 = spans2[iel2] - pi2 + il2
                            i_global3 = spans3[iel3] - pi3 + il3

                            # local spline indices (- starts --> can be negative, will therefore be written to ghost regions)
                            i_local1 = i_global1 - starts1
                            i_local2 = i_global2 - starts2
                            i_local3 = i_global3 - starts3

                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):
                                        values[iel1 * nq1 + q1, iel2 * nq2 + q2, iel3 * nq3 + q3] += (
                                            coeffs_data[pads1 + i_local1, pads2 + i_local2, pads3 + i_local3]
                                            * bi1[iel1, il1, 0, q1]
                                            * bi2[iel2, il2, 0, q2]
                                            * bi3[iel3, il3, 0, q3]
                                        )


def kernel_3d_matrixfree(
    spansi1: "int[:]",
    spansi2: "int[:]",
    spansi3: "int[:]",
    spansj1: "int[:]",
    spansj2: "int[:]",
    spansj3: "int[:]",
    pi1: int,
    pi2: int,
    pi3: int,
    pj1: int,
    pj2: int,
    pj3: int,
    startsi1: int,
    startsi2: int,
    startsi3: int,
    startsj1: int,
    startsj2: int,
    startsj3: int,
    padsi1: int,
    padsi2: int,
    padsi3: int,
    padsj1: int,
    padsj2: int,
    padsj3: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    w3: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    bi3: "float[:,:,:,:]",
    bj1: "float[:,:,:,:]",
    bj2: "float[:,:,:,:]",
    bj3: "float[:,:,:,:]",
    mat_fun: "float[:,:,:]",
    data_out: "float[:,:,:]",
    data_in: "float[:,:,:]",
):
    """
    Performs the integration of Lambda_ijk * mat_fun(eta1, eta2, eta3) * f(eta1, eta2, eta3) for the basis functions (ijk) available on the calling process,
    where f is the spline function represented by the coefficients in data_in.

    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).
    """

    ne1 = spansi1.size
    ne2 = spansi2.size
    ne3 = spansi3.size

    nq1 = shape(w1)[1]
    nq2 = shape(w2)[1]
    nq3 = shape(w3)[1]

    tmp_w1 = np.zeros(nq1)
    tmp_w2 = np.zeros(nq2)
    tmp_w3 = np.zeros(nq3)

    tmp_bi1 = np.zeros(pi1 + 1)
    tmp_bi2 = np.zeros(pi2 + 1)
    tmp_bi3 = np.zeros(pi3 + 1)

    tmp_bj1 = np.zeros(pj1 + 1)
    tmp_bj2 = np.zeros(pj2 + 1)
    tmp_bj3 = np.zeros(pj3 + 1)

    tmp_mat_fun = np.zeros((nq1, nq2, nq3))

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for iel3 in range(ne3):
                tmp_mat_fun[:, :, :] = mat_fun[
                    iel1 * nq1 : (iel1 + 1) * nq1, iel2 * nq2 : (iel2 + 1) * nq2, iel3 * nq3 : (iel3 + 1) * nq3
                ]

                tmp_w1[:] = w1[iel1, :]
                tmp_w2[:] = w2[iel2, :]
                tmp_w3[:] = w3[iel3, :]

                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            tmp_bi1[:] = bi1[iel1, :, 0, q1]
                            tmp_bi2[:] = bi2[iel2, :, 0, q2]
                            tmp_bi3[:] = bi3[iel3, :, 0, q3]

                            tmp_bj1[:] = bj1[iel1, :, 0, q1]
                            tmp_bj2[:] = bj2[iel2, :, 0, q2]
                            tmp_bj3[:] = bj3[iel3, :, 0, q3]

                            bj = 0.0
                            for jl1 in range(pj1 + 1):
                                for jl2 in range(pj2 + 1):
                                    for jl3 in range(pj3 + 1):
                                        # global spline indices
                                        j_global1 = spansj1[iel1] - pj1 + jl1
                                        j_global2 = spansj2[iel2] - pj2 + jl2
                                        j_global3 = spansj3[iel3] - pj3 + jl3

                                        # local spline indices (- starts --> can be negative, will therefore be written to ghost regions)
                                        j_local1 = j_global1 - startsj1 + padsj1
                                        j_local2 = j_global2 - startsj2 + padsj2
                                        j_local3 = j_global3 - startsj3 + padsj3

                                        bj += (
                                            tmp_bj1[jl1]
                                            * tmp_bj2[jl2]
                                            * tmp_bj3[jl3]
                                            * data_in[j_local1, j_local2, j_local3]
                                        )

                            for il1 in range(pi1 + 1):
                                for il2 in range(pi2 + 1):
                                    for il3 in range(pi3 + 1):
                                        # global spline indices
                                        i_global1 = spansi1[iel1] - pi1 + il1
                                        i_global2 = spansi2[iel2] - pi2 + il2
                                        i_global3 = spansi3[iel3] - pi3 + il3

                                        # local spline indices (- starts --> can be negative, will therefore be written to ghost regions)
                                        i_local1 = i_global1 - startsi1 + padsi1
                                        i_local2 = i_global2 - startsi2 + padsi2
                                        i_local3 = i_global3 - startsi3 + padsi3

                                        wvol = tmp_w1[q1] * tmp_w2[q2] * tmp_w3[q3] * tmp_mat_fun[q1, q2, q3]

                                        bi = tmp_bi1[il1] * tmp_bi2[il2] * tmp_bi3[il3]

                                        value = wvol * bi * bj

                                        data_out[i_local1, i_local2, i_local3] += value


def kernel_3d_diag(
    spans1: "int[:]",
    spans2: "int[:]",
    spans3: "int[:]",
    pi1: int,
    pi2: int,
    pi3: int,
    starts1: int,
    starts2: int,
    starts3: int,
    pads1: int,
    pads2: int,
    pads3: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    w3: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    bi3: "float[:,:,:,:]",
    mat_fun: "float[:,:,:]",
    data: "float[:,:,:]",
):
    """
    Computes the diagonal of a mass matrix, assuming that the domain and the codomain are the same.

    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).
    """

    ne1 = spans1.size
    ne2 = spans2.size
    ne3 = spans3.size

    nq1 = shape(w1)[1]
    nq2 = shape(w2)[1]
    nq3 = shape(w3)[1]

    nb1, nb2, nb3 = data.shape

    tmp_bi1 = np.zeros(nq1)
    tmp_bi2 = np.zeros(nq2)
    tmp_bi3 = np.zeros(nq3)

    tmp_w1 = np.zeros(nq1)
    tmp_w2 = np.zeros(nq2)
    tmp_w3 = np.zeros(nq3)

    tmp_mat_fun = np.zeros((nq1, nq2, nq3))

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for iel3 in range(ne3):
                tmp_mat_fun[:, :, :] = mat_fun[
                    iel1 * nq1 : (iel1 + 1) * nq1, iel2 * nq2 : (iel2 + 1) * nq2, iel3 * nq3 : (iel3 + 1) * nq3
                ]

                tmp_w1[:] = w1[iel1, :]
                tmp_w2[:] = w2[iel2, :]
                tmp_w3[:] = w3[iel3, :]

                for il1 in range(pi1 + 1):
                    for il2 in range(pi2 + 1):
                        for il3 in range(pi3 + 1):
                            tmp_bi1[:] = bi1[iel1, il1, 0, :]
                            tmp_bi2[:] = bi2[iel2, il2, 0, :]
                            tmp_bi3[:] = bi3[iel3, il3, 0, :]

                            # global spline indices
                            i_global1 = spans1[iel1] - pi1 + il1
                            i_global2 = spans2[iel2] - pi2 + il2
                            i_global3 = spans3[iel3] - pi3 + il3

                            # local spline indices (- starts --> can be negative, will therefore be written to ghost regions)
                            i_local1 = i_global1 - starts1
                            i_local2 = i_global2 - starts2
                            i_local3 = i_global3 - starts3

                            # Periodic case : last basis function are the first ones (no ghost regions on DiagonalStencilMatrix)
                            if i_local1 >= nb1:
                                i_local1 -= nb1

                            if i_local2 >= nb2:
                                i_local2 -= nb2

                            if i_local3 >= nb3:
                                i_local3 -= nb3

                            value = 0.0

                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):
                                        wvol = tmp_w1[q1] * tmp_w2[q2] * tmp_w3[q3] * tmp_mat_fun[q1, q2, q3]

                                        bi = tmp_bi1[q1] * tmp_bi2[q2] * tmp_bi3[q3]

                                        value += wvol * bi * bi

                            # No padding on StencilDiagonalMatrix
                            data[i_local1, i_local2, i_local3] += value
