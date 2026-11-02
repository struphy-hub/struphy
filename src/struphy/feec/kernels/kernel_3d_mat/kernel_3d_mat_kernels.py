"""Kernel kernel_3d_mat: integral kernel for mass matrices and L2-projections."""

import numpy as np
from numpy import shape


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
    Performs the integration of Lambda_(i1,i2,i3) * mat_fun(eta1, eta2, eta3) * Lambda_(j1,j2,j3) for the basis functions (i1,i2,i3, j1,j2,j3) available on the calling process.

    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).

    Parameters
    ----------
    spans1, spans2, spans3 : array[int]
        Arrays of span indices in direction 1, 2 and 3; the span is the index of the last non-vanishing spline
        on each grid element (cell). The length of each array is the number of elements (cells) in that direction.
    pi1, pi2, pi3 : int
        Degree of the codomain basis functions in direction 1, 2 and 3.
    pj1, pj2, pj3 : int
        Degree of the domain basis functions in direction 1, 2 and 3.
    starts1, starts2, starts3 : int
        Starting index on the current rank, in direction 1, 2 and 3.
    pads1, pads2, pads3 : int
        Padding (=spline degree) for ghost regions in data, in direction 1, 2 and 3.
    w1, w2, w3 : "float[:,:]"
        Quadrature weights in direction 1, 2 and 3. The indexing is [global element, quadrature point].
    bi1, bi2, bi3 : "float[:,:,:,:]"
        Values of codomain basis functions in direction 1, 2 and 3. The indexing is
        [global element, local basis function, derivative, quadrature point].
    bj1, bj2, bj3 : "float[:,:,:,:]"
        Values of domain basis functions in direction 1, 2 and 3, same indexing convention as bi1, bi2, bi3.
    mat_fun : "float[:,:,:]"
        Function under the integral evaluated at quadrature points (flattened in each direction).
        The indexing is [flattened quad. point dir. 1, flattened quad. point dir. 2, flattened quad. point dir. 3].
    data : "float[:,:,:,:,:,:]"
        _data array of StencilMatrix to store the results.
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
                    iel1 * nq1 : (iel1 + 1) * nq1,
                    iel2 * nq2 : (iel2 + 1) * nq2,
                    iel3 * nq3 : (iel3 + 1) * nq3,
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
