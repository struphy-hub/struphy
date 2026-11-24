"""Kernel kernel_3d_matrixfree: integral kernel for mass matrices and L2-projections."""

import numpy as np
from numpy import shape


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
    Performs the integration of Lambda_(i1, i2, i3) * mat_fun(eta1, eta2, eta3) * f(eta1, eta2, eta3) for the basis functions (i1, i2, i3) available on the calling process,
    where f is the spline function represented by the coefficients in data_in.

    The results are written into data_out (attention: data_out is NOT set to zero first, but the results are added to data_out).
    This computes the action of the mass matrix on a vector without ever assembling the matrix itself.

    Parameters
    ----------
    spansi1, spansi2, spansi3 : array[int]
        Arrays of span indices in direction 1, 2 and 3 for the codomain ("i") basis functions; the span is the
        index of the last non-vanishing spline on each grid element (cell).
    spansj1, spansj2, spansj3 : array[int]
        Arrays of span indices in direction 1, 2 and 3 for the domain ("j") basis functions.
    pi1, pi2, pi3 : int
        Degree of the codomain basis functions in direction 1, 2 and 3.
    pj1, pj2, pj3 : int
        Degree of the domain basis functions in direction 1, 2 and 3.
    startsi1, startsi2, startsi3 : int
        Starting index on the current rank for the codomain basis functions, in direction 1, 2 and 3.
    startsj1, startsj2, startsj3 : int
        Starting index on the current rank for the domain basis functions, in direction 1, 2 and 3.
    padsi1, padsi2, padsi3 : int
        Padding (=spline degree) for ghost regions in data_out, in direction 1, 2 and 3.
    padsj1, padsj2, padsj3 : int
        Padding (=spline degree) for ghost regions in data_in, in direction 1, 2 and 3.
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
    data_out : "float[:,:,:]"
        _data array of StencilVector to store the results of the matrix-vector product.
    data_in : "float[:,:,:]"
        _data array of StencilVector holding the spline coefficients of the input function f.
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
                    iel1 * nq1 : (iel1 + 1) * nq1,
                    iel2 * nq2 : (iel2 + 1) * nq2,
                    iel3 * nq3 : (iel3 + 1) * nq3,
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
