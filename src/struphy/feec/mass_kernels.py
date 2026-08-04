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
    Performs the integration of Lambda_(i1) * mat_fun(eta1) * Lambda_(j1) for the basis functions (i1, j1) available on the calling process.

    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).
    
    Parameters
    ----------
    spans1 : array[int]
        Array of span indices; the span is the index of the last non-vanishing spline on each grid element
        (cell). The length of the returned array is the number of elements (cells).
    pi1 : int
        Degree of the codomain basis functions.
    pj1 : int
        Degree of the domain basis functions.
    starts1 : int
        Starting index on the current rank.
    pads1 : int
        Padding (=spline degree) for ghost regions in data.
    w1 : "float[:,:]"
        Quadrature weights. The indexing is [global element, quadrature point].
    bi1 : "float[:,:,:,:]"
        Values of codomain basis functions. The indexing is [global element, local basis function, derivative, quadrature point].
    bj1 : "float[:,:,:,:]"
        Values of domain basis functions. The indexing is [global element, local basis function, derivative, quadrature point].
    mat_fun : "float[:]"
        Function under the integral evaluated at quadrature points (flattened).
    data : "float[:,:]"
        _data array of StencilMatrix to store the results.
    """

    # number of elements
    ne1 = spans1.size

    # number of quadrature points in each element
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
    Performs the integration of Lambda_(i1) * mat_fun(eta1) for the basis functions (i1) available on the calling process.

    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).

    Parameters
    ----------
    spans1 : array[int]
        Array of span indices; the span is the index of the last non-vanishing spline on each grid element
        (cell). The length of the returned array is the number of elements (cells).
    pi1 : int
        Degree of the basis functions.
    starts1 : int
        Starting index on the current rank.
    pads1 : int
        Padding (=spline degree) for ghost regions in data.
    w1 : "float[:,:]"
        Quadrature weights. The indexing is [global element, quadrature point].
    bi1 : "float[:,:,:,:]"
        Values of basis functions. The indexing is [global element, local basis function, derivative, quadrature point].
    mat_fun : "float[:]"
        Function under the integral evaluated at quadrature points (flattened).
    data : "float[:]"
        _data array of StencilVector to store the results.
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
    Evaluates sum_i1 [ coeffs_i1 * Lambda_i1(quad_eta1) ] for all quadrature points on the calling process.

    The results are written into values.

    Parameters
    ----------
    spans1 : array[int]
        Array of span indices; the span is the index of the last non-vanishing spline on each grid element
        (cell). The length of the returned array is the number of elements (cells).
    pi1 : int
        Degree of the basis functions.
    starts1 : int
        Starting index on the current rank.
    pads1 : int
        Padding (=spline degree) for ghost regions in coeffs_data.
    bi1 : "float[:,:,:,:]"
        Values of basis functions. The indexing is [global element, local basis function, derivative, quadrature point].
    coeffs_data : "float[:]"
        _data array of StencilVector holding the spline coefficients of the function to be evaluated.
    values : "float[:]"
        Output array (flattened over elements and quadrature points) holding the evaluated function values;
        it is set to zero at the start of the kernel, i.e. it is overwritten, not added to.
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
    Performs the integration of Lambda_(i1, i2) * mat_fun(eta1, eta2) * Lambda_(j1, j2) for the basis functions (i1, i2, j1, j2) available on the calling process.

    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).

    Parameters
    ----------
    spans1, spans2 : array[int]
        Arrays of span indices in direction 1 and 2; the span is the index of the last non-vanishing spline
        on each grid element (cell). The length of each array is the number of elements (cells) in that direction.
    pi1, pi2 : int
        Degree of the codomain basis functions in direction 1 and 2.
    pj1, pj2 : int
        Degree of the domain basis functions in direction 1 and 2.
    starts1, starts2 : int
        Starting index on the current rank, in direction 1 and 2.
    pads1, pads2 : int
        Padding (=spline degree) for ghost regions in data, in direction 1 and 2.
    w1, w2 : "float[:,:]"
        Quadrature weights in direction 1 and 2. The indexing is [global element, quadrature point].
    bi1, bi2 : "float[:,:,:,:]"
        Values of codomain basis functions in direction 1 and 2. The indexing is
        [global element, local basis function, derivative, quadrature point].
    bj1, bj2 : "float[:,:,:,:]"
        Values of domain basis functions in direction 1 and 2, same indexing convention as bi1, bi2.
    mat_fun : "float[:,:]"
        Function under the integral evaluated at quadrature points (flattened in each direction).
        The indexing is [flattened quadrature point in direction 1, flattened quadrature point in direction 2].
    data : "float[:,:,:,:]"
        _data array of StencilMatrix to store the results.
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
    Performs the integration of Lambda_(i1, i2) * mat_fun(eta1, eta2) for the basis functions (i1, i2) available on the calling process.

    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).

    Parameters
    ----------
    spans1, spans2 : array[int]
        Arrays of span indices in direction 1 and 2; the span is the index of the last non-vanishing spline
        on each grid element (cell). The length of each array is the number of elements (cells) in that direction.
    pi1, pi2 : int
        Degree of the basis functions in direction 1 and 2.
    starts1, starts2 : int
        Starting index on the current rank, in direction 1 and 2.
    pads1, pads2 : int
        Padding (=spline degree) for ghost regions in data, in direction 1 and 2.
    w1, w2 : "float[:,:]"
        Quadrature weights in direction 1 and 2. The indexing is [global element, quadrature point].
    bi1, bi2 : "float[:,:,:,:]"
        Values of basis functions in direction 1 and 2. The indexing is
        [global element, local basis function, derivative, quadrature point].
    mat_fun : "float[:,:]"
        Function under the integral evaluated at quadrature points (flattened in each direction).
        The indexing is [flattened quadrature point in direction 1, flattened quadrature point in direction 2].
    data : "float[:,:]"
        _data array of StencilVector to store the results.
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
    Evaluates sum_(i1, i2) [ coeffs_{i1,i2} * Lambda_{i1, i2}(quad_eta1, quad_eta2) ] for all quadrature points on the calling process.

    The results are written into values.

    Parameters
    ----------
    spans1, spans2 : array[int]
        Arrays of span indices in direction 1 and 2; the span is the index of the last non-vanishing spline
        on each grid element (cell). The length of each array is the number of elements (cells) in that direction.
    pi1, pi2 : int
        Degree of the basis functions in direction 1 and 2.
    starts1, starts2 : int
        Starting index on the current rank, in direction 1 and 2.
    pads1, pads2 : int
        Padding (=spline degree) for ghost regions in coeffs_data, in direction 1 and 2.
    bi1, bi2 : "float[:,:,:,:]"
        Values of basis functions in direction 1 and 2. The indexing is
        [global element, local basis function, derivative, quadrature point].
    coeffs_data : "float[:,:]"
        _data array of StencilVector holding the spline coefficients of the function to be evaluated.
    values : "float[:,:]"
        Output array (flattened over elements and quadrature points in each direction) holding the evaluated
        function values; it is set to zero at the start of the kernel, i.e. it is overwritten, not added to.
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
    Performs the integration of Lambda_(i1,i2,i3) * mat_fun(eta1, eta2, eta3) for the basis functions (i1,i2,i3) available on the calling process.

    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).

    Parameters
    ----------
    spans1, spans2, spans3 : array[int]
        Arrays of span indices in direction 1, 2 and 3; the span is the index of the last non-vanishing spline
        on each grid element (cell). The length of each array is the number of elements (cells) in that direction.
    pi1, pi2, pi3 : int
        Degree of the basis functions in direction 1, 2 and 3.
    starts1, starts2, starts3 : int
        Starting index on the current rank, in direction 1, 2 and 3.
    pads1, pads2, pads3 : int
        Padding (=spline degree) for ghost regions in data, in direction 1, 2 and 3.
    w1, w2, w3 : "float[:,:]"
        Quadrature weights in direction 1, 2 and 3. The indexing is [global element, quadrature point].
    bi1, bi2, bi3 : "float[:,:,:,:]"
        Values of basis functions in direction 1, 2 and 3. The indexing is
        [global element, local basis function, derivative, quadrature point].
    mat_fun : "float[:,:,:]"
        Function under the integral evaluated at quadrature points (flattened in each direction).
        The indexing is [flattened quad. point dir. 1, flattened quad. point dir. 2, flattened quad. point dir. 3].
    data : "float[:,:,:]"
        _data array of StencilVector to store the results.
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
    Evaluates sum_(i1,i2,i3) [ coeffs_{i1,i2,i3} * Lambda_{i1,i2,i3}(quad_eta1, quad_eta2, quad_eta3) ] for all quadrature points on the calling process.

    The results are written into values.

    Parameters
    ----------
    spans1, spans2, spans3 : array[int]
        Arrays of span indices in direction 1, 2 and 3; the span is the index of the last non-vanishing spline
        on each grid element (cell). The length of each array is the number of elements (cells) in that direction.
    pi1, pi2, pi3 : int
        Degree of the basis functions in direction 1, 2 and 3.
    starts1, starts2, starts3 : int
        Starting index on the current rank, in direction 1, 2 and 3.
    pads1, pads2, pads3 : int
        Padding (=spline degree) for ghost regions in coeffs_data, in direction 1, 2 and 3.
    bi1, bi2, bi3 : "float[:,:,:,:]"
        Values of basis functions in direction 1, 2 and 3. The indexing is
        [global element, local basis function, derivative, quadrature point].
    coeffs_data : "float[:,:,:]"
        _data array of StencilVector holding the spline coefficients of the function to be evaluated.
    values : "float[:,:,:]"
        Output array (flattened over elements and quadrature points in each direction) holding the evaluated
        function values; it is set to zero at the start of the kernel, i.e. it is overwritten, not added to.
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

    Parameters
    ----------
    spans1, spans2, spans3 : array[int]
        Arrays of span indices in direction 1, 2 and 3; the span is the index of the last non-vanishing spline
        on each grid element (cell). The length of each array is the number of elements (cells) in that direction.
    pi1, pi2, pi3 : int
        Degree of the basis functions in direction 1, 2 and 3.
    starts1, starts2, starts3 : int
        Starting index on the current rank, in direction 1, 2 and 3.
    pads1, pads2, pads3 : int
        Padding (=spline degree) for ghost regions, in direction 1, 2 and 3 (unused for data, which is a
        StencilDiagonalMatrix and therefore has no padding, but kept for a uniform kernel signature).
    w1, w2, w3 : "float[:,:]"
        Quadrature weights in direction 1, 2 and 3. The indexing is [global element, quadrature point].
    bi1, bi2, bi3 : "float[:,:,:,:]"
        Values of basis functions in direction 1, 2 and 3. The indexing is
        [global element, local basis function, derivative, quadrature point].
    mat_fun : "float[:,:,:]"
        Function under the integral evaluated at quadrature points (flattened in each direction).
        The indexing is [flattened quad. point dir. 1, flattened quad. point dir. 2, flattened quad. point dir. 3].
    data : "float[:,:,:]"
        _data array of StencilDiagonalMatrix to store the results. Periodic wrap-around (index -= nb) is applied
        when a local index runs beyond the array bounds, since there are no ghost regions on this matrix type.
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


def surface_kernel_3d_vec(
    spans1: "int[:]",
    spans2: "int[:]",
    pi0: int,
    pi1: int,
    pi2: int,
    starts0: int,
    starts1: int,
    starts2: int,
    pads0: int,
    pads1: int,
    pads2: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    boundary_index: int,
    mat_fun: "float[:,:]",
    data: "float[:,:,:]",
):
    """
    Performs the integration of Lambda_0ij * mat_fun(eta1, eta2) over the boundary surface at the fixed
    (normal-direction) global index boundary_index, for the basis functions (ij) available on the calling
    process in the two surface (tangential) directions.

    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).

    Parameters
    ----------
    spans1, spans2 : array[int]
        Arrays of span indices in the two surface (tangential) directions; the span is the index of the last
        non-vanishing spline on each grid element (cell) in that direction.
    pi0 : int
        Degree of the basis function in the normal direction (kept for a uniform kernel signature; not used
        directly since the normal index is fixed to boundary_index).
    pi1, pi2 : int
        Degree of the basis functions in the two surface directions.
    starts0 : int
        Starting index on the current rank in the normal direction.
    starts1, starts2 : int
        Starting index on the current rank in the two surface directions.
    pads0 : int
        Padding (=spline degree) for ghost regions in data, in the normal direction.
    pads1, pads2 : int
        Padding (=spline degree) for ghost regions in data, in the two surface directions.
    w1, w2 : "float[:,:]"
        Quadrature weights in the two surface directions. The indexing is [global element, quadrature point].
    bi1, bi2 : "float[:,:,:,:]"
        Values of basis functions in the two surface directions. The indexing is
        [global element, local basis function, derivative, quadrature point].
    boundary_index : int
        Global index in the normal direction at which the boundary surface is located.
    mat_fun : "float[:,:]"
        Function under the integral evaluated at surface quadrature points (flattened in each surface direction).
    data : "float[:,:,:]"
        _data array of StencilVector to store the results; only the slice at the fixed normal index
        (pads0 + i_local0) is written.
    """

    ne1 = spans1.size
    ne2 = spans2.size

    nq1 = shape(w1)[1]
    nq2 = shape(w2)[1]

    i_local0 = boundary_index - starts0

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for il1 in range(pi1 + 1):
                for il2 in range(pi2 + 1):
                    i_global1 = spans1[iel1] - pi1 + il1
                    i_global2 = spans2[iel2] - pi2 + il2

                    i_local1 = i_global1 - starts1
                    i_local2 = i_global2 - starts2

                    value = 0.0

                    for q1 in range(nq1):
                        for q2 in range(nq2):
                            wvol = (
                                w1[iel1, q1]
                                * w2[iel2, q2]
                                * mat_fun[iel1 * nq1 + q1, iel2 * nq2 + q2]
                            )

                            value += (
                                wvol * bi1[iel1, il1, 0, q1] * bi2[iel2, il2, 0, q2]
                            )

                    data[pads0 + i_local0, pads1 + i_local1, pads2 + i_local2] += value


def surface_kernel_3d_mat_h1(
    spans1: "int[:]",
    spans2: "int[:]",
    pi0: int,
    pi1: int,
    pi2: int,
    pj0: int,
    pj1: int,
    pj2: int,
    starts0: int,
    starts1: int,
    starts2: int,
    pads0: int,
    pads1: int,
    pads2: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    bj1: "float[:,:,:,:]",
    bj2: "float[:,:,:,:]",
    boundary_index: int,
    normal_dir: int,
    mat_fun: "float[:,:]",
    data: "float[:,:,:,:,:,:]",
):
    """
    Computes a boundary (surface) H1 mass matrix: the integration of Lambda_i * mat_fun(eta_s1, eta_s2) * Lambda_j
    over the boundary surface at the fixed global index boundary_index in the normal_dir direction, for the
    codomain ("i") and domain ("j") basis functions available on the calling process in the two tangential
    directions orthogonal to normal_dir.

    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).

    Parameters
    ----------
    spans1, spans2 : array[int]
        Arrays of span indices in the two tangential grid directions used for the surface quadrature (as
        determined by normal_dir); the span is the index of the last non-vanishing spline on each grid element.
    pi0, pi1, pi2 : int
        Degree of the codomain basis functions along logical axes 0, 1 and 2.
    pj0, pj1, pj2 : int
        Degree of the domain basis functions along logical axes 0, 1 and 2.
    starts0, starts1, starts2 : int
        Starting index on the current rank along logical axes 0, 1 and 2.
    pads0, pads1, pads2 : int
        Padding (=spline degree) for ghost regions in data, along logical axes 0, 1 and 2.
    w1, w2 : "float[:,:]"
        Quadrature weights in the two tangential directions. The indexing is [global element, quadrature point].
    bi1, bi2 : "float[:,:,:,:]"
        Values of codomain basis functions in the two tangential directions. The indexing is
        [global element, local basis function, derivative, quadrature point].
    bj1, bj2 : "float[:,:,:,:]"
        Values of domain basis functions in the two tangential directions, same indexing convention as bi1, bi2.
    boundary_index : int
        Global index along normal_dir at which the boundary surface is located.
    normal_dir : int
        Logical direction (0, 1 or 2) normal to the surface; the remaining two directions are the tangential
        directions used for the surface integration.
    mat_fun : "float[:,:]"
        Function under the integral evaluated at surface quadrature points (flattened in each tangential direction).
    data : "float[:,:,:,:,:,:]"
        _data array of StencilMatrix to store the results. 
    """

    ne1 = spans1.size
    ne2 = spans2.size

    nq1 = shape(w1)[1]
    nq2 = shape(w2)[1]

    starts = [starts0, starts1, starts2]
    pads = [pads0, pads1, pads2]
    pi = [pi0, pi1, pi2]
    pj = [pj0, pj1, pj2]

    surf_dirs = [d for d in range(3) if d != normal_dir]

    pi_s1 = pi[surf_dirs[0]]
    pi_s2 = pi[surf_dirs[1]]

    pj_s1 = pj[surf_dirs[0]]
    pj_s2 = pj[surf_dirs[1]]

    starts_n = starts[normal_dir]
    starts_s1 = starts[surf_dirs[0]]
    starts_s2 = starts[surf_dirs[1]]

    pads_n = pads[normal_dir]
    pads_s1 = pads[surf_dirs[0]]
    pads_s2 = pads[surf_dirs[1]]

    i_local_n = boundary_index - starts_n

    for iel1 in range(ne1):
        for iel2 in range(ne2):
            for il1 in range(pi_s1 + 1):
                for il2 in range(pi_s2 + 1):
                    i_global1 = spans1[iel1] - pi_s1 + il1
                    i_global2 = spans2[iel2] - pi_s2 + il2

                    i_local1 = i_global1 - starts_s1
                    i_local2 = i_global2 - starts_s2

                    for jl1 in range(pj_s1 + 1):
                        for jl2 in range(pj_s2 + 1):
                            j_local1 = jl1 - il1
                            j_local2 = jl2 - il2

                            value = 0.0

                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    wvol = (
                                        w1[iel1, q1]
                                        * w2[iel2, q2]
                                        * mat_fun[iel1 * nq1 + q1, iel2 * nq2 + q2]
                                    )

                                    value += (
                                        wvol
                                        * bi1[iel1, il1, 0, q1]
                                        * bi2[iel2, il2, 0, q2]
                                        * bj1[iel1, jl1, 0, q1]
                                        * bj2[iel2, jl2, 0, q2]
                                    )

                            if normal_dir == 0:
                                data[
                                    pads_n + i_local_n,
                                    pads_s1 + i_local1,
                                    pads_s2 + i_local2,
                                    pads_n,
                                    pads_s1 + j_local1,
                                    pads_s2 + j_local2,
                                ] += value
                            elif normal_dir == 1:
                                data[
                                    pads_s1 + i_local1,
                                    pads_n + i_local_n,
                                    pads_s2 + i_local2,
                                    pads_s1 + j_local1,
                                    pads_n,
                                    pads_s2 + j_local2,
                                ] += value
                            else:
                                data[
                                    pads_s1 + i_local1,
                                    pads_s2 + i_local2,
                                    pads_n + i_local_n,
                                    pads_s1 + j_local1,
                                    pads_s2 + j_local2,
                                    pads_n,
                                ] += value


def surface_kernel_3d_mat_hcurl(
    spans1: "int[:]",
    spans2: "int[:]",
    pi0: int,
    pi1: int,
    pi2: int,
    qi0: int,
    qi1: int,
    qi2: int,
    starts0: int,
    starts1: int,
    starts2: int,
    pads0: int,
    pads1: int,
    pads2: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    bi1: "float[:,:,:,:]",
    bi2: "float[:,:,:,:]",
    bj1: "float[:,:,:,:]",
    bj2: "float[:,:,:,:]",
    boundary_index: int,
    n_cross_weight: "float[:,:]",
    data: "float[:,:,:,:,:,:]",
):
    """
    Computes a boundary (surface) H(curl) mass matrix: the integration of (n x Lambda_i) * mat_fun(eta_s1, eta_s2)
    * (n x Lambda_j), weighted by n_cross_weight, over the boundary surface at the fixed global index
    boundary_index, for the codomain ("i") and domain ("j") tangential-trace basis functions available on the
    calling process. Mirrors surface_kernel_3d_mat_h1, but for H(curl) spaces where only the tangential
    components of the basis functions couple through the cross product with the surface normal n.

    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).

    Parameters
    ----------
    spans1, spans2 : array[int]
        Arrays of span indices in the two tangential grid directions used for the surface quadrature; the span
        is the index of the last non-vanishing spline on each grid element.
    pi0, pi1, pi2 : int
        Degree of the codomain basis functions along Cartesian axes 0, 1 and 2.
    qi0, qi1, qi2 : int
        Degree of the domain basis functions along Cartesian axes 0, 1 and 2.
    starts0, starts1, starts2 : int
        Starting index on the current rank along Cartesian axes 0, 1 and 2.
    pads0, pads1, pads2 : int
        Padding (=spline degree) for ghost regions in data, along Cartesian axes 0, 1 and 2.
    w1, w2 : "float[:,:]"
        Quadrature weights in the two tangential directions. The indexing is [global element, quadrature point].
    bi1, bi2 : "float[:,:,:,:]"
        Values of codomain basis functions in the two tangential directions. The indexing is
        [global element, local basis function, derivative, quadrature point].
    bj1, bj2 : "float[:,:,:,:]"
        Values of domain basis functions in the two tangential directions, same indexing convention as bi1, bi2.
    boundary_index : int
        Global index in the normal direction at which the boundary surface is located.
    n_cross_weight : "float[:,:]"
        Weight from the cross product with the surface normal n (and mat_fun) evaluated at surface quadrature
        points, indexed like mat_fun in surface_kernel_3d_mat_h1.
    data : "float[:,:,:,:,:,:]"
        _data array of StencilMatrix to store the results, following the same storage convention as in
        surface_kernel_3d_mat_h1.
    """
    pass