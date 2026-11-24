"""Kernel kernel_1d_mat: integral kernel for mass matrices and L2-projections."""

from numpy import shape


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
