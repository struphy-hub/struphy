"""Kernel kernel_2d_mat: integral kernel for mass matrices and L2-projections."""

from numpy import shape


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
