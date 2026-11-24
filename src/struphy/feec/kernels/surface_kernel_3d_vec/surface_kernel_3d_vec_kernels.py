"""Kernel surface_kernel_3d_vec: integral kernel for mass matrices and L2-projections."""

from numpy import shape


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
                            wvol = w1[iel1, q1] * w2[iel2, q2] * mat_fun[iel1 * nq1 + q1, iel2 * nq2 + q2]

                            value += wvol * bi1[iel1, il1, 0, q1] * bi2[iel2, il2, 0, q2]

                    data[pads0 + i_local0, pads1 + i_local1, pads2 + i_local2] += value
