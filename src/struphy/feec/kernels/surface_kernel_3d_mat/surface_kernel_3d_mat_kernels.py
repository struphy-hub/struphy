"""Kernel surface_kernel_3d_mat: integral kernel for mass matrices and L2-projections."""

from numpy import shape


def surface_kernel_3d_mat(
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
    Assembles a boundary (surface) mass matrix: the integration of Lambda_i * mat_fun(eta_s1, eta_s2) * Lambda_j
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

    # Select the normal (n) and the two tangential (s1, s2) directions with scalars only:
    # Python lists would make pyccel generate gFTL containers, which we want to avoid as a dependency.
    if normal_dir == 0:
        pi_s1 = pi1
        pi_s2 = pi2
        pj_s1 = pj1
        pj_s2 = pj2
        starts_n = starts0
        starts_s1 = starts1
        starts_s2 = starts2
        pads_n = pads0
        pads_s1 = pads1
        pads_s2 = pads2
    elif normal_dir == 1:
        pi_s1 = pi0
        pi_s2 = pi2
        pj_s1 = pj0
        pj_s2 = pj2
        starts_n = starts1
        starts_s1 = starts0
        starts_s2 = starts2
        pads_n = pads1
        pads_s1 = pads0
        pads_s2 = pads2
    else:
        pi_s1 = pi0
        pi_s2 = pi1
        pj_s1 = pj0
        pj_s2 = pj1
        starts_n = starts2
        starts_s1 = starts0
        starts_s2 = starts1
        pads_n = pads2
        pads_s1 = pads0
        pads_s2 = pads1

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
                                    wvol = w1[iel1, q1] * w2[iel2, q2] * mat_fun[iel1 * nq1 + q1, iel2 * nq2 + q2]

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
