"""Kernel assemble_dofs_for_weighted_basisfuns_2d: assembly kernel for basis projection operators."""


def assemble_dofs_for_weighted_basisfuns_2d(
    mat: "float[:,:,:,:]",
    starts_in: "int[:]",
    ends_in: "int[:]",
    pads_in: "int[:]",
    starts_out: "int[:]",
    ends_out: "int[:]",
    pads_out: "int[:]",
    fun_q: "float[:,:]",
    wts1: "float[:,:]",
    wts2: "float[:,:]",
    span1: "int[:,:]",
    span2: "int[:,:]",
    basis1: "float[:,:,:]",
    basis2: "float[:,:,:]",
    sub1: "int[:]",
    sub2: "int[:]",
    dim1_in: int,
    dim2_in: int,
    dim1_out: int,
    dim2_out: int,
    p1_out: int,
    p2_out: int,
):
    """Kernel for assembling the matrix

    A_(ij,kl) = DOFS_ij(fun*Lambda^in_kl) ,

    into the _data attribute of a StencilMatrix.
    Here, DOFS_ij are the degrees-of-freedom of the output space (codomain, must not be a product space),
    Lambda^in_kl are the basis functions of the input space (domain, must not be a product space), and fun is an arbitrary function.

    Parameters
    ----------
        mat : 4d float array
            _data attribute of StencilMatrix.

        starts_in : 1d int array
            Starting indices of the input space (domain) of a distributed StencilMatrix.

        ends_in : 1d int array
            Ending indices of the input space (domain) of a distributed StencilMatrix.

        pads_in : 1d int array
            Paddings of the input space (domain) of a distributed StencilMatrix.

        starts_out : 1d int array
            Starting indices of the output space (codomain) of a distributed StencilMatrix.

        ends_out : 1d int array
            Ending indices of the output space (codomain) of a distributed StencilMatrix.

        pads_out : 1d int array
            Paddings of the output space (codomain) of a distributed StencilMatrix.

        fun_q : 2d float array
            The function evaluated at the points (nq_i*ii + iq, nq_j*jj + jq), where iq a local quadrature point of interval ii.

        wts1 : 2d float array
            Quadrature weights in direction eta1 in format (ii, iq).

        wts2 : 2d float array
            Quadrature weights in direction eta2 in format (jj, jq).

        span1 : 2d int array
            Knot span indices in direction eta1 in format (ii, iq).

        span2 : 2d int array
            Knot span indices in direction eta2 in format (jj, jq).

        basis1 : 3d float array
            Values of p1 + 1 non-zero eta-1 basis functions at quadrature points in format (ii, iq, basis function).

        basis2 : 3d float array
            Values of p2 + 1 non-zero eta-2 basis functions at quadrature points in format (jj, jq, basis function).

        sub1 : 1d int array
            Sub-interval indices in direction 1.

        sub2 : 1d int array
            Sub-interval indices in direction 2.

        dim1_in : int
            Dimension of the first direction of the input space

        dim2_in : int
            Dimension of the second direction of the input space

        p1_out : int
            Spline degree of the first direction of the output space

        p2_out : int
            Spline degree of the second direction of the output space
    """

    # Start/end indices and paddings for distributed stencil matrix of input space
    # si1 = starts_in[0]
    # si2 = starts_in[1]
    # ei1 = ends_in[0]
    # ei2 = ends_in[1]
    pi1 = pads_in[0]
    pi2 = pads_in[1]

    # Start/end indices for distributed stencil matrix of output space
    so1 = starts_out[0]
    so2 = starts_out[1]
    # eo1 = ends_out[0]
    # eo2 = ends_out[1]
    po1 = pads_out[0]
    po2 = pads_out[1]

    # Spline degrees of input space
    p1 = basis1.shape[2] - 1
    p2 = basis2.shape[2] - 1

    # number of quadrature points
    nq1 = span1.shape[1]
    nq2 = span2.shape[1]

    # Set output to zero
    mat[:] = 0.0

    # Find column index for _data:
    if dim1_out <= dim1_in:
        cut1 = p1
    else:
        cut1 = p1_out

    if dim2_out <= dim2_in:
        cut2 = p2
    else:
        cut2 = p2_out

    # Interval (either element or sub-interval thereof)
    # -------------------------------------------------
    cumsub_i = 0  # Cumulative sub-interval index
    for ii in range(span1.shape[0]):
        cumsub_i += sub1[ii]
        i = ii - cumsub_i  # local DOF index

        cumsub_j = 0  # Cumulative sub-interval index
        for jj in range(span2.shape[0]):
            cumsub_j += sub2[jj]
            j = jj - cumsub_j  # local DOF index

            # Quadrature point index in interval
            # ----------------------------------
            for iq in range(nq1):
                for jq in range(nq2):
                    funval = fun_q[nq1 * ii + iq, nq2 * jj + jq] * wts1[ii, iq] * wts2[jj, jq]

                    # Basis function of input space:
                    # ------------------------------
                    for b1 in range(p1 + 1):
                        m = span1[ii, iq] - p1 + b1  # global index
                        # basis value
                        val1 = funval * basis1[ii, iq, b1]

                        # Diff of global indices, needs to be adjusted for boundary conditions --> col1
                        col1_tmp = m - (i + so1)
                        if col1_tmp > cut1:
                            m = m - dim1_in
                        elif col1_tmp < -cut1:
                            m = m + dim1_in
                        # add padding
                        col1 = pi1 + m - (i + so1)

                        for b2 in range(p2 + 1):
                            # global index
                            n = span2[jj, jq] - p2 + b2
                            value = val1 * basis2[jj, jq, b2]

                            # Diff of global indices, needs to be adjusted for boundary conditions --> col2
                            col2_tmp = n - (j + so2)
                            if col2_tmp > cut2:
                                n = n - dim2_in
                            elif col2_tmp < -cut2:
                                n = n + dim2_in
                            # add padding
                            col2 = pi2 + n - (j + so2)

                            # Row index: padding + local index.
                            mat[po1 + i, po2 + j, col1, col2] += value
