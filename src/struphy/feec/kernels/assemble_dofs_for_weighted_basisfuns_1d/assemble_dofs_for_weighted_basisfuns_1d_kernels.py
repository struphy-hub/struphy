"""Kernel assemble_dofs_for_weighted_basisfuns_1d: assembly kernel for basis projection operators."""


def assemble_dofs_for_weighted_basisfuns_1d(
    mat: "float[:,:]",
    starts_in: "int[:]",
    ends_in: "int[:]",
    pads_in: "int[:]",
    starts_out: "int[:]",
    ends_out: "int[:]",
    pads_out: "int[:]",
    fun_q: "float[:]",
    wts1: "float[:,:]",
    span1: "int[:,:]",
    basis1: "float[:,:,:]",
    sub1: "int[:]",
    dim1_in: int,
    dim1_out: int,
    p1_out: int,
):
    """Kernel for assembling the matrix

    A_(i,j) = DOFS_i(fun*Lambda^in_j) ,

    into the _data attribute of a StencilMatrix.
    Here, DOFS_i are the degrees-of-freedom of the output space (codomain, must not be a product space),
    Lambda^in_j are the basis functions of the input space (domain, must not be a product space), and fun is an arbitrary function.

    Parameters
    ----------
        mat : 2d float array
            _data attribute of StencilMatrix.

        starts_in : int
            Starting index of the input space (domain) of a distributed StencilMatrix.

        ends_in : int
            Ending index of the input space (domain) of a distributed StencilMatrix.

        pads_in : int
            Paddings of the input space (domain) of a distributed StencilMatrix.

        starts_out : int
            Starting indices of the output space (codomain) of a distributed StencilMatrix.

        ends_out : int
            Ending indices of the output space (codomain) of a distributed StencilMatrix.

        pads_out : int
            Paddings of the output space (codomain) of a distributed StencilMatrix.

        fun_q : 1d float array
            The function evaluated at the points (nq*ii + iq), where iq a local quadrature point of interval ii.

        wts1 : 2d float array
            Quadrature weights in format (ii, iq).

        span1 : 2d int array
            Knot span indices in direction eta1 in format (ii, iq).

        basis1 : 3d float array
            Values of p1 + 1 non-zero eta-1 basis functions at quadrature points in format (ii, iq, basis function).

        sub1 : 1d int array
            Sub-interval indices in direction 1.

        dim1_in : int
            Dimension of the first direction of the input space

        p1_out : int
            Spline degree of the first direction of the output space
    """

    # Start/end indices and paddings for distributed stencil matrix of input space
    # si1 = starts_in[0}
    # ei1 = ends_in[0]
    pi1 = pads_in[0]

    # Start/end indices for distributed stencil matrix of output space
    so1 = starts_out[0]
    # eo1 = ends_out[0]
    po1 = pads_out[0]

    # Spline degrees of input space
    p1 = basis1.shape[2] - 1

    # number of quadrature points
    nq1 = span1.shape[1]

    # Set output to zero
    mat[:] = 0.0

    # Find column index for _data:
    if dim1_out <= dim1_in:
        cut1 = p1
    else:
        cut1 = p1_out

    # Interval (either element or sub-interval thereof)
    # -------------------------------------------------
    cumsub_i = 0  # Cumulative sub-interval index
    for ii in range(span1.shape[0]):
        cumsub_i += sub1[ii]
        i = ii - cumsub_i  # local DOF index

        # Quadrature point index in interval
        # ----------------------------------
        for iq in range(nq1):
            funval = fun_q[nq1 * ii + iq] * wts1[ii, iq]

            # Basis function of input space:
            # ------------------------------
            for b1 in range(p1 + 1):
                m = span1[ii, iq] - p1 + b1  # global index
                # basis value
                value = funval * basis1[ii, iq, b1]

                # Diff of global indices, needs to be adjusted for boundary conditions --> col1
                col1_tmp = m - (i + so1)
                if col1_tmp > cut1:
                    m = m - dim1_in
                elif col1_tmp < -cut1:
                    m = m + dim1_in
                # add padding
                col1 = pi1 + m - (i + so1)

                # Row index: padding + local index.
                mat[po1 + i, col1] += value
