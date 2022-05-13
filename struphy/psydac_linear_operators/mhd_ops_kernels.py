def assemble_dofs_for_weighted_basisfuns(mat : 'double[:,:,:,:,:,:]', starts_in : 'int[:]', ends_in : 'int[:]', pads_in : 'int[:]', starts_out : 'int[:]', ends_out : 'int[:]', pads_out : 'int[:]', fun_w : 'double[:,:,:,:,:,:]', span1 : 'int[:,:]', span2 : 'int[:,:]', span3 : 'int[:,:]', basis1 : 'double[:,:,:]', basis2 : 'double[:,:,:]', basis3 : 'double[:,:,:]'):
    '''Kernel for assembling the matrix
    
    A_(ijk,mno) = DOFS_ijk(fun*Lambda^in_mno) ,

    into the _data attribute of a StencilMatrix.
    Here, DOFS_ijk are the degrees-of-freedom of the output space (codomain, must not be a product space), 
    Lambda^in_mno are the basis functions of the input space (domain, must not be a product space), and fun is an arbitrary function.

    Parameters
    ----------
        mat : 6d double array
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

        fun_w : 6d double array
            The function evaluated at the points (i, j, k, iq, jq, kq), where iq a local quadrature point in Greville element i,
            and already multiplied by the correspinding quadrature weight.

        span1 : 2d int array
            Knot span indices in direction eta1 in format (i, iq).

        span2 : 2d int array
            Knot span indices in direction eta2 in format (j, jq).

        span3 : 2d int array
            Knot span indices in direction eta3 in format (k, kq).

        basis1 : 3d double array
            Values of p1 + 1 non-zero eta-1 basis functions at quadrature points in format (i, iq, basis function).

        basis2 : 3d double array
            Values of p2 + 1 non-zero eta-2 basis functions at quadrature points in format (j, jq, basis function).

        basis3 : 3d double array
            Values of p3 + 1 non-zero eta-3 basis functions at quadrature points in format (k, kq, basis function).
    '''

    # Start/end indices and paddings for distributed stencil matrix of input space
    # si1 = starts_in[0]
    # si2 = starts_in[1]
    # si3 = starts_in[2]
    # ei1 = ends_in[0]
    # ei2 = ends_in[1]
    # ei3 = ends_in[2]
    pi1 = pads_in[0]
    pi2 = pads_in[1]
    pi3 = pads_in[2]

    # Start/end indices for distributed stencil matrix of output space
    so1 = starts_out[0]
    so2 = starts_out[1]
    so3 = starts_out[2]
    eo1 = ends_out[0]
    eo2 = ends_out[1]
    eo3 = ends_out[2]
    po1 = pads_out[0]
    po2 = pads_out[1]
    po3 = pads_out[2]

    # Spline degrees
    p1 = basis1.shape[2] - 1
    p2 = basis2.shape[2] - 1
    p3 = basis3.shape[2] - 1

    mat[:] = 0.

    # Global DOF index (ijk=Greville point index) of output space
    for i in range(so1, eo1 + 1):
        for j in range(so2, eo2 + 1):
            for k in range(so3, eo3 + 1):

                # Quadrature point index in Greville element
                for iq in range(span1.shape[1]):
                    for jq in range(span2.shape[1]):
                        for kq in range(span3.shape[1]):

                            funval = fun_w[i, j, k, iq, jq, kq]
                    
                            # Global basis function index (mno) of input space:
                            for b1 in range(p1 + 1):
                                m = span1[i, iq] - p1 + b1
                                val1 = funval * basis1[i, iq, b1]
                                for b2 in range(p2 + 1):
                                    n = span2[j, jq] - p2 + b2
                                    val2 = val1 * basis2[j, jq, b2]
                                    for b3 in range(p3 + 1):
                                        o = span3[k, kq] - p3 + b3
                                        value = val2 * basis3[k, kq, b3]

                                        # Row index: starts (e.g. so1) must be subtracted because we operate on _data attribute.
                                        # Column index is m - i plus padding of input space (m is global column, i is global row)
                                        mat[po1 + i - so1, po2 + j - so2, po3 + k - so3, pi1 + m - i, pi2 + n - j, pi3 + o - k] += value
                
                

