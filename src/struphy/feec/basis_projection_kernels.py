from pyccel.decorators import stack_array
from numpy import shape, zeros


def assemble_dofs_for_weighted_basisfuns_1d(
    mat: 'float[:,:]',
    starts_in: 'int[:]', ends_in: 'int[:]', pads_in: 'int[:]',
    starts_out: 'int[:]', ends_out: 'int[:]', pads_out: 'int[:]',
    fun_q: 'float[:]', wts1: 'float[:,:]', span1: 'int[:,:]',
    basis1: 'float[:,:,:]', sub1: 'int[:]', dim1_in: int, p1_out: int
):
    '''Kernel for assembling the matrix

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
    '''

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
    mat[:] = 0.

    # Dimensions of output space
    _sum = 0.
    for i in range(sub1.shape[0]):
        _sum += sub1[i]
    dim1_out = span1.shape[0] - _sum
    # Interval (either element or sub-interval thereof)
    # -------------------------------------------------
    cumsub_i = 0  # Cumulative sub-interval index
    for ii in range(span1.shape[0]):
        cumsub_i += sub1[ii]
        i = ii - cumsub_i  # local DOF index

        # Quadrature point index in interval
        # ----------------------------------
        for iq in range(nq1):

            funval = fun_q[nq1*ii + iq] * wts1[ii, iq]

            # Basis function of input space:
            # ------------------------------
            for b1 in range(p1 + 1):
                m = (span1[ii, iq] - p1 + b1)  # global index
                # basis value
                value = funval * basis1[ii, iq, b1]

                # Find column index for _data:
                if dim1_out <= dim1_in:
                    cut1 = p1
                else:
                    cut1 = p1_out

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


def assemble_dofs_for_weighted_basisfuns_2d(
    mat: 'float[:,:,:,:]',
    starts_in: 'int[:]', ends_in: 'int[:]', pads_in: 'int[:]',
    starts_out: 'int[:]', ends_out: 'int[:]', pads_out: 'int[:]',
    fun_q: 'float[:,:]', wts1: 'float[:,:]', wts2: 'float[:,:]',
    span1: 'int[:,:]', span2: 'int[:,:]',
    basis1: 'float[:,:,:]', basis2: 'float[:,:,:]',
    sub1: 'int[:]', sub2: 'int[:]', dim1_in: int, dim2_in: int,
    p1_out: int, p2_out: int
):
    '''Kernel for assembling the matrix

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
    '''

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
    mat[:] = 0.

    # Dimensions of output space
    _sum = 0.
    for i in range(sub1.shape[0]):
        _sum += sub1[i]
    dim1_out = span1.shape[0] - _sum

    _sum = 0.
    for i in range(sub2.shape[0]):
        _sum += sub2[i]
    dim2_out = span2.shape[0] - _sum

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

                    funval = fun_q[nq1*ii + iq, nq2*jj + jq] * \
                        wts1[ii, iq] * wts2[jj, jq]

                    # Basis function of input space:
                    # ------------------------------
                    for b1 in range(p1 + 1):
                        m = (span1[ii, iq] - p1 + b1)  # global index
                        # basis value
                        val1 = funval * basis1[ii, iq, b1]

                        # Find column index for _data:
                        if dim1_out <= dim1_in:
                            cut1 = p1
                        else:
                            cut1 = p1_out

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
                            n = (span2[jj, jq] - p2 + b2)
                            value = val1 * basis2[jj, jq, b2]

                            # Find column index for _data:
                            if dim2_out <= dim2_in:
                                cut2 = p2
                            else:
                                cut2 = p2_out

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


def assemble_dofs_for_weighted_basisfuns_3d(
    mat: 'float[:,:,:,:,:,:]',
    starts_in: 'int[:]', ends_in: 'int[:]', pads_in: 'int[:]',
    starts_out: 'int[:]', ends_out: 'int[:]', pads_out: 'int[:]',
    fun_q: 'float[:,:,:]',
    wts1: 'float[:,:]', wts2: 'float[:,:]', wts3: 'float[:,:]',
    span1: 'int[:,:]', span2: 'int[:,:]', span3: 'int[:,:]',
    basis1: 'float[:,:,:]', basis2: 'float[:,:,:]', basis3: 'float[:,:,:]',
    sub1: 'int[:]', sub2: 'int[:]', sub3: 'int[:]',
    dim1_in: int, dim2_in: int, dim3_in: int,
    p1_out: int, p2_out: int, p3_out: int
):
    '''Kernel for assembling the matrix

    A_(ijk,mno) = DOFS_ijk(fun*Lambda^in_mno) ,

    into the _data attribute of a StencilMatrix.
    Here, DOFS_ijk are the degrees-of-freedom of the output space (codomain, must not be a product space), 
    Lambda^in_mno are the basis functions of the input space (domain, must not be a product space), and fun is an arbitrary function.

    Parameters
    ----------
        mat : 6d float array
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

        fun_q : 3d float array
            The function evaluated at the points (nq_i*ii + iq, nq_j*jj + jq, nq_k*kk + kq), where iq a local quadrature point of interval ii.

        wts1 : 2d float array
            Quadrature weights in direction eta1 in format (ii, iq).

        wts2 : 2d float array
            Quadrature weights in direction eta2 in format (jj, jq).

        wts3 : 2d float array
            Quadrature weights in direction eta3 in format (kk, kq).

        span1 : 2d int array
            Knot span indices in direction eta1 in format (ii, iq).

        span2 : 2d int array
            Knot span indices in direction eta2 in format (jj, jq).

        span3 : 2d int array
            Knot span indices in direction eta3 in format (kk, kq).

        basis1 : 3d float array
            Values of p1 + 1 non-zero eta-1 basis functions at quadrature points in format (ii, iq, basis function).

        basis2 : 3d float array
            Values of p2 + 1 non-zero eta-2 basis functions at quadrature points in format (jj, jq, basis function).

        basis3 : 3d float array
            Values of p3 + 1 non-zero eta-3 basis functions at quadrature points in format (kk, kq, basis function).

        sub1 : 1d int array
            Sub-interval indices in direction 1.

        sub2 : 1d int array
            Sub-interval indices in direction 2.

        sub3 : 1d int array
            Sub-interval indices in direction 3.

        dim1_in : int
            Dimension of the first direction of the input space

        dim2_in : int
            Dimension of the second direction of the input space

        dim3_in : int
            Dimension of the third direction of the input space

        p1_out : int
            Spline degree of the first direction of the output space

        p2_out : int
            Spline degree of the second direction of the output space

        p3_out : int
            Spline degree of the third direction of the output space
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
    # eo1 = ends_out[0]
    # eo2 = ends_out[1]
    # eo3 = ends_out[2]
    po1 = pads_out[0]
    po2 = pads_out[1]
    po3 = pads_out[2]

    # Spline degrees of input space
    p1 = basis1.shape[2] - 1
    p2 = basis2.shape[2] - 1
    p3 = basis3.shape[2] - 1

    # number of quadrature points
    nq1 = span1.shape[1]
    nq2 = span2.shape[1]
    nq3 = span3.shape[1]

    # Set output to zero
    mat[:] = 0.

    # Dimensions of output space
    _sum = 0.
    for i in range(sub1.shape[0]):
        _sum += sub1[i]
    dim1_out = span1.shape[0] - _sum

    _sum = 0.
    for i in range(sub2.shape[0]):
        _sum += sub2[i]
    dim2_out = span2.shape[0] - _sum

    _sum = 0.
    for i in range(sub3.shape[0]):
        _sum += sub3[i]
    dim3_out = span3.shape[0] - _sum

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

            cumsub_k = 0  # Cumulative sub-interval index
            for kk in range(span3.shape[0]):
                cumsub_k += sub3[kk]
                k = kk - cumsub_k  # local DOF index

                # Quadrature point index in interval
                # ----------------------------------
                for iq in range(nq1):
                    for jq in range(nq2):
                        for kq in range(nq3):

                            funval = fun_q[nq1*ii + iq, nq2*jj + jq, nq3*kk +
                                           kq] * wts1[ii, iq] * wts2[jj, jq] * wts3[kk, kq]

                            # Basis function of input space:
                            # ------------------------------
                            for b1 in range(p1 + 1):
                                m = (span1[ii, iq] - p1 + b1)  # global index
                                # basis value
                                val1 = funval * basis1[ii, iq, b1]

                                # Find column index for _data:
                                if dim1_out <= dim1_in:
                                    cut1 = p1
                                else:
                                    cut1 = p1_out

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
                                    n = (span2[jj, jq] - p2 + b2)
                                    val2 = val1 * basis2[jj, jq, b2]

                                    # Find column index for _data:
                                    if dim2_out <= dim2_in:
                                        cut2 = p2
                                    else:
                                        cut2 = p2_out

                                    # Diff of global indices, needs to be adjusted for boundary conditions --> col2
                                    col2_tmp = n - (j + so2)
                                    if col2_tmp > cut2:
                                        n = n - dim2_in
                                    elif col2_tmp < -cut2:
                                        n = n + dim2_in
                                    # add padding
                                    col2 = pi2 + n - (j + so2)

                                    for b3 in range(p3 + 1):
                                        # global index
                                        o = (span3[kk, kq] - p3 + b3)
                                        value = val2 * basis3[kk, kq, b3]

                                        # Find column index for _data:
                                        if dim3_out <= dim3_in:
                                            cut3 = p3
                                        else:
                                            cut3 = p3_out

                                        # Diff of global indices, needs to be adjusted for boundary conditions --> col3
                                        col3_tmp = o - (k + so3)
                                        if col3_tmp > cut3:
                                            o = o - dim3_in
                                        elif col3_tmp < -cut3:
                                            o = o + dim3_in
                                        # add padding
                                        col3 = pi3 + o - (k + so3)

                                        # Row index: padding + local index.
                                        mat[po1 + i, po2 + j, po3 + k,
                                            col1, col2, col3] += value


@stack_array('shp')
def get_dofs_local_1_form_e1_component(
    f1: 'float[:,:,:]',
    p1: int, wts: 'float[:]', f_eval_aux: 'float[:,:,:]'
):
    '''Kernel for evaluating the degrees of freedom for the first component of 1-forms. This function is for local commuting projetors.

    Parameters
    ----------
        f1 : 3d float array
            Evaluation for the first component of the 1-form function over all the interpolation points in e2 and e3, as well as all the Gauss-Legendre quadrature point in e1.

        p1 : int
            Degree of the B-splines in the e1 direction.

        wts: 1d float list
            Gauss-Legandre quadrature weights for the intergrals in the e1 direction.
        f_eval_aux : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.
    '''

    shp = zeros(3, dtype=int)
    shp[:] = shape(f_eval_aux)

    for i in range(shp[0]):
        for j in range(shp[1]):
            for k in range(shp[2]):
                # Te following loop must be tantamount to:
                # f_eval_aux[i,j,k] = sum(numpy.multiply(f1[i*p1:(i+1)*p1,j,k],wts))
                in_start_1 = i*p1
                for ii in range(p1):
                    f_eval_aux[i, j, k] += f1[in_start_1+ii, j, k]*wts[ii]


@stack_array('shp')
def get_dofs_local_1_form_e2_component(
    f2: 'float[:,:,:]',
    p2: int, wts: 'float[:]', f_eval_aux: 'float[:,:,:]'
):
    '''Kernel for evaluating the degrees of freedom for the second component of 1-forms.  This function is for local commuting projetors.

    Parameters
    ----------
        f2 : 3d float array
            Evaluation for the second component of the 1-form function over all the interpolation points in e1 and e3, as well as all the Gauss-Legendre quadrature point in e2.

        p2 : int
            Degree of the B-splines in the e2 direction.

        wts: 1d float array
            Gauss-Legandre quadrature weights for the intergrals in the e2 direction.
        f_eval_aux : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.
    '''

    shp = zeros(3, dtype=int)
    shp[:] = shape(f_eval_aux)

    for i in range(shp[0]):
        for j in range(shp[1]):
            for k in range(shp[2]):
                # Te following loop must be tantamount to:
                # f_eval_aux[i,j,k] = np.sum(np.multiply(f2[i,j*self._p[1]:(j+1)*self._p[1],k],self._wts[1][1][0]))
                in_start_2 = j*p2
                for ii in range(p2):
                    f_eval_aux[i, j, k] += f2[i, in_start_2+ii, k]*wts[ii]


@stack_array('shp')
def get_dofs_local_1_form_e3_component(
    f3: 'float[:,:,:]',
    p3: int, wts: 'float[:]', f_eval_aux: 'float[:,:,:]'
):
    '''Kernel for evaluating the degrees of freedom for the third component of 1-forms.  This function is for local commuting projetors.

    Parameters
    ----------
        f3 : 3d float array
            Evaluation for the third component of the 1-form function over all the interpolation points in e1 and e2, as well as all the Gauss-Legendre quadrature point in e3.

        p3 : int
            Degree of the B-splines in the e3 direction.

        wts: 1d float array
            Gauss-Legandre quadrature weights for the intergrals in the e3 direction.
        f_eval_aux : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.
    '''

    shp = zeros(3, dtype=int)
    shp[:] = shape(f_eval_aux)

    for i in range(shp[0]):
        for j in range(shp[1]):
            for k in range(shp[2]):
                # Te following loop must be tantamount to:
                # f_eval_aux[i,j,k] = np.sum(np.multiply(f3[i,j,k*self._p[2]:(k+1)*self._p[2]],self._wts[2][2][0]))
                in_start_3 = k*p3
                for ii in range(p3):
                    f_eval_aux[i, j, k] += f3[i, j, in_start_3+ii]*wts[ii]

@stack_array('shp')
def get_dofs_local_2_form_e1_component(
    f1: 'float[:,:,:]',
    p2: int, p3: int, GLweightsx: 'float[:,:]', f_eval_aux: 'float[:,:,:]'
):
    '''Kernel for evaluating the degrees of freedom for the first component of 2-forms.  This function is for local commuting projetors.

    Parameters
    ----------
        f1 : 3d float array
            Evaluation for the first component of the 2-form function over all the interpolation points in e1, as well as all the Gauss-Legendre quadrature point in e2 and e3.

        p2 : int
            Degree of the B-splines in the e2 direction.

        p3 : int
            Degree of the B-splines in the e3 direction.

        GLweightsx : 2d float array
            Tensor product of the Gauss-Legandre quadrature weights for the intergrals in the e2 and e3 direction.
        f_eval_aux : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.
    '''

    shp = zeros(3, dtype=int)
    shp[:] = shape(f_eval_aux)

    for i in range(shp[0]):
        for j in range(shp[1]):
            for k in range(shp[2]):
                # The following loop must be tantamount to:
                # f_eval_aux[i,j,k] = np.sum(np.multiply(f1[i,j*p2:(j+1)*p2,k*p3:(k+1)*p3],GLweightsx))
                in_start_2 = j*p2
                in_start_3 = k*p3
                for jj in range(p2):
                    for kk in range(p3):
                        f_eval_aux[i, j, k] += f1[i, in_start_2 +
                                                  jj, in_start_3+kk] * GLweightsx[jj, kk]

@stack_array('shp')
def get_dofs_local_2_form_e2_component(
    f2: 'float[:,:,:]',
    p1: int, p3: int, GLweightsy: 'float[:,:]', f_eval_aux: 'float[:,:,:]'
):
    '''Kernel for evaluating the degrees of freedom for the second component of 2-forms.  This function is for local commuting projetors.

    Parameters
    ----------
        f2 : 3d float array
            Evaluation for the second component of the 2-form function over all the interpolation points in e2, as well as all the Gauss-Legendre quadrature point in e1 and e3.

        p1 : int
            Degree of the B-splines in the e1 direction.

        p3 : int
            Degree of the B-splines in the e3 direction.

        GLweightsy : 2d float array
            Tensor product of the Gauss-Legandre quadrature weights for the intergrals in the e1 and e3 direction.
        f_eval_aux : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.
    '''

    shp = zeros(3, dtype=int)
    shp[:] = shape(f_eval_aux)

    for i in range(shp[0]):
        for j in range(shp[1]):
            for k in range(shp[2]):
                # The following loop must be tantamount to:
                # f_eval_aux[i,j,k] = np.sum(np.multiply(f2[i*self._p[0]:(i+1)*self._p[0],j,k*self._p[2]:(k+1)*self._p[2]],self._GLweightsy))
                in_start_1 = i*p1
                in_start_3 = k*p3
                for ii in range(p1):
                    for kk in range(p3):
                        f_eval_aux[i, j, k] += f2[in_start_1+ii,
                                                  j, in_start_3+kk] * GLweightsy[ii, kk]

@stack_array('shp')
def get_dofs_local_2_form_e3_component(
    f3: 'float[:,:,:]',
    p1: int, p2: int, GLweightsz: 'float[:,:]', f_eval_aux: 'float[:,:,:]'
):
    '''Kernel for evaluating the degrees of freedom for the third component of 2-forms.  This function is for local commuting projetors.

    Parameters
    ----------
        f3 : 3d float array
            Evaluation for the third component of the 2-form function over all the interpolation points in e3, as well as all the Gauss-Legendre quadrature point in e1 and e2.

        p1 : int
            Degree of the B-splines in the e1 direction.

        p2 : int
            Degree of the B-splines in the e2 direction.

        GLweightsz : 2d float array
            Tensor product of the Gauss-Legandre quadrature weights for the intergrals in the e1 and e2 direction.
        
        f_eval_aux : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.
    '''

    shp = zeros(3, dtype=int)
    shp[:] = shape(f_eval_aux)

    for i in range(shp[0]):
        for j in range(shp[1]):
            for k in range(shp[2]):
                # The following loop must be tantamount to:
                # f_eval_aux[i,j,k] = np.sum(np.multiply(f3[i*self._p[0]:(i+1)*self._p[0],j*self._p[1]:(j+1)*self._p[1],k],self._GLweightsz))
                in_start_1 = i*p1
                in_start_2 = j*p2
                for ii in range(p1):
                    for jj in range(p2):
                        f_eval_aux[i, j, k] += f3[in_start_1+ii,
                                                  in_start_2+jj, k] * GLweightsz[ii, jj]

@stack_array('shp')
def get_dofs_local_3_form(
    faux: 'float[:,:,:]',
    p1: int, p2: int, p3: int, GLweights: 'float[:,:,:]', f_eval: 'float[:,:,:]'
):
    '''Kernel for evaluating the degrees of freedom for 3-forms.  This function is for local commuting projetors.

    Parameters
    ----------
        faux : 3d float array
            Evaluation for the 3-form function over all the Gauss-Legendre quadrature point in e1, e2 and e3.

        p1 : int
            Degree of the B-splines in the e1 direction.

        p2 : int
            Degree of the B-splines in the e2 direction.

        p3 : int
            Degree of the B-splines in the e3 direction.

        GLweights : 3d float array
            Tensor product of the Gauss-Legandre quadrature weights for the intergrals in the e1, e2 and e3 direction.
        
        f_eval : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.
    '''

    shp = zeros(3, dtype=int)
    shp[:] = shape(f_eval)

    for i in range(shp[0]):
        for j in range(shp[1]):
            for k in range(shp[2]):
                # The following loop must be tantamount to:
                # f_eval[i,j,k] = np.sum(np.multiply(faux[i*self._p[0]:(i+1)*self._p[0],j*self._p[1]:(j+1)*self._p[1],k*self._p[2]:(k+1)*self._p[2]],self._GLweights))
                in_start_1 = i*p1
                in_start_2 = j*p2
                in_start_3 = k*p3
                for ii in range(p1):
                    for jj in range(p2):
                        for kk in range(p3):
                            f_eval[i, j, k] += faux[in_start_1+ii, in_start_2 +
                                                    jj, in_start_3+kk] * GLweights[ii, jj, kk]


# We need a functions that tell us which of the quasi-interpolation points to take for a any given i
def select_quasi_points(i: int, p: int, Nbasis: int, periodic: bool):
    '''Determines the start and end indices of the quasi-interpolation points that must be taken to get the ith FEEC coefficient for local commuting projectors. 

    Parameters
    ----------
    i : int
        Index of the FEEC coefficient that must be computed.

    p : int
        B-spline degree.

    Nbasis: int
        Number of B-spline.

    periodic: bool
        Whether we have periodic boundary conditions.

    Returns
    -------
    offset : int
        Start index of the quasi-interpolation points that must be consider to obtain the ith FEEC coefficient.

    2*p-1+offset : int
        End index of the quasi-interpolation points that must be consider to obtain the ith FEEC coefficient.
    '''
    if periodic:
        return 2*i, int(2*p)-1+2*i
    else:
        # We need the number of elements n, to compute it we substract the B-spline degree from the number of B-splines.
        n = Nbasis-p
        if i >= 0 and i < p-1:
            offset = 0
        elif i >= p-1 and i <= n:
            offset = int(2*(i-p+1))
        elif i > n and i <= n+p-1:
            offset = int(2*(n-p+1))
        # else:
            # raise Exception("index i must be between 0 and n+p-1")

        return offset, int(2*p)-1+offset


def solve_local_0_form(
    original_size1: int, original_size2: int, original_size3: int, index_translation1: 'int[:]', index_translation2: 'int[:]', index_translation3: 'int[:]', starts: 'int[:]', ends: 'int[:]', pds: 'int[:]', npts: 'int[:]', periodic: 'bool[:]',
    p1: int, p2: int, p3: int, wij0: 'float[:,:]', wij1: 'float[:,:]', wij2: 'float[:,:]', rhs: 'float[:,:,:]', out: 'float[:,:,:]'
):
    '''Kernel for obtaining the FEEC coefficients of zero forms with local projectors.

    Parameters
    ----------
        original_size1: int
            Number of total quasi-interpolation points (or quasi-histopolation intervals) in the e1 direction

        original_size2: int
            Number of total quasi-interpolation points (or quasi-histopolation intervals) in the e2 direction

        original_size3: int
            Number of total quasi-interpolation points (or quasi-histopolation intervals) in the e3 direction

        index_translation1: 1d int array
            Array which translates for the e1 direction from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translation1[index_global]

        index_translation2: 1d int array
            Array which translates for the e2 direction from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translation2[index_global]

        index_translation3: 1d int array
            Array which translates for the e3 direction from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translation3[index_global]

        starts: 1d int array
            Array with the StencilVector start indices for each MPI rank.

        ends: 1d int array
            Array with the StencilVector end indices for each MPI rank.

        pds: 1d int array
            Array with the StencilVector pads for each MPI rank.

        npts: 1d int array
            Array with the number of elements the coefficient vector has in each dimension.

        periodic: 1d bool array
            Array that tell us if the splines are periodic or clamped in each dimension.

        p1 : int
            Degree of the B-splines in the e1 direction.

        p2 : int
            Degree of the B-splines in the e2 direction.

        p3 : int
            Degree of the B-splines in the e3 direction.

        wij0: 2d float array
            Array with the inverse values of the local collocation matrix for the first direction.

        wij1: 2d float array
            Array with the inverse values of the local collocation matrix for the second direction.

        wij2: 2d float array
            Array with the inverse values of the local collocation matrix for the third direction.

        rhs : 3d float array
            Array with the evaluated degrees of freedom.
        out : 3d float array
            Array of FEEC coefficients for the 0-form function.
    '''
    # We iterate over all the entries that belong to the current rank
    counteri0 = 0
    for i0 in range(starts[0], ends[0]+1):
        counteri1 = 0
        for i1 in range(starts[1], ends[1]+1):
            counteri2 = 0
            for i2 in range(starts[2], ends[2]+1):
                L123 = 0.0
                startj1, endj1 = select_quasi_points(
                    i0, p1, npts[0], periodic[0])
                startj2, endj2 = select_quasi_points(
                    i1, p2, npts[1], periodic[1])
                startj3, endj3 = select_quasi_points(
                    i2, p3, npts[2], periodic[2])
                for j1 in range(2*p1-1):
                    # position 1 to evaluate rhs. The module is only necessary for periodic boundary conditions. But it does not hurt the clamped boundary conditions so we just leave it as is to avoid an extra if.
                    if (startj1+j1 < original_size1):
                        pos1 = index_translation1[startj1+j1]
                    else:
                        pos1 = index_translation1[int(startj1+j1 - 2*npts[0])]
                    auxL2 = 0.0
                    for j2 in range(2*p2-1):
                        # position 2 to evaluate rhs
                        if (startj2+j2 < original_size2):
                            pos2 = index_translation2[startj2+j2]
                        else:
                            pos2 = index_translation2[int(
                                startj2+j2 - 2*npts[1])]
                        auxL3 = 0.0
                        for j3 in range(2*p3-1):
                            # position 3 to evaluate rhs
                            if (startj3+j3 < original_size3):
                                pos3 = index_translation3[startj3+j3]
                            else:
                                pos3 = index_translation3[int(
                                    startj3+j3 - 2*npts[2])]
                            auxL3 += wij2[i2][j3]*rhs[pos1, pos2, pos3]
                        auxL2 += wij1[i1][j2]*auxL3
                    L123 += wij0[i0][j1]*auxL2
                out[pds[0]+counteri0, pds[1]+counteri1, pds[2]+counteri2] = L123
                counteri2 += 1
            counteri1 += 1
        counteri0 += 1


def solve_local_1_form(original_pts_sizex: 'int[:]', original_pts_sizey: 'int[:]', original_pts_sizez: 'int[:]', index_translationx0: 'int[:]', index_translationx1: 'int[:]', index_translationx2: 'int[:]', index_translationy0: 'int[:]', index_translationy1: 'int[:]', index_translationy2: 'int[:]', index_translationz0: 'int[:]', index_translationz1: 'int[:]', index_translationz2: 'int[:]', nsp: int, starts: 'int[:,:]', ends: 'int[:,:]', pds: 'int[:,:]', npts: 'int[:,:]', periodic: 'bool[:,:]',
                       p1: int, p2: int, p3: int, wij0: 'float[:,:]', wij1: 'float[:,:]', wij2: 'float[:,:]',  whij0: 'float[:,:]', whij1: 'float[:,:]', whij2: 'float[:,:]', rhs0: 'float[:,:,:]', rhs1: 'float[:,:,:]', rhs2: 'float[:,:,:]', out0: 'float[:,:,:]', out1: 'float[:,:,:]', out2: 'float[:,:,:]'):
    '''Kernel for obtaining the FEEC coefficients of one forms with local projectors.

    Parameters
    ----------
        original_pts_sizex: 1d int array
            Number of total quasi-interpolation points (or quasi-histopolation intervals) in the e1,e2 and e3 direction for the x-component of the BlockVector

        original_pts_sizey: 1d int array
            Number of total quasi-interpolation points (or quasi-histopolation intervals) in the e1,e2 and e3 direction for the y-component of the BlockVector

        original_pts_sizez: 1d int array
            Number of total quasi-interpolation points (or quasi-histopolation intervals) in the e1,e2 and e3 direction for the z-component of the BlockVector

        index_translationx0: 1d int array
            For the x component of the BlockVector this array translates for the e1 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationx0[index_global]

        index_translationx1: 1d int array
            For the x component of the BlockVector this array translates for the e2 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationx1[index_global]

        index_translationx2: 1d int array
            For the x component of the BlockVector this array translates for the e3 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationx2[index_global]

        index_translationy0: 1d int array
            For the y component of the BlockVector this array translates for the e1 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationy0[index_global]

        index_translationy1: 1d int array
            For the y component of the BlockVector this array translates for the e2 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationy1[index_global]

        index_translationy2: 1d int array
            For the y component of the BlockVector this array translates for the e3 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationy2[index_global]

        index_translationz0: 1d int array
            For the z component of the BlockVector this array translates for the e1 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationz0[index_global]

        index_translationz1: 1d int array
            For the z component of the BlockVector this array translates for the e2 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationz1[index_global]

        index_translationz2: 1d int array
            For the z component of the BlockVector this array translates for the e3 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationz2[index_global]

        nsp: int
            Number of spaces.

        starts: 2d int array
            Array with the BlockVector start indices for each MPI rank.

        ends: 2d int array
            Array with the BlockVector end indices for each MPI rank.

        pds: 2d int array
            Array with the BlockVector pads for each MPI rank.

        npts: 2d int array
            Array with the number of elements the coefficient vector has in each dimension.

        periodic: 2d bool array
            Array that tell us if the splines are periodic or clamped in each dimension.

        p1 : int
            Degree of the B-splines in the e1 direction.

        p2 : int
            Degree of the B-splines in the e2 direction.

        p3 : int
            Degree of the B-splines in the e3 direction.

        wij0: 2d float array
            Array with the inverse values of the local collocation matrix for the first direction.

        wij1: 2d float array
            Array with the inverse values of the local collocation matrix for the second direction.

        wij2: 2d float array
            Array with the inverse values of the local collocation matrix for the third direction.

        whij0: 2d float array
            Array with the histopolation geometric weights for the first direction.

        whij1: 2d float array
            Array with the histopolation geometric weights for the second direction.

        whij2: 2d float array
            Array with the histopolation geometric weights for the third direction.

        rhs0 : 3d float array
            Array with the evaluated degrees of freedom for the first StencilVector.

        rhs1 : 3d float array
            Array with the evaluated degrees of freedom for the second StencilVector.

        rhs2 : 3d float array
            Array with the evaluated degrees of freedom for the third StencilVector.    

        out0 : 3d float array
            Array of FEEC coefficients for the first component of the 1-form function.

        out1 : 3d float array
            Array of FEEC coefficients for the second component of the 1-form function.

        out2 : 3d float array
            Array of FEEC coefficients for the third component of the 1-form function.
    '''
    # We iterate over the stencil vectors inside the BlockVector
    for h in range(nsp):
        # We need to know the number of iterrations to be done by the j1, j2 and j3 loops. Since they change depending on whether we have interpolation or histopolation they will be different for each h.
        if (h == 0):
            lenj1 = 2*p1
            lenj2 = 2*p2-1
            lenj3 = 2*p3-1

            # We compute the amout by which we must shift the indices to loop around the quasi-points.
            # We do it only for histopolation for it is the only case in which we might have two different values dependign on the situation.
            if (p1 == 1 and npts[1][0] != 1):
                shift1 = - 2*npts[1][0] + 1
            else:
                shift1 = - 2*npts[1][0]

            # We iterate over all the entries that belong to the current rank
            counteri0 = 0
            for i0 in range(starts[h][0], ends[h][0]+1):
                counteri1 = 0
                for i1 in range(starts[h][1], ends[h][1]+1):
                    counteri2 = 0
                    for i2 in range(starts[h][2], ends[h][2]+1):
                        L123 = 0.0
                        # For the third input I need the number of B-splines
                        startj1, endj1 = select_quasi_points(
                            i0, p1, npts[1][0], periodic[0][0])
                        startj2, endj2 = select_quasi_points(
                            i1, p2, npts[0][1], periodic[0][1])
                        startj3, endj3 = select_quasi_points(
                            i2, p3, npts[0][2], periodic[0][2])

                        for j1 in range(lenj1):
                            # position 1 to evaluate rhs
                            if (startj1+j1 < original_pts_sizex[0]):
                                pos1 = index_translationx0[startj1+j1]
                            else:
                                pos1 = index_translationx0[int(
                                    startj1+j1 + shift1)]
                            if (whij0[i0][j1] != 0.0):
                                auxL2 = 0.0
                                for j2 in range(lenj2):
                                    # position 2 to evaluate rhs
                                    if (startj2+j2 < original_pts_sizex[1]):
                                        pos2 = index_translationx1[startj2+j2]
                                    else:
                                        pos2 = index_translationx1[int(
                                            startj2+j2 - 2*npts[0][1])]
                                    auxL3 = 0.0
                                    for j3 in range(lenj3):
                                        # position 3 to evaluate rhs
                                        if (startj3+j3 < original_pts_sizex[2]):
                                            pos3 = index_translationx2[startj3+j3]
                                        else:
                                            pos3 = index_translationx2[int(
                                                startj3+j3 - 2*npts[0][2])]
                                        auxL3 += wij2[i2][j3] * \
                                            rhs0[pos1, pos2, pos3]
                                    auxL2 += wij1[i1][j2]*auxL3
                                L123 += whij0[i0][j1]*auxL2
                        out0[pds[h][0]+counteri0, pds[h][1] +
                             counteri1, pds[h][2]+counteri2] = L123
                        counteri2 += 1
                    counteri1 += 1
                counteri0 += 1

        elif (h == 1):
            lenj1 = 2*p1-1
            lenj2 = 2*p2
            lenj3 = 2*p3-1

            # We compute the amout by which we must shift the indices to loop around the quasi-points.
            # We do it only for histopolation for it is the only case in which we might have two different values dependign on the situation.
            if (p2 == 1 and npts[0][1] != 1):
                shift2 = - 2*npts[0][1] + 1
            else:
                shift2 = - 2*npts[0][1]

            # We iterate over all the entries that belong to the current rank
            counteri0 = 0
            for i0 in range(starts[h][0], ends[h][0]+1):
                counteri1 = 0
                for i1 in range(starts[h][1], ends[h][1]+1):
                    counteri2 = 0
                    for i2 in range(starts[h][2], ends[h][2]+1):
                        L123 = 0.0
                        # For the third input I need the number of B-splines
                        startj1, endj1 = select_quasi_points(
                            i0, p1, npts[1][0], periodic[0][0])
                        startj2, endj2 = select_quasi_points(
                            i1, p2, npts[0][1], periodic[0][1])
                        startj3, endj3 = select_quasi_points(
                            i2, p3, npts[0][2], periodic[0][2])
                        for j1 in range(lenj1):
                            # position 1 to evaluate rhs
                            if (startj1+j1 < original_pts_sizey[0]):
                                pos1 = index_translationy0[startj1+j1]
                            else:
                                pos1 = index_translationy0[int(
                                    startj1+j1 - 2 * npts[1][0])]
                            auxL2 = 0.0
                            for j2 in range(lenj2):
                                # position 2 to evaluate rhs
                                if (startj2+j2 < original_pts_sizey[1]):
                                    pos2 = index_translationy1[startj2+j2]
                                else:
                                    pos2 = index_translationy1[int(
                                        startj2+j2 + shift2)]
                                if (whij1[i1][j2] != 0.0):
                                    auxL3 = 0.0
                                    for j3 in range(lenj3):
                                        # position 3 to evaluate rhs
                                        if (startj3+j3 < original_pts_sizey[2]):
                                            pos3 = index_translationy2[startj3+j3]
                                        else:
                                            pos3 = index_translationy2[int(
                                                startj3+j3 - 2*npts[0][2])]
                                        auxL3 += wij2[i2][j3] * \
                                            rhs1[pos1, pos2, pos3]
                                    auxL2 += whij1[i1][j2]*auxL3
                            L123 += wij0[i0][j1]*auxL2
                        out1[pds[h][0]+counteri0, pds[h][1] +
                             counteri1, pds[h][2]+counteri2] = L123
                        counteri2 += 1
                    counteri1 += 1
                counteri0 += 1

        elif (h == 2):
            lenj1 = 2*p1-1
            lenj2 = 2*p2-1
            lenj3 = 2*p3

            # We compute the amout by which we must shift the indices to loop around the quasi-points.
            # We do it only for histopolation for it is the only case in which we might have two different values dependign on the situation.
            if (p3 == 1 and npts[0][2] != 1):
                shift3 = - 2*npts[0][2] + 1
            else:
                shift3 = - 2*npts[0][2]

            # We iterate over all the entries that belong to the current rank
            counteri0 = 0
            for i0 in range(starts[h][0], ends[h][0]+1):
                counteri1 = 0
                for i1 in range(starts[h][1], ends[h][1]+1):
                    counteri2 = 0
                    for i2 in range(starts[h][2], ends[h][2]+1):
                        L123 = 0.0
                        # For the third input I need the number of B-splines
                        startj1, endj1 = select_quasi_points(
                            i0, p1, npts[1][0], periodic[0][0])
                        startj2, endj2 = select_quasi_points(
                            i1, p2, npts[0][1], periodic[0][1])
                        startj3, endj3 = select_quasi_points(
                            i2, p3, npts[0][2], periodic[0][2])
                        for j1 in range(lenj1):
                            # position 1 to evaluate rhs
                            if (startj1+j1 < original_pts_sizez[0]):
                                pos1 = index_translationz0[startj1+j1]
                            else:
                                pos1 = index_translationz0[int(
                                    startj1+j1 - 2*npts[1][0])]
                            auxL2 = 0.0
                            for j2 in range(lenj2):
                                # position 2 to evaluate rhs
                                if (startj2+j2 < original_pts_sizez[1]):
                                    pos2 = index_translationz1[startj2+j2]
                                else:
                                    pos2 = index_translationz1[int(
                                        startj2+j2 - 2*npts[0][1])]
                                auxL3 = 0.0
                                for j3 in range(lenj3):
                                    # position 3 to evaluate rhs
                                    if (startj3+j3 < original_pts_sizez[2]):
                                        pos3 = index_translationz2[startj3+j3]
                                    else:
                                        pos3 = index_translationz2[int(
                                            startj3+j3 + shift3)]
                                    if (whij2[i2][j3] != 0.0):
                                        auxL3 += whij2[i2][j3] * \
                                            rhs2[pos1, pos2, pos3]
                                auxL2 += wij1[i1][j2]*auxL3
                            L123 += wij0[i0][j1]*auxL2
                        out2[pds[h][0]+counteri0, pds[h][1] +
                             counteri1, pds[h][2]+counteri2] = L123
                        counteri2 += 1
                    counteri1 += 1
                counteri0 += 1


def solve_local_2_form(original_pts_sizex: 'int[:]', original_pts_sizey: 'int[:]', original_pts_sizez: 'int[:]', index_translationx0: 'int[:]', index_translationx1: 'int[:]', index_translationx2: 'int[:]', index_translationy0: 'int[:]', index_translationy1: 'int[:]', index_translationy2: 'int[:]', index_translationz0: 'int[:]', index_translationz1: 'int[:]', index_translationz2: 'int[:]', nsp: int, starts: 'int[:,:]', ends: 'int[:,:]', pds: 'int[:,:]', npts: 'int[:,:]', periodic: 'bool[:,:]',
                       p1: int, p2: int, p3: int, wij0: 'float[:,:]', wij1: 'float[:,:]', wij2: 'float[:,:]',  whij0: 'float[:,:]', whij1: 'float[:,:]', whij2: 'float[:,:]', rhs0: 'float[:,:,:]', rhs1: 'float[:,:,:]', rhs2: 'float[:,:,:]', out0: 'float[:,:,:]', out1: 'float[:,:,:]', out2: 'float[:,:,:]'):
    '''Kernel for obtaining the FEEC coefficients of 2-forms with local projectors.

    Parameters
    ----------
        original_pts_sizex: 1d int array
            Number of total quasi-interpolation points (or quasi-histopolation intervals) in the e1,e2 and e3 direction for the x-component of the BlockVector

        original_pts_sizey: 1d int array
            Number of total quasi-interpolation points (or quasi-histopolation intervals) in the e1,e2 and e3 direction for the y-component of the BlockVector

        original_pts_sizez: 1d int array
            Number of total quasi-interpolation points (or quasi-histopolation intervals) in the e1,e2 and e3 direction for the z-component of the BlockVector

        index_translationx0: 1d int array
            For the x component of the BlockVector this array translates for the e1 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationx0[index_global]

        index_translationx1: 1d int array
            For the x component of the BlockVector this array translates for the e2 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationx1[index_global]

        index_translationx2: 1d int array
            For the x component of the BlockVector this array translates for the e3 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationx2[index_global]

        index_translationy0: 1d int array
            For the y component of the BlockVector this array translates for the e1 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationy0[index_global]

        index_translationy1: 1d int array
            For the y component of the BlockVector this array translates for the e2 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationy1[index_global]

        index_translationy2: 1d int array
            For the y component of the BlockVector this array translates for the e3 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationy2[index_global]

        index_translationz0: 1d int array
            For the z component of the BlockVector this array translates for the e1 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationz0[index_global]

        index_translationz1: 1d int array
            For the z component of the BlockVector this array translates for the e2 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationz1[index_global]

        index_translationz2: 1d int array
            For the z component of the BlockVector this array translates for the e3 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationz2[index_global]

        nsp: int
            Number of spaces.

        starts: 2d int array
            Array with the BlockVector start indices for each MPI rank.

        ends: 2d int array
            Array with the BlockVector end indices for each MPI rank.

        pds: 2d int array
            Array with the BlockVector pads for each MPI rank.

        npts: 2d int array
            Array with the number of elements the coefficient vector has in each dimension.

        periodic: 2d bool array
            Array that tell us if the splines are periodic or clamped in each dimension.

        p1 : int
            Degree of the B-splines in the e1 direction.

        p2 : int
            Degree of the B-splines in the e2 direction.

        p3 : int
            Degree of the B-splines in the e3 direction.

        wij0: 2d float array
            Array with the inverse values of the local collocation matrix for the first direction.

        wij1: 2d float array
            Array with the inverse values of the local collocation matrix for the second direction.

        wij2: 2d float array
            Array with the inverse values of the local collocation matrix for the third direction.

        whij0: 2d float array
            Array with the histopolation geometric weights for the first direction.

        whij1: 2d float array
            Array with the histopolation geometric weights for the second direction.

        whij2: 2d float array
            Array with the histopolation geometric weights for the third direction.

        rhs0 : 3d float array
            Array with the evaluated degrees of freedom for the first StencilVector.

        rhs1 : 3d float array
            Array with the evaluated degrees of freedom for the second StencilVector.

        rhs2 : 3d float array
            Array with the evaluated degrees of freedom for the third StencilVector.    

        out0 : 3d float array
            Array of FEEC coefficients for the first component of the 2-form function.

        out1 : 3d float array
            Array of FEEC coefficients for the second component of the 2-form function.

        out2 : 3d float array
            Array of FEEC coefficients for the third component of the 2-form function.
    '''

    # We iterate over the stencil vectors inside the BlockVector
    for h in range(nsp):
        # We need to know the number of iterrations to be done by the j1, j2 and j3 loops. Since they change depending on whether we have interpolation or histopolation they will be different for each h.
        if (h == 0):
            lenj1 = 2*p1-1
            lenj2 = 2*p2
            lenj3 = 2*p3

            # We compute the amout by which we must shift the indices to loop around the quasi-points.
            # We do it only for histopolation for it is the only case in which we might have two different values dependign on the situation.
            if (p2 == 1 and npts[1][1] != 1):
                shift2 = - 2*npts[1][1] + 1
            else:
                shift2 = - 2*npts[1][1]

            if (p3 == 1 and npts[2][2] != 1):
                shift3 = - 2*npts[2][2] + 1
            else:
                shift3 = - 2*npts[2][2]
            # We iterate over all the entries that belong to the current rank
            counteri0 = 0
            for i0 in range(starts[h][0], ends[h][0]+1):
                counteri1 = 0
                for i1 in range(starts[h][1], ends[h][1]+1):
                    counteri2 = 0
                    for i2 in range(starts[h][2], ends[h][2]+1):
                        L123 = 0.0
                        # For the third input I need the number of B-splines
                        startj1, endj1 = select_quasi_points(
                            i0, p1, npts[0][0], periodic[0][0])
                        startj2, endj2 = select_quasi_points(
                            i1, p2, npts[1][1], periodic[0][1])
                        startj3, endj3 = select_quasi_points(
                            i2, p3, npts[2][2], periodic[0][2])

                        for j1 in range(lenj1):
                            # position 1 to evaluate rhs
                            if (startj1+j1 < original_pts_sizex[0]):
                                pos1 = index_translationx0[startj1+j1]
                            else:
                                pos1 = index_translationx0[int(
                                    startj1+j1 - 2*npts[0][0])]
                            auxL2 = 0.0
                            for j2 in range(lenj2):
                                # position 2 to evaluate rhs
                                if (startj2+j2 < original_pts_sizex[1]):
                                    pos2 = index_translationx1[startj2+j2]
                                else:
                                    pos2 = index_translationx1[int(
                                        startj2+j2 + shift2)]
                                if (whij1[i1][j2] != 0.0):
                                    auxL3 = 0.0
                                    for j3 in range(lenj3):
                                        # position 3 to evaluate rhs
                                        if (startj3+j3 < original_pts_sizex[2]):
                                            pos3 = index_translationx2[startj3+j3]
                                        else:
                                            pos3 = index_translationx2[int(
                                                startj3+j3 + shift3)]
                                        if (whij2[i2][j3] != 0.0):
                                            auxL3 += whij2[i2][j3] * \
                                                rhs0[pos1, pos2, pos3]
                                    auxL2 += whij1[i1][j2]*auxL3
                            L123 += wij0[i0][j1]*auxL2
                        out0[pds[h][0]+counteri0, pds[h][1] +
                             counteri1, pds[h][2]+counteri2] = L123
                        counteri2 += 1
                    counteri1 += 1
                counteri0 += 1

        elif (h == 1):
            lenj1 = 2*p1
            lenj2 = 2*p2-1
            lenj3 = 2*p3

            # We compute the amout by which we must shift the indices to loop around the quasi-points.
            # We do it only for histopolation for it is the only case in which we might have two different values dependign on the situation.
            if (p1 == 1 and npts[0][0] != 1):
                shift1 = - 2*npts[0][0] + 1
            else:
                shift1 = - 2*npts[0][0]

            if (p3 == 1 and npts[2][2] != 1):
                shift3 = - 2*npts[2][2] + 1
            else:
                shift3 = - 2*npts[2][2]

            # We iterate over all the entries that belong to the current rank
            counteri0 = 0
            for i0 in range(starts[h][0], ends[h][0]+1):
                counteri1 = 0
                for i1 in range(starts[h][1], ends[h][1]+1):
                    counteri2 = 0
                    for i2 in range(starts[h][2], ends[h][2]+1):
                        L123 = 0.0
                        # For the third input I need the number of B-splines
                        startj1, endj1 = select_quasi_points(
                            i0, p1, npts[0][0], periodic[0][0])
                        startj2, endj2 = select_quasi_points(
                            i1, p2, npts[1][1], periodic[0][1])
                        startj3, endj3 = select_quasi_points(
                            i2, p3, npts[2][2], periodic[0][2])
                        for j1 in range(lenj1):
                            # position 1 to evaluate rhs
                            if (startj1+j1 < original_pts_sizey[0]):
                                pos1 = index_translationy0[startj1+j1]
                            else:
                                pos1 = index_translationy0[int(
                                    startj1+j1 + shift1)]
                            if (whij0[i0][j1] != 0.0):
                                auxL2 = 0.0
                                for j2 in range(lenj2):
                                    # position 2 to evaluate rhs
                                    if (startj2+j2 < original_pts_sizey[1]):
                                        pos2 = index_translationy1[startj2+j2]
                                    else:
                                        pos2 = index_translationy1[int(
                                            startj2+j2 - 2*npts[1][1])]
                                    auxL3 = 0.0
                                    for j3 in range(lenj3):
                                        # position 3 to evaluate rhs
                                        if (startj3+j3 < original_pts_sizey[2]):
                                            pos3 = index_translationy2[startj3+j3]
                                        else:
                                            pos3 = index_translationy2[int(
                                                startj3+j3 + shift3)]
                                        if (whij2[i2][j3] != 0.0):
                                            auxL3 += whij2[i2][j3] * \
                                                rhs1[pos1, pos2, pos3]
                                    auxL2 += wij1[i1][j2]*auxL3
                                L123 += whij0[i0][j1]*auxL2
                        out1[pds[h][0]+counteri0, pds[h][1] +
                             counteri1, pds[h][2]+counteri2] = L123
                        counteri2 += 1
                    counteri1 += 1
                counteri0 += 1

        elif (h == 2):
            lenj1 = 2*p1
            lenj2 = 2*p2
            lenj3 = 2*p3-1

            # We compute the amout by which we must shift the indices to loop around the quasi-points.
            # We do it only for histopolation for it is the only case in which we might have two different values dependign on the situation.
            if (p1 == 1 and npts[0][0] != 1):
                shift1 = - 2*npts[0][0] + 1
            else:
                shift1 = - 2*npts[0][0]

            if (p2 == 1 and npts[1][1] != 1):
                shift2 = - 2*npts[1][1] + 1
            else:
                shift2 = - 2*npts[1][1]

            # We iterate over all the entries that belong to the current rank
            counteri0 = 0
            for i0 in range(starts[h][0], ends[h][0]+1):
                counteri1 = 0
                for i1 in range(starts[h][1], ends[h][1]+1):
                    counteri2 = 0
                    for i2 in range(starts[h][2], ends[h][2]+1):
                        L123 = 0.0
                        # For the third input I need the number of B-splines
                        startj1, endj1 = select_quasi_points(
                            i0, p1, npts[0][0], periodic[0][0])
                        startj2, endj2 = select_quasi_points(
                            i1, p2, npts[1][1], periodic[0][1])
                        startj3, endj3 = select_quasi_points(
                            i2, p3, npts[2][2], periodic[0][2])
                        for j1 in range(lenj1):
                            # position 1 to evaluate rhs
                            if (startj1+j1 < original_pts_sizez[0]):
                                pos1 = index_translationz0[startj1+j1]
                            else:
                                pos1 = index_translationz0[int(
                                    startj1+j1 + shift1)]
                            if (whij0[i0][j1] != 0.0):
                                auxL2 = 0.0
                                for j2 in range(lenj2):
                                    # position 2 to evaluate rhs
                                    if (startj2+j2 < original_pts_sizez[1]):
                                        pos2 = index_translationz1[startj2+j2]
                                    else:
                                        pos2 = index_translationz1[int(
                                            startj2+j2 + shift2)]
                                    if (whij1[i1][j2] != 0.0):
                                        auxL3 = 0.0
                                        for j3 in range(lenj3):
                                            # position 3 to evaluate rhs
                                            if (startj3+j3 < original_pts_sizez[2]):
                                                pos3 = index_translationz2[startj3+j3]
                                            else:
                                                pos3 = index_translationz2[int(
                                                    startj3+j3 - 2*npts[2][2])]
                                            auxL3 += wij2[i2][j3] * \
                                                rhs2[pos1, pos2, pos3]
                                        auxL2 += whij1[i1][j2]*auxL3
                                L123 += whij0[i0][j1]*auxL2
                        out2[pds[h][0]+counteri0, pds[h][1] +
                             counteri1, pds[h][2]+counteri2] = L123
                        counteri2 += 1
                    counteri1 += 1
                counteri0 += 1


def solve_local_3_form(
    original_size1: int, original_size2: int, original_size3: int, index_translation1: 'int[:]', index_translation2: 'int[:]', index_translation3: 'int[:]', starts: 'int[:]', ends: 'int[:]', pds: 'int[:]', npts: 'int[:]', periodic: 'bool[:]',
    p1: int, p2: int, p3: int, whij0: 'float[:,:]', whij1: 'float[:,:]', whij2: 'float[:,:]', rhs: 'float[:,:,:]', out: 'float[:,:,:]'
):
    '''Kernel for obtaining the FEEC coefficients of three forms with local projectors.

    Parameters
    ----------
        original_size1: int
            Number of total quasi-interpolation points (or quasi-histopolation intervals) in the e1 direction

        original_size2: int
            Number of total quasi-interpolation points (or quasi-histopolation intervals) in the e2 direction

        original_size3: int
            Number of total quasi-interpolation points (or quasi-histopolation intervals) in the e3 direction

        index_translation1: 1d int array
            Array which translates for the e1 direction from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translation1[index_global]

        index_translation2: 1d int array
            Array which translates for the e2 direction from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translation2[index_global]

        index_translation3: 1d int array
            Array which translates for the e3 direction from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translation3[index_global]

        starts: 1d int array
            Array with the StencilVector start indices for each MPI rank.

        ends: 1d int array
            Array with the StencilVector end indices for each MPI rank.

        pds: 1d int array
            Array with the StencilVector pads for each MPI rank.

        npts: 1d int array
            Array with the number of elements the coefficient vector has in each dimension.

        periodic: 1d bool array
            Array that tell us if the splines are periodic or clamped in each dimension.

        p1 : int
            Degree of the B-splines in the e1 direction.

        p2 : int
            Degree of the B-splines in the e2 direction.

        p3 : int
            Degree of the B-splines in the e3 direction.

        whij0: 3d float array
            Array with the weights by which the degrees of freedom must be multiplies to obtain the FE coefficients for the e1 direction.

        whij1: 3d float array
            Array with the weights by which the degrees of freedom must be multiplies to obtain the FE coefficients for the e2 direction.

        whij2: 3d float array
            Array with the weights by which the degrees of freedom must be multiplies to obtain the FE coefficients for the e3 direction.

        rhs : 3d float array
            Array with the evaluated degrees of freedom.

        out : 3d float array
            Array of FEEC coefficients for the 3-form function.
    '''

    # We get the number of B-Splines
    if (periodic[0]):
        NB0 = npts[0]
    else:
        NB0 = npts[0]+1
    if (periodic[1]):
        NB1 = npts[1]
    else:
        NB1 = npts[1]+1
    if (periodic[2]):
        NB2 = npts[2]
    else:
        NB2 = npts[2]+1

    # We compute the amout by which we must shift the indices to loop around the quasi-points.
    # We do it only for histopolation for it is the only case in which we might have two different values dependign on the situation.
    if (p1 == 1 and NB0 != 1):
        shift1 = - 2*NB0 + 1
    else:
        shift1 = - 2*NB0

    if (p2 == 1 and NB1 != 1):
        shift2 = - 2*NB1 + 1
    else:
        shift2 = - 2*NB1

    if (p3 == 1 and NB2 != 1):
        shift3 = - 2*NB2 + 1
    else:
        shift3 = - 2*NB2

    # We iterate over all the entries that belong to the current rank
    counteri0 = 0
    for i0 in range(starts[0], ends[0]+1):
        counteri1 = 0
        for i1 in range(starts[1], ends[1]+1):
            counteri2 = 0
            for i2 in range(starts[2], ends[2]+1):
                L123 = 0.0
                startj1, endj1 = select_quasi_points(
                    i0, p1, npts[0]+1, periodic[0])
                startj2, endj2 = select_quasi_points(
                    i1, p2, npts[1]+1, periodic[1])
                startj3, endj3 = select_quasi_points(
                    i2, p3, npts[2]+1, periodic[2])
                for j1 in range(2*p1):
                    # position 1 to evaluate rhs
                    if (startj1+j1 < original_size1):
                        pos1 = index_translation1[startj1+j1]
                    else:
                        pos1 = index_translation1[int(startj1+j1 + shift1)]
                    if (whij0[i0][j1] != 0.0):
                        auxL2 = 0.0
                        for j2 in range(2*p2):
                            # position 2 to evaluate rhs
                            if (startj2+j2 < original_size2):
                                pos2 = index_translation2[startj2+j2]
                            else:
                                pos2 = index_translation2[int(
                                    startj2+j2 + shift2)]
                            if (whij1[i1][j2] != 0.0):
                                auxL3 = 0.0
                                for j3 in range(2*p3):
                                    # position 3 to evaluate rhs
                                    if (startj3+j3 < original_size3):
                                        pos3 = index_translation3[startj3+j3]
                                    else:
                                        pos3 = index_translation3[int(
                                            startj3+j3 + shift3)]
                                    if (whij2[i2][j3] != 0.0):
                                        auxL3 += whij2[i2][j3] * \
                                            rhs[pos1, pos2, pos3]
                                auxL2 += whij1[i1][j2]*auxL3
                        L123 += whij0[i0][j1]*auxL2
                out[pds[0]+counteri0, pds[1]+counteri1, pds[2]+counteri2] = L123
                counteri2 += 1
            counteri1 += 1
        counteri0 += 1


def solve_local_0V_form(original_pts_sizex: 'int[:]', original_pts_sizey: 'int[:]', original_pts_sizez: 'int[:]', index_translationx0: 'int[:]', index_translationx1: 'int[:]', index_translationx2: 'int[:]', index_translationy0: 'int[:]', index_translationy1: 'int[:]', index_translationy2: 'int[:]', index_translationz0: 'int[:]', index_translationz1: 'int[:]', index_translationz2: 'int[:]', nsp: int, starts: 'int[:,:]', ends: 'int[:,:]', pds: 'int[:,:]', npts: 'int[:,:]', periodic: 'bool[:,:]',
                        p1: int, p2: int, p3: int, wij0: 'float[:,:]', wij1: 'float[:,:]', wij2: 'float[:,:]',  rhs0: 'float[:,:,:]', rhs1: 'float[:,:,:]', rhs2: 'float[:,:,:]', out0: 'float[:,:,:]', out1: 'float[:,:,:]', out2: 'float[:,:,:]'):
    '''Kernel for obtaining the FEEC coefficients of vector 0-forms with local projectors.

    Parameters
    ----------
        original_pts_sizex: 1d int array
            Number of total quasi-interpolation points (or quasi-histopolation intervals) in the e1,e2 and e3 direction for the x-component of the BlockVector

        original_pts_sizey: 1d int array
            Number of total quasi-interpolation points (or quasi-histopolation intervals) in the e1,e2 and e3 direction for the y-component of the BlockVector

        original_pts_sizez: 1d int array
            Number of total quasi-interpolation points (or quasi-histopolation intervals) in the e1,e2 and e3 direction for the z-component of the BlockVector

        index_translationx0: 1d int array
            For the x component of the BlockVector this array translates for the e1 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationx0[index_global]

        index_translationx1: 1d int array
            For the x component of the BlockVector this array translates for the e2 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationx1[index_global]

        index_translationx2: 1d int array
            For the x component of the BlockVector this array translates for the e3 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationx2[index_global]

        index_translationy0: 1d int array
            For the y component of the BlockVector this array translates for the e1 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationy0[index_global]

        index_translationy1: 1d int array
            For the y component of the BlockVector this array translates for the e2 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationy1[index_global]

        index_translationy2: 1d int array
            For the y component of the BlockVector this array translates for the e3 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationy2[index_global]

        index_translationz0: 1d int array
            For the z component of the BlockVector this array translates for the e1 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationz0[index_global]

        index_translationz1: 1d int array
            For the z component of the BlockVector this array translates for the e2 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationz1[index_global]

        index_translationz2: 1d int array
            For the z component of the BlockVector this array translates for the e3 directions from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translationz2[index_global]

        nsp: int
            Number of spaces.

        starts: 2d int array
            Array with the BlockVector start indices for each MPI rank.

        ends: 2d int array
            Array with the BlockVector end indices for each MPI rank.

        pds: 2d int array
            Array with the BlockVector pads for each MPI rank.

        npts: 2d int array
            Array with the number of elements the coefficient vector has in each dimension.

        periodic: 2d bool array
            Array that tell us if the splines are periodic or clamped in each dimension.

        p1 : int
            Degree of the B-splines in the e1 direction.

        p2 : int
            Degree of the B-splines in the e2 direction.

        p3 : int
            Degree of the B-splines in the e3 direction.

        wij0: 2d float array
            Array with the inverse values of the local collocation matrix for the first direction.

        wij1: 2d float array
            Array with the inverse values of the local collocation matrix for the second direction.

        wij2: 2d float array
            Array with the inverse values of the local collocation matrix for the third direction.

        rhs0 : 3d float array
            Array with the evaluated degrees of freedom for the first StencilVector.

        rhs1 : 3d float array
            Array with the evaluated degrees of freedom for the second StencilVector.

        rhs2 : 3d float array
            Array with the evaluated degrees of freedom for the third StencilVector.    

        out0 : 3d float array
            Array of FEEC coefficients for the first component of the 0-form vector function.

        out1 : 3d float array
            Array of FEEC coefficients for the second component of the 0-form vector function.

        out2 : 3d float array
            Array of FEEC coefficients for the third component of the 0-form vector function.
    '''

    lenj1 = 2*p1-1
    lenj2 = 2*p2-1
    lenj3 = 2*p3-1
    # We iterate over the stencil vectors inside the BlockVector
    for h in range(nsp):
        # We need to know the number of iterrations to be done by the j1, j2 and j3 loops. Since they change depending on whether we have interpolation or histopolation they will be different for each h.
        if (h == 0):
            # We iterate over all the entries that belong to the current rank
            counteri0 = 0
            for i0 in range(starts[h][0], ends[h][0]+1):
                counteri1 = 0
                for i1 in range(starts[h][1], ends[h][1]+1):
                    counteri2 = 0
                    for i2 in range(starts[h][2], ends[h][2]+1):
                        L123 = 0.0
                        # For the third input I need the number of B-splines
                        startj1, endj1 = select_quasi_points(
                            i0, p1, npts[0][0], periodic[0][0])
                        startj2, endj2 = select_quasi_points(
                            i1, p2, npts[0][1], periodic[0][1])
                        startj3, endj3 = select_quasi_points(
                            i2, p3, npts[0][2], periodic[0][2])

                        for j1 in range(lenj1):
                            # position 1 to evaluate rhs
                            if (startj1+j1 < original_pts_sizex[0]):
                                pos1 = index_translationx0[startj1+j1]
                            else:
                                pos1 = index_translationx0[int(
                                    startj1+j1 - 2*npts[0][0])]
                            auxL2 = 0.0
                            for j2 in range(lenj2):
                                # position 2 to evaluate rhs
                                if (startj2+j2 < original_pts_sizex[1]):
                                    pos2 = index_translationx1[startj2+j2]
                                else:
                                    pos2 = index_translationx1[int(
                                        startj2+j2 - 2*npts[0][1])]
                                auxL3 = 0.0
                                for j3 in range(lenj3):
                                    # position 3 to evaluate rhs
                                    if (startj3+j3 < original_pts_sizex[2]):
                                        pos3 = index_translationx2[startj3+j3]
                                    else:
                                        pos3 = index_translationx2[int(
                                            startj3+j3 - 2*npts[0][2])]
                                    auxL3 += wij2[i2][j3] * \
                                        rhs0[pos1, pos2, pos3]
                                auxL2 += wij1[i1][j2]*auxL3
                            L123 += wij0[i0][j1]*auxL2
                        out0[pds[h][0]+counteri0, pds[h][1] +
                             counteri1, pds[h][2]+counteri2] = L123
                        counteri2 += 1
                    counteri1 += 1
                counteri0 += 1

        elif (h == 1):
            # We iterate over all the entries that belong to the current rank
            counteri0 = 0
            for i0 in range(starts[h][0], ends[h][0]+1):
                counteri1 = 0
                for i1 in range(starts[h][1], ends[h][1]+1):
                    counteri2 = 0
                    for i2 in range(starts[h][2], ends[h][2]+1):
                        L123 = 0.0
                        # For the third input I need the number of B-splines
                        startj1, endj1 = select_quasi_points(
                            i0, p1, npts[0][0], periodic[0][0])
                        startj2, endj2 = select_quasi_points(
                            i1, p2, npts[0][1], periodic[0][1])
                        startj3, endj3 = select_quasi_points(
                            i2, p3, npts[0][2], periodic[0][2])

                        for j1 in range(lenj1):
                            # position 1 to evaluate rhs
                            if (startj1+j1 < original_pts_sizey[0]):
                                pos1 = index_translationy0[startj1+j1]
                            else:
                                pos1 = index_translationy0[int(
                                    startj1+j1 - 2*npts[0][0])]
                            auxL2 = 0.0
                            for j2 in range(lenj2):
                                # position 2 to evaluate rhs
                                if (startj2+j2 < original_pts_sizey[1]):
                                    pos2 = index_translationy1[startj2+j2]
                                else:
                                    pos2 = index_translationy1[int(
                                        startj2+j2 - 2*npts[0][1])]
                                auxL3 = 0.0
                                for j3 in range(lenj3):
                                    # position 3 to evaluate rhs
                                    if (startj3+j3 < original_pts_sizey[2]):
                                        pos3 = index_translationy2[startj3+j3]
                                    else:
                                        pos3 = index_translationy2[int(
                                            startj3+j3 - 2*npts[0][2])]
                                    auxL3 += wij2[i2][j3] * \
                                        rhs1[pos1, pos2, pos3]
                                auxL2 += wij1[i1][j2]*auxL3
                            L123 += wij0[i0][j1]*auxL2
                        out1[pds[h][0]+counteri0, pds[h][1] +
                             counteri1, pds[h][2]+counteri2] = L123
                        counteri2 += 1
                    counteri1 += 1
                counteri0 += 1

        elif (h == 2):
            # We iterate over all the entries that belong to the current rank
            counteri0 = 0
            for i0 in range(starts[h][0], ends[h][0]+1):
                counteri1 = 0
                for i1 in range(starts[h][1], ends[h][1]+1):
                    counteri2 = 0
                    for i2 in range(starts[h][2], ends[h][2]+1):
                        L123 = 0.0
                        # For the third input I need the number of B-splines
                        startj1, endj1 = select_quasi_points(
                            i0, p1, npts[0][0], periodic[0][0])
                        startj2, endj2 = select_quasi_points(
                            i1, p2, npts[0][1], periodic[0][1])
                        startj3, endj3 = select_quasi_points(
                            i2, p3, npts[0][2], periodic[0][2])

                        for j1 in range(lenj1):
                            # position 1 to evaluate rhs
                            if (startj1+j1 < original_pts_sizez[0]):
                                pos1 = index_translationz0[startj1+j1]
                            else:
                                pos1 = index_translationz0[int(
                                    startj1+j1 - 2*npts[0][0])]
                            auxL2 = 0.0
                            for j2 in range(lenj2):
                                # position 2 to evaluate rhs
                                if (startj2+j2 < original_pts_sizez[1]):
                                    pos2 = index_translationz1[startj2+j2]
                                else:
                                    pos2 = index_translationz1[int(
                                        startj2+j2 - 2*npts[0][1])]
                                auxL3 = 0.0
                                for j3 in range(lenj3):
                                    # position 3 to evaluate rhs
                                    if (startj3+j3 < original_pts_sizez[2]):
                                        pos3 = index_translationz2[startj3+j3]
                                    else:
                                        pos3 = index_translationz2[int(
                                            startj3+j3 - 2*npts[0][2])]
                                    auxL3 += wij2[i2][j3] * \
                                        rhs2[pos1, pos2, pos3]
                                auxL2 += wij1[i1][j2]*auxL3
                            L123 += wij0[i0][j1]*auxL2
                        out2[pds[h][0]+counteri0, pds[h][1] +
                             counteri1, pds[h][2]+counteri2] = L123
                        counteri2 += 1
                    counteri1 += 1
                counteri0 += 1
