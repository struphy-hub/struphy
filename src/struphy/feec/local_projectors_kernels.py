from numpy import shape, zeros
from pyccel.decorators import stack_array

import struphy.kernel_arguments.local_projectors_args_kernels as local_projectors_args_kernels
from struphy.kernel_arguments.local_projectors_args_kernels import LocalProjectorsArguments


def compute_shifts(IoH: "bool[:]", p: "int[:]", B_nbasis: "int[:]", shift: "int[:]"):
    """This function computes by how much we must shift the indices in case we loop over the evaluation points.

    Parameters
    ----------
    IoH : 1d bool array
        Determines if we have interpolation or histopolation in each spatial direction. False means interpolation, True means histopolation.

    p : 1d int array
        array of 3 ints, they denote the degree of the B-splines (not the D-splines) for each one of the three spatial directions.

    B_nbasis: 1d int array
        Array with the number of B-splines in each direction.

    shifts : 1d int array
        array of 3 ints, each one denotes the amout by which we must shift the indices to loop around the quasi-points for each spatial direction.
    """
    for i, ioh in enumerate(IoH):
        # Histopolation
        if ioh:
            if p[i] == 1 and B_nbasis[i] != 1:
                shift[i] = -2 * B_nbasis[i] + 1
            else:
                shift[i] = -2 * B_nbasis[i]
        # Interpolation
        else:
            shift[i] = -2 * B_nbasis[i]


def get_local_problem_size(periodic: "bool[:]", p: "int[:]", IoH: "bool[:]"):
    """Determines the number of interpolation or histopolation weights present for a fixed index i.

    Parameters
    ----------
        periodic: 1d bool array
            Array that tell us if the splines are periodic or clamped in each dimension.

        p : 1d int array
            Degree of the B-splines in each direction.

        IoH : bool array
            Array with three bool entries, one per spatial direction. False means we are dealing with interpolation, True means we are dealing with histopolation.

    Returns
    -------
        lenj[0] : int
            The number of interpolation (or histopolation) weights we have in the first spatial direction for a given value of i.

        lenj[1] : int
            The number of interpolation (or histopolation) weights we have in the second spatial direction for a given value of i.

        lenj[2] : int
            The number of interpolation (or histopolation) weights we have in the third spatial direction for a given value of i.
    """
    lenj = zeros(3, dtype=int)

    for h in range(3):
        # Interpolation
        if not IoH[h]:
            lenj[h] = 2 * p[h] - 1
        # Histopolation
        else:
            if periodic[h]:
                lenj[h] = 2 * p[h]
            else:
                lenj[h] = 4 * p[h] - 4

    return lenj[0], lenj[1], lenj[2]


def get_dofs_local_1_form_ec_component_weighted(
    args_solve: LocalProjectorsArguments,
    fc: "float[:,:,:]",
    basis0: "float[:]",
    basis1: "float[:]",
    basis2: "float[:]",
    arezeroc: "int[:]",
    f_eval_aux: "float[:,:,:]",
    c: int,
):
    """Kernel for evaluating the degrees of freedom for the c-th component of 1-forms. This function is for local commuting projetors.

    Parameters
    ----------
        args_solve : LocalProjectorsArguments
            Class that holds basic local projectors properties as attributes.

        fc : 3d float array
            Evaluation for the c-th component of the 1-form function over all the interpolation points in e_a and e_b (a != b != c), as well as all the Gauss-Legendre quadrature point in e_c.

        basis0 : 1d float array
            Array with the evaluated basis functions for the e1 direction.

        basis1 : 1d float array
            Array with the evaluated basis functions for the e2 direction.

        basis2 : 1d float array
            Array with the evaluated basis functions for the e3 direction.

        arezeroc : 1d int array
            Array of zeros or ones. A one means that for this particular set of quadrature points, in the c-th direction, the basis function is not zero for at least one of them.

        f_eval_aux : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.

        c : int
            This int tell us whichone of the three components of the 1-form vector we are dealing with. It must be 0, 1 or 2.
    """
    p = args_solve.p[c]

    for i in range(shape(f_eval_aux)[0]):
        if c == 0:
            computei = arezeroc[i] != 0
        else:
            computei = abs(basis0[i]) >= 10.0 ** (-16)
        if computei:
            for j in range(shape(f_eval_aux)[1]):
                if c == 1:
                    computej = arezeroc[j] != 0
                else:
                    computej = abs(basis1[j]) >= 10.0 ** (-16)
                if computej:
                    for k in range(shape(f_eval_aux)[2]):
                        if c == 2:
                            computek = arezeroc[k] != 0
                        else:
                            computek = abs(basis2[k]) >= 10.0 ** (-16)
                        if computek:
                            if c == 0:
                                in_start = i * p
                            elif c == 1:
                                in_start = j * p
                            elif c == 2:
                                in_start = k * p
                            for ii in range(p):
                                if c == 0:
                                    f_eval_aux[i, j, k] += (
                                        fc[in_start + ii, j, k]
                                        * basis0[in_start + ii]
                                        * basis1[j]
                                        * basis2[k]
                                        * args_solve.wts0[args_solve.inv_index_translation0[i], ii]
                                    )
                                elif c == 1:
                                    f_eval_aux[i, j, k] += (
                                        fc[i, in_start + ii, k]
                                        * basis0[i]
                                        * basis1[in_start + ii]
                                        * basis2[k]
                                        * args_solve.wts1[args_solve.inv_index_translation1[j], ii]
                                    )
                                elif c == 2:
                                    f_eval_aux[i, j, k] += (
                                        fc[i, j, in_start + ii]
                                        * basis0[i]
                                        * basis1[j]
                                        * basis2[in_start + ii]
                                        * args_solve.wts2[args_solve.inv_index_translation2[k], ii]
                                    )


@stack_array("shp")
def get_dofs_local_1_form_ec_component(
    args_solve: LocalProjectorsArguments,
    f3: "float[:,:,:]",
    f_eval_aux: "float[:,:,:]",
    c: int,
):
    """Kernel for evaluating the degrees of freedom for the c-th component of 1-forms.  This function is for local commuting projetors.

    Parameters
    ----------
        args_solve : LocalProjectorsArguments
            Class that holds basic local projectors properties as attributes.

        f3 : 3d float array
            Evaluation for the c-th component of the 1-form function over all the interpolation points in e_a and e_b (a!=b!=c), as well as all the Gauss-Legendre quadrature point in e_c.

        f_eval_aux : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.

        c : int
            This integer determines which of the three components of the 1-form vector we are working on. Must be 0,1 or 2.
    """

    shp = zeros(3, dtype=int)
    shp[:] = shape(f_eval_aux)

    p = args_solve.p[c]
    if c == 0:
        wts = args_solve.wts0
        inv_index_translation = args_solve.inv_index_translation0
    elif c == 1:
        wts = args_solve.wts1
        inv_index_translation = args_solve.inv_index_translation1
    elif c == 2:
        wts = args_solve.wts2
        inv_index_translation = args_solve.inv_index_translation2

    for i in range(shp[0]):
        for j in range(shp[1]):
            for k in range(shp[2]):
                if c == 0:
                    in_start = i * p
                    for ii in range(p):
                        f_eval_aux[i, j, k] += f3[in_start + ii, j, k] * wts[inv_index_translation[i], ii]
                elif c == 1:
                    in_start = j * p
                    for ii in range(p):
                        f_eval_aux[i, j, k] += f3[i, in_start + ii, k] * wts[inv_index_translation[j], ii]
                elif c == 2:
                    in_start = k * p
                    for ii in range(p):
                        f_eval_aux[i, j, k] += f3[i, j, in_start + ii] * wts[inv_index_translation[k], ii]


@stack_array("shp")
def get_dofs_local_2_form_ec_component(
    args_solve: LocalProjectorsArguments,
    fc: "float[:,:,:]",
    f_eval_aux: "float[:,:,:]",
    c: int,
):
    """Kernel for evaluating the degrees of freedom for the c-th component of 2-forms.  This function is for local commuting projetors.

    Parameters
    ----------
        args_solve : LocalProjectorsArguments
            Class that holds basic local projectors properties as attributes.

        fc : 3d float array
            Evaluation for the c-th component of the 2-form function over all the interpolation points in e_c, as well as all the Gauss-Legendre quadrature point in e_a and e_b.
            Of the two spatial directions different from e_c, e_a is the one with the smaller index, and e_b is the one with the larger index.

        f_eval_aux : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.

        c : int
            This integer determines which of the three components of the 2-form vector we are working on. Must be 0, 1 or 2.
    """

    shp = zeros(3, dtype=int)
    shp[:] = shape(f_eval_aux)

    for i in range(shp[0]):
        for j in range(shp[1]):
            for k in range(shp[2]):
                if c == 0:
                    in_start_a = j * args_solve.p[1]
                    in_start_b = k * args_solve.p[2]
                    for jj in range(args_solve.p[1]):
                        for kk in range(args_solve.p[2]):
                            f_eval_aux[i, j, k] += (
                                fc[i, in_start_a + jj, in_start_b + kk]
                                * args_solve.wts1[args_solve.inv_index_translation1[j], jj]
                                * args_solve.wts2[args_solve.inv_index_translation2[k], kk]
                            )
                elif c == 1:
                    in_start_a = i * args_solve.p[0]
                    in_start_b = k * args_solve.p[2]
                    for jj in range(args_solve.p[0]):
                        for kk in range(args_solve.p[2]):
                            f_eval_aux[i, j, k] += (
                                fc[in_start_a + jj, j, in_start_b + kk]
                                * args_solve.wts0[args_solve.inv_index_translation0[i], jj]
                                * args_solve.wts2[args_solve.inv_index_translation2[k], kk]
                            )
                elif c == 2:
                    in_start_a = i * args_solve.p[0]
                    in_start_b = j * args_solve.p[1]
                    for jj in range(args_solve.p[0]):
                        for kk in range(args_solve.p[1]):
                            f_eval_aux[i, j, k] += (
                                fc[in_start_a + jj, in_start_b + kk, k]
                                * args_solve.wts0[args_solve.inv_index_translation0[i], jj]
                                * args_solve.wts1[args_solve.inv_index_translation1[j], kk]
                            )


def get_dofs_local_2_form_ec_component_weighted(
    args_solve: LocalProjectorsArguments,
    fc: "float[:,:,:]",
    basis0: "float[:]",
    basis1: "float[:]",
    basis2: "float[:]",
    arezero_a: "int[:]",
    arezero_b: "int[:]",
    f_eval_aux: "float[:,:,:]",
    c: int,
):
    """Kernel for evaluating the degrees of freedom for the c-th component of 2-forms.  This function is for local commuting projetors.

    Parameters
    ----------
        args_solve : LocalProjectorsArguments
            Class that holds basic local projectors properties as attributes.

        fc : 3d float array
            Evaluation for the c-th component of the 2-form function over all the interpolation points in e_c, as well as all the Gauss-Legendre quadrature point in e_a and e_b.
            Of the two spatial directions different from e_c, e_a is the one with the smaller index, and e_b is the one with the larger index.

        basis0 : 1d float array
            Array with the evaluated basis functions for the e1 direction.

        basis1 : 1d float array
            Array with the evaluated basis functions for the e2 direction.

        basis2 : 1d float array
            Array with the evaluated basis functions for the e3 direction.

        arezero_a : 1d int array
            Array zeros or ones. A one means that for this particular set of quadrature points, in the e_a direction, the basis function is not zero for at least one of them.

        arezero_b : 1d int array
            Array zeros or ones. A one means that for this particular set of quadrature points, in the e_b direction, the basis function is not zero for at least one of them.

        f_eval_aux : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.

        c : int
            This int tell us whichone of the three components of the 2-form vector we are dealing with. It must be 0, 1 or 2.
    """

    for i in range(shape(f_eval_aux)[0]):
        if c == 0:
            computei = abs(basis0[i]) >= 10.0 ** (-16)
        else:
            computei = arezero_a[i] != 0
        if computei:
            for j in range(shape(f_eval_aux)[1]):
                if c == 1:
                    computej = abs(basis1[j]) >= 10.0 ** (-16)
                elif c == 0:
                    computej = arezero_a[j] != 0
                elif c == 2:
                    computej = arezero_b[j] != 0
                if computej:
                    for k in range(shape(f_eval_aux)[2]):
                        if c == 2:
                            computek = abs(basis2[k]) >= 10.0 ** (-16)
                        else:
                            computek = arezero_b[k] != 0
                        if computek:
                            if c == 0:
                                in_start_a = j * args_solve.p[1]
                                in_start_b = k * args_solve.p[2]
                                for jj in range(args_solve.p[1]):
                                    for kk in range(args_solve.p[2]):
                                        f_eval_aux[i, j, k] += (
                                            fc[i, in_start_a + jj, in_start_b + kk]
                                            * basis0[i]
                                            * basis1[in_start_a + jj]
                                            * basis2[in_start_b + kk]
                                            * args_solve.wts1[args_solve.inv_index_translation1[j], jj]
                                            * args_solve.wts2[args_solve.inv_index_translation2[k], kk]
                                        )

                            elif c == 1:
                                in_start_a = i * args_solve.p[0]
                                in_start_b = k * args_solve.p[2]
                                for jj in range(args_solve.p[0]):
                                    for kk in range(args_solve.p[2]):
                                        f_eval_aux[i, j, k] += (
                                            fc[in_start_a + jj, j, in_start_b + kk]
                                            * basis0[in_start_a + jj]
                                            * basis1[j]
                                            * basis2[in_start_b + kk]
                                            * args_solve.wts0[args_solve.inv_index_translation0[i], jj]
                                            * args_solve.wts2[args_solve.inv_index_translation2[k], kk]
                                        )

                            elif c == 2:
                                in_start_a = i * args_solve.p[0]
                                in_start_b = j * args_solve.p[1]
                                for jj in range(args_solve.p[0]):
                                    for kk in range(args_solve.p[1]):
                                        f_eval_aux[i, j, k] += (
                                            fc[in_start_a + jj, in_start_b + kk, k]
                                            * basis0[in_start_a + jj]
                                            * basis1[in_start_b + kk]
                                            * basis2[k]
                                            * args_solve.wts0[args_solve.inv_index_translation0[i], jj]
                                            * args_solve.wts1[args_solve.inv_index_translation1[j], kk]
                                        )


@stack_array("shp")
def get_dofs_local_3_form(args_solve: LocalProjectorsArguments, faux: "float[:,:,:]", f_eval: "float[:,:,:]"):
    """Kernel for evaluating the degrees of freedom for 3-forms.  This function is for local commuting projetors.

    Parameters
    ----------
        args_solve : LocalProjectorsArguments
            Class that holds basic local projectors properties as attributes.

        faux : 3d float array
            Evaluation for the 3-form function over all the Gauss-Legendre quadrature point in e1, e2 and e3.

        f_eval : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.
    """
    shp = zeros(3, dtype=int)
    shp[:] = shape(f_eval)

    for i in range(shp[0]):
        for j in range(shp[1]):
            for k in range(shp[2]):
                in_start_1 = i * args_solve.p[0]
                in_start_2 = j * args_solve.p[1]
                in_start_3 = k * args_solve.p[2]
                for ii in range(args_solve.p[0]):
                    for jj in range(args_solve.p[1]):
                        for kk in range(args_solve.p[2]):
                            f_eval[i, j, k] += (
                                faux[
                                    in_start_1 + ii,
                                    in_start_2 + jj,
                                    in_start_3 + kk,
                                ]
                                * args_solve.wts0[args_solve.inv_index_translation0[i], ii]
                                * args_solve.wts1[args_solve.inv_index_translation1[j], jj]
                                * args_solve.wts2[args_solve.inv_index_translation2[k], kk]
                            )


def get_dofs_local_3_form_weighted(
    args_solve: LocalProjectorsArguments,
    faux: "float[:,:,:]",
    basis0: "float[:]",
    basis1: "float[:]",
    basis2: "float[:]",
    arezero0: "int[:]",
    arezero1: "int[:]",
    arezero2: "int[:]",
    f_eval: "float[:,:,:]",
):
    """Kernel for evaluating the degrees of freedom for 3-forms.  This function is for local commuting projetors.

    Parameters
    ----------
        args_solve : LocalProjectorsArguments
            Class that holds basic local projectors properties as attributes.

        faux : 3d float array
            Evaluation for the 3-form function over all the Gauss-Legendre quadrature point in e1, e2 and e3.

        basis0 : 1d float array
            Array with the evaluated basis functions for the e1 direction.

        basis1 : 1d float array
            Array with the evaluated basis functions for the e2 direction.

        basis2 : 1d float array
            Array with the evaluated basis functions for the e3 direction.

        arezero0 : 1d int array
            Array of zeros or ones. A one means that for this particular set of quadrature points, in the first direction, the basis function is not zero for at least one of them.

        arezero1 : 1d int array
            Array of zeros or ones. A one means that for this particular set of quadrature points, in the second direction, the basis function is not zero for at least one of them.

        arezero2 : 1d int array
            Array of zeros or ones. A one means that for this particular set of quadrature points, in the third direction, the basis function is not zero for at least one of them.

        f_eval : 3d float array
            Output array where the evaluated degrees of freedom are stored. It is passed to this function with zeros in each entry.
    """

    for i in range(shape(f_eval)[0]):
        if arezero0[i] != 0:
            for j in range(shape(f_eval)[1]):
                if arezero1[j] != 0:
                    for k in range(shape(f_eval)[2]):
                        if arezero2[k] != 0:
                            in_start_1 = i * args_solve.p[0]
                            in_start_2 = j * args_solve.p[1]
                            in_start_3 = k * args_solve.p[2]
                            for ii in range(args_solve.p[0]):
                                for jj in range(args_solve.p[1]):
                                    for kk in range(args_solve.p[2]):
                                        f_eval[i, j, k] += (
                                            faux[
                                                in_start_1 + ii,
                                                in_start_2 + jj,
                                                in_start_3 + kk,
                                            ]
                                            * basis0[in_start_1 + ii]
                                            * basis1[in_start_2 + jj]
                                            * basis2[in_start_3 + kk]
                                            * args_solve.wts0[args_solve.inv_index_translation0[i], ii]
                                            * args_solve.wts1[args_solve.inv_index_translation1[j], jj]
                                            * args_solve.wts2[args_solve.inv_index_translation2[k], kk]
                                        )


# We need a functions that tell us which of the quasi-interpolation points to take for a any given i
def select_quasi_points(i: int, p: int, Nbasis: int, periodic: bool):
    """Determines the start and end indices of the quasi-interpolation points that must be taken to get the ith FEEC coefficient.

    Parameters
    ----------
    i : int
        Index of the FEEC coefficient that must be computed.

    p : int
        B-spline degree.

    Nbasis: int
        Number of B-splines.

    periodic: bool
        Whether we have periodic boundary conditions.

    Returns
    -------
    start : int
        Start index of the quasi-interpolation points that must be consider to obtain the ith FEEC coefficient. This is an inclusive index.

    end : int
        End index of the quasi-interpolation points that must be consider to obtain the ith FEEC coefficient. This is an exclusive index.
    """
    if periodic:
        start = 2 * i
        end = int(2 * p) - 1 + 2 * i
        return start, end
    else:
        # Special case p = 1
        if p == 1:
            # raise Exception("The case clamped splines with p = 1 is not implemented at the moment. Please choose p > 1.")
            return 0, 0
        # Case p > 1
        else:
            if i <= 1:
                start = int((p + 1) * i)
                end = int((p + 1) * (i + 1))
            elif 1 < i and i <= (p - 2):
                start = int(i * p + ((i - 1) * i) / 2 + 1)
                end = int(start + p + i)
            elif (p - 1) <= i and i <= (Nbasis - p):
                start = int(p * p + ((p - 3) * (p - 2)) / 2 - 1)
                start += int(2 * (i - p + 1))
                end = int(start + 2 * p - 1)
            elif (Nbasis - p) < i and i < (Nbasis - 1):
                start = int((p - 2) * p + ((p - 3) * (p - 2)) / 2 + 2 * Nbasis)
                start += int(2 * (i + p - Nbasis - 1) * (p - 1) - ((i - Nbasis + p - 2) * (i - Nbasis + p - 1)) / 2)
                end = int(start + Nbasis + p - i - 1)
            elif i == (Nbasis - 1):
                # start, end = select_quasi_points(Nbasis-2, p, Nbasis, periodic)
                # While I figure out how to make a recursive call I will do it by hand
                ##############################################
                i = Nbasis - 2
                if i <= 1:
                    start = int((p + 1) * i)
                elif 1 < i and i <= (p - 2):
                    start = int(i * p + ((i - 1) * i) / 2 + 1)
                elif (p - 1) <= i and i <= (Nbasis - p):
                    start = int(p * p + ((p - 3) * (p - 2)) / 2 - 1)
                    start += int(2 * (i - p + 1))
                elif (Nbasis - p) < i and i < (Nbasis - 1):
                    start = int((p - 2) * p + ((p - 3) * (p - 2)) / 2 + 2 * Nbasis)
                    start += int(2 * (i + p - Nbasis - 1) * (p - 1) - ((i - Nbasis + p - 2) * (i - Nbasis + p - 1)) / 2)
                #############################################

                start += p + 1
                end = start + p + 1

            return start, end


def solve_local_main_loop(args_solve: LocalProjectorsArguments, rhs: "float[:,:,:]", out: "float[:,:,:]"):
    """Kernel for obtaining the FEEC coefficients with local projectors.

    Parameters
    ----------
        args_solve : LocalProjectorsArguments
            Class that holds basic local projectors properties as attributes.

        rhs : 3d float array
            Array with the evaluated degrees of freedom.

        out : 3d float array
            Array of FEEC coefficients.
    """
    lenj1, lenj2, lenj3 = get_local_problem_size(args_solve.periodic, args_solve.p, args_solve.IoH)

    # We iterate over all the entries that belong to the current rank
    counteri0 = 0
    for i0 in range(args_solve.starts[0], args_solve.ends[0] + 1):
        counteri1 = 0
        for i1 in range(args_solve.starts[1], args_solve.ends[1] + 1):
            counteri2 = 0
            for i2 in range(args_solve.starts[2], args_solve.ends[2] + 1):
                L123 = 0.0
                startj1, endj1 = select_quasi_points(
                    i0,
                    args_solve.p[0],
                    args_solve.B_nbasis[0],
                    args_solve.periodic[0],
                )
                startj2, endj2 = select_quasi_points(
                    i1,
                    args_solve.p[1],
                    args_solve.B_nbasis[1],
                    args_solve.periodic[1],
                )
                startj3, endj3 = select_quasi_points(
                    i2,
                    args_solve.p[2],
                    args_solve.B_nbasis[2],
                    args_solve.periodic[2],
                )
                for j1 in range(lenj1):
                    # We only bother to compute this contribution if the weight wij is not zero. For if it is zero the contribution will be zero as well.
                    if args_solve.wij0[i0][j1] != 0.0:
                        # position 1 to evaluate rhs. The module is only necessary for periodic boundary conditions. But it does not hurt the clamped boundary conditions so we just leave it as is to avoid an extra if.
                        if startj1 + j1 < args_solve.original_size[0]:
                            pos1 = args_solve.index_translation1[startj1 + j1]
                        else:
                            pos1 = args_solve.index_translation1[int(startj1 + j1 + args_solve.shift[0])]
                        auxL2 = 0.0
                        for j2 in range(lenj2):
                            # We only bother to compute this contribution if the weight wij is not zero. For if it is zero the contribution will be zero as well.
                            if args_solve.wij1[i1][j2] != 0.0:
                                # position 2 to evaluate rhs
                                if startj2 + j2 < args_solve.original_size[1]:
                                    pos2 = args_solve.index_translation2[startj2 + j2]
                                else:
                                    pos2 = args_solve.index_translation2[
                                        int(
                                            startj2 + j2 + args_solve.shift[1],
                                        )
                                    ]
                                auxL3 = 0.0
                                for j3 in range(lenj3):
                                    # We only bother to compute this contribution if the weight wij is not zero. For if it is zero the contribution will be zero as well.
                                    if args_solve.wij2[i2][j3] != 0.0:
                                        # position 3 to evaluate rhs
                                        if startj3 + j3 < args_solve.original_size[2]:
                                            pos3 = args_solve.index_translation3[startj3 + j3]
                                        else:
                                            pos3 = args_solve.index_translation3[
                                                int(
                                                    startj3 + j3 + args_solve.shift[2],
                                                )
                                            ]
                                        auxL3 += args_solve.wij2[i2][j3] * rhs[pos1, pos2, pos3]
                                auxL2 += args_solve.wij1[i1][j2] * auxL3
                        L123 += args_solve.wij0[i0][j1] * auxL2
                out[args_solve.pds[0] + counteri0, args_solve.pds[1] + counteri1, args_solve.pds[2] + counteri2] = L123
                counteri2 += 1
            counteri1 += 1
        counteri0 += 1


def solve_local_main_loop_weighted(
    args_solve: LocalProjectorsArguments,
    rhs: "float[:,:,:]",
    rows0: "int[:]",
    rows1: "int[:]",
    rows2: "int[:]",
    rowe0: "int[:]",
    rowe1: "int[:]",
    rowe2: "int[:]",
    out: "float[:,:,:]",
    basis0: "float[:]",
    basis1: "float[:]",
    basis2: "float[:]",
):
    """Kernel for obtaining the FEEC coefficients of three forms with local projectors.

    Parameters
    ----
        args_solve : LocalProjectorsArguments
            Class that holds basic local projectors properties as attributes.

        rhs : 3d float array
            Array with the evaluated degrees of freedom.

        rows0: 1d int array
            Array that tell us for which rows the basis function in the e1 direction produces non-zero entries in the BasisProjectionOperatorLocal matrix. This array contains the start indices of said regions.

        rows1: 1d int array
            Array that tell us for which rows the basis function in the e2 direction produces non-zero entries in the BasisProjectionOperatorLocal matrix. This array contains the start indices of said regions.

        rows2: 1d int array
            Array that tell us for which rows the basis function in the e3 direction produces non-zero entries in the BasisProjectionOperatorLocal matrix. This array contains the start indices of said regions.

        rowe0: 1d int array
            Array that tell us for which rows the basis function in the e1 direction produces non-zero entries in the BasisProjectionOperatorLocal matrix. This array contains the end indices of said regions.

        rowe1: 1d int array
            Array that tell us for which rows the basis function in the e2 direction produces non-zero entries in the BasisProjectionOperatorLocal matrix. This array contains the end indices of said regions.

        rowe2: 1d int array
            Array that tell us for which rows the basis function in the e3 direction produces non-zero entries in the BasisProjectionOperatorLocal matrix. This array contains the end indices of said regions.

        out : 3d float array
            Array of FEEC coefficients.

        basis0 : 1d float array
            Array with the evaluated basis functions for the e1 direction. Only relevant for the 0 and 0V spaces since they are the only ones who did not multiply the rhs
            by the basis function duting the get_dofs_weigthed function.

        basis1 : 1d float array
            Array with the evaluated basis functions for the e2 direction. Only relevant for the 0 and 0V spaces since they are the only ones who did not multiply the rhs
            by the basis function duting the get_dofs_weigthed function.

        basis2 : 1d float array
            Array with the evaluated basis functions for the e3 direction. Only relevant for the H1 and H1vec spaces since they are the only ones who did not multiply the rhs
            by the basis function during the get_dofs_weigthed function.
    """
    # First we determine if we must multiply the rhs by the basis functions. This is only required for the H1 and H1vec spaces.
    if args_solve.space_key == 0 or args_solve.space_key == 4:
        Need_basis = True
    else:
        Need_basis = False

    lenj1, lenj2, lenj3 = get_local_problem_size(args_solve.periodic, args_solve.p, args_solve.IoH)

    # We iterate over all the entries that belong to the current rank
    counteri0 = 0
    for i0 in range(args_solve.starts[0], args_solve.ends[0] + 1):
        # This bool variable tell us if this row has a non-zero FE coefficient, based on the current basis function we are using on our projection
        compute0 = False
        # We iterate over the arrays with the start and end indices of non-zero row regions to check if our current row falls in one of them.
        for i00 in range(len(rows0)):
            if counteri0 >= rows0[i00] and counteri0 <= rowe0[i00]:
                compute0 = True
                break
        if compute0:
            counteri1 = 0
            for i1 in range(args_solve.starts[1], args_solve.ends[1] + 1):
                # This bool variable tell us if this row has a non-zero FE coefficient, based on the current basis function we are using on our projection
                compute1 = False
                # We iterate over the arrays with the start and end indices of non-zero row regions to check if our current row falls in one of them.
                for i11 in range(len(rows1)):
                    if counteri1 >= rows1[i11] and counteri1 <= rowe1[i11]:
                        compute1 = True
                        break
                if compute1:
                    counteri2 = 0
                    for i2 in range(args_solve.starts[2], args_solve.ends[2] + 1):
                        # This bool variable tell us if this row has a non-zero FE coefficient, based on the current basis function we are using on our projection
                        compute2 = False
                        # We iterate over the arrays with the start and end indices of non-zero row regions to check if our current row falls in one of them.
                        for i22 in range(len(rows2)):
                            if counteri2 >= rows2[i22] and counteri2 <= rowe2[i22]:
                                compute2 = True
                                break
                        if compute2:
                            L123 = 0.0
                            startj1, endj1 = select_quasi_points(
                                i0,
                                args_solve.p[0],
                                args_solve.B_nbasis[0],
                                args_solve.periodic[0],
                            )
                            startj2, endj2 = select_quasi_points(
                                i1,
                                args_solve.p[1],
                                args_solve.B_nbasis[1],
                                args_solve.periodic[1],
                            )
                            startj3, endj3 = select_quasi_points(
                                i2,
                                args_solve.p[2],
                                args_solve.B_nbasis[2],
                                args_solve.periodic[2],
                            )
                            for j1 in range(lenj1):
                                if args_solve.wij0[i0][j1] != 0.0:
                                    # position 1 to evaluate rhs
                                    if startj1 + j1 < args_solve.original_size[0]:
                                        pos1 = args_solve.index_translation1[startj1 + j1]
                                    else:
                                        pos1 = args_solve.index_translation1[int(startj1 + j1 + args_solve.shift[0])]
                                    auxL2 = 0.0
                                    for j2 in range(lenj2):
                                        if args_solve.wij1[i1][j2] != 0.0:
                                            # position 2 to evaluate rhs
                                            if startj2 + j2 < args_solve.original_size[1]:
                                                pos2 = args_solve.index_translation2[startj2 + j2]
                                            else:
                                                pos2 = args_solve.index_translation2[
                                                    int(
                                                        startj2 + j2 + args_solve.shift[1],
                                                    )
                                                ]
                                            auxL3 = 0.0
                                            for j3 in range(lenj3):
                                                if args_solve.wij2[i2][j3] != 0.0:
                                                    # position 3 to evaluate rhs
                                                    if startj3 + j3 < args_solve.original_size[2]:
                                                        pos3 = args_solve.index_translation3[startj3 + j3]
                                                    else:
                                                        pos3 = args_solve.index_translation3[
                                                            int(
                                                                startj3 + j3 + args_solve.shift[2],
                                                            )
                                                        ]
                                                    if Need_basis:
                                                        auxL3 += (
                                                            args_solve.wij2[i2][j3]
                                                            * rhs[pos1, pos2, pos3]
                                                            * basis2[pos3]
                                                        )
                                                    else:
                                                        auxL3 += args_solve.wij2[i2][j3] * rhs[pos1, pos2, pos3]
                                            if Need_basis:
                                                auxL2 += args_solve.wij1[i1][j2] * auxL3 * basis1[pos2]
                                            else:
                                                auxL2 += args_solve.wij1[i1][j2] * auxL3
                                    if Need_basis:
                                        L123 += args_solve.wij0[i0][j1] * auxL2 * basis0[pos1]
                                    else:
                                        L123 += args_solve.wij0[i0][j1] * auxL2
                            out[counteri0, counteri1, counteri2] = L123
                        counteri2 += 1
                counteri1 += 1
        counteri0 += 1


def find_relative_col(col: int, row: int, Nbasis: int, periodic: bool):
    """Compute the relative row position of a StencilMatrix from the global column and row positions.

    Parameters
    ----------
    col : int
        Global column index.

    row : int
        Global row index.

    Nbasis : int
        Number of B(or D)-splines for this particular dimension.

    periodic : bool
        True if we have periodic boundary conditions in this direction, otherwise False.

    Returns
    -------
    relativecol : int
        The relative column position of col with respect to the the current row of the StencilMatrix.

    """
    if not periodic:
        relativecol = col - row
    # In the periodic case we must account for the possible looping of the basis functions when computing the relative row postion
    else:
        if col <= row:
            if abs(col - row) <= abs(col + Nbasis - row):
                relativecol = col - row
            else:
                relativecol = col + Nbasis - row
        else:
            if abs(col - row) <= abs(col - Nbasis - row):
                relativecol = col - row
            else:
                relativecol = col - Nbasis - row
    return relativecol


def assemble_basis_projection_operator_local(
    starts: "int[:]",
    ends: "int[:]",
    pds: "int[:]",
    periodic: "bool[:]",
    p: "int[:]",
    col: "int[:]",
    VNbasis: "int[:]",
    mat: "float[:,:,:,:,:,:]",
    coeff: "float[:,:,:]",
    rows0: "int[:]",
    rows1: "int[:]",
    rows2: "int[:]",
    rowe0: "int[:]",
    rowe1: "int[:]",
    rowe2: "int[:]",
):
    """Kernel for storing the FE coefficients into the StencilMatrix.

    Parameters
    ----------
        starts: 1d int array
            Array with the StencilVector start indices for each MPI rank.

        ends: 1d int array
            Array with the StencilVector end indices for each MPI rank.

        pds: 1d int array
            Array with the StencilVector pads for each MPI rank.

        periodic: 1d bool array
            Array that tell us if the splines are periodic or clamped in each dimension.

        p : 1d int array
            Degree of the B-splines in the [e1,e2,e3] direction.

        col : 1d int array
            Tell us the value of the [first, second, third] global column

        VNbasis : 1d int array
            Array with the number of basis functions of the input space.

        mat : 6d float array
            Data array of the StencilMatrix.

        coeff : 3d float array
            Array of FEEC coefficients.

        rows0: 1d int array
            Array that tell us for which rows the basis function in the e1 direction produces non-zero entries. This array contains the start indeces of said regions.

        rows1: 1d int array
            Array that tell us for which rows the basis function in the e2 direction produces non-zero entries. This array contains the start indeces of said regions.

        rows2: 1d int array
            Array that tell us for which rows the basis function in the e3 direction produces non-zero entries. This array contains the start indeces of said regions.

        rowe0: 1d int array
            Array that tell us for which rows the basis function in the e1 direction produces non-zero entries. This array contains the end indeces of said regions.

        rowe1: 1d int array
            Array that tell us for which rows the basis function in the e2 direction produces non-zero entries. This array contains the end indeces of said regions.

        rowe2: 1d int array
            Array that tell us for which rows the basis function in the e3 direction produces non-zero entries. This array contains the end indeces of said regions.
    """

    count0 = 0
    for row0 in range(starts[0], ends[0] + 1):
        # This bool variable tell us if this row has a non-zero FE coefficient, based on the current basis function we are using on our projection
        compute0 = False
        # We iterate over the arrays with the start and end indices of non-zero row regions to check if our current row falls in one of them.
        for i00 in range(len(rows0)):
            if count0 >= rows0[i00] and count0 <= rowe0[i00]:
                compute0 = True
                break
        relativecol0 = find_relative_col(col[0], row0, VNbasis[0], periodic[0])
        if relativecol0 >= -p[0] and relativecol0 <= p[0] and compute0:
            count1 = 0
            for row1 in range(starts[1], ends[1] + 1):
                # This bool variable tell us if this row has a non-zero FE coefficient, based on the current basis function we are using on our projection
                compute1 = False
                # We iterate over the arrays with the start and end indices of non-zero row regions to check if our current row falls in one of them.
                for i11 in range(len(rows1)):
                    if count1 >= rows1[i11] and count1 <= rowe1[i11]:
                        compute1 = True
                        break
                relativecol1 = find_relative_col(col[1], row1, VNbasis[1], periodic[1])
                if relativecol1 >= -p[1] and relativecol1 <= p[1] and compute1:
                    count2 = 0
                    for row2 in range(starts[2], ends[2] + 1):
                        # This bool variable tell us if this row has a non-zero FE coefficient, based on the current basis function we are using on our projection
                        compute2 = False
                        # We iterate over the arrays with the start and end indices of non-zero row regions to check if our current row falls in one of them.
                        for i22 in range(len(rows2)):
                            if count2 >= rows2[i22] and count2 <= rowe2[i22]:
                                compute2 = True
                                break
                        relativecol2 = find_relative_col(col[2], row2, VNbasis[2], periodic[2])
                        if relativecol2 >= -p[2] and relativecol2 <= p[2] and compute2:
                            mat[
                                count0 + pds[0],
                                count1 + pds[1],
                                count2 + pds[2],
                                relativecol0 + p[0],
                                relativecol1 + p[1],
                                relativecol2 + p[2],
                            ] = coeff[count0, count1, count2]
                        count2 += 1
                count1 += 1
        count0 += 1


def are_quadrature_points_zero(aux: "int[:]", p: int, basis: "float[:]"):
    """Kernel for determinig if a given spline is zero for all quadrature points that together are used to compute one integral.

    Parameters
    ----------
        aux : 1d int array
            Must be given as an array full of ones. After the kernel is done it shall have zeros at the points for which the basis function was zero at all p Gauss-Legandre quadrature points.

        p : int
            Degree of the B-splines in the current dirtection of work.

        basis : 1d float array
            Array with the evaluated basis functions for the current direction of work.
    """

    for i in range(shape(aux)[0]):
        in_start = i * p
        all_zero = True
        for ii in range(p):
            if basis[in_start + ii] != 0.0:
                all_zero = False
                break
        if all_zero:
            aux[i] = 0


def get_rows_periodic(starts: int, ends: int, modl: int, modr: int, Nbasis: int, col: int, aux: "int[:]"):
    """Auxiliary kernel for gettinmg the non-zero rows in the periodic case.

    Parameters
    ----------
        starts : int
            The start index for the current MPI rank.

        ends : int
            The end index for the current MPI rank

        modl : int
            Determines the left most column that is non-zero for a given row. This column has index row+modl.

        modr : int
            Determines the right most column that is non-zero for a given row. This column has index row+modr.

        Nbasis : int
            The number of basis functions in the direction of interest.

        col : int
            The index of the B or D-spline.

        aux : 1d int array
            Array where we put a one if the current row could have a non-zero FE coefficient for the column given by col.
    """

    count = 0
    for row in range(starts, ends + 1):
        for modifier in range(modl, modr):
            if col == (row + modifier) % Nbasis:
                aux[count] = 1
                break
        count += 1


def get_rows(
    col: int,
    starts: int,
    ends: int,
    p: int,
    Nbasis: int,
    periodic: bool,
    IoH: bool,
    BoD: bool,
    aux: "int[:]",
):
    """Kernel for getting the list of rows that are non-zero for the current BasisProjectionLocal column, within the start and end indices of the current MPI rank.

    Parameters
    ----------
        col : int
            The index of the B or D-spline.

        starts : int
            The start index for the current MPI rank.

        ends : int
            The end index for the current MPI rank

        p : int
            The B-spline degree for the dimension of interest.

        Nbasis : int
            The number of basis functions in the direction of interest.

        periodic : bool
            Whether we have periodic boundary conditions in this direction.

        IoH : bool
            False means we are dealing with interpolation, True means we are dealing with histopolation.

        BoD : bool
            False means we are dealing with B-splines, True means we are dealing with D-splines.

        aux : 1d int array
            Array where we put a one if the current row could have a non-zero FE coefficient for the column given by col.
    """
    # Periodic boundary conditions
    if periodic:
        # Histopolation
        if IoH:
            # D-splines
            if BoD:
                get_rows_periodic(starts, ends, -p + 1, p, Nbasis, col, aux)
            # B-splines
            if not BoD:
                get_rows_periodic(starts, ends, -p + 1, p + 1, Nbasis, col, aux)
        # Interpolation
        if not IoH:
            # D-splines
            if BoD:
                # Special case p = 1
                if p == 1:
                    get_rows_periodic(starts, ends, -1, 1, Nbasis, col, aux)
                if p != 1:
                    get_rows_periodic(starts, ends, -p + 1, p - 1, Nbasis, col, aux)
            # B-splines
            if not BoD:
                get_rows_periodic(starts, ends, -p + 1, p, Nbasis, col, aux)
    # Clamped boundary conditions
    if not periodic:
        # Histopolation
        if IoH:
            # D-splines
            if BoD:
                count = 0
                for row in range(starts, ends + 1):
                    if row >= 0 and row <= (p - 2) and col >= 0 and col <= row + p - 1:
                        aux[count] = 1
                    elif row >= (p - 1) and row < (Nbasis + 1 - p) and col >= (row - p + 1) and col <= (row + p - 1):
                        aux[count] = 1
                    elif (
                        row >= (Nbasis + 1 - p) and row <= (Nbasis - 1) and col >= (row - p + 1) and col <= (Nbasis - 1)
                    ):
                        aux[count] = 1
                    count += 1
            # B-splines
            if not BoD:
                count = 0
                for row in range(starts, ends + 1):
                    if row >= 0 and row <= (p - 2) and col >= 0 and col <= (row + p):
                        aux[count] = 1
                    elif row >= (p - 1) and row < (Nbasis - p) and col >= (row - p + 1) and col <= (row + p):
                        aux[count] = 1
                    elif row >= (Nbasis - p) and row <= (Nbasis - 2) and col >= (row - p + 1) and col <= (Nbasis - 1):
                        aux[count] = 1
                    count += 1
        # Interpolation
        if not IoH:
            # D-splines
            if BoD:
                count = 0
                for row in range(starts, ends + 1):
                    if row == 0 and col <= (p - 1):
                        aux[count] = 1
                    elif row > 0 and row < (p - 1) and col <= (row + p - 2):
                        aux[count] = 1
                    elif row >= (p - 1) and row <= (Nbasis + 1 - p) and col >= (row - p + 1) and col <= (row + p - 2):
                        aux[count] = 1
                    elif row > (Nbasis + 1 - p) and row < Nbasis and col >= (row - p + 1) and col <= (Nbasis - 1):
                        aux[count] = 1
                    elif row == Nbasis and col >= (Nbasis - p) and col <= (Nbasis - 1):
                        aux[count] = 1
                    count += 1
            # B-splines
            if not BoD:
                count = 0
                for row in range(starts, ends + 1):
                    if row == 0 and col <= p:
                        aux[count] = 1
                    elif row > 0 and row < (p - 1) and col <= (row + p - 1):
                        aux[count] = 1
                    elif row >= (p - 1) and row <= (Nbasis - p) and col >= (row - p + 1) and col <= (row + p - 1):
                        aux[count] = 1
                    elif row > (Nbasis - p) and row < (Nbasis - 1) and col >= (row - p + 1) and col <= (Nbasis - 1):
                        aux[count] = 1
                    elif row == (Nbasis - 1) and col >= (Nbasis - p - 1) and col <= (Nbasis - 1):
                        aux[count] = 1
                    count += 1


def fill_matrix_column(
    starts: "int[:]",
    ends: "int[:]",
    pds: "int[:]",
    col: int,
    VNbasis: "int[:]",
    mat: "float[:,:]",
    coeff: "float[:,:,:]",
):
    """Kernel for filling the collumn col of a matrix, with the rows of FE coefficients belonging to the current MPI rank.

    Parameters
    ----------
        starts: 1d int array
            Array with the StencilVector start indices for each MPI rank.

        ends: 1d int array
            Array with the StencilVector end indices for each MPI rank.

        pds: 1d int array
            Array with the StencilVector pads for each MPI rank.

        col : int
            Tell us the column to be filled.

        VNbasis : 1d int array
            Array with the number of basis functions of the output space.

        mat : 2d float array
            Matrix that we shall fill in this kernel.

        coeff : 3d float array
            Array of FEEC coefficients.
    """

    count0 = 0
    for row0 in range(starts[0], ends[0] + 1):
        count1 = 0
        for row1 in range(starts[1], ends[1] + 1):
            count2 = 0
            for row2 in range(starts[2], ends[2] + 1):
                row = row2 + row1 * VNbasis[2] + row0 * VNbasis[2] * VNbasis[1]
                mat[row, col] = coeff[pds[0] + count0, pds[1] + count1, pds[2] + count2]
                count2 += 1
            count1 += 1
        count0 += 1
