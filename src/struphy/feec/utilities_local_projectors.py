import cunumpy as xp

from struphy.feec.local_projectors_kernels import are_quadrature_points_zero, get_rows, select_quasi_points


def split_points(
    IoH,
    lenj,
    shift,
    pts,
    starts,
    ends,
    p,
    npts,
    periodic,
    wij,
    whij,
    localptsout,
    original_pts_size,
    index_translation,
    inv_index_translation,
):
    """Splits the interpolaton points and quadrature points between the MPI ranks. Making sure that each rank only gets the points it needs to compute the FE coefficients assigned to it.

    Parameters
    ----------
    IoH : list of strings
        Determines if we have Interpolation (I) or Histopolation (H) for each one of the three spatial dimentions.

    lenj : list of int
        Determines the number of inner itterations we need to run over all values of j for each one of the three spatial directions.

    shifts : 1d int array
        For each one of the three spatial directions it determines by which amount to shift the position index (pos) in case we have to loop over the evaluation points.

    pts : list of xp.array
        3D list of 2D array with the quasi-interpolation points
        (or Gauss-Legendre quadrature points for histopolation).
        In format (ns, nb, np) = (spatial direction, B-spline index, point) for StencilVector spaces .

    starts : 1D int array
        Array with the BlockVector (or StencilVector) start indices for each MPI rank.

    ends : 1D int array
        Array with the BlockVector (or StencilVector) end indices for each MPI rank.

    p : 1D int array
        Contains the B-splines degrees for each one of the three spatial directions.

    npts : list of ints
        Contains the number of B-splines for each one of the three spatial directions.

    periodic : 1D bool xp.array
        For each one of the three spatial directions contains the information of whether the B-splines are periodic or not.

    wij: 3d float array
        Array with the interpolation geometric weights for all three directions. In format (spatial directions, i index, j index)

    whij: 3d float array
        Array with the histopolation geometric weights for all three directions. In format (spatial directions, i index, j index)

    localptsout : empty list
        Here this function shall write the interpolation or quadrature points that are relevant for the MPI rank.

    original_pts_size : empty list
        Here this function shall write the total number of interpolation points or histopolation integrals for all three spatial diections.

    index_translation : empty list
        This function makes sure that this list translates for all three spatial direction from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translation[spatial-direction][index_global]

    inv_index_translation : empty list
        This function makes sure that this list translates for all three spatial direction from the local indices to the global indices to evaluate the right-hand-side. index_global = inv_index_translation[spatial-direction][index_local]


    """
    # We iterate over the three spatial directions
    for n, pt in enumerate(pts):
        original_pts_size[n] = xp.shape(pt)[0]
        # We initialize localpts with as many entries as the global pt, but with all entries being -1
        # This function will change the values of the needed entries from -1 to the value of the point.
        if IoH[n] == "I":
            localpts = xp.full((xp.shape(pt)[0]), fill_value=-1, dtype=float)
        elif IoH[n] == "H":
            localpts = xp.full((xp.shape(pt)), fill_value=-1, dtype=float)

        for i in range(starts[n], ends[n] + 1):
            startj1, endj1 = select_quasi_points(int(i), int(p[n]), int(npts[n]), bool(periodic[n]))
            for j1 in range(lenj[n]):
                if startj1 + j1 < xp.shape(pt)[0]:
                    pos = startj1 + j1
                else:
                    pos = int(startj1 + j1 + shift[n])
                if IoH[n] == "I":
                    if wij[n][i][j1] != 0.0:
                        localpts[pos] = pt[pos]
                elif IoH[n] == "H":
                    if whij[n][i][j1] != 0.0:
                        localpts[pos] = pt[pos]
        # We get the local points by grabing only the values different from -1.
        if IoH[n] == "I":
            localpos = xp.where(localpts != -1)[0]
        elif IoH[n] == "H":
            localpos = xp.where(localpts[:, 0] != -1)[0]
        localpts = localpts[localpos]
        localptsout.append(xp.array(localpts))

        ##
        # We build the index_translation array that shall turn global indices into local indices
        ##
        mini_indextranslation = xp.full(
            (xp.shape(pt)[0]),
            fill_value=-1,
            dtype=int,
        )
        for i, j in enumerate(localpos):
            mini_indextranslation[j] = i

        index_translation.append(xp.array(mini_indextranslation))

        ##
        # We build the inv_index_translation that shall turn local indices into global indices
        ##

        inv_mini_indextranslation = xp.full(
            (xp.shape(localptsout[-1])[0]),
            fill_value=-1,
            dtype=int,
        )
        for i, j in enumerate(localpos):
            inv_mini_indextranslation[i] = j

        inv_index_translation.append(xp.array(inv_mini_indextranslation))


def get_values_and_indices_splines(Nbasis, degree, periodic, spans, values):
    """Given an array with the values of the splines evaluated at certain points this function returns a xp.array that tell us the index of each spline. So we can know to which spline each
    value corresponds. It also modifies the evaluation values in the case we have one spline of degree one with periodic boundary conditions, so it is artificially equal to the identity.

    Parameters
    ----------
    Nbasis : int
        Number of basis functions.

    degree : int
        Degree of the B or D-splines.

    periodic : bool
        Whether we have periodic boundary conditions or nor.

    span : xp.array
    2d array indexed by (n, nq), where n is the interval and nq is the quadrature point in the interval.

    values : xp.array
    3d array of values of basis functions indexed by (n, nq, basis function).

    Returns
    -------

    eval_indeces : xp.array
    3d array of basis functions indices, indexed by (n, nq, basis function).

    values : xp.array
    3d array of values of basis functions indexed by (n, nq, basis function).

    """
    # In this case we want this spatial direction to be "neglected", that means we artificially set the values of the B-spline to 1 at all points. So it becomes the multiplicative identity.
    if Nbasis == 1 and degree == 1 and periodic:
        # Set all values to 1 for the identity case
        values = xp.ones((values.shape[0], values.shape[1], 1))
        eval_indeces = xp.zeros_like(values, dtype=int)
    else:
        eval_indeces = xp.zeros_like(values, dtype=int)
        for i in range(xp.shape(spans)[0]):
            for k in range(xp.shape(spans)[1]):
                for j in range(degree + 1):
                    eval_indeces[i, k, j] = (spans[i][k] - degree + j) % Nbasis

    return eval_indeces, values


def get_one_spline(a, values, eval_indeces):
    """Given the spline index, an array with the splines evaluated at the evaluation points and another array with the indices indicating to which spline each value corresponds, this function returns
    a 1d xp.array with the desired spline evaluated at all evaluation points.

    Parameters
    ----------
    a : int
    Spline index

    values : xp.array
    3d array of values of basis functions indexed by (n, nq, basis function).

    eval_indeces : xp.array
    3d array of basis functions indices, indexed by (n, nq, basis function).

    Returns
    -------
    my_values : xp.array
    1d array of values for the spline evaluated at all evaluation points.

    """
    my_values = xp.zeros(xp.shape(values)[0] * xp.shape(values)[1])
    for i in range(xp.shape(values)[0]):
        for j in range(xp.shape(values)[1]):
            for k in range(xp.shape(values)[2]):
                if eval_indeces[i, j, k] == a:
                    my_values[i * xp.shape(values)[1] + j] = values[i, j, k]
                    break
    return my_values


def get_span_and_basis(pts, space):
    """Compute the knot span index and the values of p + 1 basis function at each point in pts.

    Parameters
    ----------
    pts : xp.array
        2d array of points (ii, iq) = (interval, quadrature point).

    space : SplineSpace
        Psydac object, the 1d spline space to be projected.

    Returns
    -------
    span : xp.array
        2d array indexed by (n, nq), where n is the interval and nq is the quadrature point in the interval.

    basis : xp.array
        3d array of values of basis functions indexed by (n, nq, basis function).
    """

    import psydac.core.bsplines as bsp

    # Extract knot vectors, degree and kind of basis
    T = space.knots
    p = space.degree

    span = xp.zeros(pts.shape, dtype=int)
    basis = xp.zeros((*pts.shape, p + 1), dtype=float)

    for n in range(pts.shape[0]):
        for nq in range(pts.shape[1]):
            # avoid 1. --> 0. for clamped interpolation
            x = pts[n, nq] % (1.0 + 1e-14)
            span_tmp = bsp.find_span(T, p, x)
            basis[n, nq, :] = bsp.basis_funs_all_ders(
                T,
                p,
                x,
                span_tmp,
                0,
                normalization=space.basis,
            )
            span[n, nq] = span_tmp  # % space.nbasis

    return span, basis


def transform_into_ranges(numbers):
    """Given an array of zeros and ones this function annotates the start and end indices of the regions with ones.

    Parameters
    ----------
    numbers : np int array
        1d array containing zeros or ones.

    Returns
    -------
    rangestart : list
        List where the start indices of the one ranges shall be appended to.

    rangeend : list
        List where the end indices of the one ranges shall be appended to.
    """
    rangestart = []
    rangeend = []

    previous = 0
    # We itterate over all entries of the numbers array
    for i, current in enumerate(numbers):
        if current == 1 and previous == 0:
            rangestart.append(i)
            previous = 1
        if current == 0 and previous == 1:
            rangeend.append(i - 1)
            previous = 0

    if len(rangestart) == len(rangeend) + 1:
        rangeend.append(len(numbers) - 1)

    if len(rangestart) != len(rangeend):
        raise Exception(
            "The length of rangestart and rangeend are not the same. Meaning there is something wrong with the transform_into_ranges function.",
        )

    return rangestart, rangeend


def get_sparsity_pattern_periodic(p, S_nbasis, starts, ends, modr, modl):
    """Using the information about the BasisProjectionOperatorsLocals sparsity pattern this function returns a list with the
    columns that will have non-zero entries for the rows that belong to the current MPI rank. This particular function works for
    periodic boundary conditions.

    Parameters
    ----------
    p : int
        Denotes the degree of the B-splines for the relevant spatial direction.

    S_nbasis : int
        Number of splines in the relevant spatial direction (could be B or D splines depending on the case).

    starts : int
        start index of the FE coefficients the current MPI rank is responsible for in the relevant direction.

    ends : int
        end index of the FE coefficients the current MPI rank is responsible for in the relevant direction.

    modr : int
        Determines the maximum column that is not zero in the basis projection operator. This column has a value
        of j = i+p+modr, with i being the row index.

    modl : int
        Determines the minimum column that is not zero in the basis projection operator. This column has a value
        of j = i-p+modl, with i being the row index.
    """
    # Compute the number of non-zero columns
    N_non_zero = 2 * p + modr - modl + 1

    # Handle the case where all basis functions are nonzero
    if N_non_zero >= S_nbasis:
        return list(range(S_nbasis))

    # Compute the indices
    aux_indices = [(starts + j) % S_nbasis for j in range(-p + modl, p + modr + 1)]
    for cont, j in enumerate(range(starts + 1, ends + 1), start=1):
        next_index = (starts + p + modr + cont) % S_nbasis
        if next_index == aux_indices[0]:
            break
        aux_indices.append(next_index)

    return aux_indices


def get_sparsity_pattern_clamped(p, B_nbasis, S_nbasis, starts, ends, modr, modl, bordr, bordl, stuck):
    """Using the information about the BasisProjectionOperatorsLocals sparsity pattern this function returns a list with the
    columns that will have non-zero entries for the rows that belong to the current MPI rank. This particular function works for
    clamped boundary conditions.

    Parameters
    ----------
    p : int
        Denotes the degree of the B-splines for the relevant spatial direction.

    B_nbasis : int
        Number of B_splines in the relevant spatial direction.

    B_nbasis : int
        Number of splines in the relevant spatial direction (could be B or D splines depending on the case).

    starts : int
        start index of the FE coefficients the current MPI rank is responsible for in the relevant direction.

    ends : int
        end index of the FE coefficients the current MPI rank is responsible for in the relevant direction.

    modr : int
        Determines the maximum column that is not zero in the basis projection operator. This column has a value
        of j = i+p+modr, with i being the row index.

    modl : int
        Determines the minimum column that is not zero in the basis projection operator. This column has a value
        of j = i-p+modl, with i being the row index.

    bordr : int
        Determines the column for which the xij start to touch the right border

    bordl : int
        Determines the column for which the xij start to touch the left border

    stuck : bool
        If True it means that the first and last column have, respectively, the same sparsity as the second and second
        to last column.
    """
    if stuck and starts == (B_nbasis - 1):
        return list(range(starts - 1 - p + modl, S_nbasis))

    if bordr <= starts:
        return list(range(starts - p + modl, S_nbasis))

    if bordl <= starts or int(stuck) <= starts:
        aux_indices = list(
            range(
                starts - p + modl if bordl <= starts else 0,
                starts + p + modr + 1,
            ),
        )
        cont = 1
        for j in range(starts + 1, ends + 1):
            if j >= bordr:
                break
            aux_indices.append(starts + p + modr + cont)
            cont += 1
    elif stuck and starts == 0:
        aux_indices = list(range(0, starts + 1 + p + modr + 1))
        cont = 1
        for j in range(starts + 2, ends + 1):
            if j >= bordr:
                break
            aux_indices.append(starts + p + modr + 1 + cont)
            cont += 1

    return aux_indices


def get_non_zero_B_spline_indices(periodic, IoH, p, B_nbasis, starts, ends, Basis_functions_indices_B):
    """This function builds a list with the B-spline indices of those B-splines that have a non-zero contribution to the FE coefficients the current MPI rank needs for building
    the BasisProjectionOperatorLocal.

    Parameters
    ----------
    periodic : np bool array
        1d array containing the information about whether we haver periodic (True) or clamped (False) boundary conditions.

    IoH : list char
        1d list of 3 chars, they must be either an I to denote Interpolation in this direction or an H to denote Histopolation.

    p : 1D int array
        1d array of 3 ints, they denote the degree of the B-splines for each one of the three spatial directions.

    B_nbasis : np int array
        1d array containing the number of B_splines in each spatial direction.

    starts : np int array
        1d array containing for each spatial direction the start index of the FE coefficients the current MPI rank is responsible for.

    ends : np int array
        1d array containing for each spatial direction the end index of the FE coefficients the current MPI rank is responsible for.

    Basis_functions_indices_B : list
        empty list to which we append the arrays with the desire B-spline indices for each spatial direction.

    """
    for i, per in enumerate(periodic):
        if IoH[i] not in {"I", "H"}:
            raise Exception("The list IoH must have as elements either the letter I or the letter H.")

        modr = -1 if IoH[i] == "I" else 0

        if per:  # Periodic
            aux_indices = get_sparsity_pattern_periodic(p[i], B_nbasis[i], starts[i], ends[i], modr, 1)
        else:  # Clamped
            bordr = B_nbasis[i] - p[i] + (1 if IoH[i] == "I" else 0)
            stuck = IoH[i] == "I"
            aux_indices = get_sparsity_pattern_clamped(
                p[i],
                B_nbasis[i],
                B_nbasis[i],
                starts[i],
                ends[i],
                modr,
                1,
                bordr,
                p[i] - 1,
                stuck,
            )

        Basis_functions_indices_B.append(xp.array(aux_indices))


def get_non_zero_D_spline_indices(periodic, IoH, p, D_nbasis, starts, ends, Basis_functions_indices_D):
    """This function builds a list with the D-spline indices of those D-splines that have a non-zero contribution to the FE coefficients the current MPI rank needs for building
    the BasisProjectionOperatorLocal.

    Parameters
    ----------
    periodic : np bool array
        1d array containing the information about whether we haver periodic (True) or clamped (False) boundary conditions.

    IoH : list char
        1d list of 3 chars, they must be either an I to denote Interpolation in this direction or an H to denote Histopolation.

    p : 1D int array
        1d array of 3 ints, they denote the degree of the B-splines (not the D-splines) for each one of the three spatial directions.

    D_nbasis : np int array
        1d array containing the number of D_splines in each spatial direction.

    starts : np int array
        1d array containing for each spatial direction the start index of the FE coefficients the current MPI rank is responsible for.

    ends : np int array
        1d array containing for each spatial direction the end index of the FE coefficients the current MPI rank is responsible for.

    Basis_functions_indices_D : list
        empty list to which we append the arrays with the desire D-spline indices for each spatial direction.

    """
    for i, per in enumerate(periodic):
        if IoH[i] not in {"I", "H"}:
            raise Exception("The list IoH must have as elements either the letter I or the letter H.")

        if per:  # Periodic
            if IoH[i] == "I":
                modr, modl = (-2, 1) if p[i] != 1 else (-1, 0)
            else:  # IoH[i] == "H"
                modr, modl = -1, 1
            aux_indices = get_sparsity_pattern_periodic(p[i], D_nbasis[i], starts[i], ends[i], modr, modl)
        else:  # Clamped
            modr = -2 if IoH[i] == "I" else -1
            bordr = D_nbasis[i] + (2 if IoH[i] == "I" else 1) - p[i]
            stuck = IoH[i] == "I"
            aux_indices = get_sparsity_pattern_clamped(
                p[i],
                D_nbasis[i] + 1,
                D_nbasis[i],
                starts[i],
                ends[i],
                modr,
                1,
                bordr,
                p[i] - 1,
                stuck,
            )

        Basis_functions_indices_D.append(xp.array(aux_indices))


def build_translation_list_for_non_zero_spline_indices(
    B_nbasis,
    D_nbasis,
    Basis_functions_indices_B,
    Basis_functions_indices_D,
    sp_id,
    Basis_function_indices_agreggated_B=None,
    Basis_function_indices_agreggated_D=None,
):
    """This function build index translation lists that given the Basis function index tell us at which entry of self._Basis_functions_indices_B(or D) that index is found. For the vector valued spaces
    it also populates the Basis_function_indices_agreggated_B(or D) list of arrays.

    Parameters
    ----------
    B_nbasis : np int array
        1d array containing the number of B_splines in each spatial direction.

    D_nbasis : np int array
        1d array containing the number of D_splines in each spatial direction.

    Basis_functions_indices_B : list
        list of int arrays with the indices of those B-splines that produce non-zero entries in the BasisProjectionOperatorLocal for the rows relevant to the current MPI rank.
        It contains 3 arrays of ints each one taking care of one spatial direction.

    Basis_functions_indices_D : list
        list of int arrays with the indices of those D-splines that produce non-zero entries in the BasisProjectionOperatorLocal for the rows relevant to the current MPI rank.
        It contains 3 arrays of ints each one taking care of one spatial direction.

    sp_id : string
        Space id, could be 'H1', 'Hcurl', 'Hdiv', 'L2' or 'H1vec'.

    Basis_function_indices_agreggated_B : list
        List of 3 int arrays. Basis_function_indices_agreggated_B[i][j] = -1 if the jth B-spline is not necessary for any of the three block entries in the ith spatial direction,
        otherwise it is 0.

    Basis_function_indices_agreggated_D : list
        List of 3 int arrays. Basis_function_indices_agreggated_D[i][j] = -1 if the jth D-spline is not necessary for any of the three block entries in the ith spatial direction,
        otherwise it is 0.

    Returns
    -------
    translation_indices_B_or_D_splines[0] : Dictionary
        This dictionary has two np int arrays full of -1 as elements the one with key 'B' for the B-splines and the one with key 'D' for the D-splines. They shall be filled by this
        function in such a way that ith entry of the list has the index of self._Basis_functions_indices_B(or D) where the ith B(D)-spline label is stored. This applies for the first
        spatial direction.

    translation_indices_B_or_D_splines_[1] : Dictionary
        This dictionary has two np int arrays full of -1 as elements the one with key 'B' for the B-splines and the one with key 'D' for the D-splines. They shall be filled by this
        function in such a way that ith entry of the list has the index of self._Basis_functions_indices_B(or D) where the ith B(D)-spline label is stored. This applies for the second
        spatial direction.

    translation_indices_B_or_D_splines[2] : Dictionary
        This dictionary has two np int arrays full of -1 as elements the one with key 'B' for the B-splines and the one with key 'D' for the D-splines. They shall be filled by this
        function in such a way that ith entry of the list has the index of self._Basis_functions_indices_B(or D) where the ith B(D)-spline label is stored. This applies for the third
        spatial direction.

    """
    translation_indices_B_or_D_splines = [
        {
            "B": xp.full((B_nbasis[h]), fill_value=-1, dtype=int),
            "D": xp.full((D_nbasis[h]), fill_value=-1, dtype=int),
        }
        for h in range(3)
    ]

    for h in range(3):
        translation_indices_B_or_D_splines[h]["B"][Basis_functions_indices_B[h]] = xp.arange(
            len(Basis_functions_indices_B[h]),
        )
        translation_indices_B_or_D_splines[h]["D"][Basis_functions_indices_D[h]] = xp.arange(
            len(Basis_functions_indices_D[h]),
        )

        if sp_id in {"Hcurl", "Hdiv", "H1vec"}:
            Basis_function_indices_agreggated_B[h][Basis_functions_indices_B[h]] = 0
            Basis_function_indices_agreggated_D[h][Basis_functions_indices_D[h]] = 0

    return (
        translation_indices_B_or_D_splines[0],
        translation_indices_B_or_D_splines[1],
        translation_indices_B_or_D_splines[2],
    )


def evaluate_relevant_splines_at_relevant_points(
    localpts,
    Bspaces_1d,
    Dspaces_1d,
    Basis_functions_indices_B,
    Basis_functions_indices_D,
):
    """This function evaluates all the B and D-splines that produce non-zeros in the BasisProjectionOperatorLocal's rows that belong to the current MPI rank over all the local evaluation points.
    They are store as float arrays in a dictionary of lists.

    Parameters
    ----------
    localpts : list
        list of 3 float arrays, the ith array contains the points on the ith spatial direction this MPI rank needs to compute its share of FE coefficients.

    Bspaces_1d : list
        list of tuples, each tuple has three elements the ith one being the psydac.fem.splines.SplineSpace for H1 on the ith spatial direction.

    Dspaces_1d : list
        list of tuples, each tuple has three elements the ith one being the psydac.fem.splines.SplineSpace for L2 on the ith spatial direction.

    Basis_functions_indices_B : list
        list of int arrays with the indices of those B-splines that produce non-zero entries in the BasisProjectionOperatorLocal for the rows relevant to the current MPI rank.
        It contains 3 arrays of ints each one taking care of one spatial direction.

    Basis_functions_indices_D : list
        list of int arrays with the indices of those D-splines that produce non-zero entries in the BasisProjectionOperatorLocal for the rows relevant to the current MPI rank.
        It contains 3 arrays of ints each one taking care of one spatial direction.

    Returns
    -------
    values_B_or_D_splines[0] : Dictionary
        This dictionary has two lists each one full with np float arrays. The one with key 'B' for the B-splines and the one with key 'D' for the D-splines. They shall be filled by this
        function in such a way that the ith entry of the list has the values of the B-spline with index  Basis_functions_indices_B[0][i]  evaluated at all the localpoints in the first spatial direction.

    values_B_or_D_splines[1] : Dictionary
        This dictionary has two lists each one full with np float arrays. The one with key 'B' for the B-splines and the one with key 'D' for the D-splines. They shall be filled by this
        function in such a way that the ith entry of the list has the values of the B-spline with index  Basis_functions_indices_B[1][i]  evaluated at all the localpoints in the second spatial direction.

    values_B_or_D_splines[2] : Dictionary
        This dictionary has two lists each one full with np float arrays. The one with key 'B' for the B-splines and the one with key 'D' for the D-splines. They shall be filled by this
        function in such a way that the ith entry of the list has the values of the B-spline with index  Basis_functions_indices_B[2][i]  evaluated at all the localpoints in the third spatial direction.
    """
    # Initialize the result dictionary
    values_B_or_D_splines = [{"B": [], "D": []} for _ in range(3)]

    for h in range(3):
        # Reshape localpts[h] if necessary
        localpts_reshaped = (
            localpts[h].reshape((xp.shape(localpts[h])[0], 1)) if len(xp.shape(localpts[h])) == 1 else localpts[h]
        )

        # Get spans and evaluation values for B-splines and D-splines
        spans, values = get_span_and_basis(localpts_reshaped, Bspaces_1d[0][h])
        spans_D, values_D = get_span_and_basis(localpts_reshaped, Dspaces_1d[0][h])

        # Extract properties for B and D splines
        b_props = Bspaces_1d[0][h]
        d_props = Dspaces_1d[0][h]

        # Get indices and values for splines
        eval_indices_B, values_B = get_values_and_indices_splines(
            b_props.nbasis,
            b_props.degree,
            b_props.periodic,
            spans,
            values,
        )
        eval_indices_D, values_D = get_values_and_indices_splines(
            d_props.nbasis,
            d_props.degree,
            d_props.periodic,
            spans_D,
            values_D,
        )

        # Populate the dictionary with values of B and D splines
        values_B_or_D_splines[h]["B"] = [
            get_one_spline(i, values_B, eval_indices_B) for i in Basis_functions_indices_B[h]
        ]
        values_B_or_D_splines[h]["D"] = [
            get_one_spline(i, values_D, eval_indices_D) for i in Basis_functions_indices_D[h]
        ]

    return values_B_or_D_splines[0], values_B_or_D_splines[1], values_B_or_D_splines[2]


def determine_non_zero_rows_for_each_spline(
    Basis_functions_indices_B,
    Basis_functions_indices_D,
    starts,
    ends,
    p,
    B_nbasis,
    D_nbasis,
    periodic,
    IoH,
):
    """This function determines for which rows (amongst those belonging to the current MPI rank) of the BasisProjectionOperatorLocal each B and D spline, of relevance for the current MPI rank, produces
    non-zero entries and annotates this regions of non-zeros by saving the rows at which each region starts and ends.

    Parameters
    ----------
    Basis_functions_indices_B : list
        list of int arrays with the indices of those B-splines that produce non-zero entries in the BasisProjectionOperatorLocal for the rows relevant to the current MPI rank.
        It contains 3 arrays of ints each one taking care of one spatial direction.

    Basis_functions_indices_D : list
        list of int arrays with the indices of those D-splines that produce non-zero entries in the BasisProjectionOperatorLocal for the rows relevant to the current MPI rank.
        It contains 3 arrays of ints each one taking care of one spatial direction.

    starts : np int array
        1d array containing for each spatial direction the start index of the FE coefficients the current MPI rank is responsible for.

    ends : np int array
        1d array containing for each spatial direction the end index of the FE coefficients the current MPI rank is responsible for.

    p : 1D int array
        1d array of 3 ints, they denote the degree of the B-splines (not the D-splines) for each one of the three spatial directions.

    B_nbasis : np int array
        1d array containing the number of B_splines in each spatial direction.

    D_nbasis : np int array
        1d array containing the number of D_splines in each spatial direction.

    periodic : np bool array
        1d array containing the information about whether we haver periodic (True) or clamped (False) boundary conditions.

    IoH : list bool
        1d list of 3 bools, they must be either a False to denote Interpolation in this direction or a True to denote Histopolation.

    Returns
    -------
    rows_B_or_D_splines[0] : dictionary
        This dictionary contains two lists one pertaining to B-splines and the other to D-splines. For instance the ith element of the list related to B-splines is an array of integers that tell us
        the start indices of the rows of non-zeros produced by the B-spline with index given by Basis_functions_indices_B[0][i]. This is valid for the first spatial direction.

    rows_B_or_D_splines[1] : dictionary
        This dictionary contains two lists one pertaining to B-splines and the other to D-splines. For instance the ith element of the list related to B-splines is an array of integers that tell us
        the start indices of the rows of non-zeros produced by the B-spline with index given by Basis_functions_indices_B[1][i]. This is valid for the second spatial direction.

    rows_B_or_D_splines[2] : dictionary
        This dictionary contains two lists one pertaining to B-splines and the other to D-splines. For instance the ith element of the list related to B-splines is an array of integers that tell us
        the start indices of the rows of non-zeros produced by the B-spline with index given by Basis_functions_indices_B[2][i]. This is valid for the third spatial direction.

    rowe_B_or_D_splines[0] : dictionary
        This dictionary contains two lists one pertaining to B-splines and the other to D-splines. For instance the ith element of the list related to B-splines is an array of integers that tell us
        the end indices of the rows of non-zeros produced by the B-spline with index given by Basis_functions_indices_B[0][i]. This is valid for the first spatial direction.

    rowe_B_or_D_splines[1] : dictionary
        This dictionary contains two lists one pertaining to B-splines and the other to D-splines. For instance the ith element of the list related to B-splines is an array of integers that tell us
        the end indices of the rows of non-zeros produced by the B-spline with index given by Basis_functions_indices_B[0][i]. This is valid for the second spatial direction.

    rowe_B_or_D_splines[2] : dictionary
        This dictionary contains two lists one pertaining to B-splines and the other to D-splines. For instance the ith element of the list related to B-splines is an array of integers that tell us
        the end indices of the rows of non-zeros produced by the B-spline with index given by Basis_functions_indices_B[0][i]. This is valid for the third spatial direction.
    """
    rows_B_or_D_splines = [{"B": [], "D": []} for _ in range(3)]
    rowe_B_or_D_splines = [{"B": [], "D": []} for _ in range(3)]

    def process_splines(indices, nbasis, is_D, h):
        for i in indices[h]:
            aux = xp.zeros((ends[h] + 1 - starts[h]), dtype=int)
            get_rows(
                int(i),
                int(starts[h]),
                int(ends[h]),
                int(p[h]),
                int(nbasis[h]),
                bool(periodic[h]),
                bool(IoH[h]),
                is_D,
                aux,
            )
            rangestart, rangeend = transform_into_ranges(aux)
            key = "D" if is_D else "B"
            rows_B_or_D_splines[h][key].append(xp.array(rangestart, dtype=int))
            rowe_B_or_D_splines[h][key].append(xp.array(rangeend, dtype=int))

    for h in range(3):
        process_splines(Basis_functions_indices_B, B_nbasis, False, h)
        process_splines(Basis_functions_indices_D, D_nbasis, True, h)

    return (
        rows_B_or_D_splines[0],
        rows_B_or_D_splines[1],
        rows_B_or_D_splines[2],
        rowe_B_or_D_splines[0],
        rowe_B_or_D_splines[1],
        rowe_B_or_D_splines[2],
    )


def get_splines_that_are_relevant_for_at_least_one_block(
    Basis_function_indices_agreggated_B,
    Basis_function_indices_agreggated_D,
):
    """This function builds one list with all the B-spline indices (and another one for the D-splines) that are required for at least one block of the FE coefficients
    the current MPI rank needs to build its share of the BasisProjectionOperatorLocal.

    Parameters
    ----------
    Basis_function_indices_agreggated_B : list
        List of 3 int arrays. Basis_function_indices_agreggated_B[i][j] = -1 if the jth B-spline is not necessary for any of the three block entries in the ith spatial direction,
        otherwise it is 0.

    Basis_function_indices_agreggated_D : list
        List of 3 int arrays. Basis_function_indices_agreggated_D[i][j] = -1 if the jth D-spline is not necessary for any of the three block entries in the ith spatial direction,
        otherwise it is 0.

    Returns
    -------
    Basis_function_indices_mark_B : 2d int list
        Basis_function_indices_mark_B[i] contains a list of B-spline indices that are needed on the ith spatial direction by the current MPI rank for computing at least one of
        the three block entries of the FE coefficient BlockVector that is in turn used to build the BasisProjectionOperatorLocal.

    Basis_function_indices_mark_D : 2d int list
        Basis_function_indices_mark_D[i] contains a list of D-spline indices that are needed on the ith spatial direction by the current MPI rank for computing at least one of
        the three block entries of the FE coefficient BlockVector that is in turn used to build the BasisProjectionOperatorLocal.

    """

    def find_indices_with_zero(aggregated_list):
        return [[j for j, value in enumerate(sublist) if value == 0] for sublist in aggregated_list]

    Basis_function_indices_mark_B = find_indices_with_zero(Basis_function_indices_agreggated_B[:3])
    Basis_function_indices_mark_D = find_indices_with_zero(Basis_function_indices_agreggated_D[:3])

    return Basis_function_indices_mark_B, Basis_function_indices_mark_D


def is_spline_zero_at_quadrature_points(
    Basis_functions_indices_B,
    Basis_functions_indices_D,
    localpts,
    p,
    values_B_or_D_splines,
    translation_indices_B_or_D_splines,
    necessary_direction,
):
    """This function builds three dictionaries (one per spatial direction), each one of them has 2 entries one for B-splines and one for D-splines. Each entry is a list that shall be populated
    by numpy int arrays. Each array pertains to one B(D)-spline, and it has as many elements as different integrals the current MPI rank must compute for the given spatial direction. By the end of
    this function these arrays shall have a one if the given spline is non-zero for the corresponding integral, but if the spline is zero for al quadrature points of the integral then the array
    shall have a zero.

    Parameters
    ----------
    Basis_functions_indices_B : list
        list of int arrays with the indices of those B-splines that produce non-zero entries in the BasisProjectionOperatorLocal for the rows relevant to the current MPI rank.
        It contains 3 arrays of ints each one taking care of one spatial direction.

    Basis_functions_indices_D : list
        list of int arrays with the indices of those D-splines that produce non-zero entries in the BasisProjectionOperatorLocal for the rows relevant to the current MPI rank.
        It contains 3 arrays of ints each one taking care of one spatial direction.

    localpts : list
        list of 3 float arrays, the ith array contains the points on the ith spatial direction this MPI rank needs to compute its share of FE coefficients.

    p : 1D int array
        1d array of 3 ints, they denote the degree of the B-splines (not the D-splines) for each one of the three spatial directions.

    values_B_or_D_splines : list of three dictionaries
        Each dictionary takes care of one spatial direction and has two lists each one full with np float arrays. The one with key 'B' for the B-splines and the one with key 'D'
        for the D-splines. For the h-th dictionary the ith entry (with key 'B') of the list has the values of the B-spline with index  Basis_functions_indices_B[h][i]  evaluated
        at all the localpoints in the h-th spatial direction.

    translation_indices_B_or_D_splines : List of three dictionaries
        Each dictionary takes care of one spatial direction and has two np int arrays the one with key 'B' for the B-splines and the one with key 'D' for the D-splines.
        For the h-th dictionary the ith entry of the array has the index of Basis_functions_indices_B(or D) where the ith B(D)-spline label is stored.
        This applies for the h-th spatial direction.

    necessary_direction : list
        This list has three bools, if the ith element is False it means that we do not have histopolation on the direction i+1, so we do not need to compute this function for that direction.

    Returns
    -------
    are_zero_B_or_D_splines[0] : dictionary
        This dictionary has two entries one for B-splines and one for D-splines. As explained at the begining of this documentation each entry is a list of np int arrays, that tell us if a given
        spline is zero for all quadrature points used to compute one of the required integrals. This particular dictionary takes care of the first spatial direction

    are_zero_B_or_D_splines[1] : dictionary
        This dictionary has two entries one for B-splines and one for D-splines. As explained at the begining of this documentation each entry is a list of np int arrays, that tell us if a given
        spline is zero for all quadrature points used to compute one of the required integrals. This particular dictionary takes care of the second spatial direction

    are_zero_B_or_D_splines[2] : dictionary
        This dictionary has two entries one for B-splines and one for D-splines. As explained at the begining of this documentation each entry is a list of np int arrays, that tell us if a given
        spline is zero for all quadrature points used to compute one of the required integrals. This particular dictionary takes care of the third spatial direction

    """
    are_zero_B_or_D_splines = [{"B": [], "D": []} for _ in range(3)]

    for h in range(3):
        if necessary_direction[h]:
            for i in Basis_functions_indices_B[h]:
                Auxiliar = xp.ones((xp.shape(localpts[h])[0]), dtype=int)
                are_quadrature_points_zero(
                    Auxiliar,
                    int(
                        p[h],
                    ),
                    values_B_or_D_splines[h]["B"][translation_indices_B_or_D_splines[h]["B"][i]],
                )
                are_zero_B_or_D_splines[h]["B"].append(Auxiliar)

            for i in Basis_functions_indices_D[h]:
                Auxiliar = xp.ones((xp.shape(localpts[h])[0]), dtype=int)
                are_quadrature_points_zero(
                    Auxiliar,
                    int(
                        p[h],
                    ),
                    values_B_or_D_splines[h]["D"][translation_indices_B_or_D_splines[h]["D"][i]],
                )
                are_zero_B_or_D_splines[h]["D"].append(Auxiliar)

    return are_zero_B_or_D_splines[0], are_zero_B_or_D_splines[1], are_zero_B_or_D_splines[2]
