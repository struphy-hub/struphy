import inspect
import time

import cunumpy as xp
import matplotlib.pyplot as plt
import pytest
from psydac.ddm.mpi import MockComm
from psydac.ddm.mpi import mpi as MPI

from struphy.bsplines.bsplines import basis_funs, find_span
from struphy.bsplines.evaluation_kernels_1d import evaluation_kernel_1d
from struphy.feec.basis_projection_ops import BasisProjectionOperator, BasisProjectionOperatorLocal
from struphy.feec.local_projectors_kernels import fill_matrix_column
from struphy.feec.psydac_derham import Derham
from struphy.feec.utilities_local_projectors import get_one_spline, get_span_and_basis, get_values_and_indices_splines


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


@pytest.mark.parametrize("Nel", [[14, 16, 18]])
@pytest.mark.parametrize("p", [[5, 4, 3]])
@pytest.mark.parametrize("spl_kind", [[True, False, False], [False, True, False], [False, False, True]])
def test_local_projectors_compare_global(Nel, p, spl_kind):
    """Tests the Local-projectors, by comparing them to the analytical function as well as to the global projectors."""
    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    timei = time.time()
    # create derham object
    derham = Derham(Nel, p, spl_kind, comm=comm, local_projectors=True)
    timef = time.time()
    print("Time for building Derham = " + str(timef - timei))

    # constant function
    def f(e1, e2, e3):
        return xp.sin(2.0 * xp.pi * e1) * xp.cos(4.0 * xp.pi * e2) * xp.sin(6.0 * xp.pi * e3)

    # f = lambda e1, e2, e3: xp.sin(2.0*xp.pi*e1) * xp.cos(4.0*xp.pi*e2)
    # evaluation points
    e1 = xp.linspace(0.0, 1.0, 10)
    e2 = xp.linspace(0.0, 1.0, 9)
    e3 = xp.linspace(0.0, 1.0, 8)

    ee1, ee2, ee3 = xp.meshgrid(e1, e2, e3, indexing="ij")

    # loop over spaces
    for sp_id, sp_key in derham.space_to_form.items():
        P_Loc = derham.P[sp_key]

        out = derham.Vh[sp_key].zeros()

        # field for local projection output
        field = derham.create_spline_function("fh", sp_id)

        # field for global projection output
        fieldg = derham.create_spline_function("fhg", sp_id)

        # project test function
        if sp_id in ("H1", "L2"):
            f_analytic = f
        else:
            # def f_analytic(e1, e2, e3):
            # return f(e1, e2, e3), f(e1, e2, e3), f(e1, e2, e3)
            f_analytic = (f, f, f)

        timei = time.time()
        vec = P_Loc(f_analytic)
        timef = time.time()
        exectime = timef - timei

        timeig = time.time()
        vecg = derham._P[sp_key](f_analytic)
        timefg = time.time()
        exectimeg = timefg - timeig

        field.vector = vec
        field_vals = field(e1, e2, e3)

        fieldg.vector = vecg
        fieldg_vals = fieldg(e1, e2, e3)

        if sp_id in ("H1", "L2"):
            err = xp.max(xp.abs(f_analytic(ee1, ee2, ee3) - field_vals))
            # Error comparing the global and local projectors
            errg = xp.max(xp.abs(fieldg_vals - field_vals))

        else:
            err = xp.zeros(3)
            err[0] = xp.max(xp.abs(f(ee1, ee2, ee3) - field_vals[0]))
            err[1] = xp.max(xp.abs(f(ee1, ee2, ee3) - field_vals[1]))
            err[2] = xp.max(xp.abs(f(ee1, ee2, ee3) - field_vals[2]))

            # Error comparing the global and local projectors
            errg = xp.zeros(3)
            errg[0] = xp.max(xp.abs(fieldg_vals[0] - field_vals[0]))
            errg[1] = xp.max(xp.abs(fieldg_vals[1] - field_vals[1]))
            errg[2] = xp.max(xp.abs(fieldg_vals[2] - field_vals[2]))

        print(f"{sp_id =}, {xp.max(err) =}, {xp.max(errg) =},{exectime =}")
        if sp_id in ("H1", "H1vec"):
            assert xp.max(err) < 0.011
            assert xp.max(errg) < 0.011
        else:
            assert xp.max(err) < 0.1
            assert xp.max(errg) < 0.1


@pytest.mark.parametrize("direction", [0, 1, 2])
@pytest.mark.parametrize("pi", [3, 4])
@pytest.mark.parametrize("spl_kindi", [True, False])
def test_local_projectors_convergence(direction, pi, spl_kindi, do_plot=False):
    """Tests the convergence rate of the Local projectors along singleton dimensions, without mapping."""
    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # loop over different number of elements
    Nels = [2**n for n in range(3, 9)]
    errors = {"H1": [], "Hcurl": [], "Hdiv": [], "L2": [], "H1vec": []}
    figs = {}
    for sp_id in errors:
        figs[sp_id] = plt.figure(
            sp_id + ", Local-proj. convergence",
            figsize=(24, 16),
        )

    for n, Neli in enumerate(Nels):
        # test function
        def fun(eta):
            return xp.cos(4 * xp.pi * eta)

        # create derham object, test functions and evaluation points
        e1 = 0.0
        e2 = 0.0
        e3 = 0.0
        if direction == 0:
            Nel = [Neli, 1, 1]
            p = [pi, 1, 1]
            spl_kind = [spl_kindi, True, True]
            e1 = xp.linspace(0.0, 1.0, 100)
            e = e1
            c = 0

            def f(x, y, z):
                return fun(x)
        elif direction == 1:
            Nel = [1, Neli, 1]
            p = [1, pi, 1]
            spl_kind = [True, spl_kindi, True]
            e2 = xp.linspace(0.0, 1.0, 100)
            e = e2
            c = 1

            def f(x, y, z):
                return fun(y)
        elif direction == 2:
            Nel = [1, 1, Neli]
            p = [1, 1, pi]
            spl_kind = [True, True, spl_kindi]
            e3 = xp.linspace(0.0, 1.0, 100)
            e = e3
            c = 2

            def f(x, y, z):
                return fun(z)

        derham = Derham(Nel, p, spl_kind, comm=comm, local_projectors=True)

        # loop over spaces
        for sp_id, sp_key in derham.space_to_form.items():
            P_Loc = derham.P[sp_key]
            out = derham.Vh[sp_key].zeros()

            field = derham.create_spline_function("fh", sp_id)

            # project test function
            if sp_id in ("H1", "L2"):
                f_analytic = f
            else:
                f_analytic = (f, f, f)

            vec = P_Loc(f_analytic)
            veco = P_Loc(f_analytic, out=out)

            field.vector = vec
            field_vals = field(e1, e2, e3, squeeze_out=True)

            if sp_id in ("H1", "L2"):
                err = xp.max(xp.abs(f_analytic(e1, e2, e3) - field_vals))
                f_plot = field_vals
            else:
                err = [xp.max(xp.abs(exact(e1, e2, e3) - field_v)) for exact, field_v in zip(f_analytic, field_vals)]
                f_plot = field_vals[0]

            errors[sp_id] += [xp.max(err)]

            if do_plot:
                plt.figure(sp_id + ", Local-proj. convergence")
                plt.subplot(2, 4, n + 1)
                plt.plot(e, f(e1, e2, e3), "o")
                plt.plot(e, f_plot)
                plt.xlabel(f"eta{c}")
                plt.title(f"Nel[{c}] = {Nel[c]}")

            del P_Loc, out, field, vec, veco, field_vals

    rate_p1 = pi + 1
    rate_p0 = pi

    for sp_id in derham.space_to_form:
        line_for_rate_p1 = [Ne ** (-rate_p1) * errors[sp_id][0] / Nels[0] ** (-rate_p1) for Ne in Nels]
        line_for_rate_p0 = [Ne ** (-rate_p0) * errors[sp_id][0] / Nels[0] ** (-rate_p0) for Ne in Nels]

        m, _ = xp.polyfit(xp.log(Nels), xp.log(errors[sp_id]), deg=1)

        if sp_id in ("H1", "H1vec"):
            # Sometimes for very large number of elements the convergance rate falls of a bit since the error is already so small floating point impressions become relevant
            # for those cases is better to compute the convergance rate using only the information of Nel with smaller number
            if -m <= (pi + 1 - 0.1):
                m = -xp.log2(errors[sp_id][1] / errors[sp_id][2])
            print(f"{sp_id =}, fitted convergence rate = {-m}, degree = {pi}")
            assert -m > (pi + 1 - 0.1)
        else:
            # Sometimes for very large number of elements the convergance rate falls of a bit since the error is already so small floating point impressions become relevant
            # for those cases is better to compute the convergance rate using only the information of Nel with smaller number
            if -m <= (pi - 0.1):
                m = -xp.log2(errors[sp_id][1] / errors[sp_id][2])
            print(f"{sp_id =}, fitted convergence rate = {-m}, degree = {pi}")
            assert -m > (pi - 0.1)

        if do_plot:
            plt.figure(sp_id + ", Local-proj. convergence")
            plt.subplot(2, 4, 8)
            plt.loglog(Nels, errors[sp_id])
            plt.loglog(Nels, line_for_rate_p1, "k--")
            plt.loglog(Nels, line_for_rate_p0, "k--")
            plt.text(Nels[-2], line_for_rate_p1[-2], f"1/Nel^{rate_p1}")
            plt.text(Nels[-2], line_for_rate_p0[-2], f"1/Nel^{rate_p0}")
            plt.title(f"{sp_id =}, degree = {pi}")
            plt.xlabel("Nel")

    if do_plot and rank == 0:
        plt.show()


# Works only in one processor


def aux_test_replication_of_basis(Nel, plist, spl_kind):
    """Tests that the local projectors do not alter the basis functions."""
    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    derham = Derham(Nel, plist, spl_kind, comm=comm, local_projectors=True)

    # For B-splines
    sp_key = "0"
    P_Loc = derham.P[sp_key]
    spaces = derham.Vh_fem[sp_key].spaces
    space = spaces[0]
    N = space.nbasis
    ncells = space.ncells
    p = space.degree
    T = space.knots
    periodic = space.periodic
    basis = space.basis
    normalize = basis == "M"

    def make_basis_fun(i):
        def fun(etas, eta2, eta3):
            if isinstance(etas, float) or isinstance(etas, int):
                etas = xp.array([etas])
            out = xp.zeros_like(etas)
            for j, eta in enumerate(etas):
                span = find_span(T, p, eta)
                inds = xp.arange(span - p, span + 1) % N
                pos = xp.argwhere(inds == i)
                # print(f'{pos = }')
                if pos.size > 0:
                    pos = pos[0, 0]
                    out[j] = basis_funs(T, p, eta, span, normalize=normalize)[pos]
                else:
                    out[j] = 0.0
            return out

        return fun

    for j in range(N):
        fun = make_basis_fun(j)
        lambdas = P_Loc(fun).toarray()

        etas = xp.linspace(0.0, 1.0, 100)
        fun_h = xp.zeros(100)
        for k, eta in enumerate(etas):
            span = find_span(T, p, eta)
            ind1 = xp.arange(span - p, span + 1) % N
            basis = basis_funs(T, p, eta, span, normalize=normalize)
            fun_h[k] = evaluation_kernel_1d(p, basis, ind1, lambdas)

        if xp.max(xp.abs(fun(etas, 0.0, 0.0) - fun_h)) >= 10.0**-10:
            print(xp.max(xp.abs(fun(etas, 0.0, 0.0) - fun_h)))
        assert xp.max(xp.abs(fun(etas, 0.0, 0.0) - fun_h)) < 10.0**-10
        # print(f'{j = }, max error: {xp.max(xp.abs(fun(etas,0.0,0.0) - fun_h))}')

    # For D-splines

    def check_only_specified_entry_is_one(val, entry):
        # This functions verifies that all the values in the array val are zero (or very close to it) except for the one in the specified entry
        # which should be 1
        tol = 10.0**-3
        for i, value in enumerate(val):
            if i != entry:
                if abs(value) >= tol:
                    print(value, i, entry)
                assert abs(value) < tol
            else:
                if abs(value - 1.0) >= tol:
                    print(value, i, abs(value - 1.0))
                assert abs(value - 1.0) < tol

    sp_key = "3"
    sp_id = "L2"
    P_Loc = derham.P[sp_key]
    spaces = derham.Vh_fem[sp_key].spaces
    input = derham.Vh[sp_key].zeros()
    npts = derham.Vh[sp_key].npts
    field = derham.create_spline_function("fh", sp_id)

    counter = 0
    for col0 in range(npts[0]):
        for col1 in range(npts[1]):
            for col2 in range(npts[2]):
                input[col0, col1, col2] = 1.0
                input.update_ghost_regions()
                field.vector = input

                out = P_Loc(field)
                input[col0, col1, col2] = 0.0
                check_only_specified_entry_is_one(out.toarray(), counter)
                counter += 1


@pytest.mark.parametrize("Nel", [[5, 4, 1]])
@pytest.mark.parametrize("plist", [[3, 2, 1]])
@pytest.mark.parametrize("spl_kind", [[False, True, True]])
@pytest.mark.parametrize("out_sp_key", ["0", "1", "2", "3", "v"])
@pytest.mark.parametrize("in_sp_key", ["0", "1", "2", "3", "v"])
def test_basis_projection_operator_local(Nel, plist, spl_kind, out_sp_key, in_sp_key):
    import random

    from struphy.feec.utilities import compare_arrays, create_equal_random_arrays

    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    derham = Derham(Nel, plist, spl_kind, comm=comm, local_projectors=True)

    # The first step to test our BasisProjectionOperatorLocal is to build the B and D spline functions in such a way that we can evaluate them in parallel.
    # We cannot us the fields of a derham space to do this since the evaluation of the splines in this way is a collective operation, and we want our functions
    # to be able to be computed by each rank on its own.

    # We will need the FEM spline space that contains B-splines in all three directions.
    fem_space_B = derham.Vh_fem["0"]
    # FE space of one forms. That means that we have B-splines in all three spatial directions.
    W = fem_space_B
    W1ds = [W.spaces]

    # We will need the FEM spline space that contains D-splines in all three directions.
    fem_space_D = derham.Vh_fem["3"]
    # FE space of three forms. That means that we have D-splines in all three spatial directions.
    V = fem_space_D
    V1ds = [V.spaces]

    # Helper function to handle reshaping and getting spans and basis
    def process_eta(eta, w1d):
        if isinstance(eta, (float, int)):
            eta = xp.array([eta])
        if len(eta.shape) == 1:
            eta = eta.reshape((eta.shape[0], 1))
        spans, values = get_span_and_basis(eta, w1d)
        return spans, values

    # Generalized factory function
    def make_basis_fun(is_B, dim_idx, i):
        def fun(eta1, eta2, eta3):
            eta_map = [eta1, eta2, eta3]
            eta = eta_map[dim_idx]
            w1d = W1ds[0][dim_idx] if is_B else V1ds[0][dim_idx]

            out = xp.zeros_like(eta)
            for j1 in range(eta.shape[0]):
                for j2 in range(eta.shape[1]):
                    for j3 in range(eta.shape[2]):
                        spans, values = process_eta(eta[j1, j2, j3], w1d)

                        # Get spline properties
                        Nbasis = w1d.nbasis
                        degree = w1d.degree
                        periodic = w1d.periodic

                        # Evaluate spline and assign
                        eval_indices, spline_values = get_values_and_indices_splines(
                            Nbasis,
                            degree,
                            periodic,
                            spans,
                            values,
                        )
                        out[j1, j2, j3] = get_one_spline(i, spline_values, eval_indices)[0]
            return out

        return fun

    # random vectors
    if in_sp_key == "0" or in_sp_key == "3":
        varr, v = create_equal_random_arrays(derham.Vh_fem[in_sp_key], seed=4568)
        varr = varr[0].flatten()
    elif in_sp_key == "v" or in_sp_key == "1" or in_sp_key == "2":
        varraux, v = create_equal_random_arrays(derham.Vh_fem[in_sp_key], seed=4568)
        varr = []
        for i in varraux:
            aux = i.flatten()
            for j in aux:
                varr.append(j)

    # We get the local projector
    P_Loc = derham.P[out_sp_key]
    out = derham.Vh[out_sp_key].zeros()
    VFEM = derham.Vh_fem[out_sp_key]

    if out_sp_key == "0" or out_sp_key == "3":
        npts_out = derham.Vh[out_sp_key].npts
        starts = xp.array(out.starts, dtype=int)
        ends = xp.array(out.ends, dtype=int)
        pds = xp.array(out.pads, dtype=int)
        VFEM1ds = [VFEM.spaces]
        nbasis_out = xp.array([VFEM1ds[0][0].nbasis, VFEM1ds[0][1].nbasis, VFEM1ds[0][2].nbasis])
    else:
        npts_out = xp.array([sp.npts for sp in P_Loc.coeff_space.spaces])
        pds = xp.array([vi.pads for vi in P_Loc.coeff_space.spaces])
        starts = xp.array([vi.starts for vi in P_Loc.coeff_space.spaces])
        ends = xp.array([vi.ends for vi in P_Loc.coeff_space.spaces])
        starts = xp.array(starts, dtype=int)
        ends = xp.array(ends, dtype=int)
        pds = xp.array(pds, dtype=int)
        VFEM1ds = [comp.spaces for comp in VFEM.spaces]
        nbasis_out = xp.array(
            [
                [VFEM1ds[0][0].nbasis, VFEM1ds[0][1].nbasis, VFEM1ds[0][2].nbasis],
                [
                    VFEM1ds[1][0].nbasis,
                    VFEM1ds[1][1].nbasis,
                    VFEM1ds[1][2].nbasis,
                ],
                [VFEM1ds[2][0].nbasis, VFEM1ds[2][1].nbasis, VFEM1ds[2][2].nbasis],
            ],
        )

    if in_sp_key == "0" or in_sp_key == "3":
        npts_in = derham.Vh[in_sp_key].npts
    else:
        npts_in = xp.array([sp.npts for sp in derham.Vh_fem[in_sp_key].coeff_space.spaces])

    def define_basis(in_sp_key):
        def wrapper(dim, index, h=None):
            if in_sp_key == "0":
                return make_basis_fun(True, dim, index)
            elif in_sp_key == "3":
                return make_basis_fun(False, dim, index)
            elif in_sp_key == "v":
                return make_basis_fun(True, dim, index)
            elif in_sp_key == "1":
                if h == dim:
                    return make_basis_fun(False, dim, index)
                else:
                    return make_basis_fun(True, dim, index)
            elif in_sp_key == "2":
                if h != dim:
                    return make_basis_fun(False, dim, index)
                else:
                    return make_basis_fun(True, dim, index)
            else:
                raise ValueError(f"Unsupported in_sp_key: {in_sp_key}")

        # Define basis functions dynamically
        def basis1(i1, h=None):
            return wrapper(0, i1, h)

        def basis2(i2, h=None):
            return wrapper(1, i2, h)

        def basis3(i3, h=None):
            return wrapper(2, i3, h)

        return basis1, basis2, basis3

    basis1, basis2, basis3 = define_basis(in_sp_key)

    input = derham.Vh[in_sp_key].zeros()
    random.seed(42)
    if in_sp_key == "0" or in_sp_key == "3":
        npts_in = derham.Vh[in_sp_key].npts
        random_i0 = random.randrange(0, npts_in[0])
        random_i1 = random.randrange(0, npts_in[1])
        random_i2 = random.randrange(0, npts_in[2])
        starts_in = input.starts
        ends_in = input.ends
        if starts_in[0] <= random_i0 and random_i0 <= ends_in[0]:
            input[random_i0, random_i1, random_i2] = 1.0
        input.update_ghost_regions()
    else:
        npts_in = xp.array([sp.npts for sp in derham.Vh_fem[in_sp_key].coeff_space.spaces])
        random_h = random.randrange(0, 3)
        random_i0 = random.randrange(0, npts_in[random_h][0])
        random_i1 = random.randrange(0, npts_in[random_h][1])
        random_i2 = random.randrange(0, npts_in[random_h][2])
        starts_in = xp.array([sp.starts for sp in derham.Vh_fem[in_sp_key].coeff_space.spaces])
        ends_in = xp.array([sp.ends for sp in derham.Vh_fem[in_sp_key].coeff_space.spaces])
        if starts_in[random_h][0] <= random_i0 and random_i0 <= ends_in[random_h][0]:
            input[random_h][random_i0, random_i1, random_i2] = 1.0
        input.update_ghost_regions()

    # We define the matrix
    if out_sp_key == "0" or out_sp_key == "3":
        if in_sp_key == "0" or in_sp_key == "3":
            matrix = xp.zeros((npts_out[0] * npts_out[1] * npts_out[2], npts_in[0] * npts_in[1] * npts_in[2]))
        else:
            matrix = xp.zeros(
                (
                    npts_out[0] * npts_out[1] * npts_out[2],
                    npts_in[0][0] * npts_in[0][1] * npts_in[0][2]
                    + npts_in[1][0] * npts_in[1][1] * npts_in[1][2]
                    + npts_in[2][0] * npts_in[2][1] * npts_in[2][2],
                ),
            )

    else:
        if in_sp_key == "0" or in_sp_key == "3":
            matrix0 = xp.zeros((npts_out[0][0] * npts_out[0][1] * npts_out[0][2], npts_in[0] * npts_in[1] * npts_in[2]))
            matrix1 = xp.zeros((npts_out[1][0] * npts_out[1][1] * npts_out[1][2], npts_in[0] * npts_in[1] * npts_in[2]))
            matrix2 = xp.zeros((npts_out[2][0] * npts_out[2][1] * npts_out[2][2], npts_in[0] * npts_in[1] * npts_in[2]))
        else:
            matrix00 = xp.zeros(
                (
                    npts_out[0][0] * npts_out[0][1] * npts_out[0][2],
                    npts_in[0][0] * npts_in[0][1] * npts_in[0][2],
                ),
            )
            matrix10 = xp.zeros(
                (
                    npts_out[1][0] * npts_out[1][1] * npts_out[1][2],
                    npts_in[0][0] * npts_in[0][1] * npts_in[0][2],
                ),
            )
            matrix20 = xp.zeros(
                (
                    npts_out[2][0] * npts_out[2][1] * npts_out[2][2],
                    npts_in[0][0] * npts_in[0][1] * npts_in[0][2],
                ),
            )

            matrix01 = xp.zeros(
                (
                    npts_out[0][0] * npts_out[0][1] * npts_out[0][2],
                    npts_in[1][0] * npts_in[1][1] * npts_in[1][2],
                ),
            )
            matrix11 = xp.zeros(
                (
                    npts_out[1][0] * npts_out[1][1] * npts_out[1][2],
                    npts_in[1][0] * npts_in[1][1] * npts_in[1][2],
                ),
            )
            matrix21 = xp.zeros(
                (
                    npts_out[2][0] * npts_out[2][1] * npts_out[2][2],
                    npts_in[1][0] * npts_in[1][1] * npts_in[1][2],
                ),
            )

            matrix02 = xp.zeros(
                (
                    npts_out[0][0] * npts_out[0][1] * npts_out[0][2],
                    npts_in[2][0] * npts_in[2][1] * npts_in[2][2],
                ),
            )
            matrix12 = xp.zeros(
                (
                    npts_out[1][0] * npts_out[1][1] * npts_out[1][2],
                    npts_in[2][0] * npts_in[2][1] * npts_in[2][2],
                ),
            )
            matrix22 = xp.zeros(
                (
                    npts_out[2][0] * npts_out[2][1] * npts_out[2][2],
                    npts_in[2][0] * npts_in[2][1] * npts_in[2][2],
                ),
            )

    # We build the BasisProjectionOperator by hand
    if out_sp_key == "0" or out_sp_key == "3":
        if in_sp_key == "0" or in_sp_key == "3":
            # def f_analytic(e1,e2,e3): return (xp.sin(2.0*xp.pi*e1)+xp.cos(4.0*xp.pi*e2))*basis1(random_i0)(e1,e2,e3)*basis2(random_i1)(e1,e2,e3)*basis3(random_i2)(e1,e2,e3)
            # out = P_Loc(f_analytic)

            counter = 0
            for col0 in range(npts_in[0]):
                for col1 in range(npts_in[1]):
                    for col2 in range(npts_in[2]):

                        def f_analytic(e1, e2, e3):
                            return (
                                (xp.sin(2.0 * xp.pi * e1) + xp.cos(4.0 * xp.pi * e2))
                                * basis1(col0)(e1, e2, e3)
                                * basis2(col1)(e1, e2, e3)
                                * basis3(col2)(e1, e2, e3)
                            )

                        out = P_Loc(f_analytic)
                        fill_matrix_column(starts, ends, pds, counter, nbasis_out, matrix, out._data)

                        counter += 1

        else:
            counter = 0
            for h in range(3):
                for col0 in range(npts_in[h][0]):
                    for col1 in range(npts_in[h][1]):
                        for col2 in range(npts_in[h][2]):

                            def f_analytic(e1, e2, e3):
                                return (
                                    (xp.sin(2.0 * xp.pi * e1) + xp.cos(4.0 * xp.pi * e2))
                                    * basis1(col0, h)(e1, e2, e3)
                                    * basis2(col1, h)(e1, e2, e3)
                                    * basis3(col2, h)(e1, e2, e3)
                                )

                            out = P_Loc(f_analytic)
                            fill_matrix_column(starts, ends, pds, counter, nbasis_out, matrix, out._data)
                            counter += 1

    else:
        if in_sp_key == "0" or in_sp_key == "3":
            counter = 0
            for col0 in range(npts_in[0]):
                for col1 in range(npts_in[1]):
                    for col2 in range(npts_in[2]):

                        def f_analytic1(e1, e2, e3):
                            return (
                                (xp.sin(2.0 * xp.pi * e1) + xp.cos(4.0 * xp.pi * e2))
                                * basis1(col0)(e1, e2, e3)
                                * basis2(col1)(e1, e2, e3)
                                * basis3(col2)(e1, e2, e3)
                            )

                        def f_analytic2(e1, e2, e3):
                            return (
                                (xp.cos(2.0 * xp.pi * e2) + xp.cos(6.0 * xp.pi * e3))
                                * basis1(col0)(e1, e2, e3)
                                * basis2(col1)(e1, e2, e3)
                                * basis3(col2)(e1, e2, e3)
                            )

                        def f_analytic3(e1, e2, e3):
                            return (
                                (xp.sin(6.0 * xp.pi * e1) + xp.sin(4.0 * xp.pi * e3))
                                * basis1(col0)(e1, e2, e3)
                                * basis2(col1)(e1, e2, e3)
                                * basis3(col2)(e1, e2, e3)
                            )

                        out = P_Loc([f_analytic1, f_analytic2, f_analytic3])
                        fill_matrix_column(starts[0], ends[0], pds[0], counter, nbasis_out[0], matrix0, out[0]._data)
                        fill_matrix_column(starts[1], ends[1], pds[1], counter, nbasis_out[1], matrix1, out[1]._data)
                        fill_matrix_column(starts[2], ends[2], pds[2], counter, nbasis_out[2], matrix2, out[2]._data)
                        counter += 1

            matrix = xp.vstack((matrix0, matrix1, matrix2))

        else:
            for h in range(3):
                counter = 0
                for col0 in range(npts_in[h][0]):
                    for col1 in range(npts_in[h][1]):
                        for col2 in range(npts_in[h][2]):
                            if h == 0:

                                def f_analytic0(e1, e2, e3):
                                    return (
                                        (xp.sin(2.0 * xp.pi * e1) + xp.cos(4.0 * xp.pi * e2))
                                        * basis1(col0, h)(e1, e2, e3)
                                        * basis2(col1, h)(e1, e2, e3)
                                        * basis3(col2, h)(e1, e2, e3)
                                    )

                                def f_analytic1(e1, e2, e3):
                                    return (
                                        (xp.sin(10.0 * xp.pi * e1) + xp.cos(41.0 * xp.pi * e2))
                                        * basis1(col0, h)(e1, e2, e3)
                                        * basis2(col1, h)(e1, e2, e3)
                                        * basis3(col2, h)(e1, e2, e3)
                                    )

                                def f_analytic2(e1, e2, e3):
                                    return (
                                        (xp.sin(25.0 * xp.pi * e1) + xp.cos(49.0 * xp.pi * e2))
                                        * basis1(col0, h)(e1, e2, e3)
                                        * basis2(col1, h)(e1, e2, e3)
                                        * basis3(col2, h)(e1, e2, e3)
                                    )

                            elif h == 1:

                                def f_analytic0(e1, e2, e3):
                                    return (
                                        (xp.cos(2.0 * xp.pi * e2) + xp.cos(6.0 * xp.pi * e3))
                                        * basis1(col0, h)(e1, e2, e3)
                                        * basis2(col1, h)(e1, e2, e3)
                                        * basis3(col2, h)(e1, e2, e3)
                                    )

                                def f_analytic1(e1, e2, e3):
                                    return (
                                        (xp.cos(12.0 * xp.pi * e2) + xp.cos(62.0 * xp.pi * e3))
                                        * basis1(col0, h)(e1, e2, e3)
                                        * basis2(col1, h)(e1, e2, e3)
                                        * basis3(col2, h)(e1, e2, e3)
                                    )

                                def f_analytic2(e1, e2, e3):
                                    return (
                                        (xp.cos(25.0 * xp.pi * e2) + xp.cos(68.0 * xp.pi * e3))
                                        * basis1(col0, h)(e1, e2, e3)
                                        * basis2(col1, h)(e1, e2, e3)
                                        * basis3(col2, h)(e1, e2, e3)
                                    )
                            else:

                                def f_analytic0(e1, e2, e3):
                                    return (
                                        (xp.sin(6.0 * xp.pi * e1) + xp.sin(4.0 * xp.pi * e3))
                                        * basis1(col0, h)(e1, e2, e3)
                                        * basis2(col1, h)(e1, e2, e3)
                                        * basis3(col2, h)(e1, e2, e3)
                                    )

                                def f_analytic1(e1, e2, e3):
                                    return (
                                        (xp.sin(16.0 * xp.pi * e1) + xp.sin(43.0 * xp.pi * e3))
                                        * basis1(col0, h)(e1, e2, e3)
                                        * basis2(col1, h)(e1, e2, e3)
                                        * basis3(col2, h)(e1, e2, e3)
                                    )

                                def f_analytic2(e1, e2, e3):
                                    return (
                                        (xp.sin(65.0 * xp.pi * e1) + xp.sin(47.0 * xp.pi * e3))
                                        * basis1(col0, h)(e1, e2, e3)
                                        * basis2(col1, h)(e1, e2, e3)
                                        * basis3(col2, h)(e1, e2, e3)
                                    )

                            out = P_Loc([f_analytic0, f_analytic1, f_analytic2])
                            if h == 0:
                                fill_matrix_column(
                                    starts[0],
                                    ends[0],
                                    pds[0],
                                    counter,
                                    nbasis_out[0],
                                    matrix00,
                                    out[0]._data,
                                )
                                fill_matrix_column(
                                    starts[1],
                                    ends[1],
                                    pds[1],
                                    counter,
                                    nbasis_out[1],
                                    matrix10,
                                    out[1]._data,
                                )
                                fill_matrix_column(
                                    starts[2],
                                    ends[2],
                                    pds[2],
                                    counter,
                                    nbasis_out[2],
                                    matrix20,
                                    out[2]._data,
                                )

                            elif h == 1:
                                fill_matrix_column(
                                    starts[0],
                                    ends[0],
                                    pds[0],
                                    counter,
                                    nbasis_out[0],
                                    matrix01,
                                    out[0]._data,
                                )
                                fill_matrix_column(
                                    starts[1],
                                    ends[1],
                                    pds[1],
                                    counter,
                                    nbasis_out[1],
                                    matrix11,
                                    out[1]._data,
                                )
                                fill_matrix_column(
                                    starts[2],
                                    ends[2],
                                    pds[2],
                                    counter,
                                    nbasis_out[2],
                                    matrix21,
                                    out[2]._data,
                                )

                            elif h == 2:
                                fill_matrix_column(
                                    starts[0],
                                    ends[0],
                                    pds[0],
                                    counter,
                                    nbasis_out[0],
                                    matrix02,
                                    out[0]._data,
                                )
                                fill_matrix_column(
                                    starts[1],
                                    ends[1],
                                    pds[1],
                                    counter,
                                    nbasis_out[1],
                                    matrix12,
                                    out[1]._data,
                                )
                                fill_matrix_column(
                                    starts[2],
                                    ends[2],
                                    pds[2],
                                    counter,
                                    nbasis_out[2],
                                    matrix22,
                                    out[2]._data,
                                )
                            counter += 1

            matrix0 = xp.hstack((matrix00, matrix01, matrix02))
            matrix1 = xp.hstack((matrix10, matrix11, matrix12))
            matrix2 = xp.hstack((matrix20, matrix21, matrix22))
            matrix = xp.vstack((matrix0, matrix1, matrix2))

    # Now we build the same matrix using the BasisProjectionOperatorLocal
    if out_sp_key == "0" or out_sp_key == "3":
        if in_sp_key == "0" or in_sp_key == "3":

            def f_analytic(e1, e2, e3):
                return xp.sin(2.0 * xp.pi * e1) + xp.cos(4.0 * xp.pi * e2)

            matrix_new = BasisProjectionOperatorLocal(P_Loc, derham.Vh_fem[in_sp_key], [[f_analytic]], transposed=False)
        else:

            def f_analytic(e1, e2, e3):
                return xp.sin(2.0 * xp.pi * e1) + xp.cos(4.0 * xp.pi * e2)

            matrix_new = BasisProjectionOperatorLocal(
                P_Loc,
                derham.Vh_fem[in_sp_key],
                [
                    [f_analytic, f_analytic, f_analytic],
                ],
                transposed=False,
            )

    else:
        if in_sp_key == "0" or in_sp_key == "3":

            def f_analytic1(e1, e2, e3):
                return xp.sin(2.0 * xp.pi * e1) + xp.cos(4.0 * xp.pi * e2)

            def f_analytic2(e1, e2, e3):
                return xp.cos(2.0 * xp.pi * e2) + xp.cos(6.0 * xp.pi * e3)

            def f_analytic3(e1, e2, e3):
                return xp.sin(6.0 * xp.pi * e1) + xp.sin(4.0 * xp.pi * e3)

            matrix_new = BasisProjectionOperatorLocal(
                P_Loc,
                derham.Vh_fem[in_sp_key],
                [
                    [f_analytic1],
                    [
                        f_analytic2,
                    ],
                    [f_analytic3],
                ],
                transposed=False,
            )
        else:

            def f_analytic00(e1, e2, e3):
                return xp.sin(2.0 * xp.pi * e1) + xp.cos(4.0 * xp.pi * e2)

            def f_analytic01(e1, e2, e3):
                return xp.cos(2.0 * xp.pi * e2) + xp.cos(6.0 * xp.pi * e3)

            def f_analytic02(e1, e2, e3):
                return xp.sin(6.0 * xp.pi * e1) + xp.sin(4.0 * xp.pi * e3)

            def f_analytic10(e1, e2, e3):
                return xp.sin(10.0 * xp.pi * e1) + xp.cos(41.0 * xp.pi * e2)

            def f_analytic11(e1, e2, e3):
                return xp.cos(12.0 * xp.pi * e2) + xp.cos(62.0 * xp.pi * e3)

            def f_analytic12(e1, e2, e3):
                return xp.sin(16.0 * xp.pi * e1) + xp.sin(43.0 * xp.pi * e3)

            def f_analytic20(e1, e2, e3):
                return xp.sin(25.0 * xp.pi * e1) + xp.cos(49.0 * xp.pi * e2)

            def f_analytic21(e1, e2, e3):
                return xp.cos(25.0 * xp.pi * e2) + xp.cos(68.0 * xp.pi * e3)

            def f_analytic22(e1, e2, e3):
                return xp.sin(65.0 * xp.pi * e1) + xp.sin(47.0 * xp.pi * e3)

            matrix_new = BasisProjectionOperatorLocal(
                P_Loc,
                derham.Vh_fem[in_sp_key],
                [
                    [f_analytic00, f_analytic01, f_analytic02],
                    [
                        f_analytic10,
                        f_analytic11,
                        f_analytic12,
                    ],
                    [f_analytic20, f_analytic21, f_analytic22],
                ],
                transposed=False,
            )

    compare_arrays(matrix_new.dot(v), xp.matmul(matrix, varr), rank)

    print("BasisProjectionOperatorLocal test passed.")


@pytest.mark.parametrize("Nel", [[40, 1, 1]])
@pytest.mark.parametrize("plist", [[5, 1, 1]])
@pytest.mark.parametrize("spl_kind", [[False, True, True]])
@pytest.mark.parametrize("out_sp_key", ["0", "1", "2", "3", "v"])
@pytest.mark.parametrize("in_sp_key", ["0", "1", "2", "3", "v"])
def test_basis_projection_operator_local_new(Nel, plist, spl_kind, out_sp_key, in_sp_key, do_plot=False):
    import random

    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    derham = Derham(Nel, plist, spl_kind, comm=comm, local_projectors=True)

    # Building the B-splines
    # We will need the FEM spline space that contains D-splines in all three directions.
    fem_space_B = derham.Vh_fem["0"]
    # FE space of one forms. That means that we have B-splines in all three spatial directions.
    W = fem_space_B
    W1ds = [W.spaces]

    # We will need the FEM spline space that contains D-splines in all three directions.
    fem_space_D = derham.Vh_fem["3"]

    # FE space of three forms. That means that we have D-splines in all three spatial directions.
    V = fem_space_D
    V1ds = [V.spaces]

    # Helper function to handle reshaping and getting spans and basis
    def process_eta(eta, w1d):
        if isinstance(eta, (float, int)):
            eta = xp.array([eta])
        if len(eta.shape) == 1:
            eta = eta.reshape((eta.shape[0], 1))
        spans, values = get_span_and_basis(eta, w1d)
        return spans, values

    # Generalized factory function
    def make_basis_fun(is_B, dim_idx, i):
        def fun(eta1, eta2, eta3):
            eta_map = [eta1, eta2, eta3]
            eta = eta_map[dim_idx]
            w1d = W1ds[0][dim_idx] if is_B else V1ds[0][dim_idx]

            out = xp.zeros_like(eta)
            for j1 in range(eta.shape[0]):
                for j2 in range(eta.shape[1]):
                    for j3 in range(eta.shape[2]):
                        spans, values = process_eta(eta[j1, j2, j3], w1d)

                        # Get spline properties
                        Nbasis = w1d.nbasis
                        degree = w1d.degree
                        periodic = w1d.periodic

                        # Evaluate spline and assign
                        eval_indices, spline_values = get_values_and_indices_splines(
                            Nbasis,
                            degree,
                            periodic,
                            spans,
                            values,
                        )
                        out[j1, j2, j3] = get_one_spline(i, spline_values, eval_indices)[0]
            return out

        return fun

    def define_basis(in_sp_key):
        def wrapper(dim, index, h=None):
            if in_sp_key == "0":
                return make_basis_fun(True, dim, index)
            elif in_sp_key == "3":
                return make_basis_fun(False, dim, index)
            elif in_sp_key == "v":
                return make_basis_fun(True, dim, index)
            elif in_sp_key == "1":
                if h == dim:
                    return make_basis_fun(False, dim, index)
                else:
                    return make_basis_fun(True, dim, index)
            elif in_sp_key == "2":
                if h != dim:
                    return make_basis_fun(False, dim, index)
                else:
                    return make_basis_fun(True, dim, index)
            else:
                raise ValueError(f"Unsupported in_sp_key: {in_sp_key}")

        # Define basis functions dynamically
        def basis1(i1, h=None):
            return wrapper(0, i1, h)

        def basis2(i2, h=None):
            return wrapper(1, i2, h)

        def basis3(i3, h=None):
            return wrapper(2, i3, h)

        return basis1, basis2, basis3

    basis1, basis2, basis3 = define_basis(in_sp_key)

    # We get the local projector
    P_Loc = derham.P[out_sp_key]
    # We get the global projector
    P = derham._P[out_sp_key]

    input = derham.Vh[in_sp_key].zeros()
    random.seed(42)
    if in_sp_key == "0" or in_sp_key == "3":
        npts_in = derham.Vh[in_sp_key].npts
        random_i0 = random.randrange(0, npts_in[0])
        random_i1 = random.randrange(0, npts_in[1])
        random_i2 = random.randrange(0, npts_in[2])
        starts = input.starts
        ends = input.ends
        if starts[0] <= random_i0 and random_i0 <= ends[0]:
            input[random_i0, random_i1, random_i2] = 1.0
        input.update_ghost_regions()
    else:
        npts_in = xp.array([sp.npts for sp in derham.Vh_fem[in_sp_key].coeff_space.spaces])
        random_h = random.randrange(0, 3)
        random_i0 = random.randrange(0, npts_in[random_h][0])
        random_i1 = random.randrange(0, npts_in[random_h][1])
        random_i2 = random.randrange(0, npts_in[random_h][2])
        starts = xp.array([sp.starts for sp in derham.Vh_fem[in_sp_key].coeff_space.spaces])
        ends = xp.array([sp.ends for sp in derham.Vh_fem[in_sp_key].coeff_space.spaces])
        if starts[random_h][0] <= random_i0 and random_i0 <= ends[random_h][0]:
            input[random_h][random_i0, random_i1, random_i2] = 1.0
        input.update_ghost_regions()

    etas1 = xp.linspace(0.0, 1.0, 1000)
    etas2 = xp.array([0.5])

    etas3 = xp.array([0.5])
    meshgrid = xp.meshgrid(*[etas1, etas2, etas3], indexing="ij")

    # Now we build the same matrix using the BasisProjectionOperatorLocal and BasisProjectionOperator

    if out_sp_key == "0" or out_sp_key == "3":
        if in_sp_key == "0" or in_sp_key == "3":

            def f_analytic(e1, e2, e3):
                return xp.sin(2.0 * xp.pi * e1) + xp.sin(4.0 * xp.pi * e1)

            matrix_new = BasisProjectionOperatorLocal(P_Loc, derham.Vh_fem[in_sp_key], [[f_analytic]], transposed=False)
            matrix_global = BasisProjectionOperator(P, derham.Vh_fem[in_sp_key], [[f_analytic]], transposed=False)

            analytic_vals = (
                f_analytic(*meshgrid)
                * basis1(random_i0)(*meshgrid)
                * basis2(random_i1)(*meshgrid)
                * basis3(random_i2)(*meshgrid)
            )
        else:

            def f_analytic(e1, e2, e3):
                return xp.sin(2.0 * xp.pi * e1) + xp.cos(4.0 * xp.pi * e1)

            matrix_new = BasisProjectionOperatorLocal(
                P_Loc,
                derham.Vh_fem[in_sp_key],
                [
                    [f_analytic, f_analytic, f_analytic],
                ],
                transposed=False,
            )
            matrix_global = BasisProjectionOperator(
                P,
                derham.Vh_fem[in_sp_key],
                [
                    [f_analytic, f_analytic, f_analytic],
                ],
                transposed=False,
            )

            analytic_vals = (
                f_analytic(*meshgrid)
                * basis1(random_i0, random_h)(*meshgrid)
                * basis2(random_i1, random_h)(*meshgrid)
                * basis3(random_i2, random_h)(*meshgrid)
            )

    else:
        if in_sp_key == "0" or in_sp_key == "3":

            def f_analytic1(e1, e2, e3):
                return xp.sin(2.0 * xp.pi * e1) + xp.cos(4.0 * xp.pi * e1)

            def f_analytic2(e1, e2, e3):
                return xp.cos(2.0 * xp.pi * e1) + xp.cos(6.0 * xp.pi * e1)

            def f_analytic3(e1, e2, e3):
                return xp.sin(6.0 * xp.pi * e1) + xp.sin(4.0 * xp.pi * e1)

            matrix_new = BasisProjectionOperatorLocal(
                P_Loc,
                derham.Vh_fem[in_sp_key],
                [
                    [f_analytic1],
                    [
                        f_analytic2,
                    ],
                    [f_analytic3],
                ],
                transposed=False,
            )
            matrix_global = BasisProjectionOperator(
                P,
                derham.Vh_fem[in_sp_key],
                [
                    [f_analytic1],
                    [
                        f_analytic2,
                    ],
                    [f_analytic3],
                ],
                transposed=False,
            )

            analytic_vals = xp.array(
                [
                    f_analytic1(*meshgrid)
                    * basis1(random_i0)(*meshgrid)
                    * basis2(random_i1)(*meshgrid)
                    * basis3(random_i2)(*meshgrid),
                    f_analytic2(*meshgrid)
                    * basis1(random_i0)(*meshgrid)
                    * basis2(random_i1)(*meshgrid)
                    * basis3(random_i2)(*meshgrid),
                    f_analytic3(*meshgrid)
                    * basis1(random_i0)(*meshgrid)
                    * basis2(random_i1)(*meshgrid)
                    * basis3(random_i2)(*meshgrid),
                ],
            )
        else:

            def f_analytic00(e1, e2, e3):
                return xp.sin(2.0 * xp.pi * e1) + xp.cos(4.0 * xp.pi * e1)

            def f_analytic01(e1, e2, e3):
                return xp.cos(2.0 * xp.pi * e1) + xp.cos(6.0 * xp.pi * e1)

            def f_analytic02(e1, e2, e3):
                return xp.sin(6.0 * xp.pi * e1) + xp.sin(4.0 * xp.pi * e1)

            def f_analytic10(e1, e2, e3):
                return xp.sin(3.0 * xp.pi * e1) + xp.cos(4.0 * xp.pi * e1)

            def f_analytic11(e1, e2, e3):
                return xp.cos(2.0 * xp.pi * e1) + xp.cos(3.0 * xp.pi * e1)

            def f_analytic12(e1, e2, e3):
                return xp.sin(5.0 * xp.pi * e1) + xp.sin(3.0 * xp.pi * e1)

            def f_analytic20(e1, e2, e3):
                return xp.sin(5.0 * xp.pi * e1) + xp.cos(4.0 * xp.pi * e1)

            def f_analytic21(e1, e2, e3):
                return xp.cos(5.0 * xp.pi * e1) + xp.cos(6.0 * xp.pi * e1)

            def f_analytic22(e1, e2, e3):
                return xp.sin(5.0 * xp.pi * e1) + xp.sin(4.0 * xp.pi * e1)

            matrix_new = BasisProjectionOperatorLocal(
                P_Loc,
                derham.Vh_fem[in_sp_key],
                [
                    [f_analytic00, f_analytic01, f_analytic02],
                    [
                        f_analytic10,
                        f_analytic11,
                        f_analytic12,
                    ],
                    [f_analytic20, f_analytic21, f_analytic22],
                ],
                transposed=False,
            )
            matrix_global = BasisProjectionOperator(
                P,
                derham.Vh_fem[in_sp_key],
                [
                    [f_analytic00, f_analytic01, f_analytic02],
                    [
                        f_analytic10,
                        f_analytic11,
                        f_analytic12,
                    ],
                    [f_analytic20, f_analytic21, f_analytic22],
                ],
                transposed=False,
            )
            # Define the function mapping
            f_analytic_map = {
                0: [f_analytic00, f_analytic01, f_analytic02],
                1: [f_analytic10, f_analytic11, f_analytic12],
                2: [f_analytic20, f_analytic21, f_analytic22],
            }

            # Use the map to get analytic values
            analytic_vals = xp.array(
                [
                    f_analytic_map[dim][random_h](*meshgrid)
                    * basis1(random_i0, random_h)(*meshgrid)
                    * basis2(random_i1, random_h)(*meshgrid)
                    * basis3(random_i2, random_h)(*meshgrid)
                    for dim in range(3)
                ],
            )

    FE_loc = matrix_new.dot(input)
    FE_glo = matrix_global.dot(input)

    if out_sp_key == "0":
        out_sp_id = "H1"
    elif out_sp_key == "1":
        out_sp_id = "Hcurl"
    elif out_sp_key == "2":
        out_sp_id = "Hdiv"
    elif out_sp_key == "3":
        out_sp_id = "L2"
    elif out_sp_key == "v":
        out_sp_id = "H1vec"

    fieldloc = derham.create_spline_function("fh", out_sp_id)
    fieldloc.vector = FE_loc

    fieldglo = derham.create_spline_function("fh", out_sp_id)
    fieldglo.vector = FE_glo

    errorloc = xp.abs(fieldloc(*meshgrid) - analytic_vals)
    errorglo = xp.abs(fieldglo(*meshgrid) - analytic_vals)

    meanlocal = xp.mean(errorloc)
    maxlocal = xp.max(errorloc)

    meanglobal = xp.mean(errorglo)
    maxglobal = xp.max(errorglo)

    if isinstance(comm, MockComm):
        reducemeanlocal = meanlocal
    else:
        reducemeanlocal = comm.reduce(meanlocal, op=MPI.SUM, root=0)

    if rank == 0:
        reducemeanlocal = reducemeanlocal / world_size

    if isinstance(comm, MockComm):
        reducemaxlocal = maxlocal
    else:
        reducemaxlocal = comm.reduce(maxlocal, op=MPI.MAX, root=0)

    if isinstance(comm, MockComm):
        reducemeanglobal = meanglobal
    else:
        reducemeanglobal = comm.reduce(meanglobal, op=MPI.SUM, root=0)

    if rank == 0:
        reducemeanglobal = reducemeanglobal / world_size

    if isinstance(comm, MockComm):
        reducemaxglobal = maxglobal
    else:
        reducemaxglobal = comm.reduce(maxglobal, op=MPI.MAX, root=0)

    if rank == 0:
        assert reducemeanlocal < 10.0 * reducemeanglobal or reducemeanlocal < 10.0**-5
        print(f"{reducemeanlocal =}")
        print(f"{reducemaxlocal =}")
        print(f"{reducemeanglobal =}")
        print(f"{reducemaxglobal =}")

    if do_plot:
        if out_sp_key == "0" or out_sp_key == "3":
            plt.figure()
            plt.plot(etas1, fieldloc(*meshgrid)[:, 0, 0], "--", label="Local")
            plt.plot(etas1, analytic_vals[:, 0, 0], label="Analytic")
            plt.plot(etas1, fieldglo(*meshgrid)[:, 0, 0], "--", label="global")
            plt.xlabel(f"eta{0}")
            plt.title("Fitting one Basis function")
            plt.legend()
        else:
            for i in range(3):
                plt.figure()
                plt.plot(etas1, fieldloc(*meshgrid)[i][:, 0, 0], "--", label="Local")
                plt.plot(etas1, analytic_vals[i][:, 0, 0], label="Analytic")
                plt.plot(etas1, fieldglo(*meshgrid)[i][:, 0, 0], "--", label="global")
                plt.xlabel(f"eta{0}")
                plt.title("Fitting one Basis function, vector entry " + str(i))
                plt.legend()
        if rank == 0:
            plt.show()

    print("BasisProjectionOperatorLocal test passed.")


# Works only in one processor
def aux_test_spline_evaluation(Nel, plist, spl_kind):
    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    derham = Derham(Nel, plist, spl_kind, comm=comm, local_projectors=True)

    # The first step to test our BasisProjectionOperatorLocal is to build the B and D spline functions in such a way that we can evaluate them in parallel.
    # We cannot us the fields of a derham space to do this since the evaluation of the splines in this way is a collective operation, and we want our functions
    # to be able to be computed by each rank on its own.

    # Building the B-splines
    # We will need the FEM spline space that contains D-splines in all three directions.
    fem_space_B = derham.Vh_fem["0"]
    # FE space of one forms. That means that we have B-splines in all three spatial directions.
    W = fem_space_B
    W1ds = [W.spaces]

    # We will need the FEM spline space that contains D-splines in all three directions.
    fem_space_D = derham.Vh_fem["3"]

    # FE space of three forms. That means that we have D-splines in all three spatial directions.
    V = fem_space_D
    V1ds = [V.spaces]

    # Helper function to handle reshaping and getting spans and basis
    def process_eta(eta, w1d):
        if isinstance(eta, (float, int)):
            eta = xp.array([eta])
        if len(eta.shape) == 1:
            eta = eta.reshape((eta.shape[0], 1))
        spans, values = get_span_and_basis(eta, w1d)
        return spans, values

    # Generalized factory function
    def make_basis_fun(is_B, dim_idx, i):
        def fun(eta1, eta2, eta3):
            eta_map = [eta1, eta2, eta3]
            eta = eta_map[dim_idx]
            w1d = W1ds[0][dim_idx] if is_B else V1ds[0][dim_idx]

            out = xp.zeros_like(eta)
            for j1 in range(eta.shape[0]):
                for j2 in range(eta.shape[1]):
                    for j3 in range(eta.shape[2]):
                        spans, values = process_eta(eta[j1, j2, j3], w1d)

                        # Get spline properties
                        Nbasis = w1d.nbasis
                        degree = w1d.degree
                        periodic = w1d.periodic

                        # Evaluate spline and assign
                        eval_indices, spline_values = get_values_and_indices_splines(
                            Nbasis,
                            degree,
                            periodic,
                            spans,
                            values,
                        )
                        out[j1, j2, j3] = get_one_spline(i, spline_values, eval_indices)[0]
            return out

        return fun

    # FE coefficeints to get B-splines from field
    inputB = derham.Vh["0"].zeros()
    fieldB = derham.create_spline_function("fh", "H1")
    npts_in_B = derham.Vh["0"].npts

    # FE coefficeints to get D-splines from field
    inputD = derham.Vh["3"].zeros()
    fieldD = derham.create_spline_function("fh", "L2")
    npts_in_D = derham.Vh["3"].npts

    etas1 = xp.linspace(0.0, 1.0, 20)
    etas2 = xp.linspace(0.0, 1.0, 20)
    etas3 = xp.linspace(0.0, 1.0, 20)
    meshgrid = xp.meshgrid(*[etas1, etas2, etas3], indexing="ij")

    maxerrorB = 0.0

    # We test that our B-splines have the same values as the ones obtained with the field function.
    for col0 in range(npts_in_B[0]):
        for col1 in range(npts_in_B[1]):
            for col2 in range(npts_in_B[2]):
                inputB[col0, col1, col2] = 1.0
                inputB.update_ghost_regions()
                fieldB.vector = inputB

                def error(e1, e2, e3):
                    return xp.abs(
                        fieldB(e1, e2, e3)
                        - (
                            make_basis_fun(True, 0, col0)(e1, e2, e3)
                            * make_basis_fun(True, 1, col1)(e1, e2, e3)
                            * make_basis_fun(True, 2, col2)(e1, e2, e3)
                        ),
                    )

                auxerror = xp.max(error(*meshgrid))

                if auxerror > maxerrorB:
                    maxerrorB = auxerror
                inputB[col0, col1, col2] = 0.0

    print(f"{maxerrorB =}")
    assert maxerrorB < 10.0**-13

    maxerrorD = 0.0
    # We test that our D-splines have the same values as the ones obtained with the field function.
    for col0 in range(npts_in_D[0]):
        for col1 in range(npts_in_D[1]):
            for col2 in range(npts_in_D[2]):
                inputD[col0, col1, col2] = 1.0
                inputD.update_ghost_regions()
                fieldD.vector = inputD

                def error(e1, e2, e3):
                    return xp.abs(
                        fieldD(e1, e2, e3)
                        - (
                            make_basis_fun(False, 0, col0)(e1, e2, e3)
                            * make_basis_fun(False, 1, col1)(e1, e2, e3)
                            * make_basis_fun(False, 2, col2)(e1, e2, e3)
                        ),
                    )

                auxerror = xp.max(error(*meshgrid))

                if auxerror > maxerrorD:
                    maxerrorD = auxerror
                inputD[col0, col1, col2] = 0.0

    print(f"{maxerrorD =}")
    assert maxerrorD < 10.0**-13
    print("Test spline evaluation passed.")


if __name__ == "__main__":
    Nel = [14, 16, 18]
    p = [5, 4, 3]
    spl_kind = [False, True, True]

    # test_spline_evaluation(Nel, p, spl_kind)
    # test_local_projectors_compare_global(Nel, p, spl_kind)
    # test_local_projectors_convergence(2, 3, False, do_plot=False)
    # test_replication_of_basis(Nel, p, spl_kind)
    #'0', 'H1'
    #'1', 'Hcurl'
    #'2', 'Hdiv'
    #'3', 'L2'
    #'v', 'H1vec'
    # test_basis_projection_operator_local(Nel, p , spl_kind, '1', '2')
    # test_basis_projection_operator_local_new([40, 1, 1], [5, 1, 1] , [False, True, True], 'v', 'v', do_plot=True)
