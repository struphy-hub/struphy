import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpi4py import MPI

from struphy.feec.mass import WeightedMassOperators
from struphy.feec.projectors import L2Projector
from struphy.feec.psydac_derham import Derham
from struphy.geometry import domains


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("Nel", [[16, 32, 1]])
@pytest.mark.parametrize("p", [[2, 1, 1], [3, 2, 1]])
@pytest.mark.parametrize("spl_kind", [[False, True, True]])
@pytest.mark.parametrize("array_input", [False, True])
def test_l2_projectors_mappings(Nel, p, spl_kind, array_input, with_desc, do_plot=False):
    """Tests the L2-projectors for all available mappings.

    Both callable and array inputs to the projectors are tested.
    """
    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # create derham object
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # constant function
    f = lambda e1, e2, e3: np.sin(np.pi * e1) * np.cos(2 * np.pi * e2)

    # create domain object
    dom_types = []
    dom_classes = []
    for key, val in inspect.getmembers(domains):
        if inspect.isclass(val) and val.__module__ == domains.__name__ and "AxisymmMHDequilibrium" not in key:
            dom_types += [key]
            dom_classes += [val]

    # evaluation points
    e1 = np.linspace(0.0, 1.0, 30)
    e2 = np.linspace(0.0, 1.0, 40)
    e3 = 0.0

    ee1, ee2, ee3 = np.meshgrid(e1, e2, e3, indexing="ij")

    for dom_type, dom_class in zip(dom_types, dom_classes):
        print("#" * 80)
        print(f"Testing {dom_class = }")
        print("#" * 80)

        if "DESC" in dom_type and not with_desc:
            print(f"Attention: {with_desc = }, DESC not tested here !!")
            continue

        domain = dom_class()

        # mass operators
        mass_ops = WeightedMassOperators(derham, domain)

        # loop over spaces
        for sp_id, sp_key in derham.space_to_form.items():
            P_L2 = L2Projector(sp_id, mass_ops)

            out = derham.Vh[sp_key].zeros()

            field = derham.create_spline_function("fh", sp_id)

            # project test function
            if sp_id in ("H1", "L2"):
                f_analytic = f
            else:
                f_analytic = (f, f, f)

            if array_input:
                pts_q = derham.quad_grid_pts[sp_key]
                if sp_id in ("H1", "L2"):
                    ee = np.meshgrid(*[pt.flatten() for pt in pts_q], indexing="ij")
                    f_array = f(*ee)
                else:
                    f_array = []
                    for pts in pts_q:
                        ee = np.meshgrid(*[pt.flatten() for pt in pts], indexing="ij")
                        f_array += [f(*ee)]
                f_args = f_array
            else:
                f_args = f_analytic

            vec = P_L2(f_args)
            veco = P_L2(f_args, out=out)

            assert veco is out
            assert np.all(vec.toarray() == veco.toarray())

            field.vector = vec
            field_vals = field(e1, e2, e3)

            if sp_id in ("H1", "L2"):
                err = np.max(np.abs(f_analytic(ee1, ee2, ee3) - field_vals))
                f_plot = field_vals
            else:
                err = [np.max(np.abs(exact(ee1, ee2, ee3) - field_v)) for exact, field_v in zip(f_analytic, field_vals)]
                f_plot = field_vals[0]

            print(f"{sp_id = }, {np.max(err) = }")
            if sp_id in ("H1", "H1vec"):
                assert np.max(err) < 0.004
            else:
                assert np.max(err) < 0.12

            if do_plot and rank == 0:
                plt.figure(f"{dom_type}, {sp_id}")
                plt.contourf(e1, e2, np.squeeze(f_plot[:, :, 0].T))
                plt.show()


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("direction", [0, 1, 2])
@pytest.mark.parametrize("pi", [1, 2])
@pytest.mark.parametrize("spl_kindi", [True, False])
def test_l2_projectors_convergence(direction, pi, spl_kindi, do_plot=False):
    """Tests the convergence rate of the L2 projectors along singleton dimensions, without mapping."""
    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # loop over different number of elements
    Nels = [2**n for n in range(3, 9)]
    errors = {"H1": [], "Hcurl": [], "Hdiv": [], "L2": [], "H1vec": []}
    figs = {}
    for sp_id in errors:
        figs[sp_id] = plt.figure(sp_id + ", L2-proj. convergence", figsize=(12, 8))

    for n, Neli in enumerate(Nels):
        # test function
        def fun(eta):
            return np.cos(4 * np.pi * eta)

        # create derham object, test functions and evaluation points
        e1 = 0.0
        e2 = 0.0
        e3 = 0.0
        if direction == 0:
            Nel = [Neli, 1, 1]
            p = [pi, 1, 1]
            spl_kind = [spl_kindi, True, True]
            e1 = np.linspace(0.0, 1.0, 100)
            e = e1
            c = 0

            def f(x, y, z):
                return fun(x)
        elif direction == 1:
            Nel = [1, Neli, 1]
            p = [1, pi, 1]
            spl_kind = [True, spl_kindi, True]
            e2 = np.linspace(0.0, 1.0, 100)
            e = e2
            c = 1

            def f(x, y, z):
                return fun(y)
        elif direction == 2:
            Nel = [1, 1, Neli]
            p = [1, 1, pi]
            spl_kind = [True, True, spl_kindi]
            e3 = np.linspace(0.0, 1.0, 100)
            e = e3
            c = 2

            def f(x, y, z):
                return fun(z)

        derham = Derham(Nel, p, spl_kind, comm=comm)

        # create domain object
        dom_type = "Cuboid"
        dom_params = {"l1": 0.0, "r1": 1.0, "l2": 0.0, "r2": 1.0, "l3": 0.0, "r3": 1.0}
        domain_class = getattr(domains, dom_type)
        domain = domain_class(**dom_params)

        # mass operators
        mass_ops = WeightedMassOperators(derham, domain)

        # loop over spaces
        for sp_id, sp_key in derham.space_to_form.items():
            P_L2 = L2Projector(sp_id, mass_ops)

            out = derham.Vh[sp_key].zeros()

            field = derham.create_spline_function("fh", sp_id)

            # project test function
            if sp_id in ("H1", "L2"):
                f_analytic = f
            else:
                f_analytic = (f, f, f)

            vec = P_L2(f_analytic)
            veco = P_L2(f_analytic, out=out)
            assert veco is out
            assert np.all(vec.toarray() == veco.toarray())

            field.vector = vec
            field_vals = field(e1, e2, e3, squeeze_out=True)

            if sp_id in ("H1", "L2"):
                err = np.max(np.abs(f_analytic(e1, e2, e3) - field_vals))
                f_plot = field_vals
            else:
                err = [np.max(np.abs(exact(e1, e2, e3) - field_v)) for exact, field_v in zip(f_analytic, field_vals)]
                f_plot = field_vals[0]

            errors[sp_id] += [np.max(err)]

            if do_plot:
                plt.figure(sp_id + ", L2-proj. convergence")
                plt.subplot(2, 4, n + 1)
                plt.plot(e, f(e1, e2, e3), "o")
                plt.plot(e, f_plot)
                plt.xlabel(f"eta{c}")
                plt.title(f"Nel[{c}] = {Nel[c]}")

            del P_L2, out, field, vec, veco, field_vals

        del domain_class, domain, mass_ops

    rate_p1 = pi + 1
    rate_p0 = pi

    for sp_id in derham.space_to_form:
        line_for_rate_p1 = [Ne ** (-rate_p1) * errors[sp_id][0] / Nels[0] ** (-rate_p1) for Ne in Nels]
        line_for_rate_p0 = [Ne ** (-rate_p0) * errors[sp_id][0] / Nels[0] ** (-rate_p0) for Ne in Nels]

        m, _ = np.polyfit(np.log(Nels), np.log(errors[sp_id]), deg=1)
        print(f"{sp_id = }, fitted convergence rate = {-m}, degree = {pi}")
        if sp_id in ("H1", "H1vec"):
            assert -m > (pi + 1 - 0.05)
        else:
            assert -m > (pi - 0.05)

        if do_plot:
            plt.figure(sp_id + ", L2-proj. convergence")
            plt.subplot(2, 4, 8)
            plt.loglog(Nels, errors[sp_id])
            plt.loglog(Nels, line_for_rate_p1, "k--")
            plt.loglog(Nels, line_for_rate_p0, "k--")
            plt.text(Nels[-2], line_for_rate_p1[-2], f"1/Nel^{rate_p1}")
            plt.text(Nels[-2], line_for_rate_p0[-2], f"1/Nel^{rate_p0}")
            plt.title(f"{sp_id = }, degree = {pi}")
            plt.xlabel("Nel")

    if do_plot and rank == 0:
        plt.show()


if __name__ == "__main__":
    Nel = [16, 32, 1]
    p = [2, 1, 1]
    spl_kind = [False, True, True]
    array_input = True
    test_l2_projectors_mappings(Nel, p, spl_kind, array_input, do_plot=False, with_desc=False)
    # test_l2_projectors_convergence(0, 1, True, do_plot=True)
    # test_l2_projectors_convergence(1, 1, False, do_plot=True)
