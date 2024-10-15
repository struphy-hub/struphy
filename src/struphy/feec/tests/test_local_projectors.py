import numpy as np
import pytest
from mpi4py import MPI
import matplotlib.pyplot as plt
import inspect
import time
from struphy.feec.psydac_derham import Derham
from struphy.feec.projectors import CommutingProjectorLocal
from struphy.feec.mass import WeightedMassOperators
from struphy.geometry import domains


# @pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[14, 16, 18]])
@pytest.mark.parametrize('p', [[5, 4, 3]])
@pytest.mark.parametrize('spl_kind', [[True, False, False], [False, True, False], [False, False, True]])
def test_local_projectors_compare_global(Nel, p, spl_kind, do_plot=False):
    """ Tests the Local-projectors, by comparing them to the analytical function as well as to the global projectors.
    """
    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    timei = time.time()
    # create derham object
    derham = Derham(Nel, p, spl_kind, comm=comm, local_projectors=True)
    timef = time.time()
    print("Time for building Derham = "+str(timef-timei))

    # constant function
    def f(e1, e2, e3): return np.sin(2.0*np.pi*e1) * \
        np.cos(4.0*np.pi*e2) * np.sin(6.0*np.pi*e3)
    # f = lambda e1, e2, e3: np.sin(2.0*np.pi*e1) * np.cos(4.0*np.pi*e2)
    # evaluation points
    e1 = np.linspace(0., 1., 10)
    e2 = np.linspace(0., 1., 9)
    e3 = np.linspace(0., 1., 8)

    ee1, ee2, ee3 = np.meshgrid(e1, e2, e3, indexing='ij')

    # loop over spaces
    for sp_id, sp_key in derham.space_to_form.items():

        P_Loc = derham._Ploc[sp_key]

        out = derham.Vh[sp_key].zeros()

        # field for local projection output
        field = derham.create_field('fh', sp_id)

        # field for global projection output
        fieldg = derham.create_field('fhg', sp_id)

        # project test function
        if sp_id in ('H1', 'L2'):
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

        if sp_id in ('H1', 'L2'):
            err = np.max(np.abs(f_analytic(ee1, ee2, ee3) - field_vals))
            # Error comparing the global and local projectors
            errg = np.max(np.abs(fieldg_vals - field_vals))

            f_plot = field_vals
        else:
            err = np.zeros(3)
            err[0] = np.max(np.abs(f(ee1, ee2, ee3) - field_vals[0]))
            err[1] = np.max(np.abs(f(ee1, ee2, ee3) - field_vals[1]))
            err[2] = np.max(np.abs(f(ee1, ee2, ee3) - field_vals[2]))

            # Error comparing the global and local projectors
            errg = np.zeros(3)
            errg[0] = np.max(np.abs(fieldg_vals[0] - field_vals[0]))
            errg[1] = np.max(np.abs(fieldg_vals[1] - field_vals[1]))
            errg[2] = np.max(np.abs(fieldg_vals[2] - field_vals[2]))

            f_plot = field_vals[0]

        print(f'{sp_id = }, {np.max(err) = }, {np.max(errg) = },{exectime = }')
        if sp_id in ('H1', 'H1vec'):
            assert np.max(err) < 0.011
            assert np.max(errg) < 0.011
        else:
            assert np.max(err) < 0.1
            assert np.max(errg) < 0.1

        if do_plot and rank == 0:
            plt.figure(f'{sp_id}')
            plt.contourf(e1, e2, np.squeeze(f_plot[:, :, 0].T))
            plt.show()


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('direction', [0, 1, 2])
@pytest.mark.parametrize('pi', [2, 3])
@pytest.mark.parametrize('spl_kindi', [True, False])
def test_local_projectors_convergence(direction, pi, spl_kindi, do_plot=False):
    """ Tests the convergence rate of the Local projectors along singleton dimensions, without mapping.
    """
    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # loop over different number of elements
    Nels = [2**n for n in range(3, 9)]
    errors = {'H1': [], 'Hcurl': [], 'Hdiv': [], 'L2': [], 'H1vec': []}
    figs = {}
    for sp_id in errors:
        figs[sp_id] = plt.figure(
            sp_id + ', Local-proj. convergence', figsize=(24, 16))

    for n, Neli in enumerate(Nels):

        # test function
        def fun(eta): return np.cos(4*np.pi*eta)

        # create derham object, test functions and evaluation points
        e1 = 0.
        e2 = 0.
        e3 = 0.
        if direction == 0:
            Nel = [Neli, 1, 1]
            p = [pi, 1, 1]
            spl_kind = [spl_kindi, True, True]
            e1 = np.linspace(0., 1., 100)
            e = e1
            c = 0
            def f(x, y, z): return fun(x)
        elif direction == 1:
            Nel = [1, Neli, 1]
            p = [1, pi, 1]
            spl_kind = [True, spl_kindi, True]
            e2 = np.linspace(0., 1., 100)
            e = e2
            c = 1
            def f(x, y, z): return fun(y)
        elif direction == 2:
            Nel = [1, 1, Neli]
            p = [1, 1, pi]
            spl_kind = [True, True, spl_kindi]
            e3 = np.linspace(0., 1., 100)
            e = e3
            c = 2
            def f(x, y, z): return fun(z)

        derham = Derham(Nel, p, spl_kind, comm=comm, local_projectors=True)

        # loop over spaces
        for sp_id, sp_key in derham.space_to_form.items():

            P_Loc = derham._Ploc[sp_key]
            out = derham.Vh[sp_key].zeros()

            field = derham.create_field('fh', sp_id)

            # project test function
            if sp_id in ('H1', 'L2'):
                f_analytic = f
            else:
                f_analytic = (f, f, f)

            vec = P_Loc(f_analytic)
            veco = P_Loc(f_analytic, out=out)
            # assert veco is out
            # if (np.all(vec.toarray() == veco.toarray()) == False):
            # print(sp_id)
            # print("################")
            # print("################")
            # print(vec.toarray())
            # print("################")
            # print("################")
            # print("################")
            # print("################")
            # print(veco.toarray())
            # print("################")
            # print("################")
            # assert np.all(vec.toarray() == veco.toarray())

            field.vector = vec
            field_vals = field(e1, e2, e3, squeeze_output=True)

            if sp_id in ('H1', 'L2'):
                err = np.max(np.abs(f_analytic(e1, e2, e3) - field_vals))
                f_plot = field_vals
            else:
                err = [np.max(np.abs(exact(e1, e2, e3) - field_v))
                       for exact, field_v in zip(f_analytic, field_vals)]
                f_plot = field_vals[0]

            errors[sp_id] += [np.max(err)]

            if do_plot:
                plt.figure(sp_id + ', Local-proj. convergence')
                plt.subplot(2, 4, n + 1)
                plt.plot(e, f(e1, e2, e3), 'o')
                plt.plot(e, f_plot)
                plt.xlabel(f'eta{c}')
                plt.title(f'Nel[{c}] = {Nel[c]}')

            del P_Loc, out, field, vec, veco, field_vals

    rate_p1 = pi + 1
    rate_p0 = pi

    for sp_id in derham.space_to_form:

        line_for_rate_p1 = [Ne**(-rate_p1) * errors[sp_id]
                            [0] / Nels[0]**(-rate_p1) for Ne in Nels]
        line_for_rate_p0 = [Ne**(-rate_p0) * errors[sp_id]
                            [0] / Nels[0]**(-rate_p0) for Ne in Nels]

        m, _ = np.polyfit(np.log(Nels), np.log(errors[sp_id]), deg=1)
        print(f'{sp_id = }, fitted convergence rate = {-m}, degree = {pi}')
        if sp_id in ('H1', 'H1vec'):
            assert -m > (pi + 1 - 0.1)
        else:
            assert -m > (pi - 0.1)

        if do_plot:
            plt.figure(sp_id + ', Local-proj. convergence')
            plt.subplot(2, 4, 8)
            plt.loglog(Nels, errors[sp_id])
            plt.loglog(Nels, line_for_rate_p1, 'k--')
            plt.loglog(Nels, line_for_rate_p0, 'k--')
            plt.text(Nels[-2], line_for_rate_p1[-2], f'1/Nel^{rate_p1}')
            plt.text(Nels[-2], line_for_rate_p0[-2], f'1/Nel^{rate_p0}')
            plt.title(f'{sp_id = }, degree = {pi}')
            plt.xlabel('Nel')

    if do_plot and rank == 0:
        plt.show()


if __name__ == '__main__':
    Nel = [14, 16, 18]
    p = [6, 8, 9]
    spl_kind = [True, False, True]
    test_local_projectors_compare_global(Nel, p, spl_kind, do_plot=False)
    # test_local_projectors_convergence(0, 2, False, do_plot=False)
    # test_local_projectors_convergence(1, 1, False, do_plot=True)
