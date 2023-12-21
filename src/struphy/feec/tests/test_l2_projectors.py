import numpy as np
import pytest
from mpi4py import MPI
import matplotlib.pyplot as plt

from struphy.feec.psydac_derham import Derham
from struphy.feec.projectors import L2_Projector
from struphy.feec.mass import WeightedMassOperators
from struphy.geometry import domains


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[10, 10, 2], [40, 8, 2]])
@pytest.mark.parametrize('p', [[2, 2, 1], [3, 3, 1]])
@pytest.mark.parametrize('spl_kind', [[True, True, True]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}],
])
def test_l2_projector_V0(Nel, p, spl_kind, mapping, do_plot=False):
    """ Tests the L2 projector in V0.
    """
    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # create derham object
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # create domain object
    dom_type = mapping[0]
    dom_params = mapping[1]
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    # create mass matrices
    mass_ops = WeightedMassOperators(derham=derham, domain=domain)

    # create L2 projector object
    proj = L2_Projector(mass_ops.M0, derham)

    # test function in eta1 and eta2 direction
    def my_fun(e1, e2, e3):
        return np.sin(2*np.pi*e1) * np.cos(2*np.pi*e2)

    # project test function
    vec = proj(my_fun)

    # for evaluating, create Field object
    field = derham.create_field('h', 'H1')
    field.vector = vec

    # evaluate on a mesh
    x = np.linspace(0, 1., 200)

    # test optional out and tmp argument
    tmp_y = np.zeros((200,200,1), dtype=float)
    tmp_test = np.zeros((200,200,1), dtype=float)
    y = field(x, x, 0., out=tmp_y, tmp=tmp_test)

    # assert that projected function approximates values of analytical function
    # in eta1-direction
    for k in [0, 50, 75]:
        assert np.allclose(y[:, k, 0], my_fun(x, x[k], 0.), atol=1e-2), \
            np.max(np.abs(y[:, k, 0] - my_fun(x, x[k], 0.)))

    # in eta2-direction
    for k in [0, 50, 75]:
        assert np.allclose(y[k, :, 0], my_fun(x[k], x, 0.), atol=1e-2), \
            np.max(np.abs(y[k, :, 0] - my_fun(x[k], x, 0.)))

    # plot for optical confirmation
    if do_plot:
        if rank == 0:
            for k in [0, 50, 75]:
                print(np.max(np.abs(y[:, k, 0] - my_fun(x, x[k], 0.))))
                plt.plot(x, y[:, k, 0], label='discretized')
                plt.plot(x, my_fun(x, x[k], 0), label='analytical')
                plt.legend()
                plt.show()


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('p', [[2, 1, 1], [3, 1, 1], [4, 1, 1]])
@pytest.mark.parametrize('spl_kind', [[True, True, True]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}],
])
def test_convergence_l2_proj_V0(p, spl_kind, mapping, do_plot=False):
    """ Tests the convergence rate of the L2 projector in V0.
    """
    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    Nelx = [10, 100, 500]
    errors = []

    for Nx in Nelx:
        Nel = [Nx, 2, 2]
        # create derham object
        derham = Derham(Nel, p, spl_kind, comm=comm)

        # create domain object
        dom_type = mapping[0]
        dom_params = mapping[1]
        domain_class = getattr(domains, dom_type)
        domain = domain_class(**dom_params)

        # create mass matrices
        mass_ops = WeightedMassOperators(derham=derham, domain=domain)

        # create L2 projector object
        proj = L2_Projector(mass_ops.M0, derham)

        # test function in eta1
        def my_fun(e1, e2, e3):
            return np.sin(2*np.pi*e1)

        # project test function
        vec = proj(my_fun)

        # for evaluating, create Field object
        field = derham.create_field('h', 'H1')
        field.vector = vec

        # evaluate on a mesh
        x = np.linspace(0, 1., 200)
        y = field(x, 0., 0.)

        # compute the error in the L infinity norm
        errors.append(np.max(np.abs(y[:, 0, 0] - my_fun(x, 0., 0.))))

    m, _ = np.polyfit(np.log(Nelx), np.log(errors), deg=1)
    # because m is negative
    assert np.abs(p[0] + 1 + m) < 0.05

    if do_plot:
        if rank == 0:
            plt.loglog(Nelx, errors)
            plt.plot(Nelx,
                     np.array(Nelx, dtype=float)** (-p[0]-1),
                     label=f'order {p[0]+1}')
            plt.legend()
            plt.show()


if __name__ == '__main__':
    Nel = [10, 10, 2]
    p = [3, 2, 1]
    spl_kind = [True, True, True]
    mapping = ['Cuboid', {'l1': 0., 'r1': 1,
                          'l2': 0., 'r2': 1.,
                          'l3': 0., 'r3': 1.}]
    # test_l2_projector_V0(Nel, p, spl_kind, mapping)
    test_convergence_l2_proj_V0(p, spl_kind, mapping)
