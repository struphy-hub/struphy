"""
Test the solvers that are child classes of the `Propagator` class.
"""

import pytest
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from struphy.geometry import domains
from struphy.feec.psydac_derham import Derham
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.projectors import L2Projector
from struphy.propagators.base import Propagator
from struphy.propagators.propagators_fields import ImplicitDiffusion


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[32, 32, 2]])
@pytest.mark.parametrize('p', [[1, 1, 1], [2, 1, 1], [3, 1, 1]])
@pytest.mark.parametrize('spl_kind', [[True, True, True]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 0., 'r1': 2., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}], ['Orthogonal', {
            'Lx': 2., 'Ly': 2., 'alpha': .1, 'Lz': 1.}], ['Colella', {
                'Lx': 2., 'Ly': 2., 'alpha': .1, 'Lz': 1.}]
])
def test_poisson_with_M1(Nel: list[int], p: list[int], spl_kind: bool, mapping: dict[str, float] | str):
    """
    Test the Poisson solver by means of manufactured solutions in 1D & 2D .
    """
    solver_params = {
        'type': ('pcg', 'MassMatrixPreconditioner'),
        'tol': 1.e-15,
        'maxiter': 3000,
        'info': True,
        'verbose': False}

    # create domain object
    dom_type = mapping[0]
    dom_params = mapping[1]

    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    # create derham object
    derham = Derham(Nel, p, spl_kind, comm=MPI.COMM_WORLD)

    # create weighted mass operators
    mass_ops = WeightedMassOperators(derham, domain)

    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = mass_ops

    e1 = np.linspace(0., 1., 32)
    e2 = np.linspace(0., 1., 32)
    e3 = np.linspace(0., 1., 2)

    # create right-hand side
    def rho1_xyz(x, y, z):
        return np.sin(2*np.pi*x)/2

    def rho1(e1, e2, e3):
        return domain.pull(rho1_xyz, e1, e2, e3, kind='0', squeeze_out=True)

    def rho2_xyz(x, y, z):
        return np.sin(2*np.pi*x + 4*np.pi*y)/2

    def rho2(e1, e2, e3):
        return domain.pull(rho2_xyz, e1, e2, e3, kind='0', squeeze_out=True)

    def sol1(e1, e2, e3):
        return np.sin(2*np.pi*e1) / (4 * np.pi**2)

    def sol2(e1, e2, e3):
        return np.sin(2*np.pi*e1+4*np.pi*e2) / (64 * np.pi**4)

    rho_vec1 = derham.Vh['0'].zeros()
    rho_vec2 = derham.Vh['0'].zeros()
    l2_proj = L2Projector('H1', mass_ops)
    l2_proj.get_dofs(rho1, dofs=rho_vec1)
    l2_proj.get_dofs(rho2, dofs=rho_vec2)

    # Create Poisson solver
    _phi1 = derham.create_field('test1', 'H1')
    poisson_solver1 = ImplicitDiffusion(_phi1.vector,
                                        sigma=0.,
                                        phi_n=rho_vec1,
                                        A_mat='M1',
                                        **solver_params)

    _phi2 = derham.create_field('test2', 'H1')
    poisson_solver2 = ImplicitDiffusion(_phi2.vector,
                                        sigma=0.,
                                        phi_n=rho_vec2,
                                        A_mat='M1',
                                        **solver_params)

    # Solve Poisson equation (call with dt=1.)
    poisson_solver1(1.)
    poisson_solver2(1.)

    # pull solutions
    x, y, z = domain(e1, e2, e3)
    analytic_value1 = sol1(x, y, z)
    analytic_value2 = sol2(x, y, z)

    sol_val1 = domain.push(_phi1, e1, e2, e3, kind='0')
    sol_val2 = domain.push(_phi2, e1, e2, e3, kind='0')

    # compute error
    # /np.max(np.abs(analytic_value1))
    error1 = np.max(np.abs(analytic_value1 - sol_val1))
    # /np.max(np.abs(analytic_value2))
    error2 = np.max(np.abs(analytic_value2 - sol_val2))

    print(f'{error1=},{np.max(np.abs(analytic_value1))=},{np.max(np.abs(sol_val1))=}')
    print(f'{error2=},{np.max(np.abs(analytic_value2))=},{np.max(np.abs(sol_val2))=}')

    assert error1 < 1e-1
    assert error2 < 1e-1


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[32, 32, 2]])
@pytest.mark.parametrize('p', [[1, 1, 1], [2, 1, 1], [3, 1, 1]])
@pytest.mark.parametrize('spl_kind', [[True, True, True], [False, False, True]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}],
])
def test_poisson_with_M1perp(Nel: list[int], p: list[int], spl_kind: bool, mapping: dict[str, float] | str):
    """
    Test the Poisson solver by means of manufactured solutions 1D & 2D.
    """

    solver_params = {
        'type': ('pcg', 'MassMatrixPreconditioner'),
        'tol': 1.e-15,
        'maxiter': 3000,
        'info': True,
        'verbose': False}

    # create domain object
    dom_type = mapping[0]
    dom_params = mapping[1]

    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    # create derham object
    derham = Derham(Nel, p, spl_kind, comm=MPI.COMM_WORLD)

    e1 = np.linspace(0., 1., 32)
    e2 = np.linspace(0., 1., 32)
    e3 = np.linspace(0., 1., 2)

    def rho1_xyz(x, y, z):
        return np.sin(2*np.pi*x)

    def rho1(e1, e2, e3):
        return domain.pull(rho1_xyz, e1, e2, e3, kind='0', squeeze_out=True)

    def rho2_xyz(x, y, z):
        return np.sin(2*np.pi*x + 4*np.pi*y)

    def rho2(e1, e2, e3):
        return domain.pull(rho2_xyz, e1, e2, e3, kind='0', squeeze_out=True)

    def sol1(e1, e2, e3):
        return np.sin(2*np.pi*e1) / (4 * np.pi**2)

    def sol2(e1, e2, e3):
        return np.sin(2*np.pi*e1+4*np.pi*e2) / (64 * np.pi**4)

    rho_vec1 = derham.Vh['0'].zeros()
    rho_vec2 = derham.Vh['0'].zeros()

    # create weighted mass operators
    mass_ops = WeightedMassOperators(derham, domain)

    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = mass_ops

    l2_proj = L2Projector('H1', mass_ops)
    l2_proj.get_dofs(rho1, dofs=rho_vec1)
    l2_proj.get_dofs(rho2, dofs=rho_vec2)

    # Create Poisson solver
    _phi1 = derham.create_field('test1', 'H1')
    poisson_solver1 = ImplicitDiffusion(_phi1.vector,
                                        sigma=0.,
                                        phi_n=rho_vec1,
                                        A_mat='M1perp',
                                        **solver_params)

    _phi2 = derham.create_field('test2', 'H1')
    poisson_solver2 = ImplicitDiffusion(_phi2.vector,
                                        sigma=0.,
                                        phi_n=rho_vec2,
                                        A_mat='M1',
                                        **solver_params)

    # Solve Poisson equation (call with dt=1.)
    poisson_solver1(1.)
    poisson_solver2(1.)

   # pull solutions
    x, y, z = domain(e1, e2, e3)
    analytic_value1 = sol1(x, y, z)
    analytic_value2 = sol2(x, y, z)

    sol_val1 = domain.push(_phi1, e1, e2, e3, kind='0')
    sol_val2 = domain.push(_phi2, e1, e2, e3, kind='0')

    # compute error
    # /np.max(np.abs(analytic_value1))
    error1 = np.max(np.abs(analytic_value1 - sol_val1))
    # /np.max(np.abs(analytic_value2))
    error2 = np.max(np.abs(analytic_value2 - sol_val2))

    print(f'{error1=},{np.max(np.abs(analytic_value1))=},{np.max(np.abs(sol_val1))=}')
    print(f'{error2=},{np.max(np.abs(analytic_value2))=},{np.max(np.abs(sol_val2))=}')

    assert error1 < 1e-1
    assert error2 < 1e-1


@pytest.mark.parametrize('direction', [0, 1, 2])
def test_poisson_convergence(direction: list[int], show_plot=False):
    """
    Test the Poisson convergence in 1D by means of manufactured solutions.
    """

    Nels = [2**n for n in range(3, 9)]
    p_values = [1, 2, 3]
    for pi in p_values:
        errors = []
        h_vec = []
        for n, Neli in enumerate(Nels):

            e1 = np.linspace(0., 1., 100)
            e2 = np.linspace(0., 1., 100)
            e3 = np.linspace(0., 1., 100)

            if direction == 0:
                Nel = [Neli, 1, 1]
                p = [pi, 1, 1]

                def sol1(e1, e2, e3): return np.sin(
                    2*np.pi*e1) / (4 * np.pi**2)

                def rho1_xyz(x, y, z):
                    return np.sin(2*np.pi*x)
            elif direction == 1:
                Nel = [1, Neli, 1]
                p = [1, pi, 1]

                def sol1(e1, e2, e3): return np.sin(
                    2*np.pi*e2) / (4 * np.pi**2)

                def rho1_xyz(x, y, z):
                    return np.sin(2*np.pi*y)

            elif direction == 2:
                Nel = [1, 1, Neli]
                p = [1, 1, pi]

                def sol1(e1, e2, e3): return np.sin(
                    2*np.pi*e3) / (4 * np.pi**2)

                def rho1_xyz(x, y, z):
                    return np.sin(2*np.pi*z)

            else:
                print('Direction should be either 0, 1 or 2')

            def rho1(e1, e2, e3):
                return domain.pull(rho1_xyz, e1, e2, e3, kind='0', squeeze_out=True)

        # create derham object
            derham = Derham(Nel, p, [
                            True, True, True], comm=MPI.COMM_WORLD)

            dom_type = 'Cuboid'
            dom_params = {'l1': 0., 'r1': 1.,
                          'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}
            domain_class = getattr(domains, dom_type)
            domain = domain_class(**dom_params)

            rho_vec = derham.Vh['0'].zeros()
            mass_ops = WeightedMassOperators(derham, domain)

            Propagator.derham = derham
            Propagator.domain = domain
            Propagator.mass_ops = mass_ops

            L2Projector('H1', mass_ops).get_dofs(rho1, dofs=rho_vec)

            # Poisson solver
            solver_params = {
                'type': ('pcg', 'MassMatrixPreconditioner'),
                'tol': 1.e-15,
                'maxiter': 3000,
                'info': True,
                'verbose': False}

            _phi = derham.create_field('phi', 'H1')
            poisson_solver = ImplicitDiffusion(_phi.vector,
                                               sigma=0.,
                                               phi_n=rho_vec,
                                               A_mat='M1',
                                               **solver_params)
            poisson_solver(1.)

            x, y, z = domain(e1, e2, e3)
            analytic_values1 = sol1(x, y, z)
            sol_values = domain.push(_phi, e1, e2, e3, kind='0')

            error = np.max(np.abs(analytic_values1 - sol_values)
                           )/np.max(np.abs(analytic_values1))
            errors.append(error)
            h = 1/(Neli)
            h_vec.append(h)
        print(f'{np.max(np.abs(sol_values))=}, {error=}')
        m, _ = np.polyfit(np.log(Nels), np.log(errors), deg=1)
        print(f'{m=}')
        print(f'Solution converges in {direction=} with {errors =},{h =} ')
        assert -m > (pi + 1 - 0.05)

        # Plot convergence in 1D
        plt.plot(h_vec, errors, 'o', label=f'p={p[direction]}')
        plt.plot(h_vec, [h**(p[direction]+1)/h_vec[direction]**(p[direction]+1)*errors[direction]
                         for h in h_vec], 'k--')
        plt.text(h_vec[-1], errors[-2], 'p+1')

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel('Grid Spacing h')
    plt.ylabel('Error')
    plt.title('Convergence of Poisson Solver in 1D')
    plt.legend()
    if show_plot:
        plt.show()


if __name__ == '__main__':
    Nel = [32, 32, 2]
    p = [2, 1, 1]
    spl_kind = [True, True, True]
    mapping = ['Cuboid', {
        'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}]
    test_poisson_with_M1(Nel, p, spl_kind, mapping)
    test_poisson_with_M1perp(Nel, p, spl_kind, mapping)
    test_poisson_convergence(2, show_plot=True)
