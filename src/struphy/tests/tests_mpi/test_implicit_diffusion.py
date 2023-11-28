"""
Test the solvers that are child classes of the `Propagator` class.
"""

import pytest
from mpi4py import MPI
import numpy as np

from struphy.geometry import domains
from struphy.feec.psydac_derham import Derham
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.mass import WeightedMassOperator
from struphy.propagators.base import Propagator
from struphy.propagators.propagators_fields import ImplicitDiffusion
from struphy.feec.utilities import compare_arrays
from psydac.linalg.stencil import StencilVector


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[10, 2, 2], [40, 2, 2]])
@pytest.mark.parametrize('p', [[1, 1, 1], [3, 1, 1]])
@pytest.mark.parametrize('spl_kind', [[True, True, True]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}],
])
def test_poisson_solver(Nel, p, spl_kind, mapping):
    """
    Test the Poisson solver by means of manufactured solutions.
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

    def rho(e1, e2, e3):
        return np.sin(2*np.pi*e1)

    def sol(e1, e2, e3):
        return np.sin(2*np.pi*e1) / (4 * np.pi**2)

    rho_vec = StencilVector(derham.Vh['0'])
    WeightedMassOperator.assemble_vec(derham.Vh_fem['0'], rho_vec, [rho])

    sol_vec = derham.P['0'](sol)

    # create weighted mass operators
    mass_ops = WeightedMassOperators(derham, domain)

    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = mass_ops

    # Create Poisson solver
    _phi = StencilVector(derham.Vh['0'])
    poisson_solver = ImplicitDiffusion(_phi,
                                       sigma=0.,
                                       phi_n=rho_vec,
                                       x0=rho_vec,
                                       **solver_params)

    # Solve Poisson equation (call with dt=1.)
    poisson_solver(1.)

    # Compare to analytical solution
    compare_arrays(
        _phi,
        sol_vec.toarray(),
        MPI.COMM_WORLD.Get_rank(),
        atol=1e-5,
        verbose=True
    )


if __name__ == '__main__':
    Nel = [10, 2, 2]
    p = [1, 1, 1]
    spl_kind = [True, True, True]
    mapping = ['Cuboid', {'l1': 0., 'r1': 1,
                          'l2': 0., 'r2': 1.,
                          'l3': 0., 'r3': 1.}]
    test_poisson_solver(Nel, p, spl_kind, mapping)
