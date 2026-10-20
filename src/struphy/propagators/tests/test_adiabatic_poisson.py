import logging

import cunumpy as xp
import pytest
from maybempi import MPI

from struphy import domains, equils
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.adiabatic_poisson_solve import AdiabaticPoissonSolve
from struphy.propagators.base import Propagator
from struphy.propagators.implicit_diffusion import ImplicitDiffusion
from struphy.topology.grids import TensorProductGrid

logger = logging.getLogger("struphy")

comm = MPI.COMM_WORLD

Lx, Ly, Lz = 2.0, 1.0, 3.0

# HomogenSlab with p = beta / 2 = 1 and n0 = 2, i.e. T0 = p / n0 = 0.5 and n0 / T0 = 4
N0 = 2.0
N0_OVER_T0 = 4.0


def _setup(num_elements, degree):
    """Periodic Cuboid with a HomogenSlab equilibrium; sets the class attributes of Propagator."""
    domain = domains.Cuboid(l1=0.0, r1=Lx, l2=0.0, r2=Ly, l3=0.0, r3=Lz)
    eq_mhd = equils.HomogenSlab(B0z=1.0, beta=2.0, n0=N0)
    eq_mhd.domain = domain

    grid = TensorProductGrid(num_elements=num_elements)
    derham = Derham(grid, DerhamOptions(degree=degree), comm=comm)

    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)

    phi = FEECVariable(space="H1")
    phi.allocate(derham=derham, domain=domain)
    return domain, phi


def _max_error(domain, phi, sol_xyz):
    e1 = xp.linspace(0.0, 1.0, 21)
    e3 = xp.linspace(0.0, 1.0, 21)
    num = domain.push(phi.spline, e1, 0.5, e3, kind="0")
    x, y, z = domain(e1, 0.5, e3)
    exact = sol_xyz(x, y, z)
    return xp.max(xp.abs(num - exact)) / xp.max(xp.abs(exact))


def _pulled(domain, fun_xyz):
    def fun(e1, e2, e3):
        return domain.pull(fun_xyz, e1, e2, e3, kind="0", squeeze_out=False)

    return fun


SOLVER_PARAMS = SolverParameters(tol=1.0e-13, maxiter=3000)


def test_implicit_diffusion_without_diffusion_term():
    """With diffusion_mat=None and stab_mat="M0ad_withT" the solve is the weighted L2 projection
    sigma_1 * n0/T0 * phi = rho."""
    domain, phi = _setup([16, 1, 1], [3, 1, 1])
    sigma_1 = 0.7

    def rho_xyz(x, y, z):
        return xp.sin(2 * xp.pi / Lx * x) + 0.5

    def sol_xyz(x, y, z):
        return rho_xyz(x, y, z) / (sigma_1 * N0_OVER_T0)

    solver = ImplicitDiffusion(rho=_pulled(domain, rho_xyz))
    solver.variables.phi = phi
    solver.options = solver.Options(
        sigma_1=sigma_1,
        sigma_3=1.0,
        stab_mat="M0ad_withT",
        diffusion_mat=None,
        precond="MassMatrixPreconditioner",  # an M0-based preconditioner is allowed without diffusion
        solver_params=SOLVER_PARAMS,
    )
    solver.allocate()
    assert not solver.operator_is_singular
    solver(1.0)

    err = _max_error(domain, phi, sol_xyz)
    logger.info(f"{err =}")
    assert err < 1e-4


@pytest.mark.parametrize("which_geometry", [None, "cylindrical"])
def test_adiabatic_poisson_with_vacuum_term(which_geometry):
    """Manufactured solution of (1/(Z eps^2)) n0/T0 (phi - <phi>) - (1/alpha^2) Delta phi = rho / eps,
    where <phi> is the average over eta3 (cylindrical) or zero."""
    domain, phi = _setup([16, 1, 16], [3, 1, 3])
    epsilon, alpha, Z = 0.5, 2.0, 2
    kx, kz = 2 * xp.pi / Lx, 2 * xp.pi / Lz

    def sol_xyz(x, y, z):
        return xp.sin(kx * x) * (1.0 + xp.cos(kz * z))

    def sol_avg_xyz(x, y, z):
        return xp.sin(kx * x) if which_geometry == "cylindrical" else 0.0 * x

    def rho_xyz(x, y, z):
        laplace = -(kx**2) * xp.sin(kx * x) * (1.0 + xp.cos(kz * z)) - kz**2 * xp.sin(kx * x) * xp.cos(kz * z)
        adiabatic = N0_OVER_T0 / (Z * epsilon**2) * (sol_xyz(x, y, z) - sol_avg_xyz(x, y, z))
        return epsilon * (adiabatic - laplace / alpha**2)

    solver = AdiabaticPoissonSolve(rho=_pulled(domain, rho_xyz), epsilon=epsilon, alpha=alpha, Z=Z)
    solver.variables.phi = phi
    solver.options = solver.Options(
        quasi_neutral=False,
        which_geometry=which_geometry,
        precond=None,
        solver_params=SOLVER_PARAMS,
    )
    solver.allocate()
    solver(1.0)

    err = _max_error(domain, phi, sol_xyz)
    logger.info(f"{which_geometry =}, {err =}")
    assert err < 1e-3


def test_adiabatic_poisson_quasi_neutral():
    """In the quasi-neutral limit phi = Z eps T0/n0 rho (default options)."""
    domain, phi = _setup([16, 1, 1], [3, 1, 1])
    epsilon, Z = 0.1, 1

    def rho_xyz(x, y, z):
        return xp.cos(2 * xp.pi / Lx * x)

    def sol_xyz(x, y, z):
        return Z * epsilon / N0_OVER_T0 * rho_xyz(x, y, z)

    solver = AdiabaticPoissonSolve(rho=_pulled(domain, rho_xyz), epsilon=epsilon, alpha=1e-3, Z=Z)
    solver.variables.phi = phi
    solver.allocate()
    solver(1.0)

    err = _max_error(domain, phi, sol_xyz)
    logger.info(f"{err =}")
    assert err < 1e-4


def test_invalid_options():
    # no diffusion term: the stiffness-based preconditioners are not available
    with pytest.raises(AssertionError):
        ImplicitDiffusion.Options(diffusion_mat=None, precond="MultiGrid")
    with pytest.raises(AssertionError):
        ImplicitDiffusion.Options(diffusion_mat=None, precond="StiffnessPreconditioner")
    # no diffusion term: the averaged part of phi would be undetermined
    with pytest.raises(AssertionError):
        ImplicitDiffusion.Options(diffusion_mat=None, stab_average="eta3")
    # the zonal correction leaves <phi> undetermined without the vacuum term
    with pytest.raises(AssertionError):
        AdiabaticPoissonSolve.Options(quasi_neutral=True, which_geometry="cylindrical")
    with pytest.raises(AssertionError):
        AdiabaticPoissonSolve.Options(quasi_neutral=True, precond="StiffnessPreconditioner")
    # valid combinations
    ImplicitDiffusion.Options(stab_mat="M0ad_withT", diffusion_mat=None)
    AdiabaticPoissonSolve.Options(quasi_neutral=False, which_geometry="toroidal", precond="StiffnessPreconditioner")


if __name__ == "__main__":
    test_implicit_diffusion_without_diffusion_term()
    test_adiabatic_poisson_with_vacuum_term("cylindrical")
    test_adiabatic_poisson_quasi_neutral()
    test_invalid_options()
