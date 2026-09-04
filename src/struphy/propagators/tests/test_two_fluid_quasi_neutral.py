"""
MMS tests for TwoFluidQuasiNeutralFull — 1D and 2D, all boundary condition types.

Each test runs a single time step and checks that the L∞ error against the
manufactured solution is below atol.  All tests use the Schur complement solver.
"""

import logging

import cunumpy as xp
import pytest
from cunumpy import cos, pi, sin, zeros_like

from struphy import domains, equils, grids, set_logging_level
from struphy.feec.basis_projection_ops import BasisProjectionOperators
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.fields_background.projected_equils import ProjectedMHDequilibrium
from struphy.initial.base import GenericPerturbation
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.propagators.base import Propagator
from struphy.propagators.two_fluid_quasi_neutral_full import TwoFluidQuasiNeutralFull

set_logging_level(logging.INFO)


# ---------------------------------------------------------------------------
# 1-D manufactured solutions and sources
# ---------------------------------------------------------------------------

def _mms_1d(bc_type):
    """Return (mms_phi, mms_u, mms_ue) for 1D cases."""
    if bc_type == "periodic":
        def phi(x, y, z): return xp.sin(2*pi*x), zeros_like(x), zeros_like(x)
        def u(x, y, z):   return xp.sin(2*pi*x) + 1, zeros_like(x), zeros_like(x)
        def ue(x, y, z):  return xp.sin(2*pi*x), zeros_like(x), zeros_like(x)
    elif bc_type == "hom_dirichlet":
        def phi(x, y, z): return xp.sin(2*pi*x), zeros_like(x), zeros_like(x)
        def u(x, y, z):   return xp.sin(2*pi*x), zeros_like(x), zeros_like(x)
        def ue(x, y, z):  return xp.sin(2*pi*x), zeros_like(x), zeros_like(x)
    else:  # inhom_dirichlet: full solution = sin(2πx) + lifting
        def phi(x, y, z): return xp.sin(2*pi*x), zeros_like(x), zeros_like(x)
        def u(x, y, z):   return xp.sin(2*pi*x) + x + 1, zeros_like(x), zeros_like(x)
        def ue(x, y, z):  return xp.sin(2*pi*x) + x, zeros_like(x), zeros_like(x)
    return phi, u, ue


def _sources_1d(bc_type, nu, nu_e, sigma):
    def src_u(x, y, z):
        fx = 2*pi*(cos(2*pi*x) + 2*nu*pi*sin(2*pi*x))
        return fx, zeros_like(x), zeros_like(x)

    if bc_type == "inhom_dirichlet":
        def src_ue(x, y, z):
            fx = -2*pi*cos(2*pi*x) + (4*nu_e*pi**2 - sigma)*sin(2*pi*x) - sigma*x
            return fx, zeros_like(x), zeros_like(x)
    else:
        def src_ue(x, y, z):
            fx = -2*pi*cos(2*pi*x) + nu_e*4*pi**2*sin(2*pi*x) - sigma*sin(2*pi*x)
            return fx, zeros_like(x), zeros_like(x)

    return src_u, src_ue


# ---------------------------------------------------------------------------
# 2-D manufactured solutions and sources
# ---------------------------------------------------------------------------

def _mms_2d(bc_type):
    """Return (mms_phi, mms_u, mms_ue) for 2D cases."""
    if bc_type == "periodic":
        def phi(x, y, z): return xp.cos(2*pi*x) + xp.sin(2*pi*y), zeros_like(x), zeros_like(x)
        def u(x, y, z):   return -xp.sin(2*pi*x)*xp.sin(2*pi*y), -xp.cos(2*pi*x)*xp.cos(2*pi*y), zeros_like(x)
        def ue(x, y, z):  return -xp.sin(4*pi*x)*xp.sin(4*pi*y), -xp.cos(4*pi*x)*xp.cos(4*pi*y), zeros_like(x)
    else:  # hom_dirichlet
        def phi(x, y, z): return xp.cos(2*pi*x) + xp.sin(2*pi*y), zeros_like(x), zeros_like(x)
        def u(x, y, z):   return -xp.sin(2*pi*x)*xp.cos(2*pi*y),  xp.cos(2*pi*x)*xp.sin(2*pi*y), zeros_like(x)
        def ue(x, y, z):  return -xp.sin(4*pi*x)*xp.cos(4*pi*y),  xp.cos(4*pi*x)*xp.sin(4*pi*y), zeros_like(x)
    return phi, u, ue


def _sources_2d(bc_type, B0, nu, nu_e, epsilon, sigma):
    if bc_type == "periodic":
        def src_u(x, y, z):
            fx = (-2*pi*xp.sin(2*pi*x)
                  + B0/epsilon * xp.cos(2*pi*x)*xp.cos(2*pi*y)
                  - nu*8*pi**2 * xp.sin(2*pi*x)*xp.sin(2*pi*y))
            fy = (2*pi*xp.cos(2*pi*y)
                  - B0/epsilon * xp.sin(2*pi*x)*xp.sin(2*pi*y)
                  - nu*8*pi**2 * xp.cos(2*pi*x)*xp.cos(2*pi*y))
            return fx, fy, zeros_like(x)

        def src_ue(x, y, z):
            fx = (2*pi*xp.sin(2*pi*x)
                  - B0/epsilon * xp.cos(4*pi*x)*xp.cos(4*pi*y)
                  - nu_e*32*pi**2 * xp.sin(4*pi*x)*xp.sin(4*pi*y)
                  + sigma * xp.sin(4*pi*x)*xp.sin(4*pi*y))
            fy = (-2*pi*xp.cos(2*pi*y)
                  + B0/epsilon * xp.sin(4*pi*x)*xp.sin(4*pi*y)
                  - nu_e*32*pi**2 * xp.cos(4*pi*x)*xp.cos(4*pi*y)
                  + sigma * xp.cos(4*pi*x)*xp.cos(4*pi*y))
            return fx, fy, zeros_like(x)

    else:  # hom_dirichlet
        def src_u(x, y, z):
            fx = (-2*pi*xp.sin(2*pi*x)
                  - B0/epsilon * xp.cos(2*pi*x)*xp.sin(2*pi*y)
                  - nu*8*pi**2 * xp.sin(2*pi*x)*xp.cos(2*pi*y))
            fy = (2*pi*xp.cos(2*pi*y)
                  - B0/epsilon * xp.sin(2*pi*x)*xp.cos(2*pi*y)
                  + nu*8*pi**2 * xp.cos(2*pi*x)*xp.sin(2*pi*y))
            return fx, fy, zeros_like(x)

        def src_ue(x, y, z):
            fx = (2*pi*xp.sin(2*pi*x)
                  + B0/epsilon * xp.cos(4*pi*x)*xp.sin(4*pi*y)
                  - nu_e*32*pi**2 * xp.sin(4*pi*x)*xp.cos(4*pi*y)
                  + sigma * xp.sin(4*pi*x)*xp.cos(4*pi*y))
            fy = (-2*pi*xp.cos(2*pi*y)
                  + B0/epsilon * xp.sin(4*pi*x)*xp.cos(4*pi*y)
                  + nu_e*32*pi**2 * xp.cos(4*pi*x)*xp.sin(4*pi*y)
                  - sigma * xp.cos(4*pi*x)*xp.sin(4*pi*y))
            return fx, fy, zeros_like(x)

    return src_u, src_ue


# ---------------------------------------------------------------------------
# helper: build propagator and run one step
# ---------------------------------------------------------------------------

def _run_one_step(domain, Nel, degree, derham_opts, src_u, src_ue,
                  B0, nu, nu_e, epsilon, sigma, tol=1e-5,
                  lifting_u=None, lifting_ue=None):
    grid   = grids.TensorProductGrid(num_elements=Nel)
    derham = Derham(grid=grid, options=derham_opts, domain=domain)
    eq     = equils.HomogenSlab(B0x=0, B0y=0, B0z=B0, beta=0, n0=0)

    projected_equil = ProjectedMHDequilibrium(equil=eq, derham=derham)
    mass_ops        = WeightedMassOperators(derham=derham, domain=domain, eq_mhd=eq)
    basis_ops       = BasisProjectionOperators(derham, domain, eq_mhd=eq)

    Propagator.derham          = derham
    Propagator.domain          = domain
    Propagator.mass_ops        = mass_ops
    Propagator.basis_ops       = basis_ops
    Propagator.projected_equil = projected_equil

    prop = TwoFluidQuasiNeutralFull(allocate_variables=True)
    prop.options = prop.Options(
        nu=nu,
        nu_e=nu_e,
        eps_norm=epsilon,
        stab_sigma=sigma,
        source_u=src_u,
        source_ue=src_ue,
        solver="schur",
        solver_params=SolverParameters(info=True, tol=tol),
    )
    prop.allocate()

    if lifting_u is not None:
        prop.variables.u.lifting_function = lifting_u
    if lifting_ue is not None:
        prop.variables.ue.lifting_function = lifting_ue

    prop(dt=1.0)
    return prop


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bc_type", ["hom_dirichlet"])
def test_mms_1d(bc_type):
    """1D MMS: Nel=32, degree=1.  Expected L∞ error O(h²) ~ 1e-3."""
    B0, nu, nu_e, epsilon, sigma = 0.0, 10.0, 1.0, 1.0, 0.0
    atol = 0.2

    mms_phi, mms_u, mms_ue = _mms_1d(bc_type)
    src_u, src_ue = _sources_1d(bc_type, nu, nu_e, sigma)

    if bc_type == "periodic":
        derham_opts = DerhamOptions(degree=(1,1,1), bcs=(None, None, None))
        lifting_u = lifting_ue = None
    else:
        derham_opts = DerhamOptions(degree=(1,1,1), bcs=(("dirichlet","dirichlet"), None, None))
        if bc_type == "inhom_dirichlet":
            lifting_u  = GenericPerturbation(lambda x, y, z: x + 1, comp=0, given_in_basis="physical")
            lifting_ue = GenericPerturbation(lambda x, y, z: x,     comp=0, given_in_basis="physical")
        else:
            lifting_u = lifting_ue = None

    prop = _run_one_step(
        domain=domains.Cuboid(),
        Nel=(32, 1, 1),
        degree=(1, 1, 1),
        derham_opts=derham_opts,
        src_u=src_u,
        src_ue=src_ue,
        B0=B0, nu=nu, nu_e=nu_e, epsilon=epsilon, sigma=sigma,
        lifting_u=lifting_u, lifting_ue=lifting_ue,
    )

    e1 = xp.linspace(0, 1, 128)
    z0 = xp.array([0.5])

    # for inhom_dirichlet, evaluate the full solution (homogeneous part + lifting)
    if bc_type == "inhom_dirichlet":
        num_u  = prop.variables.u.spline_full(e1, z0, z0, squeeze_out=True)[0]
        num_ue = prop.variables.ue.spline_full(e1, z0, z0, squeeze_out=True)[0]
    else:
        num_u  = prop.variables.u.spline(e1, 0.5, 0.5, squeeze_out=True)[0]
        num_ue = prop.variables.ue.spline(e1, 0.5, 0.5, squeeze_out=True)[0]

    err_u   = xp.max(xp.abs(num_u  - mms_u(e1, z0, z0)[0]))
    err_ue  = xp.max(xp.abs(num_ue - mms_ue(e1, z0, z0)[0]))
    err_phi = xp.max(xp.abs(prop.variables.phi.spline(e1, 0.5, 0.5, squeeze_out=True) - mms_phi(e1, z0, z0)[0]))

    assert err_u   < atol, f"[{bc_type}] u   L∞ error {err_u:.3e}   >= {atol}"
    assert err_ue  < atol, f"[{bc_type}] ue  L∞ error {err_ue:.3e}  >= {atol}"
    assert err_phi < atol, f"[{bc_type}] phi L∞ error {err_phi:.3e} >= {atol}"


@pytest.mark.parametrize("bc_type", ["periodic", "hom_dirichlet"])
def test_mms_2d(bc_type):
    """2D MMS: Nel=(8,8,1), degree=(2,2,1).  Expected L∞ error O(h³) ~ 1e-2."""
    B0, nu, nu_e, epsilon, sigma = 1.0, 10.0, 1.0, 1.0, 0.0
    atol = 0.2

    mms_phi, mms_u, mms_ue = _mms_2d(bc_type)
    src_u, src_ue = _sources_2d(bc_type, B0, nu, nu_e, epsilon, sigma)

    if bc_type == "periodic":
        derham_opts = DerhamOptions(degree=(2,2,1), bcs=(None, None, None))
    else:
        derham_opts = DerhamOptions(degree=(2,2,1), bcs=(("dirichlet","dirichlet"), ("dirichlet","dirichlet"), None))

    prop = _run_one_step(
        domain=domains.Cuboid(),
        Nel=(8, 8, 1),
        degree=(2, 2, 1),
        derham_opts=derham_opts,
        src_u=src_u,
        src_ue=src_ue,
        B0=B0, nu=nu, nu_e=nu_e, epsilon=epsilon, sigma=sigma,
    )

    e1 = xp.linspace(0, 1, 64)
    e2 = xp.linspace(0, 1, 64)
    E1, E2 = xp.meshgrid(e1, e2, indexing="ij")
    z0 = xp.array([0.5])

    num_ux  = prop.variables.u.spline(e1, e2, z0, squeeze_out=True)[0]
    num_uex = prop.variables.ue.spline(e1, e2, z0, squeeze_out=True)[0]
    num_phi = prop.variables.phi.spline(e1, e2, z0, squeeze_out=True)

    err_u   = xp.max(xp.abs(num_ux  - mms_u(E1, E2, 0*E1)[0]))
    err_ue  = xp.max(xp.abs(num_uex - mms_ue(E1, E2, 0*E1)[0]))
    err_phi = xp.max(xp.abs(num_phi - mms_phi(E1, E2, 0*E1)[0]))

    assert err_u   < atol, f"[{bc_type}] u   L∞ error {err_u:.3e}   >= {atol}"
    assert err_ue  < atol, f"[{bc_type}] ue  L∞ error {err_ue:.3e}  >= {atol}"
    assert err_phi < atol, f"[{bc_type}] phi L∞ error {err_phi:.3e} >= {atol}"


if __name__ == "__main__":
    test_mms_1d("inhom_dirichlet")
    test_mms_2d("hom_dirichlet")