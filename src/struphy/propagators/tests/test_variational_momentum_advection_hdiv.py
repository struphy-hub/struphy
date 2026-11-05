"""VariationalMomentumAdvection with a velocity in H(div) (standalone, without a model)."""

import cunumpy as xp
import numpy as np
import pytest
from maybempi import MPI
from numpy import cos, pi, sin

from struphy import domains, equils
from struphy.feec.basis_projection_ops import BasisProjectionOperators
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.fields_background.projected_equils import ProjectedFluidEquilibriumWithB
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.variational_momentum_advection import VariationalMomentumAdvection
from struphy.topology.grids import TensorProductGrid


def _setup(map_name, num_elements, degree=(2, 2, 2)):
    if map_name == "HollowTorus":
        # the geometry of the TAE test cases
        domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=5.0, sfl=False, pol_period=1, tor_period=1)
        equil = equils.AdhocTorus(a=1.0, R0=5.0, B0=1.0, q_kind=0, p_kind=1, q0=1.25, q1=2.25, n1=0.0, n2=0.0, na=1.0)
    else:
        domain = domains.Cuboid()
        equil = equils.HomogenSlab()
    equil.domain = domain

    grid = TensorProductGrid(num_elements=num_elements)
    derham = Derham(
        grid,
        DerhamOptions(degree=degree, bcs=(("dirichlet", "dirichlet"), None, None)),
        comm=MPI.COMM_WORLD,
        domain=domain,
    )
    mass_ops = WeightedMassOperators(derham, domain, eq_mhd=equil)
    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = mass_ops
    Propagator.projected_equil = ProjectedFluidEquilibriumWithB(equil, derham)
    Propagator.basis_ops = BasisProjectionOperators(derham, domain, eq_mhd=equil)
    return derham, domain, mass_ops


def _uhat(amp):
    """Logical contravariant velocity with u^1 = 0 at eta1 = 0, 1."""

    def uhat(e1, e2, e3):
        return [
            amp * 0.5 * sin(pi * e1) * cos(2 * pi * e2),
            amp * (cos(pi * e1) * sin(2 * pi * e2 + 0.3) + 0.5),
            amp * 0.2 * cos(2 * pi * e3) * e1,
        ]

    return uhat


def _velocity(derham, domain, u_space, amp):
    uhat = _uhat(amp)
    u = FEECVariable(space=u_space)
    u._spline = derham.create_spline_function("u", u_space)
    if u_space == "Hdiv":
        comps = [lambda e1, e2, e3, i=i: domain.jacobian_det(e1, e2, e3) * uhat(e1, e2, e3)[i] for i in range(3)]
        u.spline.vector = derham.boundary_ops["2"].dot(derham.P2(comps))
    else:
        comps = [lambda e1, e2, e3, i=i: uhat(e1, e2, e3)[i] + 0 * e1 for i in range(3)]
        u.spline.vector = derham.boundary_ops["v"].dot(derham.Pv(comps))
    return u


def _propagator(u, bracket_momentum="n", nonlin_type="Newton"):
    prop = VariationalMomentumAdvection()
    prop.variables.u = u
    prop.options = prop.Options(
        bracket_momentum=bracket_momentum,
        solver_params=SolverParameters(tol=1e-13, maxiter=300),
        nonlin_solver=NonlinearSolverParameters(tol=1e-11, maxiter=30, type=nonlin_type),
    )
    prop.allocate()
    return prop


@pytest.mark.parametrize("map_name", ["Cuboid", "HollowTorus"])
@pytest.mark.parametrize("bracket_momentum, nonlin_type", [("n", "Newton"), ("n+1/2", "Newton"), ("n+1/2", "Picard")])
def test_hdiv_energy_conservation(map_name, bracket_momentum, nonlin_type):
    """The kinetic energy 1/2 u^T M2n u is conserved, the velocity changes, and the boundary conditions hold."""
    derham, domain, mass_ops = _setup(map_name, (6, 6, 4))
    # on the torus, the logical toroidal component is scaled by 2 pi R0 ~ 31
    u = _velocity(derham, domain, "Hdiv", amp=0.5 if map_name == "Cuboid" else 0.05)
    prop = _propagator(u, bracket_momentum, nonlin_type)
    M = mass_ops.M2n

    u0 = u.spline.vector.copy()
    e0 = 0.5 * u0.inner(M.dot(u0))
    for _ in range(3):
        prop(0.02)
    u1 = u.spline.vector
    e1 = 0.5 * u1.inner(M.dot(u1))
    assert abs(e1 - e0) / e0 < 1e-10, f"{e0 = }, {e1 = }"

    diff = u1 - u0
    assert (diff.inner(M.dot(diff)) / u0.inner(M.dot(u0))) ** 0.5 > 1e-3

    # vanishing normal flux at the walls
    u1_bc = derham.boundary_ops["2"].dot(u1)
    diff = u1_bc - u1
    assert diff.inner(diff) == 0.0


def test_hdiv_time_order():
    """With the momentum at t^{n+1/2} the scheme is of second order in dt, with the momentum at t^n of first order:
    errors against a reference with a small time step."""
    derham, domain, mass_ops = _setup("Cuboid", (4, 4, 4))
    M = mass_ops.M2n
    T = 0.08

    def run(bracket_momentum, nsteps):
        u = _velocity(derham, domain, "Hdiv", amp=0.5)
        prop = _propagator(u, bracket_momentum)
        for _ in range(nsteps):
            prop(T / nsteps)
        return u.spline.vector.copy()

    ref = run("n+1/2", 16)

    def err(x):
        d = x - ref
        return d.inner(M.dot(d)) ** 0.5

    e_half = [err(run("n+1/2", n)) for n in (2, 4)]
    e_n = [err(run("n", n)) for n in (2, 4)]
    assert e_half[0] / e_half[1] > 3.0, f"{e_half = }"
    assert e_n[0] / e_n[1] > 1.6, f"{e_n = }"
    assert e_half[1] < e_n[1]


def test_hdiv_agrees_with_h1vec():
    """For the same initial velocity field, the H(div) and the (H1)^3 discretizations stay close (both are
    consistent discretizations of the same equation)."""
    derham, domain, mass_ops = _setup("Cuboid", (8, 8, 4))
    pts = [np.linspace(0.05, 0.95, 5), np.linspace(0.0, 1.0, 7)[:-1], np.linspace(0.0, 1.0, 5)[:-1]]
    E = np.meshgrid(*pts, indexing="ij")
    sqrt_g = domain.jacobian_det(*E)

    mass_ops.WMMnew.spline_functions["l2_field"].vector = Propagator.projected_equil.n3
    mass_ops.WMMnew.assemble()

    vals = {}
    for u_space in ("Hdiv", "H1vec"):
        u = _velocity(derham, domain, u_space, amp=0.2)
        prop = _propagator(u)
        for _ in range(5):
            prop(0.01)
        v = u.spline(*E)
        vals[u_space] = [c / sqrt_g for c in v] if u_space == "Hdiv" else v

    exact0 = _uhat(0.2)(*E)
    norm = np.sqrt(sum(np.sum(c**2) for c in exact0))
    change = np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(vals["Hdiv"], exact0))) / norm
    diff = np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(vals["Hdiv"], vals["H1vec"]))) / norm
    assert change > 0.05, f"{change = }"
    assert diff < 0.2 * change, f"{diff = }, {change = }"


def test_hdiv_invalid_options():
    derham, domain, mass_ops = _setup("Cuboid", (4, 4, 4))

    # the momentum at t^{n+1/2} needs a velocity in Hdiv
    u = _velocity(derham, domain, "H1vec", amp=0.1)
    with pytest.raises(AssertionError):
        _propagator(u, "n+1/2")

    # no div-div regularization with a velocity in Hdiv
    u = _velocity(derham, domain, "Hdiv", amp=0.1)
    prop = VariationalMomentumAdvection(divdiv_alpha=0.1)
    prop.variables.u = u
    with pytest.raises(AssertionError):
        prop.allocate()

    with pytest.raises(AssertionError):
        VariationalMomentumAdvection.Options(bracket_momentum="n+1")


if __name__ == "__main__":
    test_hdiv_energy_conservation("HollowTorus", "n+1/2", "Newton")
    test_hdiv_time_order()
    test_hdiv_agrees_with_h1vec()
