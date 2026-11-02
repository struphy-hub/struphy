import cunumpy as xp
import pytest
from maybempi import MPI

from struphy import domains
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.maxwell_weak_ampere import MaxwellWeakAmpere
from struphy.topology.grids import TensorProductGrid

comm = MPI.COMM_WORLD


def _setup(mapping):
    polar = mapping == "IGAPolarCylinder"
    if mapping == "Cuboid":
        dom = domains.Cuboid()
    elif mapping == "HollowCylinder":
        dom = domains.HollowCylinder(a1=0.1, a2=1.0, Lz=3.0)
    else:
        dom = domains.IGAPolarCylinder(num_elements=(12, 12), degree=(2, 2), a=1.0, Lz=3.0)
    derham = Derham(
        grid=TensorProductGrid(num_elements=(12, 12, 4)),
        options=DerhamOptions(
            degree=(2, 2, 1),
            bcs=(("free", "dirichlet") if polar else ("dirichlet", "dirichlet"), None, None),
            polar_splines=polar,
        ),
        comm=comm,
        domain=dom,
    )
    Propagator.derham = derham
    Propagator.domain = dom
    Propagator.mass_ops = WeightedMassOperators(derham=derham, domain=dom)
    return derham, dom


def _propagator(derham, dom, seed=1, **options):
    e = FEECVariable(space="Hcurl")
    e.allocate(derham=derham, domain=dom)
    b = FEECVariable(space="Hdiv")
    b.allocate(derham=derham, domain=dom)
    # smooth logical components (projected, so that they are also in the polar spaces)
    funcs = [
        lambda e1, e2, e3, k=k: (
            xp.sin(xp.pi * e1) * xp.cos(2 * xp.pi * (e2 + k * seed / 7)) * xp.sin(2 * xp.pi * e3 + k)
        )
        for k in range(3)
    ]
    e.spline.vector = derham.boundary_ops["1"].dot(derham.P1(funcs))
    b.spline.vector = derham.boundary_ops["2"].dot(derham.P2(funcs[::-1]))
    prop = MaxwellWeakAmpere()
    prop.variables.e = e
    prop.variables.b = b
    prop.options = prop.Options(**options)
    prop.allocate()
    return prop


def _energy(prop):
    e = prop.variables.e.spline.vector
    b = prop.variables.b.spline.vector
    return 0.5 * e.inner(prop.mass_ops.M1.dot(e)) + 0.5 * b.inner(prop.mass_ops.M2.dot(b))


@pytest.mark.parametrize("mapping", ["Cuboid", "HollowCylinder"])
def test_stiffness_preconditioner(mapping):
    """Implicit step with the stiffness and the mass matrix preconditioner: same solution, exact energy conservation,
    and fewer iterations with the stiffness preconditioner for dt >> h."""
    derham, dom = _setup(mapping)
    dt = 1.0
    solver_params = SolverParameters(tol=1e-12, maxiter=3000)

    niter = {}
    solution = {}
    mass_pc = "PolarMassMatrixPreconditioner" if derham.polar_splines else "MassMatrixPreconditioner"
    for precond in (mass_pc, "StiffnessPreconditioner"):
        prop = _propagator(derham, dom, precond=precond, solver_params=solver_params)
        prop.prepare_time_step(dt)
        e = prop.variables.e.spline.vector
        b = prop.variables.b.spline.vector
        _, info = prop._schur_solver(e, prop._B.dot(b), dt)
        assert info["success"]
        niter[precond] = info["niter"]

        energy_0 = _energy(prop)
        prop(dt)
        assert abs(_energy(prop) - energy_0) < 1e-8 * energy_0
        solution[precond] = prop.variables.e.spline.vector.copy()

    diff = solution["StiffnessPreconditioner"] - solution[mass_pc]
    assert diff.inner(diff) ** 0.5 < 1e-8 * solution[mass_pc].inner(solution[mass_pc]) ** 0.5
    assert niter["StiffnessPreconditioner"] < niter[mass_pc]


def test_stiffness_preconditioner_dt_change():
    """The stiffness preconditioner is rebuilt only when dt changes."""
    derham, dom = _setup("Cuboid")
    prop = _propagator(derham, dom)
    prop(0.1)
    pc = prop._schur_solver.precond
    prop(0.1)
    assert prop._schur_solver.precond is pc
    prop(0.2)
    assert prop._schur_solver.precond is not pc
    assert prop._schur_precond_dt == 0.2


def test_polar_stiffness_preconditioner():
    """With polar splines, the default preconditioner is a PolarStiffnessPreconditioner. (No solve: the implicit step
    with polar splines currently fails in SchurSolver with any preconditioner.)"""
    from struphy.feec.preconditioner import PolarStiffnessPreconditioner

    derham, dom = _setup("IGAPolarCylinder")
    prop = _propagator(derham, dom)
    prop.prepare_time_step(1.0)
    assert isinstance(prop._schur_solver.precond, PolarStiffnessPreconditioner)


def test_explicit_default_precond():
    """For algo="explicit", the default stiffness preconditioner falls back to the mass matrix preconditioner."""
    derham, dom = _setup("Cuboid")
    prop = _propagator(derham, dom, algo="explicit")
    assert prop._precond == "MassMatrixPreconditioner"
    energy_0 = _energy(prop)
    prop(0.01)
    assert abs(_energy(prop) - energy_0) < 1e-3 * energy_0


@pytest.mark.mpi_skip
def test_precond_options():
    """Invalid preconditioners are rejected; solver="cg" takes no preconditioner."""
    with pytest.raises(AssertionError):
        MaxwellWeakAmpere.Options(precond="HiptmairXu")
    assert MaxwellWeakAmpere.Options().precond == "StiffnessPreconditioner"
    assert MaxwellWeakAmpere.Options(solver="cg").precond is None
