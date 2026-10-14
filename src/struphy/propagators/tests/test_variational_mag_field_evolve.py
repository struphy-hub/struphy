import cunumpy as xp
from maybempi import MPI

from struphy import domains
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.fields_background.projected_equils import ProjectedFluidEquilibriumWithB
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.variational_mag_field_evolve import VariationalMagFieldEvolve
from struphy.topology.grids import TensorProductGrid


class _ProjectedB2(ProjectedFluidEquilibriumWithB):
    """Minimal projected equilibrium that only provides given b2 coefficients."""

    def __init__(self, b2):
        self._b2 = b2

    @property
    def b2(self):
        return self._b2.copy()


def _momentum_after_one_step(derham, domain, mass_ops, model, b_init, dt):
    u = FEECVariable(space="H1vec")
    u.allocate(derham=derham, domain=domain)
    b = FEECVariable(space="Hdiv")
    b.allocate(derham=derham, domain=domain)
    b.spline.vector = b_init.copy()

    prop = VariationalMagFieldEvolve()
    prop.variables.u = u
    prop.variables.b = b
    prop.options = prop.Options(
        model=model,
        solver_params=SolverParameters(tol=1e-14, maxiter=500),
        nonlin_solver=NonlinearSolverParameters(type="Newton", tol=1e-12, maxiter=20),
    )
    prop.allocate()
    prop(dt)

    return mass_ops.WMMnew.dot(u.spline.vector).toarray()


def test_linear_model_is_linearisation_of_full_model():
    """The "linear" model must be the linearisation of the "full" model around b0 (with a current),
    i.e. [full(b0 + eps*bt) - full(b0)] / eps = linear(bt) + O(eps, dt)."""

    comm = MPI.COMM_WORLD
    domain = domains.Cuboid()
    grid = TensorProductGrid(num_elements=(4, 4, 1))
    derham = Derham(grid=grid, options=DerhamOptions(degree=(2, 2, 1)), comm=comm)
    mass_ops = WeightedMassOperators(derham=derham, domain=domain)

    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = mass_ops

    # unit density
    mass_ops.WMMnew.spline_functions["l2_field"].vector = derham.P3(lambda x, y, z: 1.0 + 0 * x)
    mass_ops.WMMnew.assemble()

    tp = 2 * xp.pi
    b0 = derham.P2(
        [
            lambda x, y, z: 0.3 + 0 * x,
            lambda x, y, z: 0.2 * xp.cos(tp * x),
            lambda x, y, z: 1.0 + 0.5 * xp.sin(tp * y),
        ],
    )
    bt = derham.P2(
        [
            lambda x, y, z: xp.sin(tp * y),
            lambda x, y, z: xp.cos(tp * x),
            lambda x, y, z: xp.sin(tp * (x + y)),
        ],
    )
    Propagator.projected_equil = _ProjectedB2(b0)

    dt = 1e-3
    eps = 1e-4
    b_pert = b0.copy()
    b_pert += eps * bt

    m_full = _momentum_after_one_step(derham, domain, mass_ops, "full", b_pert, dt)
    m_eq = _momentum_after_one_step(derham, domain, mass_ops, "full", b0, dt)
    m_lin = _momentum_after_one_step(derham, domain, mass_ops, "linear", bt, dt)

    dm_full = (m_full - m_eq) / eps
    rel_err = xp.linalg.norm(dm_full - m_lin) / xp.linalg.norm(dm_full)

    assert rel_err < 1e-3, f"{rel_err =}"


if __name__ == "__main__":
    test_linear_model_is_linearisation_of_full_model()
