import cunumpy as xp
import pytest
from feectools.linalg.solvers import UzawaSolver

from struphy import domains
from struphy.feec.basis_projection_ops import BasisProjectionOperators
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.fields_background.equils import HomogenSlab
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.two_fluid_quasi_neutral_full import TwoFluidQuasiNeutralFull
from struphy.topology.grids import TensorProductGrid


def _fill_random(v, rng):
    if hasattr(v, "blocks"):
        for blk in v.blocks:
            _fill_random(blk, rng)
    else:
        v[:] = rng.standard_normal(v[:].shape)


def _make_prop(solver, derham, domain, eq_mhd):
    u = FEECVariable(space="Hdiv")
    ue = FEECVariable(space="Hdiv")
    phi = FEECVariable(space="L2")
    for var in (u, ue, phi):
        var.allocate(derham, domain, eq_mhd)

    prop = TwoFluidQuasiNeutralFull()
    prop.variables.u = u
    prop.variables.ue = ue
    prop.variables.phi = phi
    # a single Uzawa sweep is enough to trigger the dt update, convergence is not tested here
    solver_params = SolverParameters(tol=1e-10, maxiter=1 if solver == "uzawa" else 3000, recycle=False)
    prop.options = prop.Options(stab_sigma=0.0, eps_norm=1.0, solver=solver, solver_params=solver_params)
    prop.allocate()
    return prop


@pytest.mark.filterwarnings("ignore")
def test_uzawa_A11_includes_mass_over_dt(monkeypatch):
    """The Uzawa path must solve with the same A11 + M2/dt block as the direct path (#445)."""

    # UzawaSolver.solve(out=None) builds its result on a fresh BlockVectorSpace, which fails the
    # BlockVector space check; always pass an out vector (unrelated to what is tested here).
    uzawa_dot = UzawaSolver.dot

    def dot_with_out(self, b, out=None):
        return uzawa_dot(self, b, out=self.domain.zeros() if out is None else out)

    monkeypatch.setattr(UzawaSolver, "dot", dot_with_out)

    domain = domains.Cuboid()
    grid = TensorProductGrid(num_elements=[6, 1, 1])
    derham = Derham(grid, DerhamOptions(degree=[2, 1, 1], bcs=(("dirichlet", "dirichlet"), None, None)))
    eq_mhd = HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0, beta=0.1, n0=1.0)
    eq_mhd.domain = domain

    Propagator.derham = derham
    Propagator.domain = domain
    Propagator.mass_ops = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)
    Propagator.basis_ops = BasisProjectionOperators(derham, domain, eq_mhd=eq_mhd)

    direct = _make_prop("gmres", derham, domain, eq_mhd)
    uzawa = _make_prop("uzawa", derham, domain, eq_mhd)

    rng = xp.random.default_rng(0)
    for dt in [0.1, 0.02]:
        direct(dt)
        uzawa(dt)
        M = direct._Minv.linop
        A11_dt = M[0, 0][0, 0]

        # same A11 block (Dirichlet DOFs are in the kernel of the constrained operators)
        v = A11_dt.domain.zeros()
        _fill_random(v, rng)
        uzawa._apply_essential_bc(v)
        assert xp.allclose(uzawa._Minv._A11.dot(v).toarray(), A11_dt.dot(v).toarray(), rtol=1e-12, atol=0.0)

        # the exact solution x of M x = b is a fixed point of the Uzawa velocity update
        x = M.domain.zeros()
        _fill_random(x, rng)
        uzawa._apply_essential_bc(x[0][0])
        uzawa._apply_essential_bc(x[0][1])
        uzawa._Minv._options["x0"] = x.copy()
        out = uzawa._Minv.dot(M.dot(x))
        u_ex = x[0][0].toarray()
        assert xp.allclose(out[0][0].toarray(), u_ex, rtol=0.0, atol=1e-6 * xp.max(xp.abs(u_ex)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
