import cunumpy as xp
import pytest

from struphy import EnvironmentOptions, Simulation, Time, equils, perturbations
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.multigrid.preconditioner import MultiGridOptions
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import ShearAlfven
from struphy.propagators.shear_alfven_propagator import ShearAlfvenPropagator
from struphy.topology import grids


def test_multigrid_options():
    options = ShearAlfvenPropagator.Options(precond="MultiGrid")
    assert isinstance(options.multigrid, MultiGridOptions)
    with pytest.raises(AssertionError):
        ShearAlfvenPropagator.Options(precond="MultiGrid", algo="explicit")
    with pytest.raises(AssertionError):
        ShearAlfvenPropagator.Options(precond="MultiGrid", solver="cg")


@pytest.mark.mpi_skip
def test_stiffness_preconditioner_options():
    """The Kronecker preconditioner is the default (also with polar splines, in its 2D x 1D version); without its
    requirements it falls back to the (polar) mass matrix one. The toroidal Fourier preconditioner needs the implicit
    algorithm and a periodic eta_3 direction."""
    from types import SimpleNamespace

    from struphy.feec.preconditioner import ShearAlfvenSchurPreconditioner

    assert ShearAlfvenPropagator.Options().precond == "StiffnessPreconditioner"
    tp = SimpleNamespace(polar_splines=False, with_local_projectors=False, spl_kind=(False, True, True))
    polar = SimpleNamespace(polar_splines=True, with_local_projectors=False, spl_kind=(False, True, True))
    local = SimpleNamespace(polar_splines=False, with_local_projectors=True, spl_kind=(False, True, True))
    clamped_3 = SimpleNamespace(polar_splines=True, with_local_projectors=False, spl_kind=(False, True, False))
    ok = {"algo": "implicit", "solver": "pcg", "u_space": "Hdiv"}
    assert ShearAlfvenSchurPreconditioner.resolve("StiffnessPreconditioner", tp, **ok) == "StiffnessPreconditioner"
    assert ShearAlfvenSchurPreconditioner.resolve("StiffnessPreconditioner", polar, **ok) == "StiffnessPreconditioner"
    for derham in (tp, polar):
        assert (
            ShearAlfvenSchurPreconditioner.resolve("ToroidalFourierPreconditioner", derham, **ok)
            == "ToroidalFourierPreconditioner"
        )
    with pytest.raises(AssertionError):
        ShearAlfvenSchurPreconditioner.resolve("ToroidalFourierPreconditioner", clamped_3, **ok)
    with pytest.raises(AssertionError):
        ShearAlfvenSchurPreconditioner.resolve("ToroidalFourierPreconditioner", polar, **{**ok, "algo": "explicit"})
    with pytest.raises(AssertionError):
        ShearAlfvenSchurPreconditioner.resolve("ToroidalFourierPreconditioner", tp, **{**ok, "solver": "cg"})
    for kwargs in ({"algo": "explicit"}, {"u_space": "Hcurl"}, {"solver": "cg"}):
        args = {**ok, **kwargs}
        assert (
            ShearAlfvenSchurPreconditioner.resolve("StiffnessPreconditioner", tp, **args) == "MassMatrixPreconditioner"
        )
        assert (
            ShearAlfvenSchurPreconditioner.resolve("StiffnessPreconditioner", polar, **args)
            == "PolarMassMatrixPreconditioner"
        )
    assert ShearAlfvenSchurPreconditioner.resolve("StiffnessPreconditioner", local, **ok) == "MassMatrixPreconditioner"
    assert ShearAlfvenSchurPreconditioner.resolve("MultiGrid", polar, **ok) == "MultiGrid"


def polar_step(out_folder, precond, monkeypatch, dt=0.5):
    """One implicit shear Alfven step on a polar cylinder with the preconditioner ``precond``: the propagator, the
    energies before and after, the new velocity and the numbers of PCG iterations."""
    from struphy import domains

    model = ShearAlfven()
    model.propagators.shear_alf.options = model.propagators.shear_alf.Options(
        precond=precond, solver_params=SolverParameters(tol=1e-12, maxiter=1000)
    )
    model.mhd.velocity.add_perturbation(
        perturbations.ModesCos(ls=(1,), ms=(1,), given_in_basis="2", comp=2, amps=(1e-2,))
    )
    model.mhd.velocity.add_perturbation(
        perturbations.ModesSin(ls=(2,), ms=(1,), ns=(1,), given_in_basis="2", comp=0, amps=(1e-2,))
    )
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=out_folder, sim_folder=f"polar_{precond}"),
        domain=domains.IGAPolarCylinder(num_elements=(4, 6), degree=(2, 2), a=1.0, Lz=4.0),
        equil=equils.ScrewPinch(a=1.0, R0=4.0 / (2 * xp.pi), B0=1.0),
        grid=grids.TensorProductGrid(num_elements=(4, 6, 4)),
        derham_opts=DerhamOptions(degree=(2, 2, 2), bcs=(("free", "dirichlet"), None, None), polar_splines=True),
    )
    sim.allocate()
    shear_alf = model.propagators.shear_alf

    def energy():
        u = shear_alf.variables.u.spline.vector
        b = shear_alf.variables.b.spline.vector
        return 0.5 * u.inner(shear_alf._schur_solver.A.dot(u)) + 0.5 * b.inner(shear_alf.mass_ops.M2.dot(b))

    niter = []
    call = SchurSolver.__call__

    def recording_call(self, *args, **kwargs):
        out = call(self, *args, **kwargs)
        niter.append(out[1]["niter"])
        return out

    monkeypatch.setattr(SchurSolver, "__call__", recording_call)
    energy_0 = energy()
    b_0 = shear_alf.variables.b.spline.vector.copy()
    shear_alf(dt)
    monkeypatch.undo()
    db = shear_alf.variables.b.spline.vector - b_0
    assert db.inner(db) > 0.0
    return shear_alf, energy_0, energy(), shear_alf.variables.u.spline.vector.toarray(), niter


def test_shear_alfven_polar_preconditioners(mpi_out_folder, monkeypatch):
    """With polar splines, the default preconditioner is the 2D x 1D Kronecker preconditioner of the Schur operator
    and the toroidal Fourier preconditioner is exact for the (axisymmetric) screw pinch: both give the step of the polar
    mass matrix preconditioner with fewer iterations, and the step conserves the energy (issue #810)."""
    from struphy.feec.preconditioner import PolarKroneckerBlockPreconditioner
    from struphy.linear_algebra.toroidal_fourier import ToroidalFourierSolver

    results = {}
    for precond in ("PolarMassMatrixPreconditioner", "StiffnessPreconditioner", "ToroidalFourierPreconditioner"):
        shear_alf, e_0, e_1, u, niter = polar_step(mpi_out_folder, precond, monkeypatch)
        assert abs(e_1 - e_0) < 1e-8 * e_0
        results[precond] = (u, niter[0])
        if precond == "StiffnessPreconditioner":
            assert shear_alf._precond == "StiffnessPreconditioner"
            assert isinstance(shear_alf._schur_solver.precond, PolarKroneckerBlockPreconditioner)
        if precond == "ToroidalFourierPreconditioner":
            assert isinstance(shear_alf._schur_solver.precond, ToroidalFourierSolver)
            assert shear_alf._schur_solver.precond.residual < 1e-10

    u_mass, n_mass = results["PolarMassMatrixPreconditioner"]
    for precond in ("StiffnessPreconditioner", "ToroidalFourierPreconditioner"):
        u, n = results[precond]
        assert xp.linalg.norm(u - u_mass) < 1e-8 * xp.linalg.norm(u_mass)
        assert n < n_mass, (precond, n, n_mass)
    assert results["ToroidalFourierPreconditioner"][1] <= 3


def shear_alfven_step(tmp_path, precond, dts, exb_correction=True, bcs=(None, None, None), monkeypatch=None):
    """The velocity after implicit shear Alfven steps of sizes ``dts`` with the preconditioner ``precond``, and the
    numbers of PCG iterations (recorded if ``monkeypatch`` is given)."""
    model = ShearAlfven()
    model.propagators.shear_alf.options = model.propagators.shear_alf.Options(
        precond=precond, exb_correction=exb_correction, solver_params=SolverParameters(tol=1e-12, maxiter=1000)
    )
    model.mhd.velocity.add_perturbation(
        perturbations.ModesCos(ls=(1,), ms=(1,), given_in_basis="2", comp=0, amps=(1e-2,))
    )
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder=str(precond)),
        time_opts=Time(dt=dts[0], Tend=dts[0]),
        equil=equils.HomogenSlab(B0x=0.2, B0z=1.0),
        grid=grids.TensorProductGrid(num_elements=(8, 8, 4)),
        derham_opts=DerhamOptions(degree=(2, 2, 1), bcs=bcs),
    )
    sim.allocate()
    shear_alf = model.propagators.shear_alf
    niter = []
    if monkeypatch is not None:
        call = SchurSolver.__call__

        def recording_call(self, *args, **kwargs):
            out = call(self, *args, **kwargs)
            niter.append(out[1]["niter"])
            return out

        monkeypatch.setattr(SchurSolver, "__call__", recording_call)
    for dt in dts:
        shear_alf(dt)
        if precond == "MultiGrid":
            # the multigrid preconditioner is (re)built for the current time step
            assert shear_alf._multigrid_dt == dt
        if precond == "StiffnessPreconditioner":
            assert shear_alf._schur_precond_dt == dt
    if monkeypatch is not None:
        monkeypatch.undo()
    return model.mhd.velocity.spline.vector.toarray(), niter


def test_multigrid_gives_the_same_step(tmp_path):
    """The multigrid preconditioner of the Schur operator changes the iterations, not the solution."""
    dts = (0.1, 0.05)
    u_mass, _ = shear_alfven_step(tmp_path, "MassMatrixPreconditioner", dts)
    u_multigrid, _ = shear_alfven_step(tmp_path, "MultiGrid", dts)
    assert xp.linalg.norm(u_multigrid - u_mass) < 1e-8 * xp.linalg.norm(u_mass)


@pytest.mark.parametrize("exb_correction", [True, False])
@pytest.mark.parametrize("bcs", [(None, None, None), (("dirichlet", "dirichlet"), None, None)])
def test_stiffness_preconditioner_gives_the_same_step(tmp_path, monkeypatch, exb_correction, bcs):
    """The Kronecker preconditioner of the Schur operator changes the iterations, not the solution; with the E x B
    correction it needs fewer iterations than the mass matrix preconditioner for large time steps."""
    dts = (1.0, 2.0)
    u_mass, n_mass = shear_alfven_step(tmp_path, "MassMatrixPreconditioner", dts, bcs=bcs, monkeypatch=monkeypatch)
    u_stiff, n_stiff = shear_alfven_step(
        tmp_path, "StiffnessPreconditioner", dts, exb_correction=exb_correction, bcs=bcs, monkeypatch=monkeypatch
    )
    assert xp.linalg.norm(u_stiff - u_mass) < 1e-8 * xp.linalg.norm(u_mass)
    if exb_correction:
        assert all(n < m for n, m in zip(n_stiff, n_mass)), (n_stiff, n_mass)
