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
    """The Kronecker preconditioner is the default; without its requirements it falls back to the (polar) mass matrix
    one."""
    from types import SimpleNamespace

    from struphy.feec.preconditioner import ShearAlfvenSchurPreconditioner

    assert ShearAlfvenPropagator.Options().precond == "StiffnessPreconditioner"
    tp = SimpleNamespace(polar_splines=False, with_local_projectors=False)
    polar = SimpleNamespace(polar_splines=True, with_local_projectors=False)
    local = SimpleNamespace(polar_splines=False, with_local_projectors=True)
    ok = {"algo": "implicit", "solver": "pcg", "u_space": "Hdiv"}
    assert ShearAlfvenSchurPreconditioner.resolve("StiffnessPreconditioner", tp, **ok) == "StiffnessPreconditioner"
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


@pytest.mark.mpi_skip
def test_stiffness_preconditioner_polar_fallback(tmp_path):
    """With polar splines, the default preconditioner falls back to the polar mass matrix preconditioner."""
    from struphy import domains

    model = ShearAlfven()
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder="polar"),
        domain=domains.IGAPolarCylinder(num_elements=(4, 6), degree=(2, 2), a=1.0, Lz=4.0),
        equil=equils.ScrewPinch(a=1.0, R0=4.0 / (2 * xp.pi), B0=1.0),
        grid=grids.TensorProductGrid(num_elements=(4, 6, 2)),
        derham_opts=DerhamOptions(degree=(2, 2, 1), bcs=(("free", "free"), None, None), polar_splines=True),
    )
    sim.allocate()
    shear_alf = model.propagators.shear_alf
    assert shear_alf.options.precond == "StiffnessPreconditioner"
    assert shear_alf._precond == "PolarMassMatrixPreconditioner"


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
