import cunumpy as xp
import pytest

from struphy import EnvironmentOptions, Simulation, Time, equils, perturbations
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.multigrid.preconditioner import MultiGridOptions
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


def shear_alfven_step(tmp_path, precond, dts):
    """The velocity after implicit shear Alfven steps of sizes ``dts`` with the preconditioner ``precond``."""
    model = ShearAlfven()
    model.propagators.shear_alf.options = model.propagators.shear_alf.Options(
        precond=precond, solver_params=SolverParameters(tol=1e-12, maxiter=1000)
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
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    )
    sim.allocate()
    shear_alf = model.propagators.shear_alf
    for dt in dts:
        shear_alf(dt)
        if precond == "MultiGrid":
            # the multigrid preconditioner is (re)built for the current time step
            assert shear_alf._multigrid_dt == dt
    return model.mhd.velocity.spline.vector.toarray()


def test_multigrid_gives_the_same_step(tmp_path):
    """The multigrid preconditioner of the Schur operator changes the iterations, not the solution."""
    dts = (0.1, 0.05)
    u_mass = shear_alfven_step(tmp_path, "MassMatrixPreconditioner", dts)
    u_multigrid = shear_alfven_step(tmp_path, "MultiGrid", dts)
    assert xp.linalg.norm(u_multigrid - u_mass) < 1e-8 * xp.linalg.norm(u_mass)
