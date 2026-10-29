import cunumpy as xp
import pytest

from struphy import EnvironmentOptions, Simulation, Time, equils, perturbations
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import LinearMHD
from struphy.models.variables import FEECVariable
from struphy.propagators.background_current import BackgroundCurrent
from struphy.propagators.base import Propagator
from struphy.topology import grids


@pytest.mark.mpi_skip
def test_options():
    """BackgroundCurrent reads a magnetic field in Hdiv or Hcurl only, and validates its options."""
    BackgroundCurrent(b_field=FEECVariable(space="Hcurl"))
    with pytest.raises(AssertionError):
        BackgroundCurrent(b_field=FEECVariable(space="L2"))
    with pytest.raises(AssertionError):
        BackgroundCurrent.Options(solver="no_solver")


def _force(tmp_path, b_space):
    """Velocity change of one step of BackgroundCurrent in a sheared slab (J0 != 0), with the equilibrium field B0 as
    magnetic field perturbation (2-form or 1-form), and the residual of M2n du = dt M2J b."""
    model = LinearMHD(b_space=b_space, split_sound=True)
    model.propagators.bg_current.options = BackgroundCurrent.Options(solver_params=SolverParameters(tol=1e-13))
    model.mhd.velocity.add_perturbation(
        perturbations.ModesCos(ls=(1,), ms=(1,), given_in_basis="2", comp=1, amps=(1e-2,))
    )
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder=f"sim_{b_space}"),
        time_opts=Time(dt=0.1, Tend=0.1),
        equil=equils.ShearedSlab(q0=1.05, q1=1.8, R0=1.0),
        grid=grids.TensorProductGrid(num_elements=(16, 4, 1)),
        derham_opts=DerhamOptions(degree=(3, 2, 1), bcs=(("free", "free"), None, None)),
    )
    sim.allocate()

    equil = Propagator.projected_equil
    b = model.em_fields.b_field.spline.vector
    (equil.b1 if b_space == "Hcurl" else equil.b2).copy(out=b)
    b.update_ghost_regions()

    u = model.mhd.velocity.spline.vector
    u0 = u.copy()
    dt = 0.7
    model.propagators.bg_current(dt)
    du = u - u0

    mass_ops = Propagator.mass_ops
    MJ = mass_ops.M21J if b_space == "Hcurl" else mass_ops.M2J
    force = MJ.dot(b)
    residual = mass_ops.M2n.dot(du) - dt * force
    rel_residual = xp.sqrt(residual.inner(residual)) / (dt * xp.sqrt(force.inner(force)))
    return du.toarray(), rel_residual


def test_exact_and_space_independent(tmp_path):
    """The step solves M2n du = dt M2J b (M21J for the 1-form), and the velocity change from the 2-form and the 1-form
    of the same field agree up to the projection error. The sign of the force is checked against Magnetosonic in
    test_sound_wave_linear.py::test_split_agrees_with_magnetosonic_inhomogeneous."""
    du_div, res_div = _force(tmp_path, "Hdiv")
    du_curl, res_curl = _force(tmp_path, "Hcurl")
    print(f"{res_div = :.2e}, {res_curl = :.2e}")
    assert res_div < 1e-10 and res_curl < 1e-10

    norm = xp.max(xp.abs(du_div))
    rel_diff = xp.max(xp.abs(du_div - du_curl)) / norm
    print(f"{norm = :.2e}, {rel_diff = :.2e}")
    assert norm > 1e-2
    assert rel_diff < 1e-2
