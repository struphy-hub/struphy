import cunumpy as xp
import pytest

from struphy import EnvironmentOptions, Simulation, Time, equils, perturbations
from struphy.io.options import DerhamOptions
from struphy.models import LinearMHD
from struphy.models.variables import FEECVariable
from struphy.propagators.background_pressure_gradient import BackgroundPressureGradient
from struphy.propagators.base import Propagator
from struphy.topology import grids


@pytest.mark.mpi_skip
def test_u_field_space():
    """BackgroundPressureGradient reads a velocity in Hdiv only."""
    BackgroundPressureGradient(u_field=FEECVariable(space="Hdiv"))
    with pytest.raises(AssertionError):
        BackgroundPressureGradient(u_field=FEECVariable(space="Hcurl"))


def _projection_error(tmp_path, num_elements):
    """Max error of the step dp/dt = - U . grad p0 for one smooth velocity in a sheared slab (p0 depends on x), against
    U_h . grad p0 evaluated at points (the gradient by central differences of p0)."""
    equil = equils.ShearedSlab(beta=1.0)
    model = LinearMHD()
    model.mhd.velocity.add_perturbation(
        perturbations.ModesSin(ls=(1,), ms=(1,), given_in_basis="2", comp=0, amps=(1.0,))
    )
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder=f"sim_{num_elements[0]}"),
        time_opts=Time(dt=0.1, Tend=0.1),
        equil=equil,
        grid=grids.TensorProductGrid(num_elements=num_elements),
        derham_opts=DerhamOptions(degree=(3, 2, 1), bcs=(("free", "free"), None, None)),
    )
    sim.allocate()

    # one step from p = 0
    model.mhd.pressure.spline.vector[:] = 0.0
    model.mhd.pressure.spline.vector.update_ghost_regions()
    dt = 0.5
    model.propagators.bg_pressure(dt)

    # evaluate at interior points (unit cube: the 3-form value equals the scalar)
    e1 = xp.linspace(0.05, 0.95, 19)
    e2 = xp.linspace(0.0, 1.0, 7)
    e3 = xp.array([0.25])
    p_h = xp.array(model.mhd.pressure.spline(e1, e2, e3))
    u_h = xp.array(model.mhd.velocity.spline(e1, e2, e3)[0])

    E1, E2, E3 = xp.meshgrid(e1, e2, e3, indexing="ij")
    h = 1e-5
    dp0 = (equil.p0(E1 + h, E2, E3) - equil.p0(E1 - h, E2, E3)) / (2 * h)
    expected = -dt * u_h * dp0
    assert xp.max(xp.abs(expected)) > 1e-2
    return xp.max(xp.abs(p_h - expected)) / xp.max(xp.abs(expected))


def test_convergence(tmp_path):
    """Pi^3[U_h . grad p0] (computed as D S2 - K3 D) converges to U_h . grad p0 under mesh refinement."""
    errors = [_projection_error(tmp_path, (n, 4, 1)) for n in (8, 16)]
    print(f"relative errors {errors}")
    assert errors[0] < 5e-2
    assert errors[1] < errors[0] / 3


def test_homogeneous_vanishes(tmp_path):
    """No pressure change in a homogeneous slab (grad p0 = 0), to round-off."""
    model = LinearMHD()
    for comp in range(3):
        model.mhd.velocity.add_perturbation(
            perturbations.Noise(direction="e1e2e3", amp=1e-2, seed=1234 + comp, comp=comp, given_in_basis="2")
        )
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder="homogen"),
        time_opts=Time(dt=0.1, Tend=0.1),
        equil=equils.HomogenSlab(beta=2.0),
        grid=grids.TensorProductGrid(num_elements=(8, 6, 4)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    )
    sim.allocate()

    G_u = model.propagators.bg_pressure._G.dot(model.mhd.velocity.spline.vector)
    D_u = Propagator.derham.div.dot(model.mhd.velocity.spline.vector)
    assert xp.sqrt(D_u.inner(D_u)) > 0.0
    assert xp.sqrt(G_u.inner(G_u)) < 1e-12 * xp.sqrt(D_u.inner(D_u))
