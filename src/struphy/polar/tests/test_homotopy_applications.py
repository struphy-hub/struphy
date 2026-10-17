"""Tests of the applications of the high-order polar splines: wave equations (core and Struphy models with the
leapfrog), the helical tube, the helical vacuum field (finite on the axis), field lines and the 1D/2D space demos."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------------------------------- core demos
@pytest.mark.parametrize("form", [0, 1])
def test_wave_leapfrog_conserves_energy(form):
    from struphy.polar.homotopy.demos import waves

    problem = waves.WaveProblem(form, 2, degree=3, radial_cells=4, angular_cells=12)
    _, snapshots, energy_error = problem.evolve(0.2)
    assert energy_error < 1e-10
    assert np.all(np.isfinite(snapshots))
    # the CFL step is set by the largest eigenvalue of M^{-1} K
    assert problem.dt == pytest.approx(0.9 * 2 / np.sqrt(problem.lambda_max))


def test_spaces_demos_commute():
    from struphy.polar.homotopy.demos import spaces

    _, residuals = spaces.spaces_1d(verbose=False)
    assert max(residuals.values()) < 1e-12
    curl_grad, grad_residual, curl_residual = spaces.identities(verbose=False)
    assert curl_grad == 0 and grad_residual < 1e-12 and curl_residual < 1e-12
    # the quarter annulus HollowCylinder(a1=1, a2=2, poc=4)
    V0, V1, _ = spaces.mapped_identities(verbose=False)["spaces"]
    assert abs(V1.derivative_matrix @ V0.derivative_matrix).max() == 0


# ---------------------------------------------------------------------------------------------------- geometry
def test_helical_tube():
    from struphy.geometry.domains import HelicalTube
    from struphy.geometry.domains.helical_tube.helical_tube import helical_axis

    for preset in ("stellarator", "islands"):
        domain = HelicalTube(preset, num_elements=(6, 24, 48), degree=(3, 3, 3))
        assert domain.pole
        e1, e2, e3 = np.linspace(0.05, 1, 5), np.linspace(0, 1, 7), np.linspace(0, 1, 9)
        assert np.all(domain.jacobian_det(e1, e2, e3) > 0)
        g = domain.geometry
        axis = np.array(domain(np.array([0.0]), np.array([0.3]), e3)).reshape(3, -1)
        exact = helical_axis(e3, g["R0"], g["delta"], g["delta_z"], g["periods"])
        assert np.max(np.abs(axis - exact)) < 1e-3


def test_helical_vacuum_field_is_vacuum_and_finite_on_axis():
    from struphy import equils
    from struphy.geometry.domains import HollowCylinder

    eq = equils.HelicalVacuumField(delta=0.5, k=3.0)
    eq.domain = HollowCylinder(a1=0.0, a2=0.3, Lz=2 * np.pi)
    point, h = np.array([0.1, 0.05, 0.3]), 1e-5

    def B(p):
        return np.array(eq.b_xyz(*(np.array([c]) for c in p)), dtype=float).ravel()

    d = [(B(point + h * e) - B(point - h * e)) / (2 * h) for e in np.eye(3)]
    assert abs(d[0][0] + d[1][1] + d[2][2]) < 1e-9
    assert max(abs(d[1][2] - d[2][1]), abs(d[2][0] - d[0][2]), abs(d[0][1] - d[1][0])) < 1e-9
    etas = (np.array([0.0, 0.5]), np.array([0.2]), np.array([0.4]))
    for name in ("absB0", "b1", "b2", "unit_b1", "gradB1", "curl_unit_b_dot_b0", "curl_unit_b2"):
        assert np.all(np.isfinite(np.array(getattr(eq, name)(*etas), dtype=float))), name


def test_cartesian_equilibria_unchanged_away_from_axis():
    """The direct Cartesian evaluation (finite on a polar axis) equals the former pull-back/push-forward."""
    from struphy import equils
    from struphy.fields_background.base import FluidEquilibriumWithB, MHDequilibrium
    from struphy.geometry.domains import HollowCylinder

    eq = equils.ScrewPinch(a=0.3, R0=1.0)
    eq.domain = HollowCylinder(a1=0.05, a2=0.3, Lz=2 * np.pi)
    etas = (np.linspace(0.1, 1, 4), np.linspace(0, 1, 5), np.linspace(0, 1, 3))
    for name, base in (
        ("b_cart", FluidEquilibriumWithB),
        ("b1", FluidEquilibriumWithB),
        ("b2", FluidEquilibriumWithB),
        ("j_cart", MHDequilibrium),
        ("j2", MHDequilibrium),
        ("gradB_cart", FluidEquilibriumWithB),
    ):
        new, old = getattr(eq, name)(*etas), getattr(base, name)(eq, *etas)
        if isinstance(new, tuple):
            new, old = new[0], old[0]
        assert np.allclose(np.array(new), np.array(old), atol=1e-13), name


# ---------------------------------------------------------------------------------------------------- Struphy
def test_field_lines_divergence_free_and_iota():
    from struphy.polar.homotopy.demos import fieldlines

    result = fieldlines.study(turns=2, steps_per_turn=40, verbose=False, num_elements=(6, 18, 16))
    assert result["divergence"] < 1e-12
    for (r, _, z), w in zip(result["seeds"], result["windings"]):
        if z == 0.0 and 0.2 < r < 0.6:
            assert w == pytest.approx(fieldlines.iota(r), abs=5e-3)


def _run_simulation(model, domain, derham_opts, dt, tend, folder, setup):
    from struphy import EnvironmentOptions, Simulation, Time, grids

    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(folder), sim_folder="sim"),
        time_opts=Time(dt=dt, Tend=tend),
        domain=domain,
        grid=grids.TensorProductGrid(num_elements=(4, 12, 1)),
        derham_opts=derham_opts,
    )
    setup(model)
    sim.run()
    from struphy import Output

    return Output(folder / "sim")


def test_scalar_wave_matches_core_leapfrog(tmp_path):
    """The ScalarWave model with polar splines integrates exactly the core leapfrog (same spaces and dt)."""
    from struphy import DerhamOptions, domains
    from struphy.initial.base import GenericPerturbation
    from struphy.linear_algebra.solver import SolverParameters
    from struphy.models import ScalarWave
    from struphy.polar.homotopy.demos.waves import WaveProblem

    def pulse(x, y):
        return np.exp(-((x - 0.2) ** 2 + y**2) / (2 * 0.2**2))

    problem = WaveProblem(0, 2, degree=3, radial_cells=4, angular_cells=12)
    dt = problem.dt
    c = problem.complex
    problem.initial = c.duals(0).T @ c.spaces[0].commuting_projection(pulse)

    def setup(model):
        model.em_fields.phi.save_data = True
        model.propagators.wave.options = model.propagators.wave.Options(
            precond="PolarMassMatrixPreconditioner", solver_params=SolverParameters(tol=1e-13, maxiter=500)
        )
        model.em_fields.phi.add_perturbation(
            GenericPerturbation(lambda x, y, z: pulse(x, y) + 0 * z, given_in_basis="physical")
        )

    derham_opts = DerhamOptions(
        degree=(3, 3, 1),
        bcs=(("free", "dirichlet"), None, None),
        polar_splines=True,
        polar_regularity=2,
    )
    out = _run_simulation(ScalarWave(), domains.PerturbedDiskCylinder(), derham_opts, dt, 3 * dt, tmp_path, setup)
    times, snapshots, _ = problem.evolve(3 * dt + 1e-12)
    e1, e2 = np.linspace(0.0, 1, 9), np.linspace(0, 1, 13)
    phi = out.evaluate("em_fields/phi", t=-1, eta1=e1, eta2=e2, eta3=np.array([0.5])).values.reshape(9, 13)
    core = problem.observable_space.evaluate(snapshots[3], e1, e2)
    assert np.max(np.abs(phi - core)) < 1e-9 * np.max(np.abs(core))
