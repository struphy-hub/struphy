import numpy as np
import pytest

from struphy import (
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    SavingParameters,
    Simulation,
    Time,
    WeightsParameters,
    domains,
    grids,
    maxwellians,
)
from struphy.diagnostics.beam_diagnostics import plane_crossings, rms_moments
from struphy.geometry.domains import ElectrodeSegment
from struphy.initial.base import Perturbation
from struphy.initial.perturbations import PiecewiseLinearPotential
from struphy.models import IonOpticsElectrostatic, SteadyStateOptions
from struphy.physics.ion_optics_units import IonOpticsUnits
from struphy.propagators.base import Propagator
from struphy.propagators.push_v_in_force_field import PushVinForceField


def test_ion_optics_electrostatic_wires_prescribed_potential_pusher():
    model = IonOpticsElectrostatic(alpha=2.0, epsilon=0.5)

    assert isinstance(model.propagators.push_v, PushVinForceField)
    assert model.propagators.push_v.potential is model.em_fields.phi
    assert model.propagators.push_v.variables.var is model.ions.var
    assert model.propagators.push_eta.variables.var is model.ions.var
    assert model.ions.equation_params.alpha == 2.0
    assert model.ions.equation_params.epsilon == 0.5


def test_steady_state_is_model_configuration():
    from struphy.models.ion_optics_steady_state import RayBundle
    from struphy.pic.ion_beams import LossTag, PlaneSource

    source = PlaneSource(rate=1.0, current=1.0, axis=0, eta_plane=0.0)
    options = SteadyStateOptions(
        source=source,
        n_rays=8,
        dt=0.1,
        loss_tags=(LossTag("outlet", axis=0, side=1),),
    )
    model = IonOpticsElectrostatic(steady_state=options)

    assert model.steady_state is options
    assert model.steady_state_iteration is None
    assert not hasattr(model.propagators, "inject")
    with pytest.raises(ValueError, match="not both"):
        IonOpticsElectrostatic(steady_state=options, source=source)

    rays = RayBundle(eta=[[0.0, 0.5, 0.5]], v=[[1.0, 0.0, 0.0]], current=1.0)
    explicit = SteadyStateOptions(rays=rays, dt=0.1, loss_tags=(LossTag("outlet", axis=0, side=1),))
    assert IonOpticsElectrostatic(steady_state=explicit).steady_state.rays is rays
    with pytest.raises(ValueError, match="exactly one"):
        SteadyStateOptions(source=source, rays=rays, n_rays=1, dt=0.1)


@pytest.mark.parametrize("mass_number, charge_number", [(1.0, 1), (40.0, 3)])
def test_ion_optics_units_match_model_normalization(mass_number, charge_number):
    units = IonOpticsUnits(length=2e-3, voltage=5e3, mass_number=mass_number, charge_number=charge_number)
    model = IonOpticsElectrostatic(
        base_units=units.base_units(),
        mass_number=mass_number,
        charge_number=charge_number,
    )

    assert model.units.v == pytest.approx(units.velocity, rel=1e-12)
    assert model.units.t == pytest.approx(units.time, rel=1e-12)
    # dv/dt = -grad(phi) for the bulk ion: the potential unit is `voltage`.
    assert model.ions.equation_params.epsilon == pytest.approx(1.0, rel=1e-12)


def test_ion_optics_units_energy_gain_in_si():
    units = IonOpticsUnits(length=1e-3, voltage=1e3, mass_number=4.0, charge_number=2)
    initial_eV, drop_V = 3e3, 7e3
    speed0 = units.speed(initial_eV)
    # Normalized energy conservation: v²/2 + phi = const, with phi in units of `voltage`.
    speed1 = np.sqrt(speed0**2 + 2 * units.potential(drop_V))

    assert units.kinetic_energy_eV(speed1) - initial_eV == pytest.approx(units.charge_number * drop_V, rel=1e-12)
    assert units.volts(units.potential(drop_V)) == pytest.approx(drop_V)


def test_ion_optics_units_non_bulk_species_epsilon():
    units = IonOpticsUnits(length=1e-3, voltage=1e3, mass_number=1.0, charge_number=1)
    model = IonOpticsElectrostatic(base_units=units.base_units(), mass_number=1.0, charge_number=1)
    other = model.Ions(charge_number=2, mass_number=16.0)
    other.setup_equation_params(model.units)

    assert other.equation_params.epsilon == pytest.approx(16.0 / 2.0, rel=1e-12)


def test_piecewise_linear_potential():
    potential = PiecewiseLinearPotential(nodes=(1.0, 2.0), values=(0.0, -4.0), coordinate=1)
    y = np.array([0.0, 1.0, 1.25, 2.0, 3.0])

    np.testing.assert_allclose(potential(0 * y + 7.0, y, 0 * y), [0.0, 0.0, -1.0, -4.0, -4.0])
    with pytest.raises(ValueError):
        PiecewiseLinearPotential(nodes=(1.0, 1.0), values=(0.0, 1.0))


def test_vacuum_potential_accepts_segmented_channel_electrodes(tmp_path):
    """Selected portions of the upper/lower faces provide the Laplace gauge."""
    segments = (
        ElectrodeSegment("lower", 0.0, 0.4, 0.0),
        ElectrodeSegment("lower", 0.6, 1.0, -1.0),
        ElectrodeSegment("upper", 0.0, 0.4, 0.0),
        ElectrodeSegment("upper", 0.6, 1.0, -1.0),
    )
    model = IonOpticsElectrostatic(electrode_segments=segments, electrode_length=1.0)
    model.ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    model.ions.set_markers(
        loading_params=LoadingParameters(Np=1, specific_markers=((0.5, 0.5, 0.5, 0.0, 0.0, 0.0),)),
        weights_params=WeightsParameters(control_variate=False),
        boundary_params=BoundaryParameters(bc=("remove", "remove", "periodic")),
        saving_params=SavingParameters(n_markers=1),
    )
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder="segmented_channel"),
        time_opts=Time(dt=0.1, Tend=0.1),
        domain=domains.Cuboid(),
        grid=grids.TensorProductGrid(num_elements=(12, 6, 1)),
        derham_opts=DerhamOptions(
            degree=(2, 2, 1),
            bcs=(("free", "free"), ("free", "free"), ("free", "free")),
        ),
    )
    sim.allocate()
    assert model.vacuum_solver_info["success"]
    phi = np.asarray(model.em_fields.phi.spline(np.array([0.2, 0.8]), np.array([0.0, 1.0]), np.array([0.5])))
    np.testing.assert_allclose(phi[:, :, 0], [[0.0, 0.0], [-1.0, -1.0]], atol=1e-12)


def test_plane_crossings_and_rms_moments():
    t = np.linspace(0.0, 1.0, 11)
    # Two straight rays x = t, y = y0 + y' t; the second leaves the domain (NaN) before x = 0.75.
    y0, yp = np.array([0.1, -0.2]), np.array([0.3, 0.5])
    states = np.stack(
        [np.stack([t, y0[j] + yp[j] * t, np.ones_like(t), yp[j] * np.ones_like(t)], -1) for j in range(2)], 1
    )
    states[6:, 1] = np.nan

    crossing = plane_crossings(states, 0.35)
    np.testing.assert_allclose(crossing[:, 1], y0 + 0.35 * yp)
    assert np.all(np.isnan(plane_crossings(states, 0.75)[1]))

    y = np.array([-1.0, 0.0, 1.0])
    moments = rms_moments(y, 2.0 * y)
    assert moments["size"] == pytest.approx(np.sqrt(2 / 3))
    assert moments["divergence"] == pytest.approx(2 * np.sqrt(2 / 3))
    assert moments["emittance"] == pytest.approx(0.0, abs=1e-14)
    assert rms_moments(y, np.array([1.0, -2.0, 1.0]))["emittance"] > 0.0


class _Coaxial(Perturbation):
    def __init__(self, a1, a2):
        self.params = {"a1": a1, "a2": a2}
        self.a1, self.a2 = a1, a2
        self.given_in_basis = "physical"
        self.comp = 0

    def __call__(self, x, y, z):
        return np.log(np.sqrt(x**2 + y**2) / self.a2) / np.log(self.a1 / self.a2)


def _coaxial_error(tmp_path, n, degree, a1=0.5, a2=1.0):
    model = IonOpticsElectrostatic(electrode_faces=((True, True), (False, False), (False, False)))
    # The initialized interior is discarded; only the electrode traces matter.
    model.em_fields.phi.add_perturbation(_Coaxial(a1, a2))
    model.ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    model.ions.set_markers(
        loading_params=LoadingParameters(Np=1, specific_markers=((0.5, 0.5, 0.5, 0.0, 0.0, 0.0),)),
        weights_params=WeightsParameters(control_variate=False),
        boundary_params=BoundaryParameters(bc=("remove", "periodic", "periodic")),
        saving_params=SavingParameters(n_markers=1),
    )
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder=f"coax_{n}_{degree}"),
        time_opts=Time(dt=0.1, Tend=0.1),
        domain=domains.HollowCylinder(a1=a1, a2=a2, Lz=1.0),
        grid=grids.TensorProductGrid(num_elements=(n, 4 * n, 1)),
        derham_opts=DerhamOptions(degree=(degree, degree, 1), bcs=(("free", "free"), None, ("free", "free"))),
    )
    sim.allocate()
    eta1 = np.linspace(0.0, 1.0, 51)
    numerical = np.asarray(model.em_fields.phi.spline(eta1, np.array([0.0, 0.3, 0.7]), np.array([0.5])))[:, :, 0]
    r = a1 + (a2 - a1) * eta1
    exact = (np.log(r / a2) / np.log(a1 / a2))[:, None]
    return np.max(np.abs(numerical - exact))


def test_vacuum_potential_converges_on_curved_mapping(tmp_path):
    """Coaxial capacitor: the Laplace solve converges at about order p + 1."""
    errors = [_coaxial_error(tmp_path, n, degree=2) for n in (4, 8)]
    rate = np.log2(errors[0] / errors[1])

    assert errors[1] < 1e-4
    assert rate > 2.5


def test_plane_source_and_current_ledger():
    from types import SimpleNamespace

    from struphy.pic.ion_beams import CurrentLedger, LossTag, PlaneSource

    source = PlaneSource(rate=50.0, current=2.0, axis=0, eta_plane=0.1, eta_ranges=((0.4, 0.6), (0.0, 1.0)))
    eta, v = source.sample(1000)
    assert source.weight == 0.04
    assert np.all(eta[:, 0] == 0.1) and np.all((eta[:, 1] >= 0.4) & (eta[:, 1] <= 0.6))
    np.testing.assert_allclose(v, [[1.0, 0.0, 0.0]] * 1000)

    # Records: (eta1, eta2, eta3, v1, v2, v3, weight, s0, w0, ID, axis, side).
    records = np.array(
        [
            [0.2, 1.1, 0.5, 0, 1, 0, 0.5, 1, 0.5, 7, 1, 1],  # top plate, upstream
            [0.9, -0.1, 0.5, 0, -1, 0, 0.5, 1, 0.5, 8, 1, 0],  # bottom plate, downstream
            [1.05, 0.5, 0.5, 1, 0, 0, 0.25, 1, 0.25, 9, 0, 1],  # outlet
        ]
    )
    index = {"pos": slice(0, 3), "vel": slice(3, 6), "weights": 6, "axis": 10, "side": 11}
    particles = SimpleNamespace(pop_lost_markers=lambda: records, lost_index=index)
    tags = (
        LossTag("upstream", axis=1, coordinate=0, interval=(0.0, 5.0)),
        LossTag("downstream", axis=1, coordinate=0, interval=(5.0, 10.0)),
    )
    ledger = CurrentLedger(tags, keep_records=("other",))
    ledger.update(particles, domains.Cuboid(r1=10.0))

    assert ledger.lost_charge == {"upstream": 0.5, "downstream": 0.5, "other": 0.25}
    np.testing.assert_allclose(ledger.records["other"], [[10.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.25, 0.0]])
    with pytest.raises(ValueError):
        CurrentLedger((LossTag("other", axis=0),))


def test_space_charge_requires_electrodes():
    with pytest.raises(ValueError):
        IonOpticsElectrostatic(space_charge=True)


def test_direct_poisson_matches_iterative_solve(tmp_path):
    """The serial sparse-LU path reproduces the CG solve, with and without charge."""
    from feectools.linalg.utilities import array_to_psydac

    from struphy.feec.banded_assembly import assemble_banded_operator

    models = []
    for direct in (True, False):
        model = IonOpticsElectrostatic(
            electrode_faces=((True, True), (False, False), (False, False)), poisson_direct=direct
        )
        model.em_fields.phi.add_perturbation(_Coaxial(0.5, 1.0))
        model.ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
        model.ions.set_markers(
            loading_params=LoadingParameters(Np=1, specific_markers=((0.5, 0.5, 0.5, 0.0, 0.0, 0.0),)),
            weights_params=WeightsParameters(control_variate=False),
            boundary_params=BoundaryParameters(bc=("remove", "periodic", "periodic")),
            saving_params=SavingParameters(n_markers=1),
        )
        Simulation(
            model=model,
            env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder=f"direct_{direct}"),
            time_opts=Time(dt=0.1, Tend=0.1),
            domain=domains.HollowCylinder(a1=0.5, a2=1.0, Lz=1.0),
            grid=grids.TensorProductGrid(num_elements=(6, 12, 1)),
            derham_opts=DerhamOptions(degree=(3, 2, 1), bcs=(("free", "free"), None, ("free", "free"))),
        ).allocate()
        models.append(model)

    direct, iterative = models
    assert "direct" in direct._poisson and "direct" not in iterative._poisson
    space = direct.em_fields.phi.spline.vector.space
    np.testing.assert_allclose(
        direct.em_fields.phi.spline.vector.toarray(), iterative.em_fields.phi.spline.vector.toarray(), atol=1e-9
    )

    values = np.random.default_rng(1).random(space.dimension)
    for model in models:
        model.solve_potential(array_to_psydac(values, model.em_fields.phi.spline.vector.space))
    np.testing.assert_allclose(
        direct.em_fields.phi.spline.vector.toarray(), iterative.em_fields.phi.spline.vector.toarray(), atol=1e-9
    )

    # The probed sparse stiffness equals the matrix-free operator (curved mapping, periodic direction).
    x = np.random.default_rng(2).random(space.dimension)
    stiffness = direct._poisson["stiffness"]
    np.testing.assert_allclose(
        assemble_banded_operator(stiffness, space) @ x, stiffness.dot(array_to_psydac(x, space)).toarray(), atol=1e-11
    )


def test_emittance_convergence_rule_and_steady_start():
    from struphy.diagnostics.beam_diagnostics import emittance_converged, steady_state_start

    decaying = 1.0 + 0.5 ** np.arange(30)
    assert not emittance_converged(decaying[:5])
    converged_at = next(k for k in range(1, 31) if emittance_converged(decaying[:k], n=5, tol=1e-3))
    assert 10 < converged_at < 20
    assert not emittance_converged(np.tile([1.0, 2.0], 10), n=5, tol=1e-3)  # oscillation

    # outlet records: emittance grows during a transient until t = 10, then stays constant
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 30.0, 30000))
    y = rng.standard_normal(len(t))
    spread = np.minimum(t, 10.0) / 10.0
    records = np.column_stack(
        [np.ones_like(t), y, 0 * t, np.ones_like(t), spread * rng.standard_normal(len(t)), 0 * t, np.ones_like(t), t]
    )
    start = steady_state_start(records, window=1.0, n=5, tol=0.05)
    assert start is not None and 10.0 <= start <= 20.0


def test_boltzmann_electrons_units():
    from struphy.physics.plasma_models import BoltzmannElectrons

    # protons, Te = 5 eV, n = 1e17 m^-3, units 1 mm / 5 V: lambda_D = sqrt(eps0 Te / (e n)) = 52.6 µm
    units = IonOpticsUnits(length=1e-3, voltage=5.0)
    plasma = BoltzmannElectrons.from_si(units, electron_density=1e17, electron_temperature_eV=5.0)
    assert plasma.temperature == pytest.approx(1.0)
    assert plasma.debye_length() * units.length == pytest.approx(5.2571e-5, rel=1e-3)
    assert plasma.bohm_speed() == pytest.approx(1.0)  # v_B = sqrt(kTe/m) = velocity unit when voltage = Te
    np.testing.assert_allclose(plasma.charge_density(np.array([0.0, -1.0])), -plasma.density * np.exp([0.0, -1.0]))
    with pytest.raises(ValueError):
        IonOpticsElectrostatic(
            electrode_faces=((True, True), (False, False), (False, False)), plasma=plasma, poisson_direct=False
        )


def test_axisymmetric_channel_and_per_side_particle_bc():
    from struphy.geometry.axisymmetric import AxisymmetricElectrodeChannel, meridional
    from struphy.geometry.domains import ElectrodeSegment

    domain = AxisymmetricElectrodeChannel(
        10.0,
        ((0.0, 5.0, 10.0), (3.0, 1.0, 3.0)),
        segments=(ElectrodeSegment("upper", 0.0, 4.0, 0.0),),
        axis_radius=0.01,
        tor_period=180,
        num_elements=(16, 8),
    )
    z, r = meridional(domain, np.array([[0.5, 1.0, 0.5], [0.5, 0.0, 0.2], [1.0, 0.5, 0.9]]))
    np.testing.assert_allclose(z, [5.0, 5.0, 10.0], atol=1e-3)
    np.testing.assert_allclose(r, [1.0, 0.01, 1.505], atol=2e-3)
    assert domain.wedge_fraction == pytest.approx(1 / 180)
    with pytest.raises(ValueError):
        AxisymmetricElectrodeChannel(
            10.0, ((0.0, 10.0), (3.0, 3.0)), segments=(ElectrodeSegment("lower", 0.0, 4.0, 0.0),)
        )

    BoundaryParameters(bc=("remove", ("reflect", "remove"), "reflect"))
    with pytest.raises(AssertionError):
        BoundaryParameters(bc=("remove", ("reflect", "periodic"), "reflect"))


def test_fast_poisson_boltzmann_matches_generic(tmp_path):
    """Reduced-space fast Newton (element-wise assembly, banded Cholesky) equals the generic operators."""
    from feectools.linalg.utilities import array_to_psydac

    from struphy.feec.h1_quadrature import H1QuadratureAssembler
    from struphy.physics.plasma_models import BoltzmannElectrons

    plasma = BoltzmannElectrons(density=1.0, temperature=1.0, plasma_potential=0.0)
    model = IonOpticsElectrostatic(electrode_faces=((False, False), (True, True), (False, False)), plasma=plasma)
    model.em_fields.phi.add_perturbation(PiecewiseLinearPotential((0.0, 4.0), (0.0, -3.0), coordinate=0))
    model.ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    model.ions.set_markers(
        loading_params=LoadingParameters(Np=1, specific_markers=((0.5, 0.5, 0.5, 0.0, 0.0, 0.0),)),
        weights_params=WeightsParameters(control_variate=False),
        boundary_params=BoundaryParameters(bc=("remove", "remove", "periodic")),
        saving_params=SavingParameters(n_markers=1),
    )
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder="fast_pb"),
        time_opts=Time(dt=0.1, Tend=0.1),
        domain=domains.Cuboid(l1=0.0, r1=4.0, l2=-1.0, r2=1.0),
        grid=grids.TensorProductGrid(num_elements=(24, 10, 1)),
        derham_opts=DerhamOptions(degree=(3, 3, 1), bcs=(("free", "free"),) * 3),
    )
    sim.allocate()
    space = model.em_fields.phi.spline.vector.space
    fast = model._poisson["fast"] if "fast" in model._poisson else model._setup_fast_poisson(space)
    assert fast is not None and fast["assembler"].reduced == [False, False, True]

    rng = np.random.default_rng(3)
    # a z-invariant load: the fast path solves the z-invariant problem (collapsed invariant direction)
    assembler = fast["assembler"]
    charge = array_to_psydac(assembler.expand(0.02 * rng.random(assembler.size)), space)
    start = model.em_fields.phi.spline.vector.toarray().copy()

    model._solve_poisson_boltzmann(charge)
    fast_solution = model.em_fields.phi.spline.vector.toarray().copy()

    model.em_fields.phi.spline.vector = array_to_psydac(start, space)
    model.em_fields.phi.spline.vector.update_ghost_regions()
    model._solve_poisson_boltzmann_generic(charge)
    generic_solution = model.em_fields.phi.spline.vector.toarray()
    np.testing.assert_allclose(fast_solution, generic_solution, atol=1e-8)
    # the assembler reproduces Struphy's own weighted mass matrix
    weights = np.exp(0.3 * rng.standard_normal(fast["geom_weights"].shape)) * fast["geom_weights"]
    full = H1QuadratureAssembler(Propagator.derham, space, reduce_invariant=False)
    operator = Propagator.mass_ops.create_weighted_mass(
        "H1", "H1", name="test_fast_pb", weights=[[weights]], assemble=True
    )
    reference = operator.tosparse()
    assert abs(full.matrix(weights) - reference).max() < 1e-13 * abs(reference).max()


def _iterate(fixed_point_map, rho0, relaxation, rounds):
    """Damped fixed-point iteration ``rho <- (1 - a) rho + a F(rho)``; returns the residual history."""
    rho, residuals = None, []
    for _ in range(rounds):
        new = fixed_point_map(rho0 if rho is None else rho)
        if rho is None:
            rho = new
            continue
        alpha = relaxation.update(new - rho, np.linalg.norm(new)) if hasattr(relaxation, "update") else relaxation
        residuals.append(np.linalg.norm(new - rho) / np.linalg.norm(new))
        rho = alpha * new + (1.0 - alpha) * rho
    return np.array(residuals)


def test_adaptive_relaxation_handles_stiff_alternating_and_slow_modes():
    from struphy.models.ion_optics_steady_state import AdaptiveRelaxation

    # F(rho) = A rho + b with one strongly alternating mode (-4) and one slow mode (0.9). Stability of damping
    # alone needs alpha < 2 / (1 + 4) = 0.4, so a constant alpha = 0.5 diverges and alpha = 0.2 crawls along the
    # slow mode; adaptive damping starts small, grows while the residuals agree, and beats both.
    rng = np.random.default_rng(0)
    basis, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    A = basis @ np.diag([-4.0, 0.9, 0.5, 0.2, -0.3, 0.1]) @ basis.T
    b = rng.standard_normal(6)

    def fixed_point_map(rho):
        return A @ rho + b

    rho0 = np.zeros(6)
    constant_half = _iterate(fixed_point_map, rho0, 0.5, 80)
    constant_small = _iterate(fixed_point_map, rho0, 0.2, 80)
    adaptive = _iterate(fixed_point_map, rho0, AdaptiveRelaxation(alpha=0.2), 80)
    assert constant_half[-1] > 1.0  # limit cycle or divergence
    assert adaptive[-1] < 1e-2 and adaptive[-1] < 0.2 * constant_small[-1]


def test_adaptive_relaxation_averages_noise_by_lowering_the_cap():
    from struphy.models.ion_optics_steady_state import AdaptiveRelaxation

    # a noisy map with a fixed point: the residual stalls at a noise floor, so the cap on alpha shrinks
    rng = np.random.default_rng(1)
    relaxation = AdaptiveRelaxation(alpha=0.5, patience=6)
    rho, target = np.ones(8), np.ones(8)
    for _ in range(80):
        new = 0.5 * rho + 0.5 * target + 0.5 * rng.standard_normal(8)
        alpha = relaxation.update(new - rho, np.linalg.norm(new))
        rho = alpha * new + (1.0 - alpha) * rho
    assert relaxation.cap < 0.2 and relaxation.alpha <= relaxation.cap
    assert AdaptiveRelaxation().alpha == 0.2 and AdaptiveRelaxation(alpha_min=0.1).alpha_min == 0.1


def test_steady_state_averaged_results():
    from struphy.models.ion_optics_steady_state import IterationRecord, SteadyStateIteration

    iteration = object.__new__(SteadyStateIteration)
    iteration.history = [
        IterationRecord(
            round=k,
            exit_emittance=e,
            exit_size=1.0,
            exit_current=c,
            lost_current={"outlet": c, "wall": 1.0 - c},
            steps=10,
            exit_records=np.empty((0, 8)),
        )
        for k, (c, e) in enumerate([(0.9, 1.0), (0.2, 5.0), (0.4, np.nan), (0.6, 3.0)])
    ]
    result = iteration.averaged(rounds=3)  # the last three rounds only
    assert result["exit_current"] == pytest.approx((0.4, np.std([0.2, 0.4, 0.6])))
    assert result["lost_current[wall]"][0] == pytest.approx(0.6)
    assert result["exit_emittance"] == pytest.approx((4.0, 1.0))  # NaN rounds are ignored


def test_ray_kernels_reproduce_the_original_pushers_and_the_step_rule(tmp_path):
    """Per-marker steps: scale 1 and 2 equal the original kernels (dt and 2 dt), and the step rule follows its formula."""
    from struphy.pic.ray_tracing import AdaptiveRayPusher, CellSizeMap, StepControl

    # planar accelerator: phi = 0.4 (1 - x) on the unit cube, so E = 0.4 exactly and epsilon = 1
    model = IonOpticsElectrostatic(epsilon=1.0, electrode_faces=((True, True), (False, False), (False, False)))
    model.em_fields.phi.add_perturbation(PiecewiseLinearPotential((0.0, 1.0), (0.4, 0.0), coordinate=0))
    model.ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    model.ions.set_markers(
        loading_params=LoadingParameters(Np=8),
        weights_params=WeightsParameters(control_variate=False),
        boundary_params=BoundaryParameters(bc=("remove", "periodic", "periodic")),
        saving_params=SavingParameters(n_markers=1),
    )
    Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder="rays"),
        time_opts=Time(dt=0.1, Tend=0.1),
        domain=domains.Cuboid(),
        grid=grids.TensorProductGrid(num_elements=(16, 1, 1)),
        derham_opts=DerhamOptions(degree=(3, 1, 1), bcs=(("free", "free"),) * 3),
    ).allocate()
    derham, domain, particles = Propagator.derham, Propagator.domain, model.ions.var.particles
    for propagator in model.prop_list:
        propagator.options = propagator.Options()

    n = 6
    markers = particles.markers
    markers[:, :] = -1.0
    markers[:n, :] = 0.0
    markers[:n, 0] = np.linspace(0.1, 0.6, n)
    markers[:n, 1:3] = 0.5
    markers[:n, 3] = np.linspace(0.05, 0.3, n)
    markers[:n, 4] = 0.02
    markers[:n, 6] = 1.0
    markers[:n, -1] = np.arange(n)
    particles.update_holes()
    start = markers.copy()

    cells = CellSizeMap(domain, derham.grid.num_elements, derham.degree)
    np.testing.assert_allclose(cells.lookup(start[:n, :3]), 1.0 / 16.0)  # y and z carry no structure
    pusher = AdaptiveRayPusher(
        particles,
        domain,
        derham,
        model.em_fields.e_field.spline.vector,
        1.0,
        model.propagators.push_eta.options.butcher,
        0.05,
        StepControl(),
        cells,
    )

    def original(dt):
        markers[:] = start
        particles.update_holes()
        model.propagators.push_v(0.5 * dt)
        model.propagators.push_eta(dt)
        model.propagators.push_v(0.5 * dt)
        return markers[:n, :6].copy()

    def scaled(scale):
        markers[:] = start
        particles.update_holes()
        pusher.scale[:] = scale
        pusher.step()
        return markers[:n, :6].copy()

    np.testing.assert_allclose(scaled(1.0), original(0.05), atol=1e-14, rtol=0)
    np.testing.assert_allclose(scaled(2.0), original(0.10), atol=1e-14, rtol=0)
    mixed = np.where(np.arange(markers.shape[0]) % 2 == 0, 1.0, 3.0)
    expected = np.where((np.arange(n) % 2 == 0)[:, None], original(0.05), original(0.15))
    np.testing.assert_allclose(scaled(mixed), expected, atol=1e-14, rtol=0)

    # step rule: E = 0.4 everywhere, so a = 0.4; dt = min(dt_max, C_x h / v, C_v v / a, C_a sqrt(h / a))
    markers[:] = start
    particles.update_holes()
    control = pusher.control
    speed = np.linalg.norm(start[:n, 3:6], axis=1)
    h, a = 1.0 / 16.0, 0.4
    expected_dt = np.minimum.reduce(
        [
            np.full(n, 0.05),
            control.courant * h / speed,
            control.velocity_change * speed / a,
            np.full(n, control.acceleration * np.sqrt(h / a)),
        ]
    )
    np.testing.assert_allclose(pusher.choose()[:n], expected_dt, rtol=1e-6)
    np.testing.assert_allclose(pusher._acceleration[:n], a, rtol=1e-8)
    with pytest.raises(ValueError):
        StepControl(courant=0.0)
    with pytest.raises(ValueError):
        StepControl(min_scale=2.0)
