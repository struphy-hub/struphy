import cunumpy as xp
import pytest

from struphy import EnvironmentOptions, Simulation, Time, equils, perturbations
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import LinearMHD
from struphy.propagators.background_current import BackgroundCurrent
from struphy.propagators.background_pressure_gradient import BackgroundPressureGradient
from struphy.propagators.base import Propagator
from struphy.propagators.magnetosonic import Magnetosonic
from struphy.propagators.sound_wave import SoundWave
from struphy.topology import grids

TOL = 1e-13


def allocate_model(tmp_path, name, equil, split_sound=True, b_space="Hdiv", bcs=None, noise=True):
    """A LinearMHD model with a velocity and pressure perturbation (noise, or smooth modes), allocated in the unit cube,
    with tight solver tolerances."""
    model = LinearMHD(b_space=b_space, split_sound=split_sound)
    if noise:
        for comp in range(3):
            model.mhd.velocity.add_perturbation(
                perturbations.Noise(direction="e1e2e3", amp=1e-2, seed=1234 + comp, comp=comp, given_in_basis="2")
            )
        model.mhd.pressure.add_perturbation(
            perturbations.Noise(direction="e1e2e3", amp=1e-2, seed=4321, comp=None, given_in_basis="3")
        )
    else:
        model.mhd.velocity.add_perturbation(
            perturbations.ModesCos(ls=(1,), ms=(1,), given_in_basis="2", comp=0, amps=(1e-2,))
        )
        model.mhd.velocity.add_perturbation(
            perturbations.ModesCos(ls=(1,), ns=(1,), given_in_basis="2", comp=2, amps=(1e-2,))
        )
        model.mhd.pressure.add_perturbation(perturbations.ModesCos(ms=(1,), ns=(1,), given_in_basis="3", amps=(1e-2,)))

    props = model.propagators
    if split_sound:
        props.sound.options = SoundWave.Options(solver_params=SolverParameters(tol=TOL))
        props.bg_current.options = BackgroundCurrent.Options(solver_params=SolverParameters(tol=TOL))
    else:
        props.mag_sonic.options = Magnetosonic.Options(solver_params=SolverParameters(tol=TOL))
    props.shear_alf.options = props.shear_alf.Options(solver_params=SolverParameters(tol=TOL))

    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder=name),
        time_opts=Time(dt=0.1, Tend=0.1),
        equil=equil,
        grid=grids.TensorProductGrid(num_elements=(8, 6, 4)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)) if bcs is None else DerhamOptions(degree=(2, 2, 1), bcs=bcs),
    )
    sim.allocate()
    return model


def sound_energy(model):
    """Energy 1/2 u^T M2n u + 1/(2 gamma) p^T M3p_inv p conserved by the sound wave step."""
    mass_ops = Propagator.mass_ops
    u = model.mhd.velocity.spline.vector
    p = model.mhd.pressure.spline.vector
    return u.inner(mass_ops.M2n.dot(u)) / 2 + p.inner(mass_ops.M3p_inv.dot(p)) / (2 * 5 / 3)


@pytest.mark.mpi_skip
def test_options():
    options = SoundWave.Options()
    assert options.solver_params is not None and options.solver_params_M3.tol == 1e-14
    with pytest.raises(AssertionError):
        SoundWave.Options(solver="pbicgstab_wrong")
    with pytest.raises(AssertionError):
        SoundWave.Options(precond="NoSuchPreconditioner")
    with pytest.raises(AssertionError):
        SoundWave().variables.u = LinearMHD().mhd.pressure
    model = LinearMHD(split_sound=True)
    assert not hasattr(model.propagators, "mag_sonic")
    assert list(vars(model.propagators)) == ["shear_alf", "bg_current", "sound", "bg_pressure"]
    assert model.propagators.bg_pressure.u_field is model.mhd.velocity
    assert model.propagators.bg_current.b_field is model.em_fields.b_field


@pytest.mark.parametrize("bcs", [None, (("free", "free"), None, None), (("dirichlet", "dirichlet"), None, None)])
def test_energy_conservation(tmp_path, bcs):
    """The sound wave step conserves its energy (to the solver tolerance) in an inhomogeneous equilibrium (p0 and rho0
    depend on x), while the kinetic and thermal energies change by much more."""
    equil = equils.ShearedSlab(beta=1.0, n1=2.0, n2=1.0, na=0.5)
    model = allocate_model(tmp_path, "energy", equil, bcs=bcs)
    sound = model.propagators.sound

    mass_ops = Propagator.mass_ops
    u = model.mhd.velocity.spline.vector
    energies = [sound_energy(model)]
    kinetic = [u.inner(mass_ops.M2n.dot(u)) / 2]
    for _ in range(10):
        sound(0.2)
        energies.append(sound_energy(model))
        kinetic.append(u.inner(mass_ops.M2n.dot(u)) / 2)
    energies = xp.array(energies)
    kinetic = xp.array(kinetic)

    rel_change = xp.max(xp.abs(energies - energies[0])) / energies[0]
    exchange = xp.max(xp.abs(kinetic - kinetic[0])) / energies[0]
    print(f"{bcs = }: relative energy change {rel_change:.2e}, relative energy exchange {exchange:.2e}")
    assert exchange > 1e-2
    assert rel_change < 1e-10


def test_agrees_with_magnetosonic(tmp_path):
    """In a homogeneous slab (grad p0 = 0, J0 = 0) the sound wave step and the magnetosonic step solve the same
    discrete system (M3p_inv = M3 / p0 and the basis projections of p0 are exact), so they agree to the solver
    tolerances, and the background steps do nothing."""
    equil = equils.HomogenSlab(B0x=0.3, B0z=1.0, beta=2.0, n0=0.7)
    dt = 0.3

    model = allocate_model(tmp_path, "split", equil, split_sound=True, noise=False)
    # a non-zero magnetic field perturbation, on which J0 = 0 exerts no force
    b = model.mhd.velocity.spline.vector.copy(out=model.em_fields.b_field.spline.vector)
    b.update_ghost_regions()
    u0 = model.mhd.velocity.spline.vector.toarray().copy()
    p0 = model.mhd.pressure.spline.vector.toarray().copy()
    model.propagators.bg_current(dt)
    model.propagators.bg_pressure(dt)
    assert xp.max(xp.abs(model.mhd.velocity.spline.vector.toarray() - u0)) < 1e-12 * xp.max(xp.abs(u0))
    assert xp.max(xp.abs(model.mhd.pressure.spline.vector.toarray() - p0)) < 1e-12 * xp.max(xp.abs(p0))
    model.propagators.sound(dt)
    split = [var.spline.vector.toarray().copy() for var in (model.mhd.density, model.mhd.velocity, model.mhd.pressure)]

    model = allocate_model(tmp_path, "combined", equil, split_sound=False, noise=False)
    model.propagators.mag_sonic(dt)
    combined = [var.spline.vector.toarray() for var in (model.mhd.density, model.mhd.velocity, model.mhd.pressure)]

    for name, a, b in zip(("n", "u", "p"), split, combined):
        diff = xp.max(xp.abs(a - b)) / xp.max(xp.abs(b))
        print(f"{name}: relative difference {diff:.2e}")
        assert xp.max(xp.abs(b)) > 0.0
        assert diff < 1e-9


def _changes(model, props, dt):
    """Changes of u and p (as arrays) by the given propagators applied in sequence, and the change by each one."""
    u = model.mhd.velocity.spline.vector
    p = model.mhd.pressure.spline.vector
    u0, p0 = u.toarray().copy(), p.toarray().copy()
    single = []
    for prop in props:
        u_before, p_before = u.toarray().copy(), p.toarray().copy()
        prop(dt)
        single.append((u.toarray() - u_before, p.toarray() - p_before))
    return u.toarray() - u0, p.toarray() - p0, single


def _set_b_from_u(model):
    """Magnetic field perturbation (2-form) equal to the velocity perturbation."""
    b = model.mhd.velocity.spline.vector.copy(out=model.em_fields.b_field.spline.vector)
    b.update_ghost_regions()


def test_split_agrees_with_magnetosonic_inhomogeneous(tmp_path):
    """In a sheared slab (grad p0 != 0, J0 != 0, rho0 non-uniform), one small step of the split propagators
    (background current, sound wave, background pressure gradient) changes u and p like the magnetosonic step, up to
    the difference of the spatial discretizations (weak vs. projected pressure equation) and O(dt^2). The background
    steps contribute much more than this difference, so a wrong sign or operator in them fails the test."""
    equil = equils.ShearedSlab(beta=1.0, n1=2.0, n2=1.0, na=0.5, R0=1.0)
    bcs = (("free", "free"), None, None)
    dt = 1e-3

    model = allocate_model(tmp_path, "split_inhom", equil, split_sound=True, bcs=bcs, noise=False)
    _set_b_from_u(model)
    props = model.propagators
    du_s, dp_s, single = _changes(model, (props.bg_current, props.sound, props.bg_pressure), dt)
    du_current = single[0][0]
    dp_gradient = single[2][1]

    model = allocate_model(tmp_path, "combined_inhom", equil, split_sound=False, bcs=bcs, noise=False)
    _set_b_from_u(model)
    du_c, dp_c, _ = _changes(model, (model.propagators.mag_sonic,), dt)

    def rel(a, b):
        return xp.max(xp.abs(a)) / xp.max(xp.abs(b))

    diff_u, diff_p = rel(du_s - du_c, du_c), rel(dp_s - dp_c, dp_c)
    contrib_u, contrib_p = rel(du_current, du_c), rel(dp_gradient, dp_c)
    print(f"{diff_u = :.2e}, {diff_p = :.2e}, {contrib_u = :.2e}, {contrib_p = :.2e}")
    assert diff_u < 0.05 and diff_p < 0.05
    assert contrib_u > 4 * diff_u and contrib_p > 4 * diff_p


@pytest.mark.parametrize("b_space", ["Hdiv", "Hcurl"])
def test_split_model_energy_homogeneous(tmp_path, b_space):
    """LinearMHD(split_sound=True) conserves the total perturbed energy in a homogeneous slab (the background steps
    vanish, the shear-Alfven and sound wave steps are energy conserving; with b_space="Hcurl" the default unprojected
    shear-Alfven step damps, so only the energy change of the other steps is checked there)."""
    equil = equils.HomogenSlab(B0x=0.3, B0z=1.0, beta=2.0, n0=0.7)
    model = allocate_model(tmp_path, f"split_{b_space}", equil, b_space=b_space)
    model.em_fields.b_field.spline.vector *= 0.0
    mass_ops = Propagator.mass_ops
    M_b = mass_ops.M1 if b_space == "Hcurl" else mass_ops.M2

    def total_energy():
        b = model.em_fields.b_field.spline.vector
        return sound_energy(model) + b.inner(M_b.dot(b)) / 2

    e0 = total_energy()
    changes = []
    for _ in range(5):
        for name, prop in vars(model.propagators).items():
            e_before = total_energy()
            prop(0.2)
            if b_space == "Hdiv" or name != "shear_alf":
                changes.append(abs(total_energy() - e_before) / e0)
    print(f"{b_space = }: max relative energy change per step {max(changes):.2e}")
    assert max(changes) < 1e-10
