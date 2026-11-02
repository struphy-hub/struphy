import cunumpy as xp
import pytest
from feectools.linalg.solvers import inverse
from maybempi import MPI

from struphy import EnvironmentOptions, Simulation, Time, equils, perturbations
from struphy.feec import preconditioner
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import LinearMHD, LinearMHDWithQ
from struphy.propagators.background_current import BackgroundCurrent
from struphy.propagators.base import Propagator
from struphy.propagators.sound_wave_linear import SoundWaveLinear
from struphy.propagators.sound_wave_linear_with_q import SoundWaveLinearWithQ
from struphy.topology import grids

TOL = 1e-13
GAMMA = 5 / 3


def allocate_model(tmp_path, name, equil, model_class=LinearMHDWithQ, b_space="Hdiv", bcs=None, noise=True, amp_p=1.0):
    """A LinearMHDWithQ (or LinearMHD) model with a velocity and q (or pressure, amplitude times ``amp_p``)
    perturbation (noise, or smooth modes), allocated in the unit cube, with tight solver tolerances."""
    model = model_class(b_space=b_space)
    thermo = model.mhd.sqrt_p if model_class is LinearMHDWithQ else model.mhd.pressure
    if noise:
        for comp in range(3):
            model.mhd.velocity.add_perturbation(
                perturbations.Noise(direction="e1e2e3", amp=1e-2, seed=1234 + comp, comp=comp, given_in_basis="2")
            )
        thermo.add_perturbation(
            perturbations.Noise(direction="e1e2e3", amp=1e-2 * amp_p, seed=4321, comp=None, given_in_basis="3")
        )
    else:
        model.mhd.velocity.add_perturbation(
            perturbations.ModesCos(ls=(1,), ms=(1,), given_in_basis="2", comp=0, amps=(1e-2,))
        )
        model.mhd.velocity.add_perturbation(
            perturbations.ModesCos(ls=(1,), ns=(1,), given_in_basis="2", comp=2, amps=(1e-2,))
        )
        thermo.add_perturbation(perturbations.ModesCos(ms=(1,), ns=(1,), given_in_basis="3", amps=(1e-2 * amp_p,)))

    props = model.propagators
    if model_class is LinearMHDWithQ:
        props.sound.options = SoundWaveLinearWithQ.Options(solver_params=SolverParameters(tol=TOL))
    else:
        props.sound.options = SoundWaveLinear.Options(solver_params=SolverParameters(tol=TOL))
    props.bg_current.options = BackgroundCurrent.Options(solver_params=SolverParameters(tol=TOL))
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
    """Energy 1/2 u^T M2n u + 2/gamma q^T M3 q conserved by the sound wave step."""
    mass_ops = Propagator.mass_ops
    u = model.mhd.velocity.spline.vector
    q = model.mhd.sqrt_p.spline.vector
    return u.inner(mass_ops.M2n.dot(u)) / 2 + 2 / GAMMA * q.inner(mass_ops.M3.dot(q))


def pressure_of_q(q):
    """Coefficients of the pressure perturbation 2 q0 q (L2 projection into V3) of the coefficients q."""
    mass_ops = Propagator.mass_ops
    M3_inv = inverse(mass_ops.M3, "pcg", pc=preconditioner.MassMatrixPreconditioner(mass_ops.M3), tol=1e-14)
    p = M3_inv.dot(mass_ops.M3q0.dot(q))
    p *= 2.0
    return p


@pytest.mark.mpi_skip
def test_options():
    options = SoundWaveLinearWithQ.Options()
    assert options.solver_params is not None and options.solver_params_M3.tol == 1e-14
    with pytest.raises(AssertionError):
        SoundWaveLinearWithQ.Options(solver="pbicgstab_wrong")
    with pytest.raises(AssertionError):
        SoundWaveLinearWithQ.Options(precond="NoSuchPreconditioner")
    with pytest.raises(AssertionError):
        SoundWaveLinearWithQ().variables.u = LinearMHDWithQ().mhd.sqrt_p
    with pytest.raises(AssertionError):
        SoundWaveLinearWithQ().variables.q = LinearMHDWithQ().mhd.velocity


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


def test_agrees_with_linear_mhd(tmp_path):
    """In a homogeneous slab (constant q0) the sound wave steps of LinearMHDWithQ and of LinearMHD solve the same
    discrete system for p = 2 q0 q (M3q0 = q0 M3, M3p_inv = M3 / p0), so they agree to the solver tolerances, and the
    background q step does nothing."""
    equil = equils.HomogenSlab(B0x=0.3, B0z=1.0, beta=2.0, n0=0.7)
    q0 = xp.sqrt(equil.params["beta"] * (0.3**2 + 1.0**2) / 2)
    dt = 0.3

    model = allocate_model(tmp_path, "q", equil, noise=False)
    q_before = model.mhd.sqrt_p.spline.vector.toarray().copy()
    model.propagators.bg_q(dt)
    assert xp.max(xp.abs(model.mhd.sqrt_p.spline.vector.toarray() - q_before)) < 1e-12 * xp.max(xp.abs(q_before))
    model.propagators.sound(dt)
    with_q = [var.spline.vector.toarray().copy() for var in (model.mhd.density, model.mhd.velocity, model.mhd.sqrt_p)]
    with_q[2] *= 2 * q0

    model = allocate_model(tmp_path, "p", equil, model_class=LinearMHD, noise=False, amp_p=2 * q0)
    model.propagators.sound(dt)
    with_p = [var.spline.vector.toarray() for var in (model.mhd.density, model.mhd.velocity, model.mhd.pressure)]

    for name, a, b in zip(("n", "u", "p"), with_q, with_p):
        diff = xp.max(xp.abs(a - b)) / xp.max(xp.abs(b))
        print(f"{name}: relative difference {diff:.2e}")
        assert xp.max(xp.abs(b)) > 0.0
        assert diff < 1e-9


def test_background_q_agrees_with_background_pressure(tmp_path):
    """In a sheared slab (grad p0 != 0) the change 2 q0 dq of the pressure by the background q step equals the change
    dp of the background pressure step of LinearMHD (u . grad p0 = 2 q0 u . grad q0), up to the projection errors."""
    equil = equils.ShearedSlab(beta=1.0, n1=2.0, n2=1.0, na=0.5, R0=1.0)
    bcs = (("free", "free"), None, None)
    dt = 0.1

    model = allocate_model(tmp_path, "bg_q", equil, bcs=bcs, noise=False)
    q = model.mhd.sqrt_p.spline.vector
    q_before = q.copy()
    model.propagators.bg_q(dt)
    dq = q - q_before
    dp_q = pressure_of_q(dq).toarray()

    model = allocate_model(tmp_path, "bg_p", equil, model_class=LinearMHD, bcs=bcs, noise=False)
    p = model.mhd.pressure.spline.vector
    p_before = p.toarray().copy()
    model.propagators.bg_pressure(dt)
    dp = p.toarray() - p_before

    diff = xp.max(xp.abs(dp_q - dp)) / xp.max(xp.abs(dp))
    print(f"relative difference {diff:.2e}")
    assert xp.max(xp.abs(dp)) > 0.0
    assert diff < 1e-2


def _group(model):
    """The symmetric (Strang) block of the compressional steps: bg_current(dt/2) bg_q(dt/2) sound(dt) bg_q(dt/2)
    bg_current(dt/2)."""
    props = model.propagators
    bg = props.bg_q if hasattr(props, "bg_q") else props.bg_pressure

    def half(prop):
        return lambda dt: prop(dt / 2)

    return (half(props.bg_current), half(bg), props.sound, half(bg), half(props.bg_current))


def _set_b_from_u(model):
    """Magnetic field perturbation (2-form) equal to the velocity perturbation."""
    b = model.mhd.velocity.spline.vector.copy(out=model.em_fields.b_field.spline.vector)
    b.update_ghost_regions()


def test_split_agrees_with_linear_mhd_inhomogeneous(tmp_path):
    """In a sheared slab (grad p0 != 0, J0 != 0, rho0 non-uniform), one small step of the compressional block changes
    u like the compressional block of LinearMHD, for the pressure perturbation p = 2 q0 q of the same initial state,
    up to the projection errors and O(dt^3)."""
    equil = equils.ShearedSlab(beta=1.0, n1=2.0, n2=1.0, na=0.5, R0=1.0)
    bcs = (("free", "free"), None, None)
    dt = 1e-3

    model_q = allocate_model(tmp_path, "split_q", equil, bcs=bcs, noise=False)
    _set_b_from_u(model_q)
    u_q = model_q.mhd.velocity.spline.vector
    p_init = pressure_of_q(model_q.mhd.sqrt_p.spline.vector)
    u0 = u_q.toarray().copy()
    for prop in _group(model_q):
        prop(dt)
    du_q = u_q.toarray() - u0

    model_p = allocate_model(tmp_path, "split_p", equil, model_class=LinearMHD, bcs=bcs, noise=False)
    _set_b_from_u(model_p)
    p_init.copy(out=model_p.mhd.pressure.spline.vector)
    model_p.mhd.pressure.spline.vector.update_ghost_regions()
    u_p = model_p.mhd.velocity.spline.vector
    for prop in _group(model_p):
        prop(dt)
    du_p = u_p.toarray() - u0

    diff = xp.max(xp.abs(du_q - du_p)) / xp.max(xp.abs(du_p))
    print(f"relative difference of the velocity changes {diff:.2e}")
    assert diff < 1e-4


def test_group_second_order_inhomogeneous(tmp_path):
    """Self-convergence in dt of the symmetric block of the compressional steps of LinearMHDWithQ (strang_groups) in a
    sheared slab (grad p0 != 0, J0 != 0, the steps do not commute): second order in time."""
    equil = equils.ShearedSlab(beta=1.0, n1=2.0, n2=1.0, na=0.5, R0=1.0)
    model = allocate_model(tmp_path, "order", equil, bcs=(("free", "free"), None, None), noise=False)
    _set_b_from_u(model)
    assert [name for name, _ in model.propagator_calls(1.0)][1:] == [
        "bg_current",
        "bg_q",
        "sound",
        "bg_q",
        "bg_current",
    ]
    u, q = model.mhd.velocity.spline.vector, model.mhd.sqrt_p.spline.vector
    u0, q0 = u.copy(), q.copy()
    T = 0.01

    def final(steps):
        u0.copy(out=u)
        q0.copy(out=q)
        u.update_ghost_regions()
        q.update_ghost_regions()
        for _ in range(steps):
            for prop in _group(model):
                prop(T / steps)
        return xp.concatenate((u.toarray(), q.toarray()))

    finals = [final(steps) for steps in (2, 4, 8, 16)]
    comm = MPI.COMM_WORLD
    diffs = [float(xp.max(xp.abs(a - b))) for a, b in zip(finals[:-1], finals[1:])]
    if comm.Get_size() > 1:
        diffs = [comm.allreduce(d, op=MPI.MAX) for d in diffs]
    rates = [float(xp.log2(d0 / d1)) for d0, d1 in zip(diffs[:-1], diffs[1:])]
    print(f"{diffs = }, {rates = }")
    assert all(abs(rate - 2) < 0.15 for rate in rates)


@pytest.mark.parametrize("b_space", ["Hdiv", "Hcurl"])
def test_split_model_energy_homogeneous(tmp_path, b_space):
    """LinearMHDWithQ conserves the total perturbed energy in a homogeneous slab (the background steps vanish, the
    shear-Alfven and sound wave steps are energy conserving; with b_space="Hcurl" the default unprojected shear-Alfven
    step damps, so only the energy change of the other steps is checked there)."""
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
