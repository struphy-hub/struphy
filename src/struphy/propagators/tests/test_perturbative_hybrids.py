"""Tests of the coupling steps of the hybrid perturbative models (PerturbativeMHDWithQVlasovCC/PC,
PerturbativeMHDWithQDriftkineticCC) with the inertia of the total density (``rho_field`` of CurrentCoupling6DDensity,
CurrentCoupling6DCurrent, PressureCoupling6D, ShearAlfvenCurrentCoupling5D and CurrentCoupling5DDensity/GradB/Curlb)."""

import cunumpy as xp
import pytest
from maybempi import MPI

from struphy import (
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    maxwellians,
    perturbations,
)
from struphy.linear_algebra.solver import DiscreteGradientSolverParameters, SolverParameters
from struphy.models import (
    LinearMHDDriftkineticCC,
    LinearMHDVlasovCC,
    LinearMHDVlasovPC,
    LinearMHDWithQ,
    PerturbativeMHDWithQDriftkineticCC,
    PerturbativeMHDWithQVlasovCC,
    PerturbativeMHDWithQVlasovPC,
)
from struphy.propagators.base import Propagator
from struphy.propagators.current_coupling_5d_curlb import CurrentCoupling5DCurlb
from struphy.propagators.current_coupling_5d_density import CurrentCoupling5DDensity
from struphy.propagators.current_coupling_5d_gradb import CurrentCoupling5DGradB
from struphy.propagators.current_coupling_6d_current import CurrentCoupling6DCurrent
from struphy.propagators.current_coupling_6d_density import CurrentCoupling6DDensity
from struphy.propagators.pressure_coupling_6d import PressureCoupling6D
from struphy.propagators.shear_alfven_current_coupling_5d import ShearAlfvenCurrentCoupling5D
from struphy.propagators.utils import VelocityMass

TOL = SolverParameters(tol=1e-14, maxiter=5000)


@pytest.mark.mpi_skip
def test_rho_field_wiring():
    """The hybrid perturbative models pass the density perturbation to their coupling steps, the linear hybrids don't;
    rho_field must be an L2 variable."""
    model = PerturbativeMHDWithQVlasovCC()
    assert model.propagators.couple_dens.rho_field is model.mhd.density
    assert model.propagators.couple_curr.rho_field is model.mhd.density
    model = PerturbativeMHDWithQVlasovPC()
    assert model.propagators.pc6d.rho_field is model.mhd.density
    assert LinearMHDVlasovCC().propagators.couple_dens.rho_field is None
    assert LinearMHDVlasovCC().propagators.couple_curr.rho_field is None
    assert LinearMHDVlasovPC().propagators.pc6d.rho_field is None
    model = PerturbativeMHDWithQDriftkineticCC()
    for step in ("shear_alf", "cc5d_density", "cc5d_gradb", "cc5d_curlb"):
        assert getattr(model.propagators, step).rho_field is model.mhd.density
        assert getattr(LinearMHDDriftkineticCC().propagators, step).rho_field is None
    # the shear-Alfven step with the total field replaces nonlin_alf
    assert model.propagators.shear_alf.nonlinear is True
    assert not hasattr(model.propagators, "nonlin_alf")

    u = LinearMHDWithQ().mhd.velocity
    for make in (
        lambda: CurrentCoupling6DDensity(rho_field=u),
        lambda: CurrentCoupling6DCurrent(rho_field=u),
        lambda: PressureCoupling6D(rho_field=u),
        lambda: ShearAlfvenCurrentCoupling5D(rho_field=u),
        lambda: CurrentCoupling5DDensity(rho_field=u),
        lambda: CurrentCoupling5DGradB(rho_field=u),
        lambda: CurrentCoupling5DCurlb(rho_field=u),
    ):
        with pytest.raises(AssertionError):
            make()


def _allocate(tmp_path, model, name):
    """Finite-amplitude perturbations of density, velocity and magnetic field in the Colella domain, energetic ions
    with a shifted Maxwellian, tight solver tolerances of the coupling steps."""
    model.energetic_ions.set_markers(loading_params=LoadingParameters(Np=2000, seed=1234))
    model.energetic_ions.var.add_background(maxwellians.Maxwellian3D(u1=(0.3, None), vth1=(0.5, None)))
    m = model.mhd
    for c in range(3):
        m.velocity.add_perturbation(
            perturbations.ModesSin(ls=(1,), ms=(c % 2,), amps=(0.1,), comp=c, given_in_basis="physical")
        )
    m.density.add_perturbation(perturbations.ModesCos(ls=(2,), amps=(0.3,), given_in_basis="physical"))
    b = model.em_fields.b_field
    b.add_perturbation(perturbations.ModesCos(ms=(1,), amps=(0.2,), comp=0, given_in_basis="physical"))
    props = model.propagators
    for step in ("couple_dens", "couple_curr", "pc6d"):
        if hasattr(props, step):
            prop = getattr(props, step)
            prop.options = prop.Options(solver_params=TOL)
    Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder=name),
        time_opts=Time(dt=0.05, Tend=0.05),
        domain=domains.Colella(Lx=1.0, Ly=1.0, alpha=0.1, Lz=1.0),
        equil=equils.HomogenSlab(B0x=0.2, B0z=1.0, beta=1.0, n0=1.0),
        grid=grids.TensorProductGrid(num_elements=(6, 6, 1)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    ).allocate()


def _u_energy(model, rho_field):
    """1/2 u^T M u, with the mass matrix of the total density (rho_field) or of rho0 (None)."""
    mass = VelocityMass(
        Propagator.derham, Propagator.domain, Propagator.mass_ops, Propagator.projected_equil, rho_field
    )
    u = model.mhd.velocity.spline.vector
    return u.inner(mass.M_rho.dot(u)) / 2


def _ion_energy(model):
    """Kinetic energy of the energetic ions, A_h/A_b * 1/2 sum_p w_p |v_p|^2 (all ranks)."""
    particles = model.energetic_ions.var.particles
    Ah = model.energetic_ions.var.species.mass_number
    Ab = model.mhd.mass_number
    energy = float(Ah / Ab * 0.5 * xp.sum(particles.weights * xp.sum(particles.velocities**2, axis=0)))
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        energy = comm.allreduce(energy, op=MPI.SUM)
    return energy


@pytest.mark.parametrize(
    "model_class, step, with_ions",
    [
        (PerturbativeMHDWithQVlasovCC, "couple_dens", False),
        (PerturbativeMHDWithQVlasovCC, "couple_curr", True),
        (PerturbativeMHDWithQVlasovPC, "pc6d", True),
    ],
)
def test_energy_conservation_total_density(tmp_path, model_class, step, with_ions):
    """The coupling steps with the inertia of the total density conserve 1/2 u^T M2_rho u (plus the energy of the
    energetic ions if exchanged) exactly, while the energy of the inertia rho0 alone changes, and the density does
    not change."""
    model = model_class()
    _allocate(tmp_path, model, f"{model_class.__name__}_{step}")
    n0 = model.mhd.density.spline.vector.toarray().copy()
    rho = model.mhd.density

    def energy():
        return _u_energy(model, rho) + (_ion_energy(model) if with_ions else 0.0)

    e0, k0, k0_rho0 = energy(), _u_energy(model, rho), _u_energy(model, None)
    for _ in range(3):
        getattr(model.propagators, step)(0.1)
    change = abs(energy() - e0) / e0
    exchange = abs(_u_energy(model, rho) - k0) / k0
    change_rho0 = abs(_u_energy(model, None) - k0_rho0) / k0_rho0
    print(f"{step}: relative energy change {change:.2e}, u-energy change {exchange:.2e}, rho0 {change_rho0:.2e}")
    assert change < 1e-10
    assert change_rho0 > 1e-6
    if with_ions:
        assert exchange > 1e-6
    assert xp.max(xp.abs(model.mhd.density.spline.vector.toarray() - n0)) == 0.0


def _allocate_dk(tmp_path, model, name, nonlinear=True, torus=False):
    """PerturbativeMHDWithQDriftkineticCC with finite-amplitude perturbations of density, velocity and magnetic field,
    guiding centers with a Maxwellian, tight tolerances and the energy-conserving algorithms. In a periodic box with a
    homogeneous, tilted B0 (curl B0 = 0: en_tot of the model is exactly the energy of the shear-Alfven step), or in a
    torus (``torus=True``: curvature and gradient of B0, so that the current-coupling steps do not vanish)."""
    from struphy import BoundaryParameters

    model.energetic_ions.set_markers(
        loading_params=LoadingParameters(Np=2000, seed=1234),
        boundary_params=BoundaryParameters(bc=("reflect", "periodic", "periodic")) if torus else None,
    )
    model.energetic_ions.var.add_background(maxwellians.GyroMaxwellian2D(n=(0.1, None), u_para=(0.5, None)))
    m = model.mhd
    if torus:
        m.velocity.add_perturbation(
            perturbations.TorusModesSin(ms=(1,), ns=(-1,), amps=(1e-2,), comp=0, given_in_basis="2")
        )
        m.density.add_perturbation(perturbations.TorusModesCos(ms=(1,), ns=(0,), amps=(0.3,), given_in_basis="0"))
        model.em_fields.b_field.add_perturbation(
            perturbations.TorusModesSin(ms=(1,), ns=(-1,), amps=(1e-2,), comp=1, given_in_basis="2")
        )
    else:
        m.velocity.add_perturbation(
            perturbations.ModesSin(ls=(1,), ms=(1,), amps=(0.1,), comp=0, given_in_basis="physical")
        )
        m.density.add_perturbation(perturbations.ModesCos(ls=(1,), ns=(1,), amps=(0.3,), given_in_basis="physical"))
        model.em_fields.b_field.add_perturbation(
            perturbations.ModesSin(ls=(1,), ns=(1,), amps=(0.2,), comp=1, given_in_basis="physical")
        )
    props = model.propagators
    props.shear_alf.options = props.shear_alf.Options(nonlinear=nonlinear, solver_params=TOL)
    props.cc5d_density.options = props.cc5d_density.Options(solver_params=TOL)
    props.cc5d_gradb.options = props.cc5d_gradb.Options(
        algo="discrete_gradient",
        solver_params=TOL,
        dg_solver_params=DiscreteGradientSolverParameters(tol=1e-13, maxiter=100),
    )
    props.cc5d_curlb.options = props.cc5d_curlb.Options(solver_params=TOL)
    if torus:
        geometry = dict(
            domain=domains.HollowTorus(a1=0.1, a2=1.0, R0=3.0, tor_period=1),
            equil=equils.AdhocTorus(a=1.0, R0=3.0, B0=1.0, q0=1.25, q1=2.25),
            grid=grids.TensorProductGrid(num_elements=(6, 6, 4)),
            derham_opts=DerhamOptions(degree=(2, 2, 2), bcs=(("dirichlet", "dirichlet"), None, None)),
        )
    else:
        geometry = dict(
            domain=domains.Cuboid(),
            equil=equils.HomogenSlab(B0x=0.2, B0z=1.0, beta=1.0, n0=1.0),
            grid=grids.TensorProductGrid(num_elements=(6, 4, 4)),
            derham_opts=DerhamOptions(degree=(2, 2, 1)),
        )
    Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder=name),
        time_opts=Time(dt=0.05, Tend=0.05),
        **geometry,
    ).allocate()


def _en_tot(model):
    model.update_scalar_quantities()
    return float(model.scalars.host_value("en_tot")[0])


@pytest.mark.parametrize("step", ["shear_alf", "cc5d_density", "cc5d_gradb", "cc5d_curlb"])
def test_energy_conservation_total_density_5d(tmp_path, step):
    """The 5D coupling steps with the inertia of the total density conserve the energy of the model (kinetic energy
    with the total density, magnetic energy, energies of the guiding centers), while the same energy with the inertia
    rho0 alone changes (the total-density inertia is really used), and the density does not change. The shear-Alfven
    step is tested in a homogeneous slab (in a torus the B0 . b term of en_tot is not conserved by it alone), the
    current coupling in a torus (where it does not vanish; it does not change b)."""
    model = PerturbativeMHDWithQDriftkineticCC()
    _allocate_dk(tmp_path, model, f"dk_{step}", torus=step != "shear_alf")
    n0 = model.mhd.density.spline.vector.toarray().copy()
    rho = model.mhd.density

    def energies():
        """en_tot, and en_tot with the kinetic energy of the inertia rho0 instead of rho0 + rho~."""
        en_tot = _en_tot(model)
        return en_tot, en_tot - _u_energy(model, rho) + _u_energy(model, None)

    (e0, e0_rho0), k0_rho0 = energies(), _u_energy(model, None)
    for _ in range(3):
        getattr(model.propagators, step)(0.1)
    (e1, e1_rho0), k1_rho0 = energies(), _u_energy(model, None)
    change = abs(e1 - e0) / e0
    change_rho0 = abs(e1_rho0 - e0_rho0) / e0_rho0
    # cc5d_density only rotates u (by the small factor 1 - B_par / B*_par): no exchange with the guiding centers, its
    # effect is visible in the kinetic energy of the inertia rho0 alone
    change_k_rho0 = abs(k1_rho0 - k0_rho0) / k0_rho0
    print(
        f"{step}: relative change of en_tot {change:.2e}, with rho0 inertia {change_rho0:.2e} (kinetic {change_k_rho0:.2e})"
    )
    # (discrete gradient: Picard iteration to 1e-13)
    tol = 1e-8 if step == "cc5d_gradb" else 1e-10
    assert change < tol
    if step == "cc5d_density":
        assert change_k_rho0 > 1e-10
    else:
        assert change_rho0 > 100 * tol
    assert xp.max(xp.abs(model.mhd.density.spline.vector.toarray() - n0)) == 0.0


def test_shear_alfven_cc5d_fixed_nonlinear(tmp_path):
    """The model fixes nonlinear=True of its shear-Alfven step: Options(nonlinear=False) is rejected in allocate."""
    model = PerturbativeMHDWithQDriftkineticCC()
    with pytest.raises(AssertionError, match="fixes nonlinear"):
        _allocate_dk(tmp_path, model, "dk_fixed", nonlinear=False)
