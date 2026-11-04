import cunumpy as xp
import pytest

from struphy import EnvironmentOptions, FieldsBackground, Simulation, Time, domains, equils, perturbations
from struphy.io.options import DerhamOptions
from struphy.models import LinearMHD, LinearMHDDriftkineticCC
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.magnetosonic import Magnetosonic
from struphy.propagators.magnetosonic_cross_b0 import MagnetosonicCrossB0
from struphy.topology import grids


def _link(prop, model):
    """Link the variables of an allocated LinearMHD model to a magnetosonic step and allocate it."""
    prop.variables.n = model.mhd.density
    prop.variables.u = model.mhd.velocity
    prop.variables.p = model.mhd.pressure
    prop.allocate()
    return prop


def _cross_b0(model):
    """The MagnetosonicCrossB0 step of the variables of an allocated LinearMHD model."""
    return _link(MagnetosonicCrossB0(b_field=model.em_fields.b_field), model)


def _itpa_torus(tmp_path, num_elements):
    """Allocated LinearMHD model in the geometry of the ITPA TAE, with J0 != 0 and a pressure gradient."""
    model = LinearMHD()
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder="sim"),
        time_opts=Time(dt=0.1, Tend=0.1),
        domain=domains.HollowTorus(a1=0.1, a2=1.0, R0=10.0, sfl=False, tor_period=6),
        equil=equils.AdhocTorus(
            a=1.0,
            R0=10.0,
            B0=3.0,
            q_kind=0,
            q0=1.71,
            q1=1.87,
            n1=0.0,
            n2=0.0,
            na=1.0,
            p_kind=1,
            p1=0.95,
            p2=0.05,
            beta=0.0018,
        ),  # fmt: skip
        grid=grids.TensorProductGrid(num_elements=num_elements, mpi_dims_mask=(True, True, False)),
        derham_opts=DerhamOptions(degree=(3, 3, 3), bcs=(("dirichlet", "dirichlet"), None, None)),
    )
    sim.allocate()
    return sim, model


def _pattern(space, comp, k3):
    """Coefficients sin(pi i1/N1) cos(2 pi 2 i2/N2) cos(2 pi k3 i3/N3) in one component, zero in the others."""
    v = space.zeros()
    blk = v.blocks[comp] if hasattr(v, "blocks") else v
    s, e, n = blk.starts, blk.ends, blk.space.npts
    i, j, k = (xp.arange(s[d], e[d] + 1) for d in range(3))
    blk[s[0] : e[0] + 1, s[1] : e[1] + 1, s[2] : e[2] + 1] = (
        xp.sin(xp.pi * (i + 0.5) / n[0])[:, None, None]
        * xp.cos(4 * xp.pi * j / n[1])[None, :, None]
        * xp.cos(2 * xp.pi * k3 * k / n[2])[None, None, :]
    )
    v.update_ghost_regions()
    return v


def test_magnetosonic_cross_b0_uses_shared_b_field(tmp_path):
    """MagnetosonicCrossB0 must not re-allocate the shared b_field, and its step must change the variables."""
    model = LinearMHD()
    model.mhd.velocity.add_perturbation(perturbations.ModesCos(ls=(1,), given_in_basis="2", comp=0, amps=(1e-2,)))
    model.em_fields.b_field.add_perturbation(perturbations.ModesCos(ls=(1,), given_in_basis="2", comp=1, amps=(1e-2,)))
    # needs equil at allocation time
    model.em_fields.b_field.add_background(FieldsBackground(type="FluidEquilibrium", variable="b2"))

    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder="sim"),
        time_opts=Time(dt=0.05, Tend=0.05),
        equil=equils.HomogenSlab(),
        grid=grids.TensorProductGrid(num_elements=(8, 1, 1)),
        derham_opts=DerhamOptions(degree=(2, 1, 1)),
    )
    sim.allocate()

    spline = model.em_fields.b_field.spline
    b = spline.vector
    b_init = b.toarray().copy()
    assert xp.max(xp.abs(b_init)) > 0.0
    mag_sonic = _cross_b0(model)
    assert model.em_fields.b_field.spline is spline
    assert model.em_fields.b_field.spline.vector is b
    assert xp.array_equal(b.toarray(), b_init)
    assert mag_sonic._b is b

    u0, p0 = model.mhd.velocity.spline.vector.toarray().copy(), model.mhd.pressure.spline.vector.toarray().copy()
    mag_sonic(0.05)
    assert xp.max(xp.abs(model.mhd.velocity.spline.vector.toarray() - u0)) > 0.0
    assert xp.max(xp.abs(model.mhd.pressure.spline.vector.toarray() - p0)) > 0.0
    # the magnetic field is only read
    assert xp.array_equal(b.toarray(), b_init)


@pytest.mark.mpi_skip
def test_b_field_space():
    """MagnetosonicCrossB0 accepts a velocity and a magnetic field in Hdiv only."""
    MagnetosonicCrossB0(b_field=FEECVariable(space="Hdiv"))
    with pytest.raises(AssertionError):
        MagnetosonicCrossB0(b_field=FEECVariable(space="Hcurl"))
    with pytest.raises(AssertionError):
        MagnetosonicCrossB0().variables.u = FEECVariable(space="Hcurl")


@pytest.mark.mpi_skip
def test_model_flag_replaces_the_background_and_sound_steps():
    """The flag of LinearMHDDriftkineticCC replaces the background current, background pressure and sound steps."""
    default = list(vars(LinearMHDDriftkineticCC().propagators))
    assert {"bg_current", "bg_pressure", "sound"} <= set(default)
    assert "magnetosonic" not in default

    model = LinearMHDDriftkineticCC(magnetosonic_cross_b0=True)
    assert not {"bg_current", "bg_pressure", "sound"} & set(vars(model.propagators))
    assert isinstance(model.propagators.magnetosonic, MagnetosonicCrossB0)


def _step_increments(prop, model, dt=0.1):
    """Increments of the velocity and the pressure in one step of prop from the current state, which is restored."""
    u, p, n = model.mhd.velocity.spline.vector, model.mhd.pressure.spline.vector, model.mhd.density.spline.vector
    u0, p0, n0 = u.copy(), p.copy(), n.copy()
    prop(dt)
    du, dp = u - u0, p - p0
    prop.update_feec_variables(n=n0, u=u0, p=p0)
    return du, dp


def test_step_agrees_with_magnetosonic_for_smooth_modes(tmp_path):
    """One step from a smooth state in the ITPA torus (k3 << N3): same increments as Magnetosonic."""
    _, model = _itpa_torus(tmp_path, (8, 16, 16))
    u = model.mhd.velocity.spline.vector
    b = model.em_fields.b_field.spline.vector
    _pattern(u.space, 0, 1).copy(out=u)
    _pattern(b.space, 1, 1).copy(out=b)
    u.update_ghost_regions()
    b.update_ghost_regions()

    M2 = Propagator.mass_ops.M2
    M3 = Propagator.mass_ops.M3
    du_old, dp_old = _step_increments(_link(Magnetosonic(b_field=model.em_fields.b_field), model), model)
    du_new, dp_new = _step_increments(_cross_b0(model), model)

    diff_u, diff_p = du_new - du_old, dp_new - dp_old
    assert xp.sqrt(diff_u.inner(M2.dot(diff_u)) / du_old.inner(M2.dot(du_old))) < 1e-2
    assert xp.sqrt(diff_p.inner(M3.dot(diff_p)) / dp_old.inner(M3.dot(dp_old))) < 1e-2
