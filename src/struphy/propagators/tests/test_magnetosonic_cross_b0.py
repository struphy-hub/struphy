import cunumpy as xp
import pytest

from struphy import (
    BaseUnits,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    Simulation,
    Time,
    WeightsParameters,
    domains,
    equils,
    grids,
    maxwellians,
)
from struphy.models import LinearMHDDriftkineticCC
from struphy.models.variables import FEECVariable
from struphy.propagators.magnetosonic_cross_b0 import MagnetosonicCrossB0


def _itpa_operators(tmp_path, num_elements):
    """Allocated LinearMHDDriftkineticCC in the ITPA TAE geometry, with a minimal energetic ion species."""
    model = LinearMHDDriftkineticCC(
        base_units=BaseUnits(n=0.2),
        hot_mass_number=2.0,
        magnetosonic_cross_b0=True,
    )
    model.propagators.shear_alf.options = model.propagators.shear_alf.Options(ep_scale=0.0, nonlinear=False)
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder="sim"),
        time_opts=Time(dt=0.25, Tend=0.25, split_algo="Strang"),
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
    model.energetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=1, moments=(0.0, 0.0, 0.9, 0.9), seed=1234, B0=3.0),
        weights_params=WeightsParameters(control_variate=False),
        boundary_params=BoundaryParameters(bc=("remove", "periodic", "periodic")),
    )
    model.energetic_ions.var.add_background(maxwellians.GyroMaxwellian2D(n=(1e-3, None), volume_form=True, B0=3.0))
    sim.allocate()
    return sim, model.propagators.magnetosonic


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


def test_forces_keep_smooth_modes_and_lose_eta3_nyquist(tmp_path):
    """Against the standard forms: same J0 x b and u.grad p0 for k3 << N3, drive removed at k3 = N3/2."""
    sim, prop = _itpa_operators(tmp_path, (8, 16, 16))
    mass_ops, basis_ops, div = prop.mass_ops, prop.basis_ops, sim.derham.div
    M2J, M3, K3, S2 = mass_ops.M2J, mass_ops.M3, basis_ops.K3, basis_ops.S2

    def relative_differences(k3):
        v, b, q = _pattern(M2J.domain, 0, k3), _pattern(M2J.domain, 1, k3), _pattern(M3.domain, 0, k3)
        force_old, force_new = v.inner(M2J.dot(b)), v.inner(prop._MJ.dot(b))
        # u.grad p0 of Magnetosonic: div S - K div. Here: C = (5/3 K div + G)/2.
        K_div_v = K3.dot(div.dot(v))
        gradp0_old = div.dot(S2.dot(v)) - K_div_v
        gradp0_new = 2.0 * prop._C.dot(v) - 5 / 3 * K_div_v
        work_old, work_new = q.inner(M3.dot(gradp0_old)), q.inner(M3.dot(gradp0_new))
        return abs(force_new - force_old) / abs(force_old), abs(work_new - work_old) / abs(work_old)

    force_diff, work_diff = relative_differences(1)
    assert force_diff < 1e-2
    assert work_diff < 1e-2

    force_diff, work_diff = relative_differences(8)
    assert force_diff > 0.99
    assert work_diff > 0.99


def test_schur_complement_matches_pressure_update(tmp_path):
    """The Schur complement has M3 cancelled by hand: it must equal B @ C, where C contains the inverse of M3."""
    _, prop = _itpa_operators(tmp_path, (6, 8, 8))
    u = _pattern(prop._MJ.domain, 0, 1)
    direct = prop._B.dot(prop._C.dot(u))
    diff = direct - prop._schur_solver.BC.dot(u)
    assert xp.sqrt(diff.inner(diff)) < 1e-8 * xp.sqrt(direct.inner(direct))


@pytest.mark.mpi_skip
def test_model_flag_replaces_the_background_and_sound_steps():
    default = list(vars(LinearMHDDriftkineticCC().propagators))
    assert {"bg_current", "bg_pressure", "sound"} <= set(default)
    assert "magnetosonic" not in default

    model = LinearMHDDriftkineticCC(magnetosonic_cross_b0=True)
    assert list(vars(model.propagators)) == [
        "push_bxe",
        "push_parallel",
        "shear_alf",
        "magnetosonic",
        "cc5d_density",
        "cc5d_gradb",
        "cc5d_curlb",
    ]
    assert isinstance(model.propagators.magnetosonic, MagnetosonicCrossB0)
    assert model.params["magnetosonic_cross_b0"] is True

    # the steps of a Strang time step: magnetosonic is not in a symmetric block
    assert dict(model.propagator_calls(1.0, "Strang"))["magnetosonic"] == 0.5


@pytest.mark.mpi_skip
def test_needs_hdiv():
    """Velocity and magnetic field must be 2-forms."""
    MagnetosonicCrossB0(b_field=FEECVariable(space="Hdiv"))
    with pytest.raises(AssertionError):
        MagnetosonicCrossB0(b_field=FEECVariable(space="Hcurl"))
    with pytest.raises(AssertionError):
        MagnetosonicCrossB0().variables.u = FEECVariable(space="Hcurl")
