"""Tests of the SPH pressure force (PushVinSPHpressure) on a curved mapping."""

import os

import cunumpy as xp
import pytest

from struphy import (
    BoundaryParameters,
    EnvironmentOptions,
    LoadingParameters,
    Simulation,
    SortingParameters,
    Time,
    WeightsParameters,
    domains,
    equils,
    perturbations,
)
from struphy.models import EulerSPH
from struphy.propagators.push_v_in_sph_pressure import PushVinSPHpressure


def _spurious_acceleration(domain, etas, u_prime_rho):
    """Largest acceleration U'(rho) rho |DF^{-T} grad ln sqrt(g)| that a uniform gas at rest feels without the metric
    term of the pressure force (finite differences of the Jacobian determinant)."""
    delta = 1e-5
    grad = xp.zeros_like(etas)
    for k in range(3):
        ep = etas.copy()
        em = etas.copy()
        ep[k] = xp.minimum(ep[k] + delta, 1.0)
        em[k] = xp.maximum(em[k] - delta, 0.0)
        log_p = xp.log(xp.abs(domain.jacobian_det(ep, remove_outside=False)))
        log_m = xp.log(xp.abs(domain.jacobian_det(em, remove_outside=False)))
        grad[k] = (log_p - log_m) / (ep[k] - em[k])
    dfinv = domain.jacobian_inv(etas, remove_outside=False)  # (3, 3, N)
    accel = u_prime_rho * xp.einsum("kjn,kn->jn", dfinv, grad)  # DF^{-T} grad
    return float(xp.max(xp.linalg.norm(accel, axis=0)))


@pytest.mark.parametrize(
    "thermodynamics,gamma",
    [("isothermal", 5.0 / 3.0), ("polytropic", 5.0 / 3.0), ("polytropic", 1.4)],
)
@pytest.mark.parametrize("rho0", [1.0, 2.0])
def test_uniform_gas_at_rest_on_curved_mapping(mpi_out_folder: str, thermodynamics: str, gamma: float, rho0: float):
    """A gas with uniform physical density and pressure at rest stays at rest on a curved mapping (Colella).

    For SPH, the density n of the background and perturbation is the 3-form in logical coordinates,
    n = rho sqrt(g). On the Colella mapping with Lx = Ly = 1, sqrt(g) = 1 + 2 pi alpha sin(2 pi (eta1 + eta2)),
    so n = rho0 (1 + ModesSin(ls=1, ms=1, amp=2 pi alpha)) is the uniform physical density rho0.
    The case rho0 = 2 checks the density dependence U'(rho) rho of the metric term.
    Without the metric term -U'(rho_p) rho_p grad ln sqrt(g) of the pressure force, the markers would be
    accelerated by kappa DF^{-T} grad ln sqrt(g); with it, the remaining acceleration is the (small) SPH
    discretization error.
    """
    env = EnvironmentOptions(
        out_folders=os.path.join(mpi_out_folder, "EulerSPH"),
        sim_folder=f"uniform_gas_{thermodynamics}_{gamma:.3f}_{rho0}",
    )
    time_opts = Time(dt=1e-3, Tend=1e-3)
    alpha = 0.1
    domain = domains.Colella(Lx=1.0, Ly=1.0, alpha=alpha, Lz=1.0)

    model = EulerSPH()
    model.fluid.set_markers(
        loading_params=LoadingParameters(ppb=16, loading="tesselation"),
        weights_params=WeightsParameters(),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(boxes_per_dim=(12, 12, 1), dims_mask=(True, True, False)),
        bufsize=2.0,  # room for the ghost particles of a 2D domain decomposition
    )
    kappa = 1.0
    model.propagators.push_sph_p.options = model.propagators.push_sph_p.Options(
        kernel_type="gaussian_2d",
        kappa=kappa,
        thermodynamics=thermodynamics,
        gamma=gamma,
    )
    model.fluid.var.add_background(equils.ConstantVelocity(ux=0.0, uy=0.0, uz=0.0, n=rho0))
    # n = sqrt(g): uniform physical density
    model.fluid.var.add_perturbation(
        del_n=perturbations.ModesSin(ls=(1,), ms=(1,), amps=(rho0 * 2.0 * xp.pi * alpha,), given_in_basis="0")
    )

    sim = Simulation(model=model, env=env, time_opts=time_opts, domain=domain)
    sim.allocate()

    particles = model.fluid.var.particles
    particles.markers[3:6, particles.valid_mks] = 0.0

    dt = time_opts.dt
    model.propagators.push_sph_p(dt)

    markers = particles.markers_wo_holes_and_ghost
    accel = xp.max(xp.linalg.norm(markers[3:6] / dt, axis=0))
    # U'(rho) rho: kappa (isothermal), kappa rho^(gamma - 1) (polytropic)
    u_prime_rho = kappa if thermodynamics == "isothermal" else kappa * rho0 ** (gamma - 1.0)
    spurious = _spurious_acceleration(domain, markers[:3].copy(), u_prime_rho)

    assert spurious > 0.3, f"the mapping should give a noticeable metric force, got {spurious}"
    assert accel < 0.1 * spurious, f"uniform gas accelerated: max |a| = {accel:.3e}, metric scale {spurious:.3e}"


@pytest.mark.mpi_skip
def test_gamma_must_exceed_one():
    with pytest.raises(AssertionError):
        PushVinSPHpressure.Options(thermodynamics="polytropic", gamma=1.0)
