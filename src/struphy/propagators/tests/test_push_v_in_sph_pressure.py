"""Tests of the SPH pressure force (PushVinSPHpressure) with the physical smoothing kernel on a curved mapping."""

import os

import cunumpy as xp
import pytest
from maybempi import MPI

from struphy import (
    BoundaryParameters,
    EnvironmentOptions,
    LoadingParameters,
    Simulation,
    SortingParameters,
    SPHParameters,
    Time,
    WeightsParameters,
    domains,
    equils,
    perturbations,
)
from struphy.models import EulerSPH
from struphy.propagators.push_v_in_sph_pressure import PushVinSPHpressure


def _setup(mpi_out_folder, name, thermodynamics, gamma, rho0, amp, kernel="gaussian"):
    """EulerSPH on Colella (alpha = 0.1) at rest, with the physical density rho0 (1 + amp sin(2 pi eta1)).

    For SPH, the density n of the background and perturbation is the 3-form in logical coordinates,
    n = rho sqrt(g). On the Colella mapping with Lx = Ly = 1, sqrt(g) = 1 + 2 pi alpha sin(2 pi (eta1 + eta2)),
    so n = rho0 (1 + ModesSin(ls=1, ms=1, amp=2 pi alpha)) is the uniform physical density rho0.
    """
    env = EnvironmentOptions(out_folders=os.path.join(mpi_out_folder, "EulerSPH"), sim_folder=name)
    time_opts = Time(dt=1e-3, Tend=1e-3)
    alpha = 0.1
    domain = domains.Colella(Lx=1.0, Ly=1.0, alpha=alpha, Lz=1.0)

    model = EulerSPH()
    model.fluid.set_markers(
        loading_params=LoadingParameters(ppb=256, loading="tesselation"),
        weights_params=WeightsParameters(),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(boxes_per_dim=(8, 8, 1), dims_mask=(True, True, False)),
        sph_params=SPHParameters(kernel=kernel),
        bufsize=2.0,  # room for the ghost particles of a 2D domain decomposition
    )
    model.propagators.push_sph_p.options = model.propagators.push_sph_p.Options(
        kappa=1.0,
        thermodynamics=thermodynamics,
        gamma=gamma,
    )
    model.fluid.var.add_background(equils.ConstantVelocity(ux=0.0, uy=0.0, uz=0.0, n=rho0))
    # n = sqrt(g): uniform physical density
    model.fluid.var.add_perturbation(
        del_n=perturbations.ModesSin(ls=(1,), ms=(1,), amps=(rho0 * 2.0 * xp.pi * alpha,), given_in_basis="0")
    )
    if amp != 0.0:
        model.fluid.var.add_perturbation(del_n=perturbations.ModesSin(ls=(1,), amps=(rho0 * amp,), given_in_basis="3"))

    sim = Simulation(model=model, env=env, time_opts=time_opts, domain=domain)
    sim.allocate()

    particles = model.fluid.var.particles
    particles.markers[3:6, particles.valid_mks] = 0.0
    return model, particles, time_opts.dt


@pytest.mark.parametrize(
    "thermodynamics,gamma",
    [("isothermal", 5.0 / 3.0), ("polytropic", 5.0 / 3.0), ("polytropic", 1.4)],
)
@pytest.mark.parametrize("rho0", [1.0, 2.0])
def test_uniform_gas_at_rest_on_curved_mapping(mpi_out_folder: str, thermodynamics: str, gamma: float, rho0: float):
    """A gas with uniform physical density and pressure at rest stays at rest on a curved mapping (Colella).

    With the kernel in physical space, the SPH density is the physical density directly and no metric term
    appears in the pressure force. The remaining acceleration is the (small) SPH discretization error, compared
    with the scale U'(rho) rho / h of the single sums of the force. The markers form a lattice distorted by the
    mapping (tesselation in logical space); the error depends strongly on h / dx (about 0.1 x the scale at
    h / dx = 3, 2e-3 at h / dx = 6), hence ppb = 256 (h / dx = 6).
    """
    model, particles, dt = _setup(
        mpi_out_folder, f"uniform_{thermodynamics}_{gamma:.3f}_{rho0}", thermodynamics, gamma, rho0, 0.0
    )
    model.propagators.push_sph_p(dt)

    markers = particles.markers_wo_holes_and_ghost
    accel = xp.max(xp.linalg.norm(markers[3:6] / dt, axis=0)) if markers.shape[1] > 0 else 0.0
    accel = particles.mpi_comm.allreduce(accel, op=MPI.MAX) if particles.mpi_comm is not None else accel
    # U'(rho) rho: kappa (isothermal), kappa rho^(gamma - 1) (polytropic)
    u_prime_rho = 1.0 if thermodynamics == "isothermal" else rho0 ** (gamma - 1.0)
    scale = u_prime_rho / particles.sph_kernel.h

    assert accel < 0.01 * scale, f"uniform gas accelerated: max |a| = {accel:.3e}, force scale {scale:.3e}"


@pytest.mark.parametrize("kernel", ["gaussian", "wendland_c2", "cubic_spline"])
@pytest.mark.parametrize("thermodynamics", ["isothermal", "polytropic"])
def test_momentum_conservation_on_curved_mapping(mpi_out_folder: str, kernel: str, thermodynamics: str):
    """The pressure force is antisymmetric in each pair of markers (radial kernel in physical space), so the
    total momentum sum_p w_p v_p is conserved up to round-off on a periodic curved mapping, also for a
    non-uniform density."""
    model, particles, dt = _setup(
        mpi_out_folder, f"momentum_{kernel}_{thermodynamics}", thermodynamics, 5.0 / 3.0, 1.0, 0.3, kernel=kernel
    )
    model.propagators.push_sph_p(dt)

    markers = particles.markers_wo_holes_and_ghost
    w = markers[particles.index["weights"]]
    mom = xp.ascontiguousarray(xp.sum(w * markers[3:6], axis=1), dtype=float)
    # scale: the largest single momentum change
    ref = xp.array([xp.max(xp.abs(w * markers[3 + k])) if markers.shape[1] > 0 else 0.0 for k in range(3)])
    if particles.mpi_comm is not None:
        # buffer-based reductions (the pickle-based allreduce cannot take the max of arrays)
        mom_glob = xp.zeros(3)
        ref_glob = xp.zeros(3)
        particles.mpi_comm.Allreduce(mom, mom_glob, op=MPI.SUM)
        particles.mpi_comm.Allreduce(ref, ref_glob, op=MPI.MAX)
        mom, ref = mom_glob, ref_glob

    assert xp.max(ref) > 0.0, "the density perturbation should accelerate the markers"
    assert xp.max(xp.abs(mom)) < 1e-10 * xp.max(ref) * particles.Np, f"momentum not conserved: {mom}, ref {ref}"


@pytest.mark.mpi_skip
def test_gamma_must_exceed_one():
    with pytest.raises(AssertionError):
        PushVinSPHpressure.Options(thermodynamics="polytropic", gamma=1.0)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("name,value", [("kernel_type", "gaussian_2d"), ("kernel_width", (0.1, 0.1, 1.0))])
def test_removed_kernel_options(name, value):
    """The logical kernel options were replaced by the per-species SPHParameters: clear error."""
    with pytest.raises(ValueError, match="SPHParameters"):
        PushVinSPHpressure.Options(**{name: value})
