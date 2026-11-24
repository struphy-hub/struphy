"""Loading of the magnetic moment of Particles5D with the local field strength (issue #841)."""

import pytest

pytestmark = pytest.mark.mpi_pic

VTH = 0.8


def _make_particles(loading_B0, background_B0=None, loading="pseudo_random", spatial="uniform", ppc=200):
    """Particles5D in a hollow torus with the ad hoc tokamak equilibrium (|B_0| ~ 1/R varies by a factor ~2)."""
    from maybempi import MPI

    from struphy import LoadingParameters, domains
    from struphy.feec.psydac_derham import Derham
    from struphy.fields_background import equils
    from struphy.fields_background.projected_equils import ProjectedMHDequilibrium
    from struphy.io.options import DerhamOptions
    from struphy.kinetic_background import maxwellians
    from struphy.pic.particles import Particles5D
    from struphy.topology.grids import TensorProductGrid

    comm = MPI.COMM_WORLD

    domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=2.0, tor_period=1)
    equil = equils.AdhocTorus(a=1.0, R0=2.0, B0=1.0)
    equil.domain = domain

    grid = TensorProductGrid(num_elements=[8, 8, 4])
    derham = Derham(grid, DerhamOptions(degree=[1, 1, 1], bcs=[("free", "free"), None, None]), comm=comm, domain=domain)
    projected_equil = ProjectedMHDequilibrium(equil, derham)

    loading_params = LoadingParameters(
        ppc=ppc,
        seed=1234,
        loading=loading,
        spatial=spatial,
        moments=(0.0, 0.0, 1.0, VTH),
        B0=loading_B0,
    )
    if background_B0 is None:
        background_B0 = loading_B0
    background = maxwellians.GyroMaxwellian2D(
        n=(1.0, None),
        vth_para=(1.0, None),
        vth_perp=(VTH, None),
        B0=background_B0,
    )

    particles = Particles5D(
        comm_world=comm if comm.Get_size() > 1 else None,
        domain_decomp=(derham.domain_array, derham.domain_decomposition.nprocs),
        loading_params=loading_params,
        domain=domain,
        equil=equil,
        projected_equil=projected_equil,
        background=background,
    )
    particles.draw_markers()
    particles.initialize_weights()
    return particles, equil, domain


def _global_sum(local, comm):
    """Sum over all ranks (the serial stand-in comm does not reduce)."""
    if comm.Get_size() > 1:
        from maybempi import MPI

        return comm.allreduce(local, op=MPI.SUM)
    return local


def _exp_means_by_field_strength(particles, equil):
    """Mean of mu |B_0| / vth^2 (perpendicular energy over temperature, Exp(1) for a local Maxwellian)
    for the markers below and above the median field strength 1/(1 + r cos(theta)/R0) ~ 1."""
    import cunumpy as xp
    from maybempi import MPI

    comm = MPI.COMM_WORLD
    mu = particles.velocities[1]
    absB = equil.absB0(xp.ascontiguousarray(particles.positions))
    x = mu * absB / VTH**2
    sums = []
    for mask in (absB < 1.0, absB >= 1.0):
        sums += [float(xp.sum(x[mask])), float(xp.sum(mask))]
    sums = [_global_sum(v, comm) for v in sums]
    return sums[0] / sums[1], sums[2] / sums[3], sums[1] + sums[3]


@pytest.mark.parametrize(
    "loading, spatial",
    [("pseudo_random", "uniform"), ("sobol_standard", "disc")],
)
def test_mu_is_local_maxwellian(loading, spatial):
    """With B0="equil" (default), mu |B_0(x)| / vth^2 is Exp(1) distributed wherever the marker is; with a
    constant B0 it is not (its mean is |B_0(x)| / B0)."""
    particles, equil, _ = _make_particles("equil", loading=loading, spatial=spatial)
    mean_low, mean_high, n_total = _exp_means_by_field_strength(particles, equil)
    # statistical error of the means ~ 1/sqrt(n/2) ~ 0.01 for ~5e4 markers
    assert n_total > 2e4
    assert abs(mean_low - 1.0) < 0.05, mean_low
    assert abs(mean_high - 1.0) < 0.05, mean_high

    particles, equil, _ = _make_particles(1.0, loading=loading, spatial=spatial)
    mean_low, mean_high, _ = _exp_means_by_field_strength(particles, equil)
    assert mean_low < 0.95 and mean_high > 1.05, (mean_low, mean_high)


@pytest.mark.parametrize("spatial", ["uniform", "disc"])
def test_weights_match_sampling_density(spatial):
    """If the initial condition has the moments and the B0 of the loading, f_init / s0 does not depend on the
    velocities: the weights are w0 = |det DF| / s_eta / Np, with s_eta = 1 (uniform) or 2 eta1 (disc).
    A background with B0="equil" and a loading with constant B0 gives velocity-dependent weights."""
    import cunumpy as xp

    def ratio_spread(particles, domain):
        pos = xp.ascontiguousarray(particles.positions)
        detj = xp.abs(domain.jacobian_det(pos))
        s_eta = 2.0 * pos[0] if spatial == "disc" else 1.0
        r = particles.weights0 * s_eta / detj
        return float(xp.max(r) / xp.min(r) - 1.0)

    particles, _, domain = _make_particles("equil", spatial=spatial, ppc=20)
    assert ratio_spread(particles, domain) < 1e-10

    particles, _, domain = _make_particles(1.0, background_B0="equil", spatial=spatial, ppc=20)
    assert ratio_spread(particles, domain) > 1e-2


@pytest.mark.mpi_skip
def test_B0_option_validation():
    """LoadingParameters and GyroMaxwellian2D accept a positive number, a callable or "equil"."""
    import cunumpy as xp

    from struphy import LoadingParameters
    from struphy.kinetic_background import maxwellians

    assert LoadingParameters().B0 == "equil"
    LoadingParameters(B0=1.5)
    LoadingParameters(B0=lambda etas: 1.0 + etas[0])
    for bad in ("absB0", -1.0, 0.0):
        with pytest.raises(AssertionError):
            LoadingParameters(B0=bad)
        with pytest.raises(AssertionError):
            maxwellians.GyroMaxwellian2D(B0=bad)

    # NumPy scalars are numbers too
    LoadingParameters(B0=xp.float32(1.5))
    maxwellians.GyroMaxwellian2D(B0=xp.int64(2))

    etas = xp.random.rand(3, 5)
    assert xp.allclose(maxwellians.GyroMaxwellian2D(B0=1.5).eval_B0(etas), 1.5)
    assert xp.allclose(maxwellians.GyroMaxwellian2D(B0=lambda e: 1.0 + e[0]).eval_B0(etas), 1.0 + etas[0])
    # a callable returning a scalar is broadcast
    assert xp.allclose(maxwellians.GyroMaxwellian2D(B0=lambda e: 3.0).eval_B0(etas), 3.0)

    # on a phase-space meshgrid, B0 is evaluated on the spatial points and broadcast to the velocity axes
    maxw = maxwellians.GyroMaxwellian2D(B0=lambda e: 1.0 + e[0] + 2.0 * e[2])
    e1, e2, e3, v1, v2 = xp.meshgrid(
        xp.linspace(0, 1, 3),
        xp.linspace(0, 1, 4),
        xp.linspace(0, 1, 2),
        xp.linspace(-1, 1, 5),
        xp.linspace(0, 1, 6),
        indexing="ij",
    )
    assert xp.allclose(maxw.eval_B0_on_args((e1, e2, e3)), 1.0 + e1 + 2.0 * e3)
    flat = maxw(e1.ravel(), e2.ravel(), e3.ravel(), v1.ravel(), v2.ravel())
    assert xp.allclose(maxw(e1, e2, e3, v1, v2).ravel(), flat)

    # the cache of eval_B0 belongs to the B0 option: replacing it invalidates the cached values
    maxw = maxwellians.GyroMaxwellian2D(B0=lambda e: 1.0 + e[0])
    assert xp.allclose(maxw.eval_B0(etas), 1.0 + etas[0])
    maxw.params["B0"] = lambda e: 5.0 + e[1]
    assert xp.allclose(maxw.eval_B0(etas), 5.0 + etas[1])

    # "equil" without attached equilibrium (e.g. outside of a Particles5D species) gives a clear error
    with pytest.raises(ValueError, match="no equilibrium is attached"):
        maxwellians.GyroMaxwellian2D().eval_B0(etas)


@pytest.mark.mpi_skip
def test_attach_different_equil_warns():
    """A background attached to a second, different equilibrium (shared by two simulations) logs a warning;
    attaching the same equilibrium again does not."""
    from unittest import mock

    from struphy.fields_background import equils
    from struphy.kinetic_background import maxwellians

    equil1, equil2 = equils.AdhocTorus(), equils.AdhocTorus(B0=2.0)
    maxw = maxwellians.GyroMaxwellian2D()
    background = maxw + 2.0 * maxw
    with mock.patch.object(maxwellians.logger, "warning") as warning:
        background.attach_equil(equil1)
        background.attach_equil(equil1)
        warning.assert_not_called()
        background.attach_equil(equil2)
        # maxw appears twice in the sum: attached twice, but warned once
        warning.assert_called_once()
    assert maxw._B0_equil is equil2

    # a constant B0 needs no equilibrium: nothing attached, no warning
    const = maxwellians.GyroMaxwellian2D(B0=1.0)
    with mock.patch.object(maxwellians.logger, "warning") as warning:
        const.attach_equil(equil1)
        const.attach_equil(equil2)
        warning.assert_not_called()
    assert const._B0_equil is None
