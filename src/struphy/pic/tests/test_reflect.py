"""Reflection uses the same arguments on NumPy and CUDA."""

import cunumpy as xp
import numpy as np
import pytest

from struphy.geometry.domains import Cuboid
from struphy.pic.pushing.kernels.reflect import reflect

requires_cupy = pytest.mark.skipif(not xp.cupy_available(), reason="CuPy/GPU not available")

BACKENDS = ["numpy", pytest.param("cupy", marks=requires_cupy)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("count", [0, 1, 129])
def test_reflect_in_place(backend, axis, count):
    """Only selected velocity entries change, including strided arrays and indices."""
    if count == 0 and backend == "numpy":
        # Zero-size NumPy arrays have stride 0, which the compiled pyccel reflect divides by (garbage on arm64,
        # SIGFPE on x86-64). Particles.apply_kinetic_bc never passes an empty index array; the CUDA launch of
        # zero threads is a no-op and stays tested.
        pytest.skip("pyccel kernels must not be called with zero-size arrays")
    original = np.random.default_rng(32).random((16, 140))
    indices = np.arange(139, 139 - count, -1, dtype=np.int64)
    expected = original.copy()
    expected[2 * (3 + axis), indices] *= -1
    with xp.use_backend(backend):
        # every other row of a marker buffer (one column per marker): strided, in the marker layout
        storage = xp.asarray(original)
        markers = storage[::2]
        outside_inds = xp.asarray(np.repeat(indices, 2))[::2]
        domain = Cuboid(r1=2.0, r2=3.0, r3=4.0)
        reflect(markers, domain.args_domain, outside_inds, axis, n_threads=outside_inds.size)
        np.testing.assert_allclose(xp.to_numpy(storage), expected, rtol=1e-13, atol=1e-14)


def reflecting_particles(domain):
    """Particles6D with reflecting boundaries in all directions, 100 markers drawn uniformly."""
    from maybempi import MPI

    from struphy import LoadingParameters
    from struphy.particles.parameters import BoundaryParameters
    from struphy.pic.particles import Particles6D

    loading_params = LoadingParameters(Np=100, seed=1234, moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), spatial="uniform")
    particles = Particles6D(
        comm_world=MPI.COMM_WORLD,
        loading_params=loading_params,
        boundary_params=BoundaryParameters(bc=("reflect", "reflect", "reflect")),
        domain=domain,
    )
    particles.draw_markers()
    return particles


@requires_cupy
def test_reflect_rejects_unsupported_mapping():
    """On the CuPy backend, reflection with a mapping other than Cuboid fails when the particles are created."""
    from struphy.geometry.domains import HollowCylinder

    with xp.use_backend("cupy"), pytest.raises(NotImplementedError, match="Cuboid"):
        reflecting_particles(HollowCylinder())


@requires_cupy
@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_reflect_rejects_wrong_index_dtype(dtype):
    with xp.use_backend("cupy"):
        with pytest.raises(TypeError, match="Array1D<long long>"):
            reflect(xp.zeros((6, 2)), Cuboid().args_domain, xp.asarray([0], dtype=dtype), 0, n_threads=1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_particle_boundary_reflection(backend):
    with xp.use_backend(backend):
        particles = reflecting_particles(Cuboid())
        selected = xp.nonzero(particles.valid_mks)[0][:3]
        particles.markers[:3, selected] = xp.asarray([[-0.1, 0.5, 0.5], [0.5, 1.1, 0.5], [-0.2, 1.2, -0.3]]).T
        particles.markers[3:6, selected] = xp.asarray([1.0, 2.0, 3.0])[:, None]
        particles.apply_kinetic_bc()
        np.testing.assert_allclose(
            xp.to_numpy(particles.markers[:3, selected]).T, [[0.1, 0.5, 0.5], [0.5, 0.9, 0.5], [0.2, 0.8, 0.3]]
        )
        np.testing.assert_allclose(
            xp.to_numpy(particles.markers[3:6, selected]).T, [[-1.0, 2.0, 3.0], [1.0, -2.0, 3.0], [-1.0, -2.0, -3.0]]
        )
        assert bool(xp.all(particles.markers[particles.first_pusher_idx, selected] == -1.0))


def assert_reflected_markers_not_pushed_again(markers, start, first_pusher_idx):
    """Checks after one Runge-Kutta step with reflecting boundaries (issue #866) on the markers of
    :func:`~struphy.pic.tests.kernel_test_args.marker_arguments`: some markers were reflected, all stay in the unit
    cube and near their start position (a reflected marker pushed again from eta1 = -1 ends far away), and the
    boundary particle (marker 1) is unchanged."""
    reflected = markers[first_pusher_idx, 2:] == -1.0
    assert reflected.any(), "no marker was reflected; increase the time step"
    positions = markers[:3, 2:]
    assert np.all((positions >= 0.0) & (positions <= 1.0))
    # one step moves a marker by less than 0.5 here; a reflection only shortens the distance
    assert np.all(np.abs(positions - start[:3, 2:]) < 0.5)
    np.testing.assert_array_equal(markers[:, 1], start[:, 1])


@pytest.mark.mpi_skip
@pytest.mark.parametrize("backend", BACKENDS)
# time steps that reflect markers but keep each stage displacement below the domain size (one reflection must bring
# the marker back): the random U gives logical velocities of up to 20 (Hcurl), 400 (Hdiv) and 3 (H1vec)
@pytest.mark.parametrize("space, dt", [("Hcurl", 0.01), ("Hdiv", 0.001), ("H1vec", 0.05)])
def test_pc_eta_stage_skips_reflected_markers(backend, space, dt):
    """A marker reflected in an RK stage of push_pc_eta_stage is not pushed again in the later stages (issue #866).

    Reflection overwrites its start position (row first_pusher_idx is set to -1); pushing it again from there sent
    it far outside the domain."""
    import importlib

    from struphy.pic.tests.kernel_test_args import (
        butcher_arguments,
        derham_arguments,
        marker_arguments,
        spline_coefficients,
    )

    name = f"push_pc_eta_stage_{space}"
    kernel = getattr(importlib.import_module(f"struphy.pic.pushing.kernels.{name}"), name)
    with xp.use_backend(backend):
        args_markers, args_domain = marker_arguments((1, 1, 1))
        start = xp.to_numpy(args_markers.markers).copy()
        args = (args_markers, args_domain, derham_arguments(), *spline_coefficients(n=3, seed=62), False)
        a, b, c, n_stages = butcher_arguments("rk4")
        for stage in range(n_stages):
            kernel(dt, stage, *args, a, b, c, n_threads=args_markers.n_markers)
        markers = xp.to_numpy(args_markers.markers)

    assert_reflected_markers_not_pushed_again(markers, start, args_markers.first_pusher_idx)


@pytest.mark.mpi_skip
def test_deterministic_diffusion_stage_skips_reflected_markers():
    """Like test_pc_eta_stage_skips_reflected_markers, for push_deterministic_diffusion_stage (pyccel only)."""
    from struphy.pic.pushing.kernels.push_deterministic_diffusion_stage import push_deterministic_diffusion_stage
    from struphy.pic.tests.kernel_test_args import (
        butcher_arguments,
        derham_arguments,
        marker_arguments,
        spline_coefficients,
    )

    args_markers, args_domain = marker_arguments((1, 1, 1))
    start = args_markers.markers.copy()
    pi_u, *pi_grad_u = spline_coefficients(n=4, seed=64)
    # a positive density; diffusion coefficient 0.1 with dt = 0.1 moves markers by up to 0.1 per stage
    pi_u = 1.0 + 0.5 * np.abs(pi_u)
    a, b, c, n_stages = butcher_arguments("rk4")
    for stage in range(n_stages):
        push_deterministic_diffusion_stage(
            0.1, stage, args_markers, args_domain, derham_arguments(), pi_u, *pi_grad_u, 0.1, a, b, c
        )

    assert_reflected_markers_not_pushed_again(args_markers.markers, start, args_markers.first_pusher_idx)
