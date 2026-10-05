"""Reflection uses the same arguments on NumPy and CUDA."""

import cunumpy as xp
import numpy as np
import pytest

from struphy.geometry.domains import Cuboid
from struphy.pic.pushing.pusher_utilities import reflect

requires_cupy = pytest.mark.skipif(not xp.cupy_available(), reason="CuPy/GPU not available")
BACKENDS = ["numpy", pytest.param("cupy", marks=requires_cupy)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("count", [0, 1, 129])
def test_reflect_in_place(backend, axis, count):
    """Only selected velocity entries change, including strided arrays and indices."""
    original = np.random.default_rng(32).random((140, 16))
    indices = np.arange(139, 139 - count, -1, dtype=np.int64)
    expected = original.copy()
    expected[indices, 2 * (3 + axis)] *= -1
    with xp.use_backend(backend):
        storage = xp.asarray(original.copy())
        markers = storage[:, ::2]
        outside_inds = xp.asarray(np.repeat(indices, 2))[::2]
        domain = Cuboid(r1=2.0, r2=3.0, r3=4.0)
        reflect(markers, domain.args_domain, outside_inds, axis)
        np.testing.assert_allclose(xp.to_numpy(storage), expected, rtol=1e-13, atol=1e-14)


@requires_cupy
def test_reflect_rejects_unsupported_mapping():
    with xp.use_backend("cupy"):
        args = Cuboid().args_domain
        args.kind_map = 999
        with pytest.raises(NotImplementedError, match="Cuboid"):
            reflect(xp.zeros((2, 6)), args, xp.asarray([0], dtype=xp.int64), 0)


@requires_cupy
@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_reflect_rejects_wrong_index_dtype(dtype):
    with xp.use_backend("cupy"):
        with pytest.raises(TypeError, match="Array1D<long long>"):
            reflect(xp.zeros((2, 6)), Cuboid().args_domain, xp.asarray([0], dtype=dtype), 0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_particle_boundary_reflection(backend):
    from struphy.pic.pushing.kernels import catalog
    from struphy.pic.tests.test_kernel_backends import make_pusher

    with xp.use_backend(backend):
        particles = make_pusher(catalog["push_eta_stage"])().particles
        particles._periodic_axes = ()
        particles._remove_axes = ()
        particles._reflect_axes = (0, 1, 2)
        selected = xp.nonzero(particles.valid_mks)[0][:3]
        particles.markers[selected, :3] = xp.asarray([[-0.1, 0.5, 0.5], [0.5, 1.1, 0.5], [-0.2, 1.2, -0.3]])
        particles.markers[selected, 3:6] = xp.asarray([1.0, 2.0, 3.0])
        particles.apply_kinetic_bc()
        np.testing.assert_allclose(
            xp.to_numpy(particles.markers[selected, :3]), [[0.1, 0.5, 0.5], [0.5, 0.9, 0.5], [0.2, 0.8, 0.3]]
        )
        np.testing.assert_allclose(
            xp.to_numpy(particles.markers[selected, 3:6]), [[-1.0, 2.0, 3.0], [1.0, -2.0, 3.0], [-1.0, -2.0, -3.0]]
        )
        assert bool(xp.all(particles.markers[selected, particles.first_pusher_idx] == -1.0))
