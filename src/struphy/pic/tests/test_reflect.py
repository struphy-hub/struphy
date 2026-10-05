"""Reflection uses the same arguments on NumPy and CUDA."""

from pathlib import Path

import cunumpy as xp
import numpy as np
import pytest
from cunumpy import PyccelKernel

import struphy
from struphy.geometry.domains import Cuboid
from struphy.pic.pushing import pusher_utilities_kernels
from struphy.utils.kernel_backends import CudaKernel, Kernel

requires_cupy = pytest.mark.skipif(not xp.cupy_available(), reason="CuPy/GPU not available")

# the same pair as Particles builds in __init__
reflect = Kernel(
    PyccelKernel(pusher_utilities_kernels.reflect),
    CudaKernel.from_file(Path(struphy.__file__).parent / "pic" / "pushing" / "reflect_cuda.cu"),
)
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
        reflect(markers, domain.args_domain, outside_inds, axis, n_threads=outside_inds.size)
        np.testing.assert_allclose(xp.to_numpy(storage), expected, rtol=1e-13, atol=1e-14)


def reflecting_particles(domain):
    """Particles6D with reflecting boundaries in all directions, 100 markers drawn uniformly."""
    from feectools.ddm.mpi import mpi as MPI

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
            reflect(xp.zeros((2, 6)), Cuboid().args_domain, xp.asarray([0], dtype=dtype), 0, n_threads=1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_particle_boundary_reflection(backend):
    with xp.use_backend(backend):
        particles = reflecting_particles(Cuboid())
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
