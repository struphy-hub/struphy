"""Dispatch between pyccel and CUDA kernels, and the CUDA versions of the kernel argument classes."""

import copy

import cunumpy
import numpy as np
import pytest
from cunumpy import PyccelKernel

from struphy.geometry.domains import Cuboid
from struphy.pic.pushing.demo_cuda import make_demo_arguments, push_eta_linear, run_push_eta_linear
from struphy.utils.cuda_arguments import CudaDerhamArguments, CudaMarkerArguments
from struphy.utils.kernel_backends import Kernel, KernelCatalog, catalog, is_cuda_backend

requires_cupy = pytest.mark.skipif(not cunumpy.cupy_available(), reason="CuPy/GPU not available")

BACKENDS = ["numpy", pytest.param("cupy", marks=requires_cupy)]

N_MARKERS = 1000
N_COLS = 25
MARKER_INDICES = (N_MARKERS, 3, 6, 7, 8, 14, 17, 18, 4)  # Np, vdim, weight_idx, ..., mu_idx


def expected_push(markers, valid_mks, dt, n_steps=1):
    out = markers.copy()
    for _ in range(n_steps):
        out[valid_mks, 0:3] += dt * out[valid_mks, 3:6]
    return out


def device_marker_arrays():
    import cupy as cp

    return cp.random.random((N_MARKERS, N_COLS)), cp.ones(N_MARKERS, dtype=bool), cp.zeros(3, dtype=int)


# ---------------------------
# kernel dispatch and catalog
# ---------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_kernel_dispatch(backend):
    with cunumpy.use_backend(backend):
        assert is_cuda_backend() == (backend == "cupy")
        kernel = push_eta_linear.get_kernel()
        if backend == "cupy":
            assert kernel is push_eta_linear.cuda_kernel
        else:
            assert kernel is push_eta_linear.pyccel_kernel


def test_kernel_without_cuda_falls_back_to_pyccel():
    kernel = Kernel(push_eta_linear.pyccel_kernel)
    for backend in ("numpy", "cupy"):
        with cunumpy.use_backend(backend):
            assert kernel.get_kernel() is kernel.pyccel_kernel


def test_catalog():
    assert catalog.get("push_eta_linear") is push_eta_linear

    local = KernelCatalog()
    local.register(push_eta_linear, name="foo")
    assert "foo" in local and local.names == ["foo"]
    with pytest.raises(AssertionError):
        local.register(push_eta_linear, name="foo")


def test_kernel_type_checks():
    with pytest.raises(AssertionError):
        Kernel(lambda: None)
    with pytest.raises(AssertionError):
        Kernel(push_eta_linear.pyccel_kernel, cuda_kernel=PyccelKernel(lambda: None))


# ------------------------
# CUDA argument classes
# ------------------------


@requires_cupy
def test_cuda_marker_args_reference_device_arrays():
    """The CUDA arguments hold the very same device arrays (no copies)."""
    markers, valid_mks, bc_type = device_marker_arrays()
    args = CudaMarkerArguments(markers, valid_mks, *MARKER_INDICES, bc_type)

    assert args.markers is markers and args.valid_mks is valid_mks and args.bc_type is bc_type
    assert args.n_threads == N_MARKERS
    assert len(args.values) == 14
    assert args.values[0] is markers
    assert all(isinstance(v, np.int32) for v in args.values[2:13])
    assert args.n_markers == N_MARKERS and args.n_cols == N_COLS
    assert args.first_init_idx == 8

    with pytest.raises(AttributeError, match="read-only"):
        args.markers = markers


@requires_cupy
def test_cuda_marker_args_reject_host_and_bad_arrays():
    """Host arrays are never copied to the device; wrong dtypes or layouts are not converted."""
    markers, valid_mks, bc_type = device_marker_arrays()

    with pytest.raises(TypeError, match="must be a CuPy array"):
        CudaMarkerArguments(markers.get(), valid_mks, *MARKER_INDICES, bc_type)
    with pytest.raises(TypeError, match="must be a CuPy array"):
        CudaMarkerArguments(markers, valid_mks, *MARKER_INDICES, bc_type.get())
    with pytest.raises(TypeError, match="dtype"):
        CudaMarkerArguments(markers.astype(np.float32), valid_mks, *MARKER_INDICES, bc_type)
    with pytest.raises(ValueError, match="C-contiguous"):
        CudaMarkerArguments(markers[:, ::2], valid_mks, *MARKER_INDICES, bc_type)


@requires_cupy
def test_cuda_derham_args():
    import cupy as cp

    knots = cp.array([0.0, 0.0, 1.0, 1.0])
    args = CudaDerhamArguments(cp.ones(3, dtype=int), knots, knots, knots, cp.zeros(3, dtype=int))
    assert len(args.values) == 5 and args.tn1 is knots


@requires_cupy
def test_domain_cuda_args():
    """Domain.cuda_args_domain references the device arrays of the domain."""
    with cunumpy.use_backend("cupy"):
        domain = Cuboid(l1=0.5, r1=2.0)
        args = domain.cuda_args_domain

        assert domain.cuda_args_domain is args  # cached
        assert args.t1 is domain.T[0] and args.ind3 is domain.indN[2]
        assert args.params is domain.params_numpy
        assert args.kind_map == domain.kind_map
        assert len(args.values) == 12

        # not deep-copied/pickled along with the domain, rebuilt from the copy's own arrays
        domain_copy = copy.deepcopy(domain)
        assert domain_copy.cuda_args_domain.t1 is domain_copy.T[0]

    # a domain created on the NumPy backend has host arrays
    with pytest.raises(TypeError, match="must be a CuPy array"):
        Cuboid().cuda_args_domain


@requires_cupy
def test_particles_cuda_args_markers():
    """Particles.cuda_args_markers references the marker arrays of the particles.

    Particles cannot be created on the CuPy backend yet, hence the arrays of a NumPy-backend
    instance are replaced by device arrays here.
    """
    import cupy as cp
    from feectools.ddm.mpi import mpi as MPI

    from struphy import LoadingParameters
    from struphy.pic.particles import Particles6D

    loading_params = LoadingParameters(Np=100, seed=1234, moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), spatial="uniform")
    particles = Particles6D(comm_world=MPI.COMM_WORLD, loading_params=loading_params, domain=Cuboid())
    particles.draw_markers()

    host_args = particles.args_markers
    particles._markers = cp.asarray(particles.markers)
    particles._valid_mks = cp.asarray(particles.valid_mks)
    particles._bc_type = cp.asarray(particles._bc_type)
    particles._cuda_args_markers = None

    args = particles.cuda_args_markers
    assert particles.cuda_args_markers is args  # cached
    assert args.markers is particles.markers
    assert args.valid_mks is particles.valid_mks
    assert args.bc_type is particles._bc_type
    for name in (
        "n_markers",
        "Np",
        "vdim",
        "weight_idx",
        "first_diagnostics_idx",
        "first_init_idx",
        "first_shift_idx",
        "residual_idx",
        "first_free_idx",
        "mu_idx",
    ):
        assert getattr(args, name) == getattr(host_args, name), name


# ------------------------
# pushing
# ------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_push_eta_linear(backend):
    """The kernel updates the owner's marker array in place."""
    dt = 0.1
    with cunumpy.use_backend(backend):
        args_markers, args_domain = make_demo_arguments(N_MARKERS)
        markers = args_markers.markers
        expected = expected_push(cunumpy.to_numpy(markers), cunumpy.to_numpy(args_markers.valid_mks), dt)

        push_eta_linear(dt, 0, args_markers, args_domain)

        if backend == "cupy":
            assert args_markers.markers is markers
        assert np.allclose(cunumpy.to_numpy(markers), expected, rtol=1e-14, atol=0.0)


@requires_cupy
def test_pyccel_cuda_agree():
    dt, n_steps = 0.01, 5
    results = {}
    for backend in ("numpy", "cupy"):
        with cunumpy.use_backend(backend):
            args_markers, args_domain = make_demo_arguments(N_MARKERS, seed=2)
            if backend == "numpy":
                expected = expected_push(args_markers.markers, args_markers.valid_mks, dt, n_steps)
            results[backend], _ = run_push_eta_linear(args_markers, args_domain, dt, n_steps)

    assert np.allclose(results["numpy"], expected, rtol=1e-13, atol=0.0)
    assert np.allclose(results["cupy"], results["numpy"], rtol=1e-14, atol=0.0)


@requires_cupy
def test_cuda_kernel_rejects_pyccel_args():
    """Passing the pyccel argument classes to the CUDA kernel fails instead of copying."""
    with cunumpy.use_backend("numpy"):
        host_markers, host_domain = make_demo_arguments(N_MARKERS)
    with cunumpy.use_backend("cupy"):
        cuda_markers, cuda_domain = make_demo_arguments(N_MARKERS)
        with pytest.raises(ValueError, match="number of CUDA threads"):
            push_eta_linear(0.1, 0, host_markers, cuda_domain)
        with pytest.raises(TypeError):
            push_eta_linear(0.1, 0, cuda_markers, host_domain)
