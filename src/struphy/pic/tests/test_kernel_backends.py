"""Dispatch between pyccel and CUDA kernels, and transformation of kernel arguments."""

import cunumpy
import numpy as np
import pytest
from cunumpy import PyccelKernel

from struphy.geometry.domains import Cuboid
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, MarkerArguments
from struphy.pic.pushing.demo_cuda import push_eta_linear
from struphy.utils.kernel_backends import Kernel, KernelCatalog, catalog, is_cuda_backend
from struphy.utils.kernel_transform import CudaMarkerArguments, _cupy, transform

requires_cupy = pytest.mark.skipif(not cunumpy.cupy_available(), reason="CuPy/GPU not available")

BACKENDS = ["numpy", pytest.param("cupy", marks=requires_cupy)]

N_MARKERS = 1000
N_COLS = 25


def make_marker_args(seed=0):
    rng = np.random.default_rng(seed)
    markers = rng.random((N_MARKERS, N_COLS))
    valid_mks = rng.random(N_MARKERS) > 0.1  # some holes/ghosts
    return MarkerArguments(markers, valid_mks, N_MARKERS, 3, 6, 7, 8, 14, 17, 18, 4, np.zeros(3, dtype=int))


@pytest.fixture
def domain_args():
    return Cuboid().args_domain


def expected_push(markers, valid_mks, dt):
    out = markers.copy()
    out[valid_mks, 0:3] += dt * markers[valid_mks, 3:6]
    return out


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


def test_push_eta_linear_pyccel(domain_args):
    dt = 0.1
    args_markers = make_marker_args()
    expected = expected_push(args_markers.markers, args_markers.valid_mks, dt)

    with cunumpy.use_backend("numpy"):
        push_eta_linear(dt, 0, args_markers, domain_args)

    assert np.allclose(args_markers.markers, expected, rtol=1e-14, atol=0.0)


@requires_cupy
def test_push_eta_linear_cuda(domain_args):
    dt = 0.1
    args_markers = make_marker_args()
    expected = expected_push(args_markers.markers, args_markers.valid_mks, dt)

    # transform once at setup, the kernel works on the device arrays only
    cuda_markers = transform(args_markers)
    cuda_domain = transform(domain_args)

    with cunumpy.use_backend("cupy"):
        push_eta_linear(dt, 0, cuda_markers, cuda_domain)

    assert np.allclose(cuda_markers.markers.get(), expected, rtol=1e-14, atol=0.0)


@requires_cupy
def test_pyccel_cuda_agree(domain_args):
    dt = 0.05
    args_pyccel = make_marker_args(seed=1)
    cuda_markers = transform(make_marker_args(seed=1))
    cuda_domain = transform(domain_args)

    for _ in range(3):
        with cunumpy.use_backend("numpy"):
            push_eta_linear(dt, 0, args_pyccel, domain_args)
        with cunumpy.use_backend("cupy"):
            push_eta_linear(dt, 0, cuda_markers, cuda_domain)

    assert np.allclose(args_pyccel.markers, cuda_markers.markers.get(), rtol=1e-14, atol=0.0)


@requires_cupy
def test_cuda_kernel_rejects_untransformed_args(domain_args):
    """Host arrays are never moved to the device at call time."""
    with cunumpy.use_backend("cupy"):
        with pytest.raises(ValueError, match="number of CUDA threads"):
            push_eta_linear(0.1, 0, make_marker_args(), domain_args)
        with pytest.raises(TypeError):
            push_eta_linear.cuda_kernel(0.1, 0, transform(make_marker_args()), domain_args)


@requires_cupy
def test_transform_marker_args():
    import cupy as cp

    args_markers = make_marker_args()
    out = transform(args_markers)

    assert isinstance(out, CudaMarkerArguments)
    assert len(out.values) == 14
    assert out.n_threads == N_MARKERS
    assert isinstance(out.markers, cp.ndarray) and out.markers.flags.c_contiguous
    assert out.markers.dtype == np.float64 and out.valid_mks.dtype == np.bool_
    assert np.array_equal(out.markers.get(), args_markers.markers)
    assert all(isinstance(v, np.int32) for v in out.values[2:13])
    assert out.n_markers == N_MARKERS and out.n_cols == N_COLS
    assert out.bc_type.dtype == np.int64
    with pytest.raises(AttributeError):
        out.markers = None  # frozen, values stay consistent


@requires_cupy
def test_transform_does_not_copy_device_arrays():
    import cupy as cp

    x = cp.zeros((4, 3))
    assert _cupy(x, np.float64) is x


@requires_cupy
def test_transform_domain_and_derham_args(domain_args):
    import cupy as cp

    out = transform(domain_args)
    assert len(out.values) == 12 and out.n_threads is None
    assert isinstance(out.kind_map, np.int32) and out.kind_map == domain_args.kind_map
    assert all(isinstance(v, cp.ndarray) for v in out.values[1:])

    knots = np.array([0.0, 0.0, 1.0, 1.0])
    args_derham = DerhamArguments(np.ones(3, dtype=int), knots, knots, knots, np.zeros(3, dtype=int))
    out = transform(args_derham)
    assert len(out.values) == 5
    assert out.pn.dtype == np.int64 and out.tn1.dtype == np.float64


def test_transform_unknown_type():
    with pytest.raises(TypeError):
        transform(np.zeros(3))


def test_kernel_type_checks():
    with pytest.raises(AssertionError):
        Kernel(lambda: None)
    with pytest.raises(AssertionError):
        Kernel(push_eta_linear.pyccel_kernel, cuda_kernel=PyccelKernel(lambda: None))


@requires_cupy
def test_demo_run_push_eta_linear():
    from struphy.pic.pushing.demo_cuda import make_demo_arguments, run_push_eta_linear

    dt, n_steps = 0.01, 5
    args_markers, args_domain = make_demo_arguments(N_MARKERS, seed=2)
    expected = args_markers.markers.copy()
    for _ in range(n_steps):
        expected = expected_push(expected, args_markers.valid_mks, dt)

    markers_pyccel, _ = run_push_eta_linear("numpy", args_markers, args_domain, dt, n_steps)
    args_markers, args_domain = make_demo_arguments(N_MARKERS, seed=2)
    markers_cuda, _ = run_push_eta_linear("cupy", args_markers, args_domain, dt, n_steps)

    assert np.allclose(markers_pyccel, expected, rtol=1e-13, atol=0.0)
    assert np.allclose(markers_cuda, expected, rtol=1e-13, atol=0.0)
