"""Proof of concept: 1:1 pyccel and CUDA kernels, dispatched by the cunumpy backend.

The pyccel kernel :func:`push_eta_linear` is compiled with ``epyccel`` at test time (``struphy compile``
skips test files); its CUDA counterpart is :data:`PUSH_ETA_LINEAR_SRC`. See ``CUDA_STRATEGY.md``.
"""

import importlib
import inspect
import sys

import cunumpy
import numpy as np
import pytest
from cunumpy import PyccelKernel

from struphy.geometry.domains import Cuboid
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments
from struphy.utils.cuda_arguments import CudaDomainArguments, CudaMarkerArguments
from struphy.utils.kernel_backends import CudaKernel, Kernel, KernelCatalog, is_cuda_backend

requires_cupy = pytest.mark.skipif(not cunumpy.cupy_available(), reason="CuPy/GPU not available")

N_COLS = 25
MARKER_INDICES = (3, 6, 7, 8, 14, 17, 18, 4)  # vdim, weight_idx, ..., mu_idx


# ---------------------------------
# the kernel pair: pyccel and CUDA
# ---------------------------------


def push_eta_linear(
    dt: float,
    stage: int,
    args_markers: "MarkerArguments",
    args_domain: "DomainArguments",
):
    """Explicit Euler step eta <- eta + dt * v for each valid marker (pyccel kernel)."""

    markers = args_markers.markers
    n_markers = args_markers.n_markers
    valid_mks = args_markers.valid_mks

    for ip in range(n_markers):
        # only do something if particle is valid (i.e. not a hole or ghost)
        if not valid_mks[ip]:
            continue

        markers[ip, 0] += dt * markers[ip, 3]
        markers[ip, 1] += dt * markers[ip, 4]
        markers[ip, 2] += dt * markers[ip, 5]


# Arguments: (dt, stage, CudaMarkerArguments, CudaDomainArguments), see struphy.utils.cuda_arguments.
CUDA_ARGS = r"""
    double dt, int stage,
    double* markers, bool* valid_mks, int n_markers, int n_cols,
    int Np, int vdim, int weight_idx, int first_diagnostics_idx, int first_init_idx,
    int first_shift_idx, int residual_idx, int first_free_idx, int mu_idx, long long* bc_type,
    int kind_map, double* params, long long* degree,
    double* t1, double* t2, double* t3,
    long long* ind1, long long* ind2, long long* ind3,
    double* cx, double* cy, double* cz
"""

PUSH_ETA_LINEAR_SRC = f"""
extern "C" __global__
void push_eta_linear({CUDA_ARGS})
{{
    int ip = blockDim.x * blockIdx.x + threadIdx.x;

    // only do something if particle is valid (i.e. not a hole or ghost)
    if (ip >= n_markers || !valid_mks[ip]) return;

    double* mk = markers + (long long)ip * n_cols;
    mk[0] += dt * mk[3];
    mk[1] += dt * mk[4];
    mk[2] += dt * mk[5];
}}
"""

# writes the scalar arguments into the markers, to check that they arrive with the right types
WRITE_SCALARS_SRC = f"""
extern "C" __global__
void write_scalars({CUDA_ARGS})
{{
    int ip = blockDim.x * blockIdx.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* mk = markers + (long long)ip * n_cols;
    mk[0] = dt;
    mk[1] = stage;
    mk[2] = n_cols;
    mk[3] = first_init_idx;
    mk[4] = mu_idx;
    mk[5] = kind_map;
}}
"""


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    """The Kernel pair, with the pyccel kernel compiled by epyccel."""
    from pyccel import epyccel

    src_dir = tmp_path_factory.mktemp("pyccel_src")
    src = "from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments\n\n\n"
    src += inspect.getsource(push_eta_linear)
    (src_dir / "poc_push_kernels.py").write_text(src)

    sys.path.insert(0, str(src_dir))
    try:
        module = epyccel(importlib.import_module("poc_push_kernels"), language="fortran")
    finally:
        sys.path.remove(str(src_dir))

    return Kernel(
        pyccel_kernel=PyccelKernel(module.push_eta_linear),
        cuda_kernel=CudaKernel(PUSH_ETA_LINEAR_SRC, "push_eta_linear"),
    )


def make_arguments(n_markers: int, seed: int = 0):
    """Random markers (some holes) and a Cuboid domain, as kernel arguments for the active cunumpy backend.

    The arrays are created on the active backend (on the device for CuPy); the arguments reference them without copies.
    """
    rng = np.random.default_rng(seed)
    markers = cunumpy.asarray(rng.random((n_markers, N_COLS)))
    valid_mks = cunumpy.asarray(rng.random(n_markers) > 0.1)
    bc_type = cunumpy.zeros(3, dtype=int)

    domain = Cuboid()
    if not is_cuda_backend():
        args_markers = MarkerArguments(markers, valid_mks, n_markers, *MARKER_INDICES, bc_type)
        return args_markers, domain.args_domain

    args_markers = CudaMarkerArguments(markers, valid_mks, n_markers, *MARKER_INDICES, bc_type)
    return args_markers, domain.cuda_args_domain


def expected_push(markers, valid_mks, dt, n_steps=1):
    out = markers.copy()
    for _ in range(n_steps):
        out[valid_mks, 0:3] += dt * out[valid_mks, 3:6]
    return out


BACKENDS = ["numpy", pytest.param("cupy", marks=requires_cupy)]


# ---------------------------------
# tests
# ---------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_kernel_dispatch(kernel, backend):
    with cunumpy.use_backend(backend):
        assert is_cuda_backend() == (backend == "cupy")
        expected = kernel.cuda_kernel if backend == "cupy" else kernel.pyccel_kernel
        assert kernel.get_kernel() is expected


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_markers", [1, 129, 1000])
def test_push_eta_linear(kernel, backend, n_markers):
    """One step, compared to the analytic result; holes are not touched. 129 is not a multiple of the block size."""
    dt = 0.1
    with cunumpy.use_backend(backend):
        args_markers, args_domain = make_arguments(n_markers)
        markers = args_markers.markers
        valid = cunumpy.to_numpy(args_markers.valid_mks)
        before = cunumpy.to_numpy(markers).copy()

        kernel(dt, 0, args_markers, args_domain, n_threads=n_markers)

        after = cunumpy.to_numpy(markers)
        assert np.allclose(after, expected_push(before, valid, dt), rtol=1e-14, atol=0.0)
        assert np.array_equal(after[~valid], before[~valid])


@requires_cupy
def test_pyccel_cuda_agree(kernel):
    """Same markers pushed for many steps on both backends (what used to be the demo)."""
    dt, n_steps, n_markers = 1e-3, 100, 100_000
    results = {}
    for backend in ("numpy", "cupy"):
        with cunumpy.use_backend(backend):
            args_markers, args_domain = make_arguments(n_markers, seed=1)
            if backend == "numpy":
                expected = expected_push(args_markers.markers, args_markers.valid_mks, dt, n_steps)
            for _ in range(n_steps):
                kernel(dt, 0, args_markers, args_domain, n_threads=n_markers)
            results[backend] = cunumpy.to_numpy(args_markers.markers)

    # not bitwise equal: nvcc contracts x + dt * v into fused multiply-adds by default
    assert np.allclose(results["numpy"], expected, rtol=1e-13, atol=0.0)
    assert np.allclose(results["cupy"], results["numpy"], rtol=1e-12, atol=0.0)


@requires_cupy
def test_cuda_kernel_updates_device_array_in_place(kernel):
    """The CUDA kernel works on the very array created on the device; nothing is replaced or copied."""
    with cunumpy.use_backend("cupy"):
        args_markers, args_domain = make_arguments(1000)
        markers = args_markers.markers
        ptr = markers.data.ptr

        for _ in range(10):
            kernel(0.1, 0, args_markers, args_domain, n_threads=1000)

        assert args_markers.markers is markers
        assert markers.data.ptr == ptr
        assert args_markers.values[0] is markers


@requires_cupy
def test_cuda_scalar_arguments():
    """Python scalars (not cast) and the flattened argument classes arrive in the CUDA kernel correctly and in order."""
    write_scalars = CudaKernel(WRITE_SCALARS_SRC, "write_scalars")
    with cunumpy.use_backend("cupy"):
        args_markers, args_domain = make_arguments(10)
        write_scalars(0.25, 3, args_markers, args_domain, n_threads=10)

        row = cunumpy.to_numpy(args_markers.markers)[0, :6]
        first_pusher_idx, mu_idx = MARKER_INDICES[3], MARKER_INDICES[7]
        assert np.array_equal(row, [0.25, 3, N_COLS, first_pusher_idx, mu_idx, Cuboid().kind_map])


@requires_cupy
def test_cuda_domain_arguments_reference_domain_arrays():
    with cunumpy.use_backend("cupy"):
        domain = Cuboid()
        args = domain.cuda_args_domain
        assert isinstance(args, CudaDomainArguments)
        assert len(args.values) == 12
        assert args.values[3] is domain.T[0] and args.values[8] is domain.indN[2] and args.values[9] is domain.cx


@requires_cupy
def test_cuda_arguments_reject_host_and_bad_arrays():
    """Host arrays are never copied to the device, and wrong dtypes or layouts are not converted."""
    import cupy as cp

    markers = cp.zeros((10, N_COLS))
    valid_mks = cp.ones(10, dtype=bool)
    bc_type = cp.zeros(3, dtype=int)

    CudaMarkerArguments(markers, valid_mks, 10, *MARKER_INDICES, bc_type)  # ok
    for bad_markers in (markers.get(), markers.astype(np.float32), cp.zeros((N_COLS, 10)).T):
        with pytest.raises(TypeError):
            CudaMarkerArguments(bad_markers, valid_mks, 10, *MARKER_INDICES, bc_type)
    with pytest.raises(TypeError):
        CudaMarkerArguments(markers, valid_mks.get(), 10, *MARKER_INDICES, bc_type)


@requires_cupy
def test_cuda_kernel_rejects_host_arrays(kernel):
    """Arrays are never converted: host arrays and the pyccel argument classes fail; n_threads is required."""
    with cunumpy.use_backend("numpy"):
        host_markers, host_domain = make_arguments(10)
    with cunumpy.use_backend("cupy"):
        cuda_markers, cuda_domain = make_arguments(10)
        with pytest.raises(TypeError):
            kernel(0.1, 0, host_markers, cuda_domain, n_threads=10)
        with pytest.raises(TypeError):
            kernel(0.1, 0, cuda_markers, host_domain, n_threads=10)
        with pytest.raises(ValueError, match="n_threads"):
            kernel(0.1, 0, cuda_markers, cuda_domain)


@requires_cupy
def test_cuda_kernel_from_file(kernel, tmp_path):
    """CUDA kernels can be loaded from <name>_cuda.cu files; the name is taken from the file name."""
    path = tmp_path / "push_eta_linear_cuda.cu"
    path.write_text(PUSH_ETA_LINEAR_SRC)
    cuda_kernel = CudaKernel.from_file(path)
    assert cuda_kernel.name == "push_eta_linear"

    results = {}
    for backend in ("numpy", "cupy"):
        with cunumpy.use_backend(backend):
            args_markers, args_domain = make_arguments(1000)
            if backend == "cupy":
                cuda_kernel(0.1, 0, args_markers, args_domain, n_threads=1000)
            else:
                kernel.pyccel_kernel(0.1, 0, args_markers, args_domain)
            results[backend] = cunumpy.to_numpy(args_markers.markers)
    assert np.allclose(results["cupy"], results["numpy"], rtol=1e-14, atol=0.0)

    with pytest.raises(AssertionError, match="naming convention"):
        CudaKernel.from_file(tmp_path / "push_eta_linear.cu")


@pytest.fixture
def catalog_package(tmp_path, monkeypatch):
    """A package with one folder per kernel: push_a has a CUDA version, push_b has not.

    The pyccel kernels are not compiled here (PyccelKernel also wraps plain Python functions).
    """
    root = tmp_path / "poc_catalog_pkg"
    header = "from struphy.kernel_arguments.pusher_args_kernels import DomainArguments, MarkerArguments\n\n\n"
    src = inspect.getsource(push_eta_linear)
    for name, cuda in (("push_a", True), ("push_b", False)):
        (root / name).mkdir(parents=True)
        (root / name / "__init__.py").write_text("")
        (root / name / f"{name}_kernels.py").write_text(header + src.replace("push_eta_linear", name))
        if cuda:
            (root / name / f"{name}_cuda.cu").write_text(PUSH_ETA_LINEAR_SRC.replace("push_eta_linear", name))
    (root / "not_a_kernel").mkdir()
    (root / "__init__.py").write_text(
        "from struphy.utils.kernel_backends import KernelCatalog\n\ncatalog = KernelCatalog.from_package(__name__)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield importlib.import_module("poc_catalog_pkg").catalog
    for mod in [m for m in sys.modules if m.startswith("poc_catalog_pkg")]:
        del sys.modules[mod]


def test_catalog_discovers_kernels(catalog_package):
    catalog = catalog_package
    assert catalog.names == ["push_a", "push_b"]
    assert "push_a" in catalog and "not_a_kernel" not in catalog
    assert catalog.missing_cuda == ["push_b"]
    assert catalog["push_a"].name == "push_a"
    assert catalog["push_a"].cuda_kernel.name == "push_a"
    assert catalog["push_b"].cuda_path.name == "push_b_cuda.cu"


@pytest.mark.parametrize("backend", BACKENDS)
def test_catalog_kernels_run(catalog_package, backend):
    """The kernels from the catalog push on both backends; a missing CUDA kernel raises on the GPU."""
    dt = 0.1
    with cunumpy.use_backend(backend):
        args_markers, args_domain = make_arguments(10)
        valid = cunumpy.to_numpy(args_markers.valid_mks)
        expected = expected_push(cunumpy.to_numpy(args_markers.markers), valid, dt)

        catalog_package["push_a"](dt, 0, args_markers, args_domain, n_threads=10)
        assert np.allclose(cunumpy.to_numpy(args_markers.markers), expected, rtol=1e-14, atol=0.0)

        if backend == "numpy":
            catalog_package["push_b"](dt, 0, args_markers, args_domain, n_threads=10)
        else:
            with pytest.raises(NotImplementedError, match="No CUDA version of kernel 'push_b'.*push_b_cuda.cu"):
                catalog_package["push_b"](dt, 0, args_markers, args_domain, n_threads=10)


def test_kernel_without_cuda_version():
    kernel = Kernel(PyccelKernel(push_eta_linear))
    assert kernel.name == "push_eta_linear"
    with cunumpy.use_backend("numpy"):
        assert kernel.get_kernel() is kernel.pyccel_kernel
    if cunumpy.cupy_available():
        with cunumpy.use_backend("cupy"), pytest.raises(NotImplementedError, match="push_eta_linear"):
            kernel.get_kernel()


def make_pusher(kernel):
    """Pusher for push_eta_stage (forward Euler) on 100 particles in a Cuboid."""
    from feectools.ddm.mpi import mpi as MPI

    from struphy import LoadingParameters
    from struphy.ode.utils import ButcherTableau
    from struphy.pic.particles import Particles6D
    from struphy.pic.pushing.pusher import Pusher

    domain = Cuboid()
    loading_params = LoadingParameters(Np=100, seed=1234, moments=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), spatial="uniform")
    particles = Particles6D(comm_world=MPI.COMM_WORLD, loading_params=loading_params, domain=domain)
    particles.draw_markers()
    butcher = ButcherTableau("forward_euler")
    return lambda: Pusher(
        particles,
        kernel,
        (butcher.a_stage, butcher.b, butcher.c),
        domain.args_domain,
        pushes_eta=True,
        alpha_in_kernel=1.0,
        n_stages=butcher.n_stages,
        local_eval_only=True,
    )


@pytest.mark.parametrize("wrap", [False, True])
def test_pusher_accepts_kernel(wrap):
    """Pusher takes a PyccelKernel (wrapped into a Kernel) or a Kernel, and runs the pyccel kernel on NumPy."""
    from struphy.pic.pushing import pusher_kernels

    pyccel_kernel = PyccelKernel(pusher_kernels.push_eta_stage)
    with cunumpy.use_backend("numpy"):
        pusher = make_pusher(Kernel(pyccel_kernel) if wrap else pyccel_kernel)()
        assert pusher.kernel is pyccel_kernel
        pusher(0.1)


@requires_cupy
def test_pusher_without_cuda_kernel_fails_at_setup():
    """On the CuPy backend, a pusher whose kernel has no CUDA version fails when it is created, not in the time loop."""
    from struphy.pic.pushing import pusher_kernels

    with cunumpy.use_backend("numpy"):
        create_pusher = make_pusher(PyccelKernel(pusher_kernels.push_eta_stage))
    with cunumpy.use_backend("cupy"), pytest.raises(NotImplementedError, match="push_eta_stage"):
        create_pusher()
