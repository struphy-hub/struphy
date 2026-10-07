"""Struphy's argument ABI and cunumpy integration (dispatch itself is tested upstream)."""

import ast
import copy
import inspect
import pickle
from pathlib import Path

import cunumpy
import numpy as np
import pytest
from cunumpy.arguments import CudaStruct
from cunumpy.kernel_testing import requires_cupy
from cunumpy.kernels import Kernel

import struphy
from struphy.geometry.domains import Cuboid
from struphy.kernel_arguments.local_projectors_args_cuda import CudaLocalProjectorsArguments
from struphy.kernel_arguments.pusher_args_cuda import CudaDerhamArguments, CudaDomainArguments, CudaMarkerArguments
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments, MarkerArguments
from struphy.kernel_arguments.spline_args_cuda import CudaSplineArguments
from struphy.kernel_arguments.spline_args_kernels import SplineArguments
from struphy.pic.tests.kernel_test_args import N_GEOMETRY_DOMAINS
from struphy.utils.cuda_arguments import (
    CUDA_OPTIONS,
    write_local_projectors_header,
    write_pusher_header,
    write_spline_header,
)

N_COLS = 25
MARKER_INDICES = (3, 6, 7, 8, 14, 17, 18, 4)
ARGS_DIR = Path(struphy.__file__).parent / "kernel_arguments"
HEADER = ARGS_DIR / "pusher_args.cuh"
LOCAL_PROJECTORS_HEADER = ARGS_DIR / "local_projectors_args.cuh"
SPLINE_HEADER = ARGS_DIR / "spline_args.cuh"

# (CUDA class, pyccel source, pyccel class, struct members that only the CUDA class has, its header)
ARGUMENT_PAIRS = (
    (CudaMarkerArguments, "pusher_args_kernels.py", "MarkerArguments", {"n_markers"}, HEADER),
    (CudaDerhamArguments, "pusher_args_kernels.py", "DerhamArguments", {"nt1", "nt2", "nt3"}, HEADER),
    (CudaDomainArguments, "pusher_args_kernels.py", "DomainArguments", set(), HEADER),
    (
        CudaLocalProjectorsArguments,
        "local_projectors_args_kernels.py",
        "LocalProjectorsArguments",
        set(),
        LOCAL_PROJECTORS_HEADER,
    ),
    (CudaSplineArguments, "spline_args_kernels.py", "SplineArguments", set(), SPLINE_HEADER),
)


def make_arguments(n_markers, seed=0):
    rng = np.random.default_rng(seed)
    markers = cunumpy.asarray(rng.random((n_markers, N_COLS)))
    valid = cunumpy.asarray(rng.random(n_markers) > 0.1)
    bc = cunumpy.zeros(3, dtype=np.int64)
    args_class = CudaMarkerArguments if cunumpy.get_backend() == "cupy" else MarkerArguments
    return args_class(markers, valid, n_markers, *MARKER_INDICES, bc), Cuboid().args_domain


def make_pusher(kernel):
    """Pusher for push_eta_stage (forward Euler) on 100 particles in a Cuboid."""
    from maybempi import MPI

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
        (butcher.a_stage, butcher.b, butcher.c, butcher.n_stages),
        domain.args_domain,
        pushes_eta=True,
        alpha_in_kernel=1.0,
        n_stages=butcher.n_stages,
        local_eval_only=True,
    )


def pyccel_init_parameters(source, class_name):
    """Constructor parameter names of a pyccel class, read from its source (pyccel classes are compiled)."""
    tree = ast.parse((ARGS_DIR / source).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    return [arg.arg for arg in init.args.args[1:]]


def test_generated_header(tmp_path):
    assert HEADER.read_text() == write_pusher_header(tmp_path / "pusher_args.cuh")


def test_generated_local_projectors_header(tmp_path):
    assert LOCAL_PROJECTORS_HEADER.read_text() == write_local_projectors_header(tmp_path / "local_projectors_args.cuh")


def test_generated_spline_header(tmp_path):
    assert SPLINE_HEADER.read_text() == write_spline_header(tmp_path / "spline_args.cuh")


@pytest.mark.parametrize("cuda_class, source, class_name, cuda_only, header", ARGUMENT_PAIRS)
def test_argument_classes_correspond(cuda_class, source, class_name, cuda_only, header):
    """Each CUDA argument class mirrors its pyccel class: same constructor, same attributes in the same order."""
    assert cuda_class.__name__ == "Cuda" + class_name
    cuda_parameters = list(inspect.signature(cuda_class.__init__).parameters)[1:]
    assert cuda_parameters == pyccel_init_parameters(source, class_name)
    # attributes set from constructor arguments; derived ones (e.g. n_markers, bn1) are skipped by the parser
    pyccel_fields = [field.name for field in CudaStruct.from_pyccel_class(ARGS_DIR / source, class_name).fields]
    cuda_fields = [field.name for field in cuda_class.struct.fields]
    assert [name for name in cuda_fields if name not in cuda_only] == pyccel_fields


def test_owners_select_pyccel_classes_on_numpy():
    from maybempi import MPI

    from struphy import LoadingParameters
    from struphy.feec.tests.test_derham_gpu import make_derham
    from struphy.pic.particles import Particles6D

    with cunumpy.use_backend("numpy"):
        domain = Cuboid()
        particles = Particles6D(
            comm_world=MPI.COMM_WORLD, loading_params=LoadingParameters(Np=10, seed=1234), domain=domain
        )
        derham = make_derham()
        assert type(domain.args_domain) is DomainArguments
        assert type(particles.args_markers) is MarkerArguments
        assert type(derham.args_derham) is DerhamArguments
        for space in ("H1", "Hcurl"):
            spline = derham.create_spline_function("f", space)
            assert all(type(args) is SplineArguments for args in spline._args_spline)


@requires_cupy
@pytest.mark.parametrize("cuda_class, source, class_name, cuda_only, header", ARGUMENT_PAIRS)
def test_cuda_struct_layout(cuda_class, source, class_name, cuda_only, header):
    include = f"struphy/kernel_arguments/{header.name}"
    cuda_class.struct.verify_layout(include, include_dirs=CUDA_OPTIONS["include_dirs"])


@requires_cupy
def test_cuda_classes_reject_host_arrays():
    with cunumpy.use_backend("numpy"):
        markers = np.zeros((10, N_COLS))
        with pytest.raises(TypeError, match="CuPy array"):
            CudaMarkerArguments(markers, np.ones(10, dtype=bool), 10, *MARKER_INDICES, np.zeros(3, dtype=np.int64))


@requires_cupy
def test_device_bundle_copies_rebuild():
    with cunumpy.use_backend("cupy"):
        m, _ = make_arguments(10)
        for other in (copy.deepcopy(m), pickle.loads(pickle.dumps(m))):
            assert other.markers is not m.markers
            assert other.packed["markers"]["data"] == other.markers.data.ptr


@requires_cupy
def test_device_replacement_repacked():
    with cunumpy.use_backend("cupy"):
        m, _ = make_arguments(10)
        m.markers = m.markers.copy()
        view = m.packed["markers"]
        assert view["data"] == m.markers.data.ptr
        assert tuple(view["shape"]) == m.markers.shape
        assert tuple(view["strides"]) == tuple(s // m.markers.itemsize for s in m.markers.strides)


@requires_cupy
def test_scalar_types_checked():
    with cunumpy.use_backend("cupy"):
        m, _ = make_arguments(10)
        m.Np = 1.5
        with pytest.raises(TypeError):
            m.pack()
        m.Np = 2**31
        with pytest.raises(OverflowError):
            m.pack()


def test_catalog_signatures():
    from struphy.pic.tests.test_cuda_parity import CATALOGS

    for catalog in CATALOGS.values():
        catalog.check_signatures()
    assert {package.split(".", 1)[1]: len(catalog) for package, catalog in CATALOGS.items()} == {
        "pic.pushing.kernels": 44,
        "pic.accumulation.kernels": 16,
        "pic.diagnostics.kernels": 10,
        "pic.sph.kernels": 4,
        "bsplines.kernels": 3,
        "geometry.kernels": 4,
        "feec.kernels": 17,
        "feec.local_projectors.kernels": 8,
    }


@pytest.mark.parametrize("wrap", [False, True])
def test_pusher_accepts_kernel(wrap):
    from struphy.pic.pushing.kernels.push_eta_stage import push_eta_stage

    with cunumpy.use_backend("numpy"):
        host = push_eta_stage.host_kernel
        pusher = make_pusher(Kernel(host) if wrap else host)()
        pusher(0.001)


def test_spline_mappings_checked_only_on_cupy():
    """Spline mappings (kind_map < 10) have no CUDA version yet (PR 19); the check passes on NumPy."""
    from struphy.utils.cuda_arguments import check_mapping_on_device

    with cunumpy.use_backend("numpy"):
        for kind_map in (0, 1, 2, 10, 22, 32):
            check_mapping_on_device(kind_map, "Geometry evaluations")


@requires_cupy
@pytest.mark.parametrize("domain_index", range(N_GEOMETRY_DOMAINS))
def test_geometry_evaluation_on_cupy(domain_index):
    """Geometry evaluations run on CuPy for every analytic mapping and four spline mappings and agree with NumPy."""
    from struphy.pic.tests.kernel_test_args import geometry_domain

    markers = np.random.default_rng(3).uniform(-0.1, 1.0, (50, 3))
    eta = (np.linspace(0.1, 0.9, 5), np.linspace(0.0, 1.0, 4), np.linspace(0.2, 0.8, 3))
    results = []
    for backend in ("numpy", "cupy"):
        with cunumpy.use_backend(backend):
            domain = geometry_domain(domain_index)
            device_markers = cunumpy.asarray(markers)
            device_eta = tuple(cunumpy.asarray(e) for e in eta)
            values = (
                domain.jacobian_det(device_markers),
                domain.jacobian_inv(device_markers, remove_outside=False),
                domain.metric(*device_eta),
                domain.pull(lambda x, y, z: x * y + z, device_markers, kind="3"),
                domain.push((1.0, 2.0, 3.0), *device_eta, kind="2"),
            )
            results.append([cunumpy.to_numpy(v) for v in values])
    for host, device in zip(*results):
        np.testing.assert_allclose(device, host, rtol=1e-10, atol=1e-10)
