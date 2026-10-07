"""Struphy's argument ABI and cunumpy integration (dispatch itself is tested upstream)."""

import copy
import pickle
from pathlib import Path

import cunumpy
import numpy as np
import pytest
from cunumpy.cuda import CudaKernel
from cunumpy.kernel_testing import requires_cupy
from cunumpy.kernels import Kernel, PyccelKernel

import struphy
from struphy.geometry.domains import Cuboid
from struphy.utils.cuda_arguments import (
    CUDA_OPTIONS,
    CudaDerhamArguments,
    CudaDomainArguments,
    CudaMarkerArguments,
    write_pusher_header,
)

N_COLS = 25
MARKER_INDICES = (3, 6, 7, 8, 14, 17, 18, 4)
STRUCT_CLASSES = (CudaMarkerArguments, CudaDerhamArguments, CudaDomainArguments)
HEADER = Path(struphy.__file__).parent / "kernel_arguments" / "pusher_args.cuh"


def make_arguments(n_markers, seed=0):
    rng = np.random.default_rng(seed)
    markers = cunumpy.asarray(rng.random((n_markers, N_COLS)))
    valid = cunumpy.asarray(rng.random(n_markers) > 0.1)
    bc = cunumpy.zeros(3, dtype=np.int64)
    return CudaMarkerArguments(markers, valid, n_markers, *MARKER_INDICES, bc), Cuboid().args_domain


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
        (butcher.a_stage, butcher.b, butcher.c, butcher.n_stages),
        domain.args_domain,
        pushes_eta=True,
        alpha_in_kernel=1.0,
        n_stages=butcher.n_stages,
        local_eval_only=True,
    )


def test_generated_header(tmp_path):
    assert HEADER.read_text() == write_pusher_header(tmp_path / "pusher_args.cuh")


def test_catalog_signatures():
    from struphy.pic.pushing.kernels import catalog

    catalog.check_signatures()
    assert len(catalog) == 43


def same_buffer(a, b):
    return a.__array_interface__["data"][0] == b.__array_interface__["data"][0] and a.shape == b.shape


def test_host_bundle_references_owner_arrays():
    with cunumpy.use_backend("numpy"):
        markers, domain = make_arguments(10)
        for bundle in (markers, domain):
            host = bundle.__host_args__()
            assert bundle.__host_args__() is host
            for name in bundle.host_fields:
                value = getattr(bundle, name)
                if isinstance(value, np.ndarray):
                    # pyccel getters return a new view of the same buffer
                    assert same_buffer(getattr(host, name), value)


def test_host_bundle_copies_rebuild():
    with cunumpy.use_backend("numpy"):
        m, _ = make_arguments(10)
        for other in (copy.deepcopy(m), pickle.loads(pickle.dumps(m))):
            assert other.markers is not m.markers
            assert same_buffer(other.__host_args__().markers, other.markers)
            np.testing.assert_array_equal(other.markers, m.markers)


@requires_cupy
@pytest.mark.parametrize("cls", STRUCT_CLASSES)
def test_cuda_struct_layout(cls):
    cls.struct.verify_layout("struphy/kernel_arguments/pusher_args.cuh", include_dirs=CUDA_OPTIONS["include_dirs"])


@requires_cupy
def test_device_bundle_copies_rebuild():
    with cunumpy.use_backend("cupy"):
        m, _ = make_arguments(10)
        for other in (copy.deepcopy(m), pickle.loads(pickle.dumps(m))):
            assert other.markers is not m.markers
            assert other.packed["markers"]["data"] == other.markers.data.ptr
        with pytest.raises(RuntimeError, match="device array"):
            m.__host_args__()


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


@pytest.mark.parametrize("wrap", [False, True])
def test_pusher_accepts_kernel(wrap):
    from struphy.pic.pushing.kernels import catalog

    with cunumpy.use_backend("numpy"):
        host = catalog["push_eta_stage"].host_kernel
        pusher = make_pusher(Kernel(host) if wrap else host)()
        pusher(0.001)


@requires_cupy
def test_missing_cuda_rejected_at_setup():
    from struphy.pic.pushing.kernels import catalog

    with cunumpy.use_backend("cupy"), pytest.raises(NotImplementedError):
        catalog["push_gc_cc_J1_Hdiv"].get_kernel()
