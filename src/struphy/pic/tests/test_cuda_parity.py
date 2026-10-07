"""Parity of every CUDA kernel with its pyccel kernel, on the cases of ``cuda_parity_cases.PARITY_CASES``."""

import importlib

import cunumpy as xp
import numpy as np
import pytest
from cunumpy.kernel_testing import assert_kernels_agree, requires_cupy
from cunumpy.kernels import Kernel, KernelCatalog
from cunumpy.profiling import assert_no_transfers

from struphy.pic.accumulation.kernels.charge_density_0form import charge_density_0form
from struphy.pic.accumulation.kernels.linear_vlasov_ampere import linear_vlasov_ampere
from struphy.pic.pushing.kernels.push_eta_stage import push_eta_stage
from struphy.pic.pushing.kernels.push_v_with_efield import push_v_with_efield
from struphy.pic.pushing.kernels.push_vxb_analytic import push_vxb_analytic
from struphy.pic.pushing.kernels.push_vxb_implicit import push_vxb_implicit
from struphy.pic.pushing.kernels.push_weights_with_efield_lin_va import push_weights_with_efield_lin_va
from struphy.pic.tests.cuda_parity_cases import PARITY_CASES
from struphy.utils.cuda_arguments import CUDA_STRUCTS

# all kernel packages (one folder per kernel), for tests that go through every kernel
PACKAGES = (
    "struphy.pic.pushing.kernels",
    "struphy.pic.accumulation.kernels",
    "struphy.pic.diagnostics.kernels",
    "struphy.pic.sph.kernels",
    "struphy.bsplines.kernels",
    "struphy.geometry.kernels",
    "struphy.feec.kernels",
    "struphy.feec.local_projectors.kernels",
)
CATALOGS = {package: KernelCatalog.from_package(package, structs=CUDA_STRUCTS) for package in PACKAGES}
# every kernel with a CUDA version, by name
CUDA_KERNELS = {name: kernel for catalog in CATALOGS.values() for name, kernel in catalog.parity_cases()}
# (kernel name, case index) of every parity case, as pytest parameters
PARITY_PARAMS = [
    pytest.param(name, index, id=f"{name}-{index}")
    for name in PARITY_CASES
    for index in range(len(PARITY_CASES[name].cases))
]


def check_case(name, index, compare):
    """Run case `index` of kernel `name` with `compare(kernel, make_args, **settings)` (cunumpy's signature)."""
    spec = PARITY_CASES[name]
    case = spec.cases[index]
    return compare(
        CUDA_KERNELS[name],
        lambda backend, seed: spec.build(case),
        n_threads=spec.n_threads,
        rtol=spec.rtol,
        atol=spec.atol,
    )


def test_folders_declare_their_kernels():
    """Each kernel folder's __init__.py declares its kernel under the folder name (the import used in the code)."""
    for package, catalog in CATALOGS.items():
        for name in catalog:
            kernel = getattr(importlib.import_module(f"{package}.{name}"), name)
            assert isinstance(kernel, Kernel) and kernel.name == name
            assert kernel.has_cuda == catalog[name].has_cuda


def test_cuda_kernels_have_parity_cases():
    """Every kernel with a CUDA version has parity cases (add them to cuda_parity_cases.PARITY_CASES), and only those."""
    assert set(PARITY_CASES) == set(CUDA_KERNELS)
    assert all(len(spec.cases) > 0 for spec in PARITY_CASES.values())


@requires_cupy
@pytest.mark.parametrize("name, index", PARITY_PARAMS)
def test_parity(name, index):
    check_case(name, index, assert_kernels_agree)


@requires_cupy
def test_device_pusher_time_loop(monkeypatch):
    """Single-rank device push; setup and result inspection are outside the guard."""
    import cupy as cp

    from struphy.pic.tests.test_kernel_backends import make_pusher

    with xp.use_backend("cupy"):
        pusher = make_pusher(push_eta_stage)()
        if pusher.particles.mpi_size != 1:
            pytest.skip("Single-rank transfer guard; multi-rank exchange needs CUDA-aware MPI")
        pusher(0.001)  # warm up NVRTC and CuPy operations
        before = cp.asnumpy(pusher.particles.markers).copy()
        valid_mks = cp.asnumpy(pusher.particles.valid_mks).copy()
        original = cp.asarray

        def device_only(value, *args, **kwargs):
            assert isinstance(value, cp.ndarray), "Host array conversion inside time loop"
            return original(value, *args, **kwargs)

        with assert_no_transfers(), monkeypatch.context() as patch:
            patch.setattr(cp, "asarray", device_only)
            for _ in range(5):
                pusher(0.001)
        after = cp.asnumpy(pusher.particles.markers)
        np.testing.assert_allclose(
            after[valid_mks, :3],
            (before[valid_mks, :3] + 0.005 * before[valid_mks, 3:6]) % 1.0,
            atol=1e-13,
        )
        np.testing.assert_array_equal(after[~valid_mks], before[~valid_mks])


def test_vlasov_kernel_coverage():
    """Vlasov (PushEta, both PushVxB algorithms) and the ported (linear) Vlasov-Ampere kernels have device kernels."""
    for kernel in (
        push_eta_stage,
        push_vxb_analytic,
        push_vxb_implicit,
        push_v_with_efield,
        push_weights_with_efield_lin_va,
        charge_density_0form,
        linear_vlasov_ampere,
    ):
        assert kernel.has_cuda, kernel.name
