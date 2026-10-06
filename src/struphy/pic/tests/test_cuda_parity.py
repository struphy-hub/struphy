"""Parity of every CUDA kernel with its pyccel kernel, from the ``<name>_test_args.py`` modules of the folders."""

import importlib

import cunumpy as xp
import numpy as np
import pytest
from cunumpy.kernel_testing import check_parity, parity_cases, requires_cupy
from cunumpy.kernels import Kernel, KernelCatalog
from cunumpy.profiling import assert_no_transfers

from struphy.pic.accumulation.kernels.charge_density_0form import charge_density_0form
from struphy.pic.pushing.kernels.push_eta_stage import push_eta_stage
from struphy.pic.pushing.kernels.push_v_with_efield import push_v_with_efield
from struphy.pic.pushing.kernels.push_vxb_analytic import push_vxb_analytic
from struphy.pic.pushing.kernels.push_vxb_implicit import push_vxb_implicit
from struphy.pic.pushing.kernels.push_weights_with_efield_lin_va import push_weights_with_efield_lin_va
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
CUDA_CASES = [case for catalog in CATALOGS.values() for case in parity_cases(catalog)]


def test_folders_declare_their_kernels():
    """Each kernel folder's __init__.py declares its kernel under the folder name (the import used in the code)."""
    for package, catalog in CATALOGS.items():
        for name in catalog:
            kernel = getattr(importlib.import_module(f"{package}.{name}"), name)
            assert isinstance(kernel, Kernel) and kernel.name == name
            assert kernel.has_cuda == catalog[name].has_cuda


def test_cuda_kernels_have_test_args():
    """Every kernel with a CUDA version has a <name>_test_args.py with make_args and CASES."""
    for catalog in CATALOGS.values():
        for name, kernel in catalog.parity_cases():
            assert kernel.test_args_module is not None, f"add {name}_test_args.py to the folder of {name}"
            assert callable(kernel.test_args.make_args) and len(kernel.test_args.CASES) > 0


@requires_cupy
@pytest.mark.parametrize("kernel", CUDA_CASES)
def test_parity(kernel):
    for seed in range(len(kernel.test_args.CASES)):
        check_parity(kernel, seed=seed)


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
    """Vlasov (PushEta, both PushVxB algorithms) and the ported Vlasov-Ampere kernels have device kernels."""
    for kernel in (
        push_eta_stage,
        push_vxb_analytic,
        push_vxb_implicit,
        push_v_with_efield,
        push_weights_with_efield_lin_va,
        charge_density_0form,
    ):
        assert kernel.has_cuda, kernel.name
