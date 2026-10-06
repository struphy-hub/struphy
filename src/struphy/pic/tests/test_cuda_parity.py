"""Every CUDA pusher must have a parity argument factory here."""

import cunumpy as xp
import numpy as np
import pytest

from struphy.ode.utils import ButcherTableau
from struphy.pic.pushing.kernels import catalog
from struphy.pic.tests.test_kernel_backends import make_arguments, requires_cupy


def eta_arguments(bc, method):
    m, d = make_arguments(129)
    rng = np.random.default_rng(7)
    markers = rng.random((129, 25))
    markers[:, 3:6] = rng.uniform(-2, 2, (129, 3))
    markers[:, 8:14] = markers[:, :6]
    markers[:, 18:21] = 0.0
    markers[0, 8] = -1.0
    markers[1, -1] = -2.0
    m.markers[:] = xp.asarray(markers)
    m.bc_type[:] = xp.asarray(bc, dtype=np.int64)
    butcher = ButcherTableau(method)
    return (m, d, xp.asarray(butcher.a_stage), xp.asarray(butcher.b), xp.asarray(butcher.c), butcher.n_stages)


def field_arguments(bc, method, electric=False):
    from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments
    from struphy.utils.cuda_arguments import CudaDerhamArguments

    m, d, *_ = eta_arguments(bc, method)
    degree = np.array([2, 3, 1], dtype=np.int64)
    knots = [np.r_[np.zeros(p), np.linspace(0, 1, 9), np.ones(p)] for p in degree]
    cls = CudaDerhamArguments if xp.get_backend() == "cupy" else DerhamArguments
    a = cls(xp.asarray(degree), *(xp.asarray(t) for t in knots), xp.zeros(3, dtype=np.int64))
    rng = np.random.default_rng(11)
    coeffs = tuple(xp.asarray(rng.normal(size=(18, 20, 16))) for _ in range(3))
    return (m, d, a, *coeffs, 0.7) if electric else (m, d, a, *coeffs)


FACTORIES = {
    "push_eta_stage": eta_arguments,
    "push_vxb_analytic": field_arguments,
    "push_vxb_implicit": field_arguments,
}
CUDA_NAMES = [n for n in catalog.names if n not in catalog.missing_cuda]


def test_cuda_factories_cover_catalog():
    assert set(CUDA_NAMES) == set(FACTORIES)


@requires_cupy
@pytest.mark.parametrize("name", CUDA_NAMES)
@pytest.mark.parametrize("bc", [(0, 0, 0), (1, 1, 1), (2, 0, 1)])
@pytest.mark.parametrize("method", ["forward_euler", "rk4"])
def test_catalog_parity(name, bc, method):
    results = []
    for backend in ("numpy", "cupy"):
        with xp.use_backend(backend):
            args = FACTORIES[name](bc, method)
            for stage in range(args[-1] if name == "push_eta_stage" else 1):
                catalog[name](0.2, stage, *args, n_threads=129)
            # Compare the mutable bundle and every explicit array argument.
            results.append(
                [
                    xp.to_numpy(a).copy()
                    for a in (args[0].markers, args[0].valid_mks, args[0].bc_type, *args[2:])
                    if hasattr(a, "shape")
                ]
            )
    for host, device in zip(*results):
        np.testing.assert_allclose(host, device, rtol=1e-13, atol=1e-14)


@requires_cupy
def test_device_pusher_time_loop(monkeypatch):
    """Single-rank device push; setup and result inspection are outside the guard."""
    import cupy as cp

    from struphy.pic.tests.test_kernel_backends import make_pusher

    with xp.use_backend("cupy"):
        pusher = make_pusher(catalog["push_eta_stage"])()
        if pusher.particles.mpi_size != 1:
            pytest.skip("Single-rank transfer guard; multi-rank exchange needs CUDA-aware MPI")
        pusher(0.001)  # warm up NVRTC and CuPy operations
        before = cp.asnumpy(pusher.particles.markers).copy()
        valid_mks = cp.asnumpy(pusher.particles.valid_mks).copy()
        original = cp.asarray

        def device_only(value, *args, **kwargs):
            assert isinstance(value, cp.ndarray), "Host array conversion inside time loop"
            return original(value, *args, **kwargs)

        with monkeypatch.context() as patch:
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
    """Both selectable PushVxB algorithms and PushEta must have device kernels."""
    for name in ("push_eta_stage", "push_vxb_analytic", "push_vxb_implicit"):
        assert catalog[name].cuda_kernel is not None
