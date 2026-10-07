"""Every CUDA pusher and accumulation kernel must have a parity argument factory here."""

import cunumpy as xp
import numpy as np
import pytest
from cunumpy.kernel_testing import assert_kernels_agree
from cunumpy.profiling import assert_no_transfers

from struphy.ode.utils import ButcherTableau
from struphy.pic.accumulation.kernels import catalog as accum_catalog
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


def field_arguments(bc, method):
    from struphy.utils.cuda_arguments import CudaDerhamArguments

    m, d, *_ = eta_arguments(bc, method)
    degree = np.array([2, 3, 1], dtype=np.int64)
    knots = [np.r_[np.zeros(p), np.linspace(0, 1, 9), np.ones(p)] for p in degree]
    a = CudaDerhamArguments(xp.asarray(degree), *(xp.asarray(t) for t in knots), xp.zeros(3, dtype=np.int64))
    rng = np.random.default_rng(11)
    coeffs = tuple(xp.asarray(rng.normal(size=(18, 20, 16))) for _ in range(3))
    return (m, d, a, *coeffs)


def efield_arguments(bc, method):
    return (*field_arguments(bc, method), 0.7)


def weights_arguments(bc, method):
    f0_values = xp.asarray(np.random.default_rng(13).random(129))
    return (*field_arguments(bc, method), f0_values, 1.3, 0.8)


FACTORIES = {
    "push_eta_stage": eta_arguments,
    "push_vxb_analytic": field_arguments,
    "push_vxb_implicit": field_arguments,
    "push_v_with_efield": efield_arguments,
    "push_weights_with_efield_lin_va": weights_arguments,
}
CUDA_NAMES = [name for name, kernel in catalog.parity_cases()]


def density_arguments(bc):
    """Accumulation into a 0-form vector; the arguments are markers, Derham, domain and the vector."""
    m, d, a, *_ = field_arguments(bc, "forward_euler")
    return (m, a, d, xp.zeros((18, 20, 16)))


ACCUM_FACTORIES = {
    "charge_density_0form": density_arguments,
}
ACCUM_CUDA_NAMES = [name for name, kernel in accum_catalog.parity_cases()]


def test_cuda_factories_cover_catalog():
    assert set(CUDA_NAMES) == set(FACTORIES)
    assert set(ACCUM_CUDA_NAMES) == set(ACCUM_FACTORIES)


@requires_cupy
@pytest.mark.parametrize("name", CUDA_NAMES)
@pytest.mark.parametrize("bc", [(0, 0, 0), (1, 1, 1), (2, 0, 1)])
@pytest.mark.parametrize("method", ["forward_euler", "rk4"])
def test_catalog_parity(name, bc, method):
    stages = ButcherTableau(method).n_stages if name == "push_eta_stage" else 1
    for stage in range(stages):

        def make_args(backend, seed):
            args = FACTORIES[name](bc, method)
            # Reach this RK stage independently on both backends, then compare
            # the complete bundle and array arguments after the selected stage.
            for previous in range(stage):
                catalog[name](0.2, previous, *args, n_threads=129)
            return (0.2, stage, *args)

        assert_kernels_agree(catalog[name], make_args, n_threads=129, rtol=1e-13, atol=1e-14)


@requires_cupy
@pytest.mark.parametrize("name", ACCUM_CUDA_NAMES)
@pytest.mark.parametrize("bc", [(0, 0, 0), (2, 0, 1)])
def test_accum_catalog_parity(name, bc):
    # atomics add in another order than the serial loop, hence the tolerance
    assert_kernels_agree(
        accum_catalog[name], lambda backend, seed: ACCUM_FACTORIES[name](bc), n_threads=129, rtol=1e-12, atol=1e-13
    )


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
    """PushEta, both selectable PushVxB algorithms and the electric-field push must have device kernels."""
    for name in ("push_eta_stage", "push_vxb_analytic", "push_vxb_implicit", "push_v_with_efield"):
        assert catalog[name].cuda_kernel is not None
    for name in ("push_weights_with_efield_lin_va",):
        assert catalog[name].cuda_kernel is not None
    for name in ("charge_density_0form",):
        assert accum_catalog[name].cuda_kernel is not None
