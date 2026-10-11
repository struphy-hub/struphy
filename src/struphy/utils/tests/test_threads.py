import pytest
from threadpoolctl import ThreadpoolController, threadpool_info

import struphy.utils.threads as threads
from struphy.utils.threads import BLAS_THREAD_ENV_VARS, limit_blas_threads


class FakeComm:
    """Stand-in for an MPI communicator with ``size`` ranks; only ``Get_size`` is used."""

    def __init__(self, size):
        self.size = size

    def Get_size(self):
        return self.size


def num_threads(user_api):
    return [lib["num_threads"] for lib in threadpool_info() if lib["user_api"] == user_api]


@pytest.fixture
def restore_thread_pools(monkeypatch):
    """Clear the BLAS environment variables and restore all thread pools after the test."""
    for var in BLAS_THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # load scipy's BLAS before recording, limit_blas_threads loads it
    import scipy.sparse.linalg  # noqa: F401

    controller = ThreadpoolController()
    original = [(lib, lib.num_threads) for lib in controller.lib_controllers]
    yield
    for lib, n in original:
        lib.set_num_threads(n)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("comm", [None, FakeComm(1)])
def test_serial_unchanged(comm, restore_thread_pools):
    blas = num_threads("blas")
    assert not limit_blas_threads(comm)
    assert num_threads("blas") == blas


@pytest.mark.mpi_skip
@pytest.mark.parametrize("var", BLAS_THREAD_ENV_VARS)
def test_user_setting_wins(var, monkeypatch, restore_thread_pools):
    monkeypatch.setenv(var, "3")
    blas = num_threads("blas")
    assert not limit_blas_threads(FakeComm(2))
    assert num_threads("blas") == blas


@pytest.mark.mpi_skip
@pytest.mark.parametrize("n", [1, 2])
def test_limit_blas_only(n, restore_thread_pools):
    omp = num_threads("openmp")
    assert limit_blas_threads(FakeComm(4), num_threads=n)

    blas = num_threads("blas")
    assert len(blas) > 0
    assert all(b <= n for b in blas)
    assert num_threads("openmp") == omp


@pytest.mark.mpi_skip
def test_openmp_restored(monkeypatch, restore_thread_pools):
    """A BLAS built with OpenMP may also reduce the OpenMP threads, which must be undone."""
    # loads the OpenMP runtime if the kernels are compiled with --openmp
    import struphy.pic.pushing.kernels.push_vxb_analytic.push_vxb_analytic_kernels  # noqa: F401

    controller = ThreadpoolController()
    omp_libs = controller.select(user_api="openmp")
    if not omp_libs.lib_controllers:
        pytest.skip("no OpenMP runtime loaded")
    omp_libs.limit(limits=2)
    omp = num_threads("openmp")

    real_limits = threads.threadpool_limits

    def blas_with_openmp_side_effect(limits, user_api):
        real_limits(limits=limits, user_api=user_api)
        if user_api == "blas":
            real_limits(limits=1, user_api="openmp")

    monkeypatch.setattr(threads, "threadpool_limits", blas_with_openmp_side_effect)
    assert limit_blas_threads(FakeComm(2))
    assert num_threads("openmp") == omp
