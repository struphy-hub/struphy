import ctypes
import ctypes.util
import logging

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


def num_threads(user_api, internal_api=None):
    return [
        lib["num_threads"]
        for lib in threadpool_info()
        if lib["user_api"] == user_api and internal_api in (None, lib["internal_api"])
    ]


def lib_class(user_api):
    """Class of the first loaded threadpoolctl controller of ``user_api``, skip if there is none."""
    libs = ThreadpoolController().select(user_api=user_api).lib_controllers
    if not libs:
        pytest.skip(f"no {user_api} library loaded")
    return type(libs[0])


@pytest.fixture
def restore_thread_pools(monkeypatch):
    """Clear the BLAS environment variables and restore all thread pools after the test."""
    for env_vars in BLAS_THREAD_ENV_VARS.values():
        for var in env_vars:
            monkeypatch.delenv(var, raising=False)

    # load the libraries before recording: scipy's BLAS (limit_blas_threads loads it) and an
    # OpenMP runtime (the kernels are not compiled with OpenMP in CI)
    import scipy.sparse.linalg  # noqa: F401

    gomp = ctypes.util.find_library("gomp")
    if gomp is not None:
        ctypes.CDLL(gomp)

    original = [(lib, lib.num_threads) for lib in ThreadpoolController().lib_controllers]
    yield
    # remove patched set_num_threads before restoring
    monkeypatch.undo()
    for lib, n in original:
        lib.set_num_threads(n)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("comm", [None, FakeComm(1)])
def test_serial_unchanged(comm, restore_thread_pools):
    blas = num_threads("blas")
    assert not limit_blas_threads(comm)
    assert num_threads("blas") == blas


@pytest.mark.mpi_skip
@pytest.mark.parametrize("n", [1, 2])
def test_limit_blas_only(n, restore_thread_pools):
    lib_class("blas")
    omp = num_threads("openmp")
    assert limit_blas_threads(FakeComm(4), num_threads=n)
    assert all(b <= n for b in num_threads("blas"))
    assert num_threads("openmp") == omp


@pytest.mark.mpi_skip
@pytest.mark.parametrize("var", BLAS_THREAD_ENV_VARS["openblas"])
def test_openblas_user_setting_wins(var, monkeypatch, restore_thread_pools):
    if not num_threads("blas", "openblas"):
        pytest.skip("OpenBLAS not loaded")
    monkeypatch.setenv(var, "3")
    openblas = num_threads("blas", "openblas")
    limit_blas_threads(FakeComm(2))
    assert num_threads("blas", "openblas") == openblas


@pytest.mark.mpi_skip
def test_other_blas_setting_ignored(monkeypatch, restore_thread_pools):
    """MKL_NUM_THREADS (e.g. exported by HPC modules) does not stop the limit of OpenBLAS."""
    if not num_threads("blas", "openblas"):
        pytest.skip("OpenBLAS not loaded")
    monkeypatch.setenv("MKL_NUM_THREADS", "3")
    limit_blas_threads(FakeComm(2))
    assert all(b == 1 for b in num_threads("blas", "openblas"))


@pytest.mark.mpi_skip
def test_no_controllable_blas(monkeypatch, restore_thread_pools):
    """E.g. Apple Accelerate, which threadpoolctl cannot detect."""

    class NoLibs:
        lib_controllers = []

    monkeypatch.setattr(threads, "ThreadpoolController", NoLibs)
    assert not limit_blas_threads(FakeComm(2))


@pytest.fixture
def blas_with_openmp(monkeypatch, restore_thread_pools):
    """Emulate a BLAS built with OpenMP: setting its threads also sets the OpenMP threads."""
    blas_cls = lib_class("blas")
    omp_cls = lib_class("openmp")
    ThreadpoolController().select(user_api="openmp").limit(limits=2)

    blas_set = blas_cls.set_num_threads
    omp_set = omp_cls.set_num_threads

    def set_blas(self, n):
        blas_set(self, n)
        for lib in ThreadpoolController().select(user_api="openmp").lib_controllers:
            omp_set(lib, n)

    monkeypatch.setattr(blas_cls, "set_num_threads", set_blas)
    return blas_cls, blas_set, omp_cls, omp_set


@pytest.mark.mpi_skip
def test_openmp_restored(blas_with_openmp):
    omp = num_threads("openmp")
    assert limit_blas_threads(FakeComm(2))
    assert num_threads("openmp") == omp
    assert all(b == 1 for b in num_threads("blas"))


@pytest.mark.mpi_skip
def test_openmp_coupled_blas_warns(blas_with_openmp, monkeypatch, caplog):
    """If the BLAS follows the OpenMP threads again after the restore, warn and return False."""
    blas_cls, blas_set, omp_cls, omp_set = blas_with_openmp

    def set_omp(self, n):
        omp_set(self, n)
        for lib in ThreadpoolController().select(user_api="blas").lib_controllers:
            blas_set(lib, n)

    monkeypatch.setattr(omp_cls, "set_num_threads", set_omp)

    logger = logging.getLogger("struphy")
    logger.addHandler(caplog.handler)
    try:
        assert not limit_blas_threads(FakeComm(2))
    finally:
        logger.removeHandler(caplog.handler)
    assert num_threads("openmp") == [2] * len(num_threads("openmp"))
    assert "Could not limit the BLAS threads" in caplog.text
