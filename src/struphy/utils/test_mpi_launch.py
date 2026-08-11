import sys

import pytest

from struphy.utils.mpi_launch import _LAUNCHER_ENV_VARS, _OVERRIDE_ENV_VAR, launched_under_mpi


@pytest.fixture
def clean_env(monkeypatch):
    """An environment with no launcher variables and no override set."""
    for var in (*_LAUNCHER_ENV_VARS, _OVERRIDE_ENV_VAR):
        monkeypatch.delenv(var, raising=False)


def test_no_launcher_means_no_mpi(clean_env, monkeypatch):
    # mpi4py may well be imported by another test; the detection must not read it
    # as "launched under MPI" unless MPI was actually initialized.
    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)
    assert launched_under_mpi() is False


@pytest.mark.parametrize("var", _LAUNCHER_ENV_VARS)
def test_each_launcher_variable_is_detected(clean_env, monkeypatch, var):
    monkeypatch.setenv(var, "0")
    assert launched_under_mpi() is True


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_override_forces_mpi_on(clean_env, monkeypatch, value):
    monkeypatch.setenv(_OVERRIDE_ENV_VAR, value)
    assert launched_under_mpi() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_override_forces_mpi_off_even_under_a_launcher(clean_env, monkeypatch, value):
    monkeypatch.setenv(_LAUNCHER_ENV_VARS[0], "3")
    monkeypatch.setenv(_OVERRIDE_ENV_VAR, value)
    assert launched_under_mpi() is False


def test_unrecognized_override_falls_back_to_detection(clean_env, monkeypatch):
    monkeypatch.setenv(_OVERRIDE_ENV_VAR, "maybe")
    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)
    assert launched_under_mpi() is False

    monkeypatch.setenv(_LAUNCHER_ENV_VARS[0], "0")
    assert launched_under_mpi() is True


def test_slurm_procid_alone_is_not_an_mpi_launch(clean_env, monkeypatch):
    # A plain `sbatch` script gets SLURM_PROCID without being an MPI launch; `srun`
    # is covered by the PMI/PMIX variables its MPI plugin exports.
    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)
    monkeypatch.setenv("SLURM_PROCID", "0")
    assert launched_under_mpi() is False


def test_already_initialized_mpi_is_detected(clean_env, monkeypatch):
    """An application that imported and initialized mpi4py itself counts as MPI."""

    class _FakeMPI:
        @staticmethod
        def Is_initialized():
            return True

    monkeypatch.setitem(sys.modules, "mpi4py.MPI", _FakeMPI)
    assert launched_under_mpi() is True

    class _NotInitialized:
        @staticmethod
        def Is_initialized():
            return False

    monkeypatch.setitem(sys.modules, "mpi4py.MPI", _NotInitialized)
    assert launched_under_mpi() is False
