"""Tests for ``struphy backend``: the cunumpy backend stored in the state file."""

import functools

import pytest

from struphy.console import backend as backend_cli
from struphy.utils import utils


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Redirect the state file to a temporary folder."""
    monkeypatch.setattr(utils, "read_state", functools.partial(utils.read_state, libpath=str(tmp_path)))
    monkeypatch.setattr(utils, "save_state", functools.partial(utils.save_state, libpath=str(tmp_path)))
    return tmp_path


def test_default_backend_is_numpy(state_dir):
    assert backend_cli.configured_backend() == "numpy"


def test_set_get_unset(state_dir, capsys):
    backend_cli.struphy_backend("set", "cupy")
    assert utils.read_state() == {"backend": "cupy"}
    assert backend_cli.configured_backend() == "cupy"

    capsys.readouterr()  # drop the output of set
    backend_cli.struphy_backend("get", value=True)
    assert capsys.readouterr().out.strip() == "cupy"

    backend_cli.struphy_backend("unset")
    assert "backend" not in utils.read_state()
    assert backend_cli.configured_backend() == "numpy"


def test_set_keeps_other_state(state_dir):
    utils.save_state({"i_path": "inp", "kernels": ["a.py"]})
    backend_cli.struphy_backend("set", "numpy")
    assert utils.read_state() == {"i_path": "inp", "kernels": ["a.py"], "backend": "numpy"}


def test_set_rejects_unknown_backend(state_dir):
    with pytest.raises(ValueError, match="one of"):
        backend_cli.struphy_backend("set", "torch")
    with pytest.raises(ValueError, match="one of"):
        backend_cli.struphy_backend("set")


def test_get_reports_stored_env_and_effective(state_dir, monkeypatch, capsys):
    monkeypatch.delenv("CUNUMPY_BACKEND", raising=False)
    backend_cli.struphy_backend("get")
    out = capsys.readouterr().out
    assert "stored backend (struphy backend set): numpy (default)" in out
    assert "CUNUMPY_BACKEND: (not set)" in out
    assert "backend in this process: numpy" in out


def test_invalid_backend_in_state_file_is_rejected(state_dir):
    utils.save_state({"backend": "torch"})
    with pytest.raises(ValueError, match="invalid backend"):
        backend_cli.configured_backend()
