"""Tests for reading and writing the Struphy state file (``state.yml``)."""

import os

from struphy.utils import utils


def test_empty_state_file_is_read_as_an_empty_dict(tmp_path):
    """An empty state file (e.g. seen while another process rewrites it) gives {} and default paths."""
    (tmp_path / "state.yml").write_text("")
    state = utils.read_state(libpath=str(tmp_path))
    assert state == {}
    utils.update_state(state=state)
    assert state["i_path"].endswith(os.path.join("io", "inp"))


def test_saved_state_is_read_back_without_leftover_files(tmp_path):
    state = {"i_path": "inp", "o_path": "out", "b_path": "batch", "kernels": ["a.py", "b.py"]}
    utils.save_state(state, libpath=str(tmp_path))
    assert utils.read_state(libpath=str(tmp_path)) == state
    assert (tmp_path / "kernels.txt").read_text() == "a.py\nb.py\n"
    assert sorted(os.listdir(tmp_path)) == ["kernels.txt", "state.yml"]
