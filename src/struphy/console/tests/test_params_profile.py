"""Tests for ``struphy params --check-file`` and ``struphy profile``."""

import time

import pytest

from struphy.console.params import struphy_params
from struphy.console.profile import struphy_profile


def test_params_check_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    struphy_params("Maxwell", yes=True)
    capsys.readouterr()

    struphy_params("Maxwell", check_file="params_Maxwell.py")
    assert "is valid for the model Maxwell" in capsys.readouterr().out
    assert not (tmp_path / "__pycache__").exists()

    # wrong model, missing file, file that fails to execute, file without a Simulation
    (tmp_path / "broken.py").write_text("raise ValueError('bad option')\n")
    (tmp_path / "no_sim.py").write_text("x = 1\n")
    for model_name, path, message in [
        ("Vlasov", "params_Maxwell.py", "its model is Maxwell, not Vlasov"),
        ("Maxwell", "missing.py", "does not exist"),
        ("Maxwell", "broken.py", "ValueError: bad option"),
        ("Maxwell", "no_sim.py", "does not define a Simulation"),
    ]:
        with pytest.raises(SystemExit) as exc:
            struphy_params(model_name, check_file=path)
        assert exc.value.code == 1
        assert message in capsys.readouterr().err


def test_profile(tmp_path, monkeypatch, capsys):
    from scope_profiler import ProfileManager

    monkeypatch.chdir(tmp_path)
    for name in ("sim_a", "sim_b"):
        (tmp_path / name).mkdir()
        with ProfileManager.session(file_path=str(tmp_path / name / "profiling_data.h5"), verbose=False):
            for _ in range(2):
                with ProfileManager.profile_region("prop: Maxwell"):
                    time.sleep(1e-3)
            with ProfileManager.profile_region("kernel: push"):
                time.sleep(1e-3)
    capsys.readouterr()

    struphy_profile(["sim_a", "sim_b"], prefix="prop:", savefig="profile.png")
    out = capsys.readouterr().out
    assert "Profile: sim_a" in out and "Profile: sim_b" in out
    assert "prop: Maxwell" in out and "kernel: push" not in out
    assert (tmp_path / "profile.png").is_file()

    with pytest.raises(SystemExit) as exc:
        struphy_profile(["sim_a", "missing"])
    assert exc.value.code == 1
    assert "profiling_activated=True" in capsys.readouterr().err
