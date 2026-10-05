import shutil

import pytest

from struphy.console.format import check_omp_flags, check_ruff


@pytest.mark.parametrize("verbose", [False, True])
def test_check_omp_flags(tmp_path, verbose):
    good = tmp_path / "good.py"
    good.write_text("def f():\n    #$ omp parallel\n    pass\n")
    bad = tmp_path / "bad.py"
    bad.write_text("def f():\n    # $ omp parallel\n    pass\n")

    assert check_omp_flags(str(good), verbose=verbose)
    assert not check_omp_flags(str(bad), verbose=verbose)


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
def test_check_ruff(tmp_path):
    good = tmp_path / "good.py"
    good.write_text("import os\nimport sys\n\nx = 1\n\n\ndef f():\n    #$ omp parallel\n    pass\n")
    bad_format = tmp_path / "bad_format.py"
    bad_format.write_text("x=1\n")
    bad_blank_lines = tmp_path / "bad_blank_lines.py"
    bad_blank_lines.write_text("x = 1\n\n\n\n\ny = 2\n")
    bad_imports = tmp_path / "bad_imports.py"
    bad_imports.write_text("import sys\nimport os\n\nprint(os, sys)\n")

    # "#$" -> "# $" reformatting of OpenMP flags is ignored
    assert check_ruff(str(good))
    assert not check_ruff(str(bad_format))
    assert not check_ruff(str(bad_blank_lines))
    assert not check_ruff(str(bad_imports))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
