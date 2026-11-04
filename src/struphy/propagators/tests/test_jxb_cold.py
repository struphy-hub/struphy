import pytest

from struphy.propagators.jxb_cold import JxBCold


@pytest.mark.mpi_skip
def test_jxb_cold_solver_options():
    """The Crank-Nicolson system of JxBCold is not symmetric: the default is a solver for general matrices, and the
    symmetric solvers of older parameter files are replaced (pcg diverged for dt > epsilon, issue #856)."""
    assert JxBCold.Options().solver == "pbicgstab"
    for solver in ("pcg", "pcg_sr", "cg"):
        assert JxBCold.Options(solver=solver).solver == "pbicgstab"
    for solver in ("bicgstab", "gmres"):
        assert JxBCold.Options(solver=solver).solver == solver
    with pytest.raises(AssertionError):
        JxBCold.Options(solver="minres")
