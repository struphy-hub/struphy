"""Regression tests of the polar-FEEC demos (ports of PolarFEEC.jl examples 04-06 and the talk studies)."""

import numpy as np
import pytest


def test_regularity_study():
    """The k-th derivative is continuous at the centre exactly when k <= r (PolarFEEC.jl example 04)."""
    from struphy.polar.homotopy.demos import regularity

    results = regularity.study(verbose=False)
    for r in range(4):
        delta = results["defects"][f"r = {r}"]
        for k in range(1, 4):
            if k <= r:
                assert delta[k - 1] < 1e-7, (r, k, delta)
            else:
                assert delta[k - 1] > 1e-6, (r, k, delta)
    assert np.allclose(results["defects"]["tensor product"], results["defects"]["r = 0"])


def test_pde_checks_and_reference_errors():
    """Gauges, boundary conditions and the L2 errors of PolarFEEC.jl example 05 (p = 3, n = 6)."""
    from struphy.polar.homotopy.demos import pde

    solution = pde.solve_all(3, 6)
    checks = pde.checks(solution)
    assert abs(checks["mean_of_source"]) < 1e-12
    assert abs(checks["u_centre"]) < 1e-12
    assert checks["max_u_on_boundary"] < 1e-12 and checks["tangential_trace"] < 1e-12
    assert checks["gauge_free"] < 1e-10 and checks["gauge_dirichlet"] < 1e-10
    # values of PolarFEEC.jl
    reference = [3.4128468787e-03, 4.2082992517e-04, 1.5702930010e-03, 1.2889397521e-03]
    assert np.allclose(pde.errors(solution), reference, rtol=1e-8)


@pytest.mark.convergence
def test_pde_convergence_rates():
    from struphy.polar.homotopy.demos import pde

    cells = (6, 8, 12)
    table = pde.convergence(cells, (2, 3), verbose=False)
    for p, errors in table.items():
        rates = np.log(errors[:-1] / errors[1:]) / np.log(np.array(cells[1:]) / np.array(cells[:-1]))[:, None]
        assert np.all(rates[-1, :2] > p + 1 - 0.3), (p, rates)
        assert np.all(rates[-1, 2:] > p - 0.3), (p, rates)


def test_maxwell_spectrum():
    """No spurious modes: kernel = dim V0, dim V2 - 1 nonzero eigenvalues, upper bounds of the exact ones."""
    from struphy.polar.homotopy.demos import maxwell

    s = maxwell.study(meshes=((8, 16),), compared=20, verbose=False)
    r = s["results"][0]
    polar = r["polar"]
    assert r["kernel"] == polar.dimension(0)
    assert len(r["lam"]) == polar.dimension(2) - 1
    assert np.max(r["error"]) < 2e-2


def test_projection_commutation():
    from struphy.polar.homotopy.demos import projections

    defects = projections.commutation((2,), (4, 8), method="commuting")
    assert np.max(defects[2]) < 1e-12
    errors = projections.convergence((2,), (4, 8, 16))
    rates = np.log2(errors[2][:-1] / errors[2][1:])[-1]
    assert rates[0] > 2.5 and rates[1] > 1.6 and rates[2] > 1.6
