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


def test_spectrum_explorer(tmp_path):
    """Curl-curl eigenvalues on the unit disk: no spurious modes, upper bounds of the exact ones; the deficit with
    respect to a higher regularity vanishes for the modes that already are in the more regular space."""
    import json

    from struphy.polar.homotopy.demos import spectrum

    kwargs = dict(degree=4, cells=(4, 14))
    reference = spectrum.complex_(3, **kwargs)
    deficits = []
    for r in (0, 1, 2):
        s = spectrum.series(r, reference=reference, count=8, **kwargs)
        assert s["kernel"] == s["complex"].dimension(0)
        exact = np.array(s["exact"], dtype=float)
        assert np.all(s["lam"] >= exact * (1 - 1e-12))
        assert np.max(np.abs(s["lam"] - exact) / exact) < 2e-3
        deficits.append(s["deficit"][0])
    assert deficits[2] < 1e-10 < deficits[0]

    html = spectrum.write_explorer(tmp_path, count=3, pixels=16, regularities=(0, 1), verbose=False, **kwargs)
    assert html.exists()
    text = (tmp_path / "spectrum_data.js").read_text()
    data = json.loads(text[text.index("=") + 1 :].strip().rstrip(";"))
    assert [s["enrichment"] for s in data["spaces"]] == ["cochain_chain", "cochain"]
    for s in data["spaces"]:
        for entry in s["series"]:
            assert len(entry["lambda"]) == 3
            assert all((tmp_path / "spectrum" / name).exists() for name in entry["image"])


def test_wave_3d_helical_tube():
    """Leapfrog Maxwell in the helical tube with polar splines: a stable time step and a small energy error."""
    import warnings

    from struphy.polar.homotopy.demos import waves3d

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wave = waves3d.Wave3D(1, 3, degree=2)
    markers = waves3d.sample_points(3)
    values, energy_error = wave.evolve(
        0.05, np.linspace(0, 0.05, 3), lambda f: np.concatenate([np.asarray(v) for v in f(markers)])
    )
    assert np.all(np.isfinite(values)) and np.max(np.abs(values[0])) > 0
    assert energy_error < 0.1


@pytest.mark.parametrize("angular", ["bspline", "trig"])
def test_guiding_centre_orbits(angular):
    """Guiding centres in the projected vacuum field near the axis: finite orbits, conserved energy."""
    from struphy.polar.homotopy.demos import orbits

    field = orbits.ProjectedVacuumField(2, angular, num_elements=(4, 8, 8), degree=(3, 3, 3))
    path, errors = orbits.integrate(field, 0.05, periods=0.2)
    assert np.all(np.isfinite(path))
    assert np.max(np.hypot(path[..., 0], path[..., 1])) < orbits.A
    assert np.max(errors) < 1e-6


def test_talk_figures_cli(tmp_path):
    """The figure script runs (one quick group)."""
    import matplotlib

    matplotlib.use("Agg")
    from struphy.polar.homotopy.demos import talk_figures

    talk_figures.main([str(tmp_path), "--only", "regularity"])
    assert (tmp_path / "result_regularity_zeroform.png").exists()
    assert (tmp_path / "result_regularity_twoform.png").exists()


def test_wave_convergence_is_finite():
    """The sampling grid of the wave studies leaves out the pole, where the 2-form pushforward is singular."""
    from struphy.polar.homotopy.demos import waves

    x, _ = waves.sample_points(21)
    assert np.all(x > 0)
    study = waves.convergence(
        1, regularities=(0, 1), radial_cells=(3, 4), reference_cells=6, degree=2, final_time=0.05, verbose=False
    )
    for errors in study["final"].values():
        assert np.all(np.isfinite(errors))


def test_schematic_diagrams():
    """The de Rham diagrams are tikzfigure matrices with labelled arrows (TikZ code only, no LaTeX needed)."""
    matrix = pytest.importorskip("tikzfigure.core.matrix")
    if not hasattr(matrix.Matrix, "arrow"):
        pytest.skip("needs tikzfigure >= 0.5.0")
    from struphy.polar.homotopy.demos import schematics

    code = schematics.derham_logical_and_physical().generate_tikz()
    assert code.count("\\matrix") == 2
    assert "(l-1-1) to node[above, font=\\scriptsize] {grad} (l-1-2)" in code
    assert "(l.east) to node[above, font=\\scriptsize] {pushforward $F_*$} (p.west)" in code
    code = schematics.derham_polar_cochain_chain().generate_tikz()
    assert "(h-1-2) to node[above, font=\\scriptsize] {$\\mathrm{grad}^\\dagger$} (h-1-1)" in code
    assert code.count("color=struphyblue") == 6
