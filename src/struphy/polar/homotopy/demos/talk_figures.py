"""All figures (and videos) of R. Remmerswaal's NMPP seminar talk on high-order polar splines, computed with Struphy.

Run as ``python -m struphy.polar.homotopy.demos.talk_figures OUT_FOLDER [--full] [--videos] [--only NAME ...]``.
Without ``--full`` the studies run at a coarse resolution (minutes); ``--full`` uses the resolution of the talk
(the wave, 3D wave and orbit studies then take hours). The file names are those of the talk (``figures/tikz`` and
``media``); the schematic diagrams of the talk are drawn by :mod:`struphy.polar.homotopy.demos.schematics`.

Every figure is computed with Struphy objects: the polar splines of Struphy Derham complexes (B-splines or
``angular_basis="trig"``), Struphy domains (``PerturbedDiskCylinder``, ``HollowCylinder(a1=0)``, ``HelicalTube``) and
Struphy's mass operators, projectors and field evaluation.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, folder, name):
    path = Path(folder) / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------------------------------------- figures
def basis_figures(folder, full=False):
    from struphy.polar.homotopy.demos import basis

    _save(basis.figure_mapping(), folder, "disk_mapping")
    _save(basis.figure_tensor_basis("bspline"), folder, "disc_basis_functions_tensor_bspline")
    _save(basis.figure_tensor_basis("trig"), folder, "disc_basis_functions_tensor_fourier")
    for r in (1, 2, 3):
        _save(basis.figure_polar_zero_forms(r), folder, f"disc_basis_functions_r{r}")
    _save(basis.figure_all_forms(3), folder, "allforms")


def regularity_figures(folder, full=False):
    from struphy.polar.homotopy.demos import regularity

    _save(regularity.figure_regularity_lines(0), folder, "result_regularity_zeroform")
    _save(regularity.figure_regularity_lines(2), folder, "result_regularity_twoform")


def projection_figures(folder, full=False):
    from struphy.polar.homotopy.demos import projections

    regularities = (1, 2, 3, 4, 5, 6) if full else (1, 2, 3)
    cells = (8, 12, 16, 24, 32, 48) if full else (4, 8, 16)
    for method, label in (("commuting", "commuting projection"), ("l2", "$L^2$ projection")):
        errors = projections.convergence(regularities, cells, method=method)
        _save(projections.figure_convergence(errors, cells, title=label), folder, f"polar_projection_{method}")
        defects = projections.commutation(regularities, cells, method=method)
        _save(
            projections.figure_commutation(defects, cells, title=label),
            folder,
            f"polar_projection_commutation_{method}",
        )


def _wave_settings(full):
    from struphy.polar.homotopy.demos import waves

    if full:
        return dict(regularities=(0, 1, 2, 3, 4, 5), radial_cells=(8, 16, 32, 48), reference_cells=64, degree=5), 1.0
    initial = lambda x, y: waves.gaussian(x, y, width=0.15)  # noqa: E731 (the talk's pulse is not resolved coarse)
    return dict(regularities=(0, 1, 2), radial_cells=(4, 8, 16), reference_cells=32, degree=3, initial=initial), 0.3


def wave_figures(folder, full=False):
    from struphy.polar.homotopy.demos import waves

    settings, final_time = _wave_settings(full)
    names = [
        ("polar_0form_wave", dict(form=0)),
        ("polar_1form_wave", dict(form=1)),
        ("polar_1form_wave_convergence_nonmatching", dict(form=1, non_matching=True)),
    ]
    for name, extra in names:
        form = extra.pop("form")
        study = waves.convergence(form, final_time=final_time, verbose=True, **settings, **extra)
        stem = name if "nonmatching" in name else f"{name}_convergence"
        _save(waves.figure_convergence(study, degree=settings["degree"]), folder, f"{stem}_linf")
        for index, n in enumerate(study["cells"]):
            if n in (8, 48) or (not full and index in (0, len(study["cells"]) - 1)):
                tag = f"nr{n}"
                _save(waves.figure_time_traces(study, index), folder, f"{stem}_linf_time_{tag}")


def wave3d_figures(folder, full=False):
    from struphy.polar.homotopy.demos import waves3d

    if full:
        study = waves3d.convergence((1, 2, 3, 4), (5, 8, 12), reference_mesh=16, final_time=0.4)
    else:
        study = waves3d.convergence((1, 2), (3, 4), reference_mesh=5, final_time=0.1, degree=3)
    _save(waves3d.figure_convergence(study), folder, "polar_3d_wave_convergence_linf")
    from struphy.polar.homotopy.demos.waves import figure_time_traces

    _save(figure_time_traces(study, 0), folder, f"polar_3d_wave_convergence_linf_time_nr{study['cells'][0]}")


def orbit_figures(folder, full=False):
    from struphy.polar.homotopy.demos import orbits

    field = orbits.ProjectedVacuumField(2)
    path, _ = orbits.integrate(field, 0.04, periods=20.0 if full else 2.0)
    _save(orbits.figure_orbits(path), folder, "result_vacuum_orbits_overview")
    _save(orbits.figure_orbits(path, extent=0.05), folder, "result_vacuum_orbits_closeup")
    kwargs = (
        dict(regularities=(0, 1, 2, 3, 4), dts=(0.04, 0.02, 0.01, 0.005, 0.0025), periods=20.0)
        if full
        else dict(regularities=(0, 2, 4), dts=(0.08, 0.04, 0.02), periods=1.0)
    )
    for angular, name in (("bspline", "bspline"), ("trig", "compatible")):
        study = orbits.energy_study(angular=angular, **kwargs)
        _save(orbits.figure_energy(study), folder, f"result_vacuum_orbits_energy_{name}")


def extra_figures(folder, full=False):
    """Figures of PolarFEEC.jl that the talk does not show: Poisson/curl-curl (example 05), the Maxwell spectrum
    (example 06), field lines in the twisted tube (example 07)."""
    from struphy.polar.homotopy.demos import fieldlines, maxwell, pde

    solution = pde.solve_all(3, 6)
    _save(pde.figure_poisson(solution), folder, "polarfeec_05_poisson")
    _save(pde.figure_curl_curl(solution), folder, "polarfeec_05_curl_curl")
    spectrum = maxwell.study(meshes=maxwell.MESHES if full else ((8, 16), (16, 32)), compared=40, verbose=False)
    _save(maxwell.figure_spectrum(spectrum), folder, "polarfeec_06_maxwell_spectrum")
    result = fieldlines.study(turns=150 if full else 25, verbose=False)
    _save(fieldlines.figure_poincare(result), folder, "polarfeec_07_poincare")


# ---------------------------------------------------------------------------------------------------------- videos
def videos(folder, full=False):
    """The wave videos of the talk (mp4, needs ffmpeg)."""
    from struphy.polar.homotopy.demos import waves

    folder = Path(folder)
    final_time = 1.0 if full else 0.3
    kwargs = dict(degree=5, radial_cells=8) if full else dict(degree=3, radial_cells=6)
    series = [
        ("polar_0form_wave_p5_cochain_bspline16", 0, dict(enrichment="cochain", angular_cells=16)),
        (
            "polar_0form_wave_p5_cochain_trigonometric15",
            0,
            dict(enrichment="cochain", angular="trig", angular_cells=15),
        ),
        ("polar_0form_wave_p5_none_bspline16", 0, dict(enrichment="none", angular_cells=16)),
        ("polar_1form_wave_p5_cochain_chain_bspline16", 1, dict(angular_cells=16)),
        ("polar_1form_wave_p5_cochain_chain_bspline16_nonmatching", 1, dict(angular_cells=16, non_matching=True)),
        (
            "polar_1form_wave_p5_cochain_trigonometric15",
            1,
            dict(enrichment="cochain", angular="trig", angular_cells=15),
        ),
    ]
    regularities = (0, 1, 2, 3, 4, 5) if full else (0, 2)
    for name, form, extra in series:
        for r in regularities:
            if extra.get("enrichment") == "none" and r > 0 and form == 1:
                continue
            result = waves.run_family(form, (r,), final_time=final_time, verbose=False, **kwargs, **extra)
            path = folder / f"{name}_r{r}.mp4"
            waves.animate(result, path, regularity=r)
            print("wrote", path)


def schematic_figures(folder, full=False):
    """The schematic diagrams (de Rham diagrams with tikzfigure and LaTeX, the 3D coordinate transformation, the
    boundary-compatible homotopy)."""
    from struphy.polar.homotopy.demos import schematics

    schematics.write_all(folder)


GROUPS = {
    "basis": basis_figures,
    "schematics": schematic_figures,
    "regularity": regularity_figures,
    "projections": projection_figures,
    "waves": wave_figures,
    "waves3d": wave3d_figures,
    "orbits": orbit_figures,
    "extra": extra_figures,
    "videos": videos,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="output folder")
    parser.add_argument("--full", action="store_true", help="the resolution of the talk (hours)")
    parser.add_argument("--videos", action="store_true", help="also write the wave videos (mp4)")
    parser.add_argument("--only", nargs="+", choices=list(GROUPS), help="only these groups of figures")
    args = parser.parse_args(argv)
    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)
    groups = args.only or [g for g in GROUPS if g != "videos" or args.videos]
    for group in groups:
        print(f"--- {group}")
        GROUPS[group](folder, args.full)


if __name__ == "__main__":
    main()
