"""Numerical convergence of the slit-lens results (plan phase 10).

Run from the repository root with::

    python examples/IonOpticsElectrostatic/convergence_study/convergence_study.py

The metrics of ``../steady_state_iteration`` are: exit rms emittance, exit rms size, and
waist position and size. They are computed for the slit lens at zero current and at
300 µA per mm of slit (at 400 µA/mm the waist leaves the 80 mm domain). The steady-state iteration is refined in one computational
parameter at a time, from a base case:

* mesh: elements (x × y) at spline degree p = 3;
* spline degree p;
* rays;
* time step dt of the ray tracing.

A few time-dependent PIC runs at 300 µA/mm, steady window from the emittance
criterion, are compared against the converged steady state. The thesis defaults
(Kalvas 2013, §5.9.3) serve as the starting point: more than 100 trajectories per
cell, and α as close to 1 as possible.

By default only a quick check runs: 7 steady runs at 300 µA/mm, about 4 min. Pass
``--full`` for all sweeps plus the time-dependent runs, about 30 min, cached in
``convergence_study.json``. Add ``--replot`` to only redraw the figure from the cache.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
for sibling in (
    "child_langmuir_diode",
    "slit_immersion_lens",
    "slit_lens_injection",
    "slit_lens_space_charge",
):
    sys.path.insert(0, str(HERE.parent / sibling))

import slit_lens_space_charge as lens_sc  # noqa: E402
from slit_immersion_lens import DESIGN, UNITS, analytic_lens  # noqa: E402
from slit_lens_injection import loss_tags, source  # noqa: E402

from struphy import domains  # noqa: E402
from struphy.initial.perturbations import PiecewiseLinearPotential  # noqa: E402
from struphy.models import IonOpticsElectrostatic  # noqa: E402
from struphy.models.ion_optics_steady_state import SteadyStateOptions, build_steady_state_simulation  # noqa: E402

CURRENTS = (0.0, 300.0)  # µA per mm of slit; at 400 the waist leaves the domain
BASE = {"num_elements": (160, 20), "degree": 3, "n_rays": 2000, "dt": 0.04}
SWEEPS = {
    "num_elements": [(80, 10), (160, 20), (320, 40), (640, 80)],
    "degree": [2, 3, 4],
    "n_rays": [500, 1000, 2000, 4000, 8000],
    "dt": [0.08, 0.04, 0.02, 0.01],
}
# Default: a quick check (7 steady runs, about 4 min). `--full` runs the sweeps above
# plus the time-dependent runs (about 30 min); its results are in convergence_study.json.
QUICK_CURRENTS = (300.0,)
QUICK_SWEEPS = {
    "num_elements": [(80, 10), (160, 20), (320, 40)],
    "n_rays": [1000, 2000, 4000],
    "dt": [0.08, 0.04],
}
TIME_DEPENDENT = [
    {"num_elements": (160, 20), "markers_per_ns": 80.0, "dt": 0.04},
    {"num_elements": (320, 40), "markers_per_ns": 80.0, "dt": 0.04},
    {"num_elements": (160, 20), "markers_per_ns": 160.0, "dt": 0.04},
    {"num_elements": (160, 20), "markers_per_ns": 80.0, "dt": 0.02},
]
PLANES_MM = np.arange(40.0, 79.76, 0.25)


def steady_case(output_dir, current, num_elements, degree, n_rays, dt):
    """One converged steady-state iteration; returns the metrics."""
    lens = analytic_lens()
    length = DESIGN.length / UNITS.length
    h = DESIGN.half_gap / UNITS.length
    # Zero-current rays carry a tiny current so exit statistics are weighted equally.
    src = source(lens_sc.LAMINAR_BEAM, current=lens_sc.normalized_current(current) if current > 0 else 1e-12)
    model = IonOpticsElectrostatic(
        base_units=UNITS.base_units(),
        electrode_faces=((False, False), (True, True), (False, False)),
        steady_state=SteadyStateOptions(
            source=src,
            n_rays=n_rays,
            dt=dt,
            loss_tags=loss_tags(),
            exit_tag="outlet",
            max_rounds=40,
            planes=(0, PLANES_MM / length),
        ),
    )
    model.em_fields.phi.add_perturbation(PiecewiseLinearPotential(lens.plate_nodes, (lens.v1, lens.v2), coordinate=0))
    name = f"ss_{current:g}_{num_elements[0]}x{num_elements[1]}_p{degree}_n{n_rays}_dt{dt:g}"
    sim = build_steady_state_simulation(
        output_dir,
        name,
        model,
        n_rays,
        domains.Cuboid(l1=0.0, r1=length, l2=-h, r2=h),
        (*num_elements, 1),
        (degree, degree, 1),
        ("remove", "remove", "periodic"),
    )
    sim.run()
    iteration = model.steady_state_iteration
    record = iteration.history[-1]
    y_planes = -h + 2 * h * record.plane_crossings[:, :, 1]
    sizes = np.array([np.nanstd(y) for y in y_planes])
    i = int(np.nanargmin(sizes))
    return {
        "rounds": len(iteration.history),
        "converged": bool(iteration.converged),
        "exit_emittance": 1e3 * record.exit_emittance,  # mm·mrad
        "exit_size": record.exit_size,
        "waist_x": float(PLANES_MM[i]),
        "waist_size": float(sizes[i]),
        "transmission": record.exit_current,
    }


def time_dependent_case(output_dir, num_elements, markers_per_ns, dt, current=300.0):
    beam = replace(lens_sc.LAMINAR_BEAM, markers_per_ns=markers_per_ns)
    sim = lens_sc.run(output_dir, current, beam=beam, num_elements=num_elements, dt=dt)
    result = lens_sc.steady_beam(sim)
    x_waist, size = lens_sc.waist(result)
    return {
        "exit_emittance": 1e3 * result["outlet"]["emittance"],
        "exit_size": result["outlet"]["size"],
        "waist_x": float(x_waist),
        "waist_size": float(size),
        "t_steady": result["t_steady"],
    }


def run_study(output_dir, currents=CURRENTS, sweeps=SWEEPS, time_dependent=TIME_DEPENDENT):
    results = {"steady": [], "time_dependent": []}
    done = {}
    for current in currents:
        for parameter, values in sweeps.items():
            for value in values:
                case = dict(BASE, **{parameter: value})
                key = (current, *(tuple(v) if isinstance(v, (list, tuple)) else v for v in case.values()))
                if key in done:  # the base case appears in every sweep
                    results["steady"].append({"current": current, "parameter": parameter, **case, **done[key]})
                    continue
                metrics = done[key] = steady_case(output_dir, current, **case)
                results["steady"].append({"current": current, "parameter": parameter, **case, **metrics})
                print(f"steady {current:5.0f} µA/mm {parameter}={value}: {metrics}", flush=True)
    for case in time_dependent:
        metrics = time_dependent_case(output_dir, **case)
        results["time_dependent"].append({**case, **metrics})
        print(f"time-dependent {case}: {metrics}", flush=True)
    return results


METRICS = (
    ("exit_emittance", "Exit rms emittance (mm·mrad)"),
    ("exit_size", "Exit rms size (mm)"),
    ("waist_x", "Waist position (mm)"),
    ("waist_size", "Waist rms size (mm)"),
)


def _value_label(parameter, value):
    if parameter == "num_elements":
        return value[1]  # elements across the slit (2h = 10 mm)
    return value


def convergence_table(results):
    """Relative change of each metric between the two finest values of each sweep."""
    rows = []
    currents = sorted({r["current"] for r in results["steady"]})
    for current in currents:
        for parameter in dict.fromkeys(r["parameter"] for r in results["steady"]):
            runs = [r for r in results["steady"] if r["current"] == current and r["parameter"] == parameter]
            finest, second = runs[-1], runs[-2]
            row = {"current": current, "parameter": parameter}
            for key, _ in METRICS:
                if key == "waist_x":
                    row[key] = finest[key] - second[key]  # absolute, mm
                else:
                    row[key] = (second[key] - finest[key]) / finest[key]
            rows.append(row)
    return rows


def plot_results(results, output):
    parameters = list(dict.fromkeys(r["parameter"] for r in results["steady"]))
    currents = sorted({r["current"] for r in results["steady"]})
    fig, axes = plt.subplots(
        len(METRICS), len(parameters), figsize=(4 * len(parameters), 13), constrained_layout=True, squeeze=False
    )
    fig.suptitle("Convergence of the slit-lens results (steady-state iteration; ◆ time-dependent PIC at 300 µA/mm)")
    labels = {
        "num_elements": "Elements across the slit (p = 3)",
        "degree": "Spline degree p",
        "n_rays": "Rays",
        "dt": "Time step dt",
    }
    colours = {0.0: "#4e79a7", 300.0: "#e15759"}
    for col, parameter in enumerate(parameters):
        for row, (key, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            for current in currents:
                runs = [r for r in results["steady"] if r["current"] == current and r["parameter"] == parameter]
                xs = [_value_label(parameter, r[parameter]) for r in runs]
                ax.plot(xs, [r[key] for r in runs], "o-", color=colours[current], label=f"{current:g} µA/mm")
            if parameter in ("num_elements", "n_rays", "dt"):
                ax.set_xscale("log")
            for td in results["time_dependent"]:
                if parameter == "num_elements":
                    ax.plot(td["num_elements"][1], td[key], "D", color="k", fillstyle="none")
                elif parameter == "dt":
                    ax.plot(td["dt"], td[key], "D", color="k", fillstyle="none")
            if row == len(METRICS) - 1:
                ax.set_xlabel(labels[parameter])
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.savefig(output, dpi=120)
    plt.close(fig)
    print(f"Wrote {output}")


def main():
    full = "--full" in sys.argv
    cache = HERE / ("convergence_study.json" if full else "convergence_study_quick.json")
    if "--replot" in sys.argv and cache.exists():
        results = json.loads(cache.read_text())
    elif full:
        results = run_study(HERE / "output")
        cache.write_text(json.dumps(results, indent=1))
    else:
        results = run_study(HERE / "output", QUICK_CURRENTS, QUICK_SWEEPS, time_dependent=())
        cache.write_text(json.dumps(results, indent=1))
    for row in convergence_table(results):
        print(
            f"{row['current']:5.0f} µA/mm, {row['parameter']:12s}: "
            + ", ".join(f"{key} {row[key]:+.2e}" for key, _ in METRICS)
        )
    plot_results(results, HERE / ("convergence_study.png" if full else "convergence_study_quick.png"))


if __name__ == "__main__":
    main()
