#!/usr/bin/env python3
"""
Post-process current StruPhy FEEC Run 1, CV on: total-energy conservation.

The saved CV-on scalar/en_f has a one-sample initialization convention change:
at t=0 it contains the sampled equilibrium hot-particle kinetic energy, while
from the next saved time onward it contains the delta-f perturbation contribution.
This script separates that constant equilibrium reference from the perturbation
energy before calculating the physical total-energy error.

No simulation is launched and no StruPhy source or HDF5 data is modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUN = "thesis_feec_run1_cv_final"
N_HOT = 0.06
VTH_PARALLEL = 0.20
VTH_PERP = 0.53
ALPHA_HOT = -2.0
DOMAIN_VOLUME = np.pi

# Independent analytic reference for validating the sampled equilibrium energy.
E_HOT_EQUILIBRIUM_ANALYTIC = (
    ALPHA_HOT**2 * 0.5 * N_HOT * DOMAIN_VOLUME
    * (VTH_PARALLEL**2 + 2.0 * VTH_PERP**2)
)


def load_data(path: Path) -> dict[str, np.ndarray]:
    keys = (
        "time/value", "scalar/en_B", "scalar/en_E",
        "scalar/en_J", "scalar/en_f", "scalar/en_tot",
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"HDF5 not found: {path}\n"
            "Check --repo-root and --run; no simulation was started."
        )
    with h5py.File(path, "r") as h5:
        missing = [k for k in keys if k not in h5]
        if missing:
            raise KeyError(f"Missing HDF5 datasets: {missing}")
        data = {k: np.asarray(h5[k], dtype=np.float64) for k in keys}

    t = data["time/value"]
    if t.ndim != 1 or t.size < 2 or not np.all(np.isfinite(t)):
        raise ValueError("Invalid time array.")
    if not np.all(np.diff(t) > 0):
        raise ValueError("Time array is not strictly increasing.")
    for k in keys[1:]:
        if data[k].shape != t.shape or not np.all(np.isfinite(data[k])):
            raise ValueError(f"Invalid shape or non-finite values in {k}.")
    return data


def make_figure(data: dict[str, np.ndarray], outdir: Path, run_name: str) -> dict:
    t = data["time/value"]
    en_b = data["scalar/en_B"]
    en_e = data["scalar/en_E"]
    en_j = data["scalar/en_J"]
    en_f_saved = data["scalar/en_f"]
    en_tot_saved = data["scalar/en_tot"]

    # First verify the HDF5 scalar identity exactly as recorded.
    saved_sum = en_b + en_e + en_j + en_f_saved
    scale = max(float(np.max(np.abs(en_tot_saved))),
                float(np.max(np.abs(saved_sum))), 1e-30)
    decomposition_abs = float(np.max(np.abs(en_tot_saved - saved_sum)))
    decomposition_rel = decomposition_abs / scale
    if decomposition_rel > 1e-9:
        raise ValueError(
            "HDF5 scalar identity failed: en_tot != en_B + en_E + en_J + en_f "
            f"(relative discrepancy {decomposition_rel:.3e})."
        )

    # In this CV-on output, en_f[0] is the sampled equilibrium kinetic energy;
    # en_f[1:] is the evolving delta-f kinetic-energy contribution. Use the
    # sampled value as the constant equilibrium reference. At t=0, delta-f=0.
    equilibrium_sampled = float(en_f_saved[0])
    if equilibrium_sampled <= 0:
        raise ValueError("Initial sampled equilibrium particle energy is not positive.")

    analytic_ratio = equilibrium_sampled / E_HOT_EQUILIBRIUM_ANALYTIC
    if not 0.90 <= analytic_ratio <= 1.10:
        raise ValueError(
            "The initial en_f value does not resemble the expected sampled "
            "equilibrium energy (outside ±10% of analytic reference); refusing "
            "to assume the CV initialization convention."
        )

    en_f_delta = en_f_saved.copy()
    en_f_delta[0] = 0.0
    perturbation_energy = en_b + en_e + en_j + en_f_delta
    physical_total = equilibrium_sampled + perturbation_energy
    initial_total = float(physical_total[0])
    if not np.isfinite(initial_total) or initial_total <= 0:
        raise ValueError("Reconstructed initial physical total energy is invalid.")

    relative_error = np.abs(physical_total - initial_total) / abs(initial_total)
    if not np.all(np.isfinite(relative_error)):
        raise ValueError("Non-finite relative energy error.")
    if relative_error[0] != 0.0:
        raise AssertionError("Initial relative energy error must be exactly zero.")

    imax = int(np.argmax(relative_error))
    positive = relative_error[relative_error > 0]
    floor = max(0.5 * float(np.min(positive)), 1e-14) if positive.size else 1e-14

    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })
    plotted = relative_error.copy()
    plotted[0] = floor  # log-axis display only; numerical value remains zero
    fig, ax = plt.subplots(figsize=(5.4, 3.55))
    ax.semilogy(t, np.maximum(plotted, floor), color="tab:purple")
    ax.set_xlim(0.0, 200.0)
    ax.set_xlabel(r"$t\,|\Omega_{ce}|$")
    ax.set_ylabel(
        r"$|\mathcal{E}(t)-\mathcal{E}(0)|/"
        r"|\mathcal{E}_{\mathrm{total}}(0)|$"
    )
    ax.set_title("Run 1: total-energy conservation (FEEC, CV on)")
    ax.set_ylim(bottom=floor, top=max(float(np.max(relative_error)) * 2.0, floor * 10))
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(outdir / "Figure_4_4_FEEC.png", dpi=400)
    fig.savefig(outdir / "Figure_4_4_FEEC.pdf")
    plt.close(fig)

    report = {
        "run": run_name,
        "method": "current FEEC, control variate on",
        "definition": (
            "abs(Ephysical(t)-Ephysical(0))/abs(Ephysical(0)); "
            "Ephysical=Eeq_sampled+en_B+en_E+en_J+en_f_delta"
        ),
        "initialization_handling": (
            "scalar/en_f[0] is the sampled equilibrium hot-particle kinetic "
            "energy; scalar/en_f[1:] is treated as delta-f perturbation energy. "
            "The t=0 delta-f contribution is set to zero."
        ),
        "sampled_equilibrium_energy": equilibrium_sampled,
        "analytic_equilibrium_energy_reference": float(E_HOT_EQUILIBRIUM_ANALYTIC),
        "sampled_to_analytic_equilibrium_ratio": float(analytic_ratio),
        "initial_perturbation_energy": float(perturbation_energy[0]),
        "initial_physical_total_energy": initial_total,
        "final_physical_total_energy": float(physical_total[-1]),
        "initial_relative_energy_error": float(relative_error[0]),
        "maximum_relative_energy_error": float(np.max(relative_error)),
        "time_of_maximum_error": float(t[imax]),
        "final_relative_energy_error": float(relative_error[-1]),
        "saved_scalar_decomposition_relative_discrepancy": decomposition_rel,
        "number_of_snapshots": int(t.size),
        "time_interval": [float(t[0]), float(t[-1])],
        "log_axis_floor_display_only": floor,
        "comparison_note": (
            "This is the current-FEEC CV-on total-energy diagnostic. For "
            "thesis comparison, use the geometric-methods energy-conservation "
            "curve (thesis Fig. 4.10), not historical Standard-FEM Fig. 4.4."
        ),
    }
    (outdir / "diagnostics_fig4_4.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None,
                        help="StruPhy repository root (default: current directory).")
    parser.add_argument("--run", default=DEFAULT_RUN,
                        help=f"Completed run directory (default: {DEFAULT_RUN}).")
    parser.add_argument("--output", default="post_processing_FEEC_Fig4_4",
                        help="Output directory, relative to repository root unless absolute.")
    args = parser.parse_args()

    repo = Path(args.repo_root).expanduser().resolve() if args.repo_root else Path.cwd().resolve()
    run_dir = Path(args.run).expanduser()
    if not run_dir.is_absolute():
        run_dir = repo / run_dir
    h5_path = run_dir / "data" / "data_proc0.hdf5"
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = repo / output

    print(f"Repository: {repo}")
    print(f"Run data:   {h5_path}")
    print(f"Output:     {output}")
    print("Simulation will NOT be run.")
    report = make_figure(load_data(h5_path), output, run_dir.name)
    print("\nPOST-PROCESSING COMPLETE")
    print(json.dumps(report, indent=2))
    print(f"\nPNG:  {output / 'Figure_4_4_FEEC.png'}")
    print(f"PDF:  {output / 'Figure_4_4_FEEC.pdf'}")
    print(f"JSON: {output / 'diagnostics_fig4_4.json'}")


if __name__ == "__main__":
    main()
