"""Post-process and plot the shear Alfvén wave of GyrokineticMaxwellZLR.

Run as ``python pproc_shear_alfven.py [flags of params_shear_alfven.py]``; the flags must be those of the run.
Plots the energies (fields, species, total) over time and compares the oscillation frequency of the magnetic
energy with the shear Alfvén frequency omega_A = k_par / sqrt(1 + sum_s alpha_s^2 n_0 / B_0^2) of the model
at k_perp = 0 (light units), see Remmerswaal, Hatzky, Sonnendrücker, J. Plasma Phys. 91, E127 (2025).
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from struphy import Output

SCRIPT_DIR = Path(__file__).resolve().parent

# The setup of the run (parsed from the same command line).
sys.path.insert(0, str(SCRIPT_DIR))
import params_shear_alfven as params  # noqa: E402

DEFAULT_OUTPUT = Path(params.args.out_folder) / params.args.sim_folder


def alfven_frequency(run):
    """Shear Alfvén frequency k_par / sqrt(1 + sum_s alpha_s^2 n_0 / B_0^2) of the run (light units, k_par = 2 pi / L_z)."""
    model = params.model
    polarization = sum(
        species.var.species.equation_params.alpha**2 for species in (model.kinetic_ions, model.kinetic_electrons)
    )
    n0_over_B02 = params.equil.params["n0"] / params.equil.params["B0z"] ** 2
    k_par = 2 * np.pi / params.domain.params["r3"]
    return k_par / np.sqrt(1.0 + polarization * n0_over_B02)


def oscillation_frequency(time, values):
    """Angular frequency of a signal: a least-squares fit of ``c0 + c1 cos(omega t + phi)``,
    started from the dominant peak of the (zero-padded) spectrum."""
    from scipy.optimize import curve_fit

    values = values - np.mean(values)
    n = 16 * len(values)
    spectrum = np.abs(np.fft.rfft(values, n=n))
    freqs = np.fft.rfftfreq(n, d=time[1] - time[0])
    omega_fft = 2 * np.pi * freqs[np.argmax(spectrum[1:]) + 1]

    def model(t, c0, c1, omega, phi):
        return c0 + c1 * np.cos(omega * t + phi)

    try:
        popt, _ = curve_fit(model, time, values, p0=(0.0, np.max(np.abs(values)), omega_fft, 0.0))
        return abs(popt[2])
    except RuntimeError:
        return omega_fft


def main(path_out=DEFAULT_OUTPUT, save_results=False):
    run = Output(path_out).pproc().with_time_units("normalized")

    names = ("en_E", "en_B", "en_ions", "en_electrons", "en_tot")
    scalars = run.evaluate("scalars", variables=names)
    time = scalars["en_tot"].t.values

    def show(fig, name):
        """Show the figure, or with ``save_results`` save it as ``<name>.png`` into the results folder of the run."""
        if save_results:
            results_dir = run.path_out / "results"
            results_dir.mkdir(exist_ok=True)
            fig.savefig(results_dir / f"{name}.png")
            plt.close(fig)
        else:
            plt.show()

    # energies
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in ("en_E", "en_B"):
        axs[0].plot(time, scalars[name], label=name)
    axs[0].set(xlabel="time", ylabel="energy", title="Field energies")
    axs[0].legend()

    en_tot = scalars["en_tot"].values
    axs[1].plot(time, (en_tot - en_tot[0]) / en_tot[0])
    axs[1].set(xlabel="time", ylabel=r"$(H - H_0) / H_0$", title="Relative energy error")
    fig.tight_layout()
    show(fig, "energies")

    # shear Alfvén frequency: the magnetic energy of a standing wave oscillates at 2 omega
    omega_A = alfven_frequency(run)
    period = 2 * np.pi / omega_A
    print(f"Shear Alfvén frequency of the model: omega_A = {omega_A:.5f} (period {period:.2f})")
    if time[-1] - time[0] >= period:
        omega = oscillation_frequency(time, scalars["en_B"].values) / 2
        print(
            f"Measured from the magnetic energy:   omega   = {omega:.5f} (relative deviation {omega / omega_A - 1:+.3f})"
        )
    else:
        print("Run too short to measure the frequency (needs at least one period).")

    print(f"Relative energy error at the end: {(en_tot[-1] - en_tot[0]) / en_tot[0]:.3e}")


if __name__ == "__main__":
    main()
