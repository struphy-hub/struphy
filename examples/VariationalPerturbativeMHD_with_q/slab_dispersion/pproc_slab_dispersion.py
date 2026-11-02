"""Post-process and plot the MHD slab dispersion.

Run as ``python pproc_slab_dispersion.py [sim_folder] [flags of params_slab_dispersion.py] [--show]``; the flags
must be those of the run. The figures are written into ``<sim_folder>/results`` (``--show`` shows them instead):
the space-time power spectra of u_x (Alfvén branch) and of the pressure perturbation (slow and fast branches) with
the fitted and exact phase speeds, and the energies against time.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt

from struphy import Output

SCRIPT_DIR = Path(__file__).resolve().parent

# The setup of the run (parsed from the same command line).
sys.path.insert(0, str(SCRIPT_DIR))
import params_slab_dispersion as params  # noqa: E402

DEFAULT_OUTPUT = Path(params.args.out_folder) / params.args.sim_folder


def exact_speeds():
    """Alfvén, slow and fast phase speeds for k along z in the homogeneous slab."""
    Bsq = params.B0x**2 + params.B0y**2 + params.B0z**2
    p0 = params.beta * Bsq / 2.0
    vA2 = Bsq / params.n0
    cs2 = params.gamma * p0 / params.n0
    cos2 = params.B0z**2 / Bsq
    delta = 4.0 * cs2 * vA2 * cos2 / (cs2 + vA2) ** 2
    v_alfven = np.sqrt(vA2 * cos2)
    v_slow = np.sqrt(0.5 * (cs2 + vA2) * (1.0 - np.sqrt(1.0 - delta)))
    v_fast = np.sqrt(0.5 * (cs2 + vA2) * (1.0 + np.sqrt(1.0 - delta)))
    return v_alfven, v_slow, v_fast


def space_time_spectrum(t, data):
    """Absolute value of the space-time Fourier transform of data[t, z], for omega >= 0 and k >= 0."""
    nt, nz = data.shape
    dt = t[1] - t[0]
    dz = params.Lz / nz
    spec = np.abs(np.fft.fftshift(np.fft.fft2(data - np.mean(data, axis=0))))
    omega = np.fft.fftshift(np.fft.fftfreq(nt, dt)) * 2 * np.pi
    k = np.fft.fftshift(np.fft.fftfreq(nz, dz)) * 2 * np.pi
    # waves in -z and +z have the same speed: fold k -> |k|, keep omega >= 0
    i_om, i_k = omega >= 0, k >= 0
    spec_pos = spec[i_om][:, i_k]
    spec_neg = spec[i_om][:, np.flatnonzero(k <= 0)[::-1]][:, : spec_pos.shape[1]]
    return omega[i_om], k[i_k], spec_pos + spec_neg


def fit_speed(omega, k, spec, v_guess, v_other=None, width=2):
    """Least-squares fit of omega = v k to the spectral peaks near omega = v_guess k; wave numbers where the branch
    is closer than two frequency bins to the branch with speed v_other are skipped."""
    domega = omega[1] - omega[0]
    ks, oms = [], []
    for j in range(len(k) // 8, len(k) // 2):
        if v_other is not None and abs(v_guess - v_other) * k[j] < 2 * domega:
            continue
        center = int(np.argmin(np.abs(omega - v_guess * k[j])))
        lo, hi = max(center - width, 1), min(center + width + 1, len(omega) - 1)
        if hi <= lo:
            continue
        ks.append(k[j])
        oms.append(omega[lo + int(np.argmax(spec[lo:hi, j]))])
    if len(ks) < 2:
        return np.nan
    ks, oms = np.array(ks), np.array(oms)
    return float(np.sum(ks * oms) / np.sum(ks**2))


def plot_spectrum(omega, k, spec, branches, title):
    """Normalized spectral power with lines omega = v k for each (label, v, style) in branches."""
    power = (spec / np.max(spec)) ** 2
    fig, ax = plt.subplots(figsize=(7, 6))
    pc = ax.pcolormesh(k, omega, np.maximum(power, 1e-12), norm=colors.LogNorm(vmin=1e-8, vmax=1.0), cmap="plasma")
    fig.colorbar(pc, ax=ax, label="normalized spectral power")
    for label, v, style in branches:
        if np.isfinite(v):
            ax.plot(k, v * k, style, lw=1.5, label=rf"{label}: $v={v:.3f}$")
    ax.set(xlabel=r"$k$", ylabel=r"$\omega$", title=title, ylim=(0.0, omega[-1]))
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def saver(run, save_figs):
    """A function saving figures as ``<name>.png`` into ``<path_out>/results`` and closing them (``save_figs``)."""
    results_dir = Path(run.path_out) / "results"
    results_dir.mkdir(exist_ok=True)

    def save(figs):
        if save_figs:
            for name, fig in figs.items():
                fig.savefig(results_dir / f"{name}.png")
                plt.close(fig)

    return save


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    run = Output(path_out)
    save = saver(run, save_figs)

    # fields on the line x = y = 0.5, at the cell centres in z
    nz = run.grid.num_elements[2]
    eta3 = (np.arange(nz) + 0.5) / nz
    ux = run.evaluate("mhd/velocity", eta1=0.5, eta2=0.5, eta3=eta3, representation="v", component=0)
    q = run.evaluate("mhd/sqrt_p", eta1=0.5, eta2=0.5, eta3=eta3, representation="0")
    t = ux.t.values
    ux = np.asarray(ux.transpose("t", "eta3").values)
    q = np.asarray(q.transpose("t", "eta3").values)

    # pressure perturbation p - p0 = 2 q0 q~ + q~^2, with q0 = sqrt(p0)
    Bsq = params.B0x**2 + params.B0y**2 + params.B0z**2
    q0 = np.sqrt(params.beta * Bsq / 2.0)
    p = 2.0 * q0 * q + q**2

    v_alfven, v_slow, v_fast = exact_speeds()

    if len(t) >= 16:
        omega, k, spec_u = space_time_spectrum(t, ux)
        _, _, spec_p = space_time_spectrum(t, p)
        v_alfven_fit = fit_speed(omega, k, spec_u, v_alfven)
        v_slow_fit = fit_speed(omega, k, spec_p, v_slow, v_other=v_alfven)
        v_fast_fit = fit_speed(omega, k, spec_p, v_fast)
        print(f"Alfvén: fit {v_alfven_fit:.4f}, exact {v_alfven:.4f}")
        print(f"slow:   fit {v_slow_fit:.4f}, exact {v_slow:.4f}")
        print(f"fast:   fit {v_fast_fit:.4f}, exact {v_fast:.4f}")

        fig = plot_spectrum(
            omega,
            k,
            spec_u,
            [("fit", v_alfven_fit, "w:"), ("exact", v_alfven, "c--")],
            r"Alfvén branch: spectrum of $u_x$",
        )
        save({"spectrum_ux": fig})
        fig = plot_spectrum(
            omega,
            k,
            spec_p,
            [
                ("slow fit", v_slow_fit, "w:"),
                ("slow exact", v_slow, "c--"),
                ("fast fit", v_fast_fit, "r:"),
                ("fast exact", v_fast, "g--"),
            ],
            "Slow and fast branches: spectrum of the pressure",
        )
        save({"spectrum_p": fig})
    else:
        print(f"Only {len(t)} saved time steps, no spectra.")

    names = ["en_U", "en_mag_1", "en_mag_2", "en_thermo_1", "en_thermo_2"]
    energies = run.evaluate("scalars", variables=names + ["en_tot"])
    fig, axs = plt.subplots(2, 1, sharex=True)
    for name in names:
        axs[0].plot(energies.t, energies[name] - energies[name][0], label=name)
    axs[0].set(ylabel="energy change")
    axs[0].legend()
    en_tot = energies["en_tot"]
    axs[1].plot(energies.t, (en_tot - en_tot[0]) / en_tot[0])
    axs[1].set(xlabel="time", ylabel="relative change of en_tot")
    fig.tight_layout()
    save({"energies": fig})

    if not save_figs:
        plt.show()


if __name__ == "__main__":
    # the flags of the run are parsed by params_slab_dispersion.py, the rest here
    _, other_flags = params.parser.parse_known_args()
    parser = argparse.ArgumentParser(description="Plot a saved simulation run.")
    parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: the output folder of the run flags, beside this script)",
    )
    parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    args = parser.parse_args(other_flags)
    main(args.path_out, save_figs=not args.show)
