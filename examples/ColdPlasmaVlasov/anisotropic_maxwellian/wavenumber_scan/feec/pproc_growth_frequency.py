#!/usr/bin/env python3
"""
Self-contained FEEC Run 3 post-processing for thesis Figure 4.6.

Inputs:
    Seven thesis_feec_run3_k* simulation directories.

Calculates:
    - Analytical complex frequencies from the hot-plasma dispersion relation.
    - Numerical growth rates from the upper envelope of en_B.
    - Numerical frequencies from spatially resolved Bx^2 spectra.
    - Frequency-window uncertainty from three time intervals.

No intermediate NPZ, CSV, or /tmp script is required.
No simulations are run.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import root
from scipy.special import wofz
from scipy.signal import find_peaks

from struphy.post_processing.output import Output


# ---------------------------------------------------------------------
# Paths and physical parameters
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parent
FIGURES = BASE / "output" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

RUN_PREFIX = "thesis_feec_run3"
K_VALUES = np.arange(1.50, 3.0001, 0.25)

OCE = -1.0
OPE = 2.0
NUH = 0.05
VPAR = 0.20
VPERP = 0.60

GROWTH_T0 = 30.0
GROWTH_T1 = 100.0

FREQUENCY_WINDOWS = (
    (20.0, 100.0),
    (30.0, 110.0),
    (40.0, 120.0),
)

FREQUENCY_MIN = 0.40
FREQUENCY_MAX = 1.60

# Acceptance limits used for the final numerical validation.
MAX_FREQUENCY_ERROR_PERCENT = 3.0
MAX_WINDOW_SPREAD_PERCENT = 1.0
MIN_SPECTRAL_PROMINENCE = 0.50
MIN_GROWTH_R2 = 0.95


# ---------------------------------------------------------------------
# Analytical hot-plasma dispersion relation
# ---------------------------------------------------------------------

def plasma_Z(x):
    return 1j * np.sqrt(np.pi) * wofz(x)


def dispersion_relation(w, k):
    xi = (w + OCE) / (k * np.sqrt(2.0) * VPAR)

    return (
        1.0
        - k**2 / w**2
        - OPE**2 / (w * (w + OCE))
        + NUH * OPE**2 / w**2
        * (
            w / (k * np.sqrt(2.0) * VPAR) * plasma_Z(xi)
            - (1.0 - VPERP**2 / VPAR**2)
            * (1.0 + xi * plasma_Z(xi))
        )
    )


def analytical_frequency(k, guess):
    def residual(x):
        w = x[0] + 1j * x[1]
        value = dispersion_relation(w, k)
        return [value.real, value.imag]

    solution = root(
        residual,
        [guess.real, guess.imag],
        method="hybr",
    )

    if not solution.success:
        raise RuntimeError(
            f"Analytical dispersion root failed at k={k:.2f}: "
            f"{solution.message}"
        )

    w = solution.x[0] + 1j * solution.x[1]
    residual_size = abs(dispersion_relation(w, k))

    if residual_size > 1e-7:
        raise RuntimeError(
            f"Dispersion residual at k={k:.2f} is "
            f"{residual_size:.3e}"
        )

    if w.imag <= 0:
        raise RuntimeError(
            f"Unexpected analytical growth rate at k={k:.2f}: {w.imag}"
        )

    return w


# ---------------------------------------------------------------------
# Numerical growth rate from the magnetic-energy upper envelope
# ---------------------------------------------------------------------

def numerical_growth_rate(t, eb, k):
    mask = (
        (t >= GROWTH_T0)
        & (t <= GROWTH_T1)
        & np.isfinite(eb)
        & (eb > 0)
    )

    tw = np.asarray(t[mask], dtype=float)
    ew = np.asarray(eb[mask], dtype=float)

    if len(tw) < 20:
        raise RuntimeError(
            f"k={k:.2f}: insufficient en_B samples for growth fitting."
        )

    dt = float(np.median(np.diff(tw)))

    if not np.allclose(np.diff(tw), dt, rtol=1e-7, atol=1e-10):
        raise RuntimeError(f"k={k:.2f}: nonuniform en_B time grid.")

    peak_distance = max(1, int(round(2.0 / dt)))
    peaks, _ = find_peaks(ew, distance=peak_distance)

    if len(peaks) < 5:
        raise RuntimeError(
            f"k={k:.2f}: only {len(peaks)} upper-envelope peaks found."
        )

    tp = tw[peaks]
    ep = ew[peaks]

    log_energy = np.log(ep)
    slope, intercept = np.polyfit(tp, log_energy, 1)
    fitted = slope * tp + intercept

    ss_res = np.sum((log_energy - fitted)**2)
    ss_tot = np.sum((log_energy - np.mean(log_energy))**2)

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Magnetic energy grows as exp(2*gamma*t).
    gamma = 0.5 * float(slope)

    if not np.isfinite(gamma) or gamma <= 0:
        raise RuntimeError(f"k={k:.2f}: invalid fitted growth rate.")

    return gamma, float(r2), len(peaks)


# ---------------------------------------------------------------------
# Numerical frequency from spatially resolved Bx^2 spectra
# ---------------------------------------------------------------------

def numerical_frequency_from_bx(bx, t, gamma, k, window):
    t0, t1 = window

    if bx.ndim != 2 or bx.shape[0] != len(t):
        raise RuntimeError(f"k={k:.2f}: unexpected Bx dimensions {bx.shape}")

    if not np.all(np.isfinite(bx)):
        raise RuntimeError(f"k={k:.2f}: non-finite Bx values.")

    dt_values = np.diff(t)
    dt = float(np.median(dt_values))

    if not np.allclose(dt_values, dt, rtol=1e-7, atol=1e-10):
        raise RuntimeError(f"k={k:.2f}: nonuniform field time grid.")

    # Correctly remove the exponential magnetic-energy growth:
    # Bx^2 ~ exp(2*gamma*t).
    corrected = bx**2 * np.exp(
        -2.0 * gamma * (t - t0)
    )[:, None]

    # Remove temporal mean and linear drift independently at each z.
    corrected -= np.mean(corrected, axis=0, keepdims=True)

    coefficients = np.polyfit(t, corrected, 1)
    corrected -= (
        coefficients[0][None, :] * t[:, None]
        + coefficients[1][None, :]
    )

    corrected *= np.hanning(len(t))[:, None]

    nfft = max(32 * len(t), 65536)

    spectrum = np.fft.rfft(corrected, n=nfft, axis=0)
    power = np.mean(np.abs(spectrum)**2, axis=1)
    angular = 2.0 * np.pi * np.fft.rfftfreq(nfft, d=dt)

    band = (
        (angular >= FREQUENCY_MIN)
        & (angular <= FREQUENCY_MAX)
    )

    indices = np.flatnonzero(band)
    band_power = power[band]

    peaks, properties = find_peaks(
        band_power,
        prominence=0,
    )

    if len(peaks) == 0:
        raise RuntimeError(
            f"k={k:.2f}, window {t0:g}-{t1:g}: no resolved spectral peak."
        )

    selected = int(np.argmax(band_power[peaks]))
    local_peak = int(peaks[selected])
    j = int(indices[local_peak])

    # Parabolic interpolation in log-power around the spectral maximum.
    delta = 0.0

    if 0 < j < len(power) - 1:
        q = np.log(
            np.maximum(
                power[j-1:j+2],
                np.finfo(float).tiny,
            )
        )

        denominator = q[0] - 2.0*q[1] + q[2]

        if abs(denominator) > 1e-14:
            delta = float(
                np.clip(
                    0.5 * (q[0] - q[2]) / denominator,
                    -1.0,
                    1.0,
                )
            )

    energy_peak = (
        2.0 * np.pi * (j + delta) / (nfft * dt)
    )

    # Bx^2 energy oscillates at twice the field frequency.
    omega = 0.5 * energy_peak

    prominence = float(
        properties["prominences"][selected]
        / max(float(np.max(band_power)), np.finfo(float).tiny)
    )

    return float(omega), prominence


# ---------------------------------------------------------------------
# Main post-processing
# ---------------------------------------------------------------------

def main():
    print("=" * 86)
    print("FEEC FIGURE 4.6 — SELF-CONTAINED RUN 3 POST-PROCESSING")
    print("=" * 86)

    rows = []
    analytical_guess = 0.35 + 0.03j

    for k in K_VALUES:
        tag = f"{k:.2f}".replace(".", "p")
        run = ROOT / f"{RUN_PREFIX}_k{tag}"
        hdf5 = run / "data" / "data_proc0.hdf5"

        if not hdf5.is_file():
            raise FileNotFoundError(
                f"Missing Run 3 simulation data: {hdf5}"
            )

        print(f"\nProcessing k={k:.2f}", flush=True)

        output = Output(str(run))
        scalars = output.evaluate("scalars")

        t_scalar = np.asarray(scalars["t"].values, dtype=float)
        eb = np.asarray(scalars["en_B"].values, dtype=float)

        if len(t_scalar) != 8001 or len(eb) != 8001:
            raise RuntimeError(
                f"k={k:.2f}: expected 8001 scalar samples."
            )

        if not np.isclose(t_scalar[-1], 400.0):
            raise RuntimeError(f"k={k:.2f}: unexpected final simulation time.")

        if not np.allclose(
            np.diff(t_scalar), 0.05, rtol=0.0, atol=1e-9
        ):
            raise RuntimeError(f"k={k:.2f}: unexpected scalar time step.")

        gamma, growth_r2, envelope_count = numerical_growth_rate(
            t_scalar, eb, float(k)
        )

        analytical = analytical_frequency(
            float(k), analytical_guess
        )
        analytical_guess = analytical

        # Read the actual evolved magnetic field once for all three windows.
        # The interval 20..120 is sampled every 0.1 time units.
        ids = np.arange(400, 2401, 2)
        t_field = ids * 0.05

        field = output.evaluate(
            "em_fields/b_field",
            t=ids.tolist(),
            representation="v",
        )

        bx_all = np.asarray(
            field.isel(
                component=0,
                eta1=0,
                eta2=0,
            ).values,
            dtype=float,
        )

        if bx_all.ndim != 2 or bx_all.shape[0] != len(t_field):
            raise RuntimeError(
                f"k={k:.2f}: unexpected Bx field shape {bx_all.shape}."
            )

        window_values = []

        for window in FREQUENCY_WINDOWS:
            t0, t1 = window
            mask = (t_field >= t0) & (t_field <= t1)

            omega, prominence = numerical_frequency_from_bx(
                bx_all[mask, :],
                t_field[mask],
                gamma,
                float(k),
                window,
            )

            error = (
                100.0 * abs(omega - analytical.real)
                / analytical.real
            )

            window_values.append(
                (omega, prominence, error)
            )

            print(
                f"  {t0:.0f}-{t1:.0f}: "
                f"omega={omega:.6f}, "
                f"analytical={analytical.real:.6f}, "
                f"error={error:.3f}%, "
                f"prominence={prominence:.4f}",
                flush=True,
            )

        window_array = np.asarray(window_values, dtype=float)

        omega_values = window_array[:, 0]
        omega_mean = float(np.mean(omega_values))
        omega_std = float(np.std(omega_values, ddof=1))

        mean_error = float(np.mean(window_array[:, 2]))
        spread = float(
            100.0
            * (np.max(omega_values) - np.min(omega_values))
            / omega_mean
        )
        minimum_prominence = float(
            np.min(window_array[:, 1])
        )

        rows.append([
            k,
            omega_mean,
            omega_std,
            analytical.real,
            gamma,
            analytical.imag,
            growth_r2,
            mean_error,
            spread,
            minimum_prominence,
            envelope_count,
            *omega_values,
        ])

        print(
            f"  FINAL: omega={omega_mean:.6f} +/- {omega_std:.6f}; "
            f"gamma={gamma:.7e}; analytical gamma={analytical.imag:.7e}; "
            f"growth R2={growth_r2:.5f}",
            flush=True,
        )

    data = np.asarray(rows, dtype=float)

    # -------------------------------------------------------------
    # Validation before producing final outputs
    # -------------------------------------------------------------

    checks = {
        "Seven wavenumbers processed": len(data) == 7,
        "Growth-envelope R2 >= 0.95": np.all(data[:, 6] >= MIN_GROWTH_R2),
        "Mean frequency error <= 3%": np.all(
            data[:, 7] <= MAX_FREQUENCY_ERROR_PERCENT
        ),
        "Three-window spread <= 1%": np.all(
            data[:, 8] <= MAX_WINDOW_SPREAD_PERCENT
        ),
        "Spectral prominence >= 0.50": np.all(
            data[:, 9] >= MIN_SPECTRAL_PROMINENCE
        ),
    }

    print("\nVALIDATION")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    if not all(checks.values()):
        raise RuntimeError(
            "Figure 4.6 validation failed. No final figure was generated."
        )

    # -------------------------------------------------------------
    # Save the independent numerical analysis
    # -------------------------------------------------------------

    csv_path = FIGURES / "fig4_6_feec_analysis_run3_FINAL.csv"
    npz_path = FIGURES / "fig4_6_feec_analysis_run3_FINAL.npz"

    header = (
        "k,omega_numerical_mean,omega_window_std,omega_analytical,"
        "gamma_numerical,gamma_analytical,growth_R2,"
        "mean_frequency_error_percent,frequency_window_spread_percent,"
        "min_spectral_prominence,envelope_peak_count,"
        "omega_window_20_100,omega_window_30_110,omega_window_40_120"
    )

    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.10g",
    )

    np.savez(
        npz_path,
        k=data[:, 0],
        omega_numerical=data[:, 1],
        omega_window_std=data[:, 2],
        omega_analytical=data[:, 3],
        gamma_numerical=data[:, 4],
        gamma_analytical=data[:, 5],
        growth_R2=data[:, 6],
        mean_frequency_error_percent=data[:, 7],
        frequency_window_spread_percent=data[:, 8],
        min_spectral_prominence=data[:, 9],
        envelope_peak_count=data[:, 10],
        omega_window_20_100=data[:, 11],
        omega_window_30_110=data[:, 12],
        omega_window_40_120=data[:, 13],
    )

    # -------------------------------------------------------------
    # Thesis-style final Figure 4.6
    # -------------------------------------------------------------

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    })

    k = data[:, 0]
    omega_num = data[:, 1]
    omega_std = data[:, 2]
    omega_ana = data[:, 3]
    gamma_num = data[:, 4]
    gamma_ana = data[:, 5]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5.8, 2.35)
    )

    ax1.plot(
        k, omega_ana,
        "-", linewidth=0.9,
        label="analytical",
    )
    ax1.errorbar(
        k, omega_num, yerr=omega_std,
        fmt="o", markersize=3,
        capsize=2, linewidth=0.7,
        label="FEEC",
    )

    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks([0, 2, 4, 6])
    ax1.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax1.set_xlabel(r"$kc/|\Omega_{ce}|$")
    ax1.set_ylabel(r"$\omega_r/|\Omega_{ce}|$")
    ax1.set_title("Comparison real frequencies")
    ax1.legend(loc="lower right", frameon=True)

    ax2.plot(
        k, gamma_ana,
        "-", linewidth=0.9,
        label="analytical",
    )
    ax2.plot(
        k, gamma_num,
        "o", markersize=3,
        label="FEEC",
    )

    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 0.05)
    ax2.set_xticks([0, 2, 4, 6])
    ax2.set_yticks([0, 0.02, 0.04])
    ax2.set_xlabel(r"$kc/|\Omega_{ce}|$")
    ax2.set_ylabel(r"$\gamma/|\Omega_{ce}|$")
    ax2.set_title("Comparison growth rates")
    ax2.legend(loc="upper right", frameon=True)

    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.22,
        top=0.80,
        wspace=0.38,
    )

    png_path = FIGURES / "fig4_6_feec_reproduction_run3_FINAL.png"
    pdf_path = FIGURES / "fig4_6_feec_reproduction_run3_FINAL.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print("\n" + "=" * 86)
    print("FIGURE 4.6 GENERATED SUCCESSFULLY")
    print("PNG:", png_path)
    print("PDF:", pdf_path)
    print("CSV:", csv_path)
    print("NPZ:", npz_path)
    print("No simulations were run.")
    print("=" * 86)


if __name__ == "__main__":
    main()
