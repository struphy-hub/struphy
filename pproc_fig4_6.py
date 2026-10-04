from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from scipy.optimize import root
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent

K_VALUES = np.arange(1.50, 3.0001, 0.25)

# ============================================================
# Run-2 parameters from thesis
# ============================================================

C = 1.0
OCE = -1.0
OPE = 2.0
NUH = 0.05
VPAR = 0.20
VPERP = 0.60

# Linear-phase analysis window.
# Chosen before nonlinear saturation in all seven runs.
FIT_T0 = 30.0
FIT_T1 = 100.0

# Frequency-analysis window.
FFT_T0 = 20.0
FFT_T1 = 120.0


# ============================================================
# Plasma dispersion function
# ============================================================

def Z(x):
    """
    Plasma dispersion function for Im(x) > 0:
        Z(x) = i sqrt(pi) w(x)
    """
    return 1j * np.sqrt(np.pi) * wofz(x)


# ============================================================
# Dispersion relation (thesis Eq. 2.46)
# ============================================================

def dispersion(w, k):
    xi = (w + OCE) / (k * np.sqrt(2.0) * VPAR)

    return (
        1.0
        - k**2 / w**2
        - OPE**2 / (w * (w + OCE))
        + NUH * OPE**2 / w**2
        * (
            w / (k * np.sqrt(2.0) * VPAR) * Z(xi)
            - (1.0 - VPERP**2 / VPAR**2)
            * (1.0 + xi * Z(xi))
        )
    )


def solve_analytical(k, guess):
    """Solve Eq. (2.46) for the unstable R-wave."""

    def residual(x):
        w = x[0] + 1j * x[1]
        d = dispersion(w, k)
        return [d.real, d.imag]

    sol = root(residual, [guess.real, guess.imag], method="hybr")

    if not sol.success:
        raise RuntimeError(
            f"Analytical dispersion solve failed for k={k:.2f}: "
            f"{sol.message}"
        )

    w = sol.x[0] + 1j * sol.x[1]

    if abs(dispersion(w, k)) > 1e-7:
        raise RuntimeError(
            f"Poor dispersion residual for k={k:.2f}: "
            f"{abs(dispersion(w, k)):.3e}"
        )

    return w


# ============================================================
# Numerical extraction
# ============================================================

def load_run(k):
    tag = f"{k:.2f}".replace(".", "p")
    path = ROOT / f"thesis_run2_k{tag}" / "run3_fig4_7.npz"

    if not path.exists():
        raise FileNotFoundError(path)

    return np.load(path)


def numerical_growth_rate(d):
    """
    Magnetic energy grows as exp(2 gamma t).

    Therefore:
        slope[log(E_B)] = 2 gamma
    """

    t = d["time"]
    EB = d["en_B"]

    E0 = (
        d["en_E"][0]
        + d["en_B"][0]
        + d["en_jc"][0]
        + d["en_jh"][0]
    )

    y = EB / E0

    mask = (
        (t >= FIT_T0)
        & (t <= FIT_T1)
        & np.isfinite(y)
        & (y > 0)
    )

    xfit = t[mask]
    yfit = np.log(y[mask])

    coeff = np.polyfit(xfit, yfit, 1)

    slope = coeff[0]
    gamma = slope / 2.0

    # R^2 diagnostic
    pred = np.polyval(coeff, xfit)
    ss_res = np.sum((yfit - pred)**2)
    ss_tot = np.sum((yfit - np.mean(yfit))**2)
    r2 = 1.0 - ss_res / ss_tot

    return gamma, r2


def numerical_frequency(d):
    t = np.asarray(d["time"], dtype=float)
    Bx = np.asarray(d["Bx"])
    z = np.asarray(d["z"], dtype=float)

    k = float(d_k_current)

    # ------------------------------------------------------------
    # 1. Determine numerical growth rate from magnetic-field energy
    # ------------------------------------------------------------
    EB = np.asarray(d["en_B"], dtype=float)

    fit_mask = (t >= 30.0) & (t <= 100.0)
    if np.count_nonzero(fit_mask) < 2:
        return np.nan

    coeff = np.polyfit(
        t[fit_mask],
        np.log(np.maximum(EB[fit_mask], 1e-300)),
        1
    )

    # Magnetic energy ~ exp(2 gamma t)
    gamma = 0.5 * coeff[0]

    # ------------------------------------------------------------
    # 2. Frequency-analysis window
    # ------------------------------------------------------------
    mask = (t >= 20.0) & (t <= 120.0)

    tw = t[mask]
    B = Bx[mask, :]

    # ------------------------------------------------------------
    # 3. Extract complex spatial Fourier coefficient at +k
    # ------------------------------------------------------------
    phase = np.exp(-1j * k * z)

    signal = np.trapezoid(
        B * phase[None, :],
        z,
        axis=1
    )

    # ------------------------------------------------------------
    # 4. Remove exponential growth
    # ------------------------------------------------------------
    signal *= np.exp(-gamma * tw)

    # Remove DC component
    signal -= np.mean(signal)

    # Hann window
    signal *= np.hanning(len(signal))

    # ------------------------------------------------------------
    # 5. Complex FFT
    # ------------------------------------------------------------
    nfft = max(16 * len(signal), 16384)

    spectrum = np.fft.fft(signal, n=nfft)

    dt_sample = float(np.mean(np.diff(tw)))

    freqs = np.fft.fftfreq(
        nfft,
        d=dt_sample
    )

    # Positive frequencies only
    positive = freqs > 0.0

    freqs_pos = freqs[positive]
    power_pos = np.abs(spectrum[positive])**2

    # Physically relevant search range
    search = (
        (freqs_pos > 0.05) &
        (freqs_pos < 1.5)
    )

    if not np.any(search):
        return np.nan

    idx = np.argmax(power_pos[search])

    frequency_cycles = freqs_pos[search][idx]

    # FFT frequency is cycles/time; convert to angular frequency
    omega = 2.0 * np.pi * frequency_cycles

    return float(omega)

# ============================================================
# Main analysis
# ============================================================

numerical_gamma = []
numerical_omega = []

analytical_gamma = []
analytical_omega = []

# Continuation guesses for analytical root.
guess = 0.35 + 0.03j

print()
print("=" * 78)
print("THESIS FIGURE 4.6 — RUN 2 ANALYSIS")
print("=" * 78)
print()
print(
    f"Growth fit: t = {FIT_T0:.0f} ... {FIT_T1:.0f}"
)
print(
    f"Frequency analysis: t = {FFT_T0:.0f} ... {FFT_T1:.0f}"
)
print()

for k in K_VALUES:

    d = load_run(k)

    # Make k available to numerical_frequency without copying arrays.
    d_k_current = k

    gamma_num, r2 = numerical_growth_rate(d)
    omega_num = numerical_frequency(d)

    w_guess = guess
    w_ana = solve_analytical(k, w_guess)

    # Use current solution as continuation guess.
    guess = w_ana

    numerical_gamma.append(gamma_num)
    numerical_omega.append(omega_num)

    analytical_gamma.append(w_ana.imag)
    analytical_omega.append(w_ana.real)

    print(
        f"k={k:.2f}  "
        f"omega_num={omega_num:.6f}  "
        f"omega_ana={w_ana.real:.6f}  "
        f"gamma_num={gamma_num:.6f}  "
        f"gamma_ana={w_ana.imag:.6f}  "
        f"R2={r2:.5f}"
    )

print()
print("=" * 78)


numerical_gamma = np.asarray(numerical_gamma)
numerical_omega = np.asarray(numerical_omega)

analytical_gamma = np.asarray(analytical_gamma)
analytical_omega = np.asarray(analytical_omega)


# ============================================================
# Save numerical/analytical values
# ============================================================

np.savez(
    ROOT / "fig4_6_analysis.npz",
    k=K_VALUES,
    omega_numerical=numerical_omega,
    omega_analytical=analytical_omega,
    gamma_numerical=numerical_gamma,
    gamma_analytical=analytical_gamma,
)

np.savetxt(
    ROOT / "fig4_6_analysis.txt",
    np.column_stack([
        K_VALUES,
        numerical_omega,
        analytical_omega,
        numerical_gamma,
        analytical_gamma,
    ]),
    header=(
        "k omega_numerical omega_analytical "
        "gamma_numerical gamma_analytical"
    ),
)


# ============================================================
# Thesis-style Figure 4.6
# ============================================================

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

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(5.8, 2.35)
)

# ------------------------------------------------------------
# Left: real frequency
# ------------------------------------------------------------

kplot = np.linspace(0.01, 6.0, 500)

# Analytical unstable branch over plotting range
omega_curve = []

for kk in kplot:
    try:
        # Initial guess follows the numerical branch.
        # For the continuation curve use the previous solution.
        if len(omega_curve) == 0:
            g = 0.2 + 0.01j
        else:
            g = omega_curve[-1]

        w = solve_analytical(kk, g)
        omega_curve.append(w)
    except Exception:
        omega_curve.append(np.nan + 1j*np.nan)

omega_curve = np.asarray(omega_curve)

ax1.plot(
    kplot,
    omega_curve.real,
    linewidth=0.9,
    label="analytical"
)

ax1.plot(
    K_VALUES,
    numerical_omega,
    "o",
    markersize=3.0,
    markeredgewidth=0.5,
    label="numerics"
)

ax1.set_xlim(0, 6)
ax1.set_ylim(0, 1.0)

ax1.set_xticks([0, 2, 4, 6])
ax1.set_yticks([0, 0.25, 0.50, 0.75, 1.00])

ax1.set_xlabel(r"$kc/|\Omega_{ce}|$")
ax1.set_ylabel(r"$\omega_r/|\Omega_{ce}|$")
ax1.set_title("Comparison real frequencies", pad=3)

ax1.legend(
    loc="lower right",
    frameon=True,
    framealpha=1.0,
    borderpad=0.3
)

# ------------------------------------------------------------
# Right: growth rate
# ------------------------------------------------------------

ax2.plot(
    kplot,
    omega_curve.imag,
    linewidth=0.9,
    label="analytical"
)

ax2.plot(
    K_VALUES,
    numerical_gamma,
    "o",
    markersize=3.0,
    markeredgewidth=0.5,
    label="numerics"
)

ax2.set_xlim(0, 6)
ax2.set_ylim(0, 0.05)

ax2.set_xticks([0, 2, 4, 6])
ax2.set_yticks([0, 0.02, 0.04])

ax2.set_xlabel(r"$kc/|\Omega_{ce}|$")
ax2.set_ylabel(r"$\gamma/|\Omega_{ce}|$")
ax2.set_title("Comparison growth rates", pad=3)

ax2.legend(
    loc="upper right",
    frameon=True,
    framealpha=1.0,
    borderpad=0.3
)

fig.subplots_adjust(
    left=0.095,
    right=0.985,
    bottom=0.22,
    top=0.80,
    wspace=0.38
)

png = ROOT / "fig4_6_reproduction.png"
pdf = ROOT / "fig4_6_reproduction.pdf"

fig.savefig(png, dpi=300)
fig.savefig(pdf)

plt.close(fig)

print()
print("Saved:", png)
print("Saved:", pdf)
print("Saved:", ROOT / "fig4_6_analysis.npz")
print("Saved:", ROOT / "fig4_6_analysis.txt")
