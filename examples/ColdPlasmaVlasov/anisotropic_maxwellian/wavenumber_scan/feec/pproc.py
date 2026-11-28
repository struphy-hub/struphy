#!/usr/bin/env python3
"""Figure 4.5 post-processing for the existing FEEC Run-3 simulations.

This reproduces the historical Standard-FEM normalisation without rerunning
simulations or modifying any StruPhy source or historical Standard-FEM files.

Run from the repository root:
    python examples/ColdPlasmaVlasov/anisotropic_maxwellian/wavenumber_scan/feec/pproc.py

The historical Standard-FEM hot-energy diagnostic is
    en_jh = alpha^2/2 * sum_p(w_p |v_p|^2) + Eh_eq
with Eh_eq = Lz*n_h/2*(vth_parallel^2 + 2*vth_perp^2)
when the control variate is enabled.

In FEEC, en_f is explicitly the delta-f kinetic energy:
    en_f = alpha_hot^2/2 * sum_p(w_p |v_p|^2).
It therefore excludes the equilibrium Maxwellian energy. The FEEC equivalent
of historical en_jh is en_f + Eh_eq, not en_f alone.

All energies below are in the same normalised units as the FEEC scalar
diagnostics. For these parameters alpha_hot=-2 and n_input=0.05, so the
physical hot density alpha_hot^2*n_input equals 0.20, matching the historical
n_h = nu_h*Omega_pe^2 = 0.05*2^2 = 0.20.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

from struphy.post_processing.output import Output


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
FIGURES = ROOT / "output" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

K_VALUES = np.arange(3.00, 1.49, -0.25)  # historical legend order
K_ASCENDING = K_VALUES[::-1]

# Thesis Run-2 parameters (dimensionless, c = me = |Omega_ce| = 1).
NU_H_INPUT = 0.05
ALPHA_HOT = -2.0
N_HOT_PHYSICAL = ALPHA_HOT**2 * NU_H_INPUT  # 0.20, matches nu_h*Omega_pe^2
VTH_PARALLEL = 0.20
VTH_PERP = 0.60
CONTROL_VARIATE = True

EXPECTED_TEND = 400.0
EXPECTED_DT = 0.05
EXPECTED_SNAPSHOTS = 8001

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})


def equilibrium_hot_energy(k: float) -> float:
    """Return the control-variate Maxwellian baseline used by historical en_jh."""
    length_z = 2.0 * np.pi / k
    return (
        length_z
        * N_HOT_PHYSICAL
        / 2.0
        * (VTH_PARALLEL**2 + 2.0 * VTH_PERP**2)
    )


def load_feec_run(k: float):
    tag = f"{k:.2f}".replace(".", "p")
    run = REPO / f"thesis_feec_run3_k{tag}"
    raw = run / "data" / "data_proc0.hdf5"
    if not raw.is_file():
        raise FileNotFoundError(f"Missing completed FEEC run: {raw}")

    ds = Output(str(run)).evaluate("scalars")
    t = np.asarray(ds["t"].values, dtype=float)
    eb = np.asarray(ds["en_B"].values, dtype=float)
    ee = np.asarray(ds["en_E"].values, dtype=float)
    ej = np.asarray(ds["en_J"].values, dtype=float)
    ef = np.asarray(ds["en_f"].values, dtype=float)

    arrays = (t, eb, ee, ej, ef)
    if any(a.ndim != 1 for a in arrays):
        raise ValueError(f"k={k:.2f}: expected 1-D scalar histories.")
    if len({len(a) for a in arrays}) != 1:
        raise ValueError(f"k={k:.2f}: scalar-history lengths do not match.")
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError(f"k={k:.2f}: non-finite scalar data.")

    if len(t) != EXPECTED_SNAPSHOTS:
        raise ValueError(
            f"k={k:.2f}: expected {EXPECTED_SNAPSHOTS} snapshots, got {len(t)}."
        )
    if not np.isclose(t[0], 0.0) or not np.isclose(t[-1], EXPECTED_TEND):
        raise ValueError(
            f"k={k:.2f}: unexpected time interval {t[0]}..{t[-1]}."
        )
    if not np.allclose(np.diff(t), EXPECTED_DT, rtol=0.0, atol=1e-9):
        raise ValueError(f"k={k:.2f}: saved time step is not 0.05.")

    return t, eb, ee, ej, ef


def main() -> None:
    curves = {}
    rows = []

    print("=" * 88)
    print("FEEC FIGURE 4.5 — HISTORICALLY CONSISTENT TOTAL-ENERGY NORMALISATION")
    print("=" * 88)
    print("Historical en_jh = delta-f particle kinetic energy + control-variate Eh_eq")
    print("FEEC en_f         = delta-f particle kinetic energy only")
    print("Therefore FEEC hot-energy counterpart = en_f + Eh_eq")
    print()
    print(" k       Eh_eq          E0(correct)       EB(0)/E0       EB(400)/E0")

    for k in K_ASCENDING:
        t, eb, ee, ej, ef = load_feec_run(float(k))
        eh_eq = equilibrium_hot_energy(float(k))

        # Exact structural counterpart of the historical denominator:
        # E0 = en_E(0) + en_B(0) + en_jc(0) + en_jh(0)
        #    = en_E(0) + en_B(0) + en_J(0) + en_f(0) + Eh_eq.
        hot_energy = ef + (eh_eq if CONTROL_VARIATE else 0.0)
        total_energy = ee + eb + ej + hot_energy
        e0 = float(total_energy[0])

        if not np.isfinite(e0) or e0 <= 0.0:
            raise ValueError(f"k={k:.2f}: invalid corrected E0={e0}.")

        ratio = eb / e0
        if np.any(~np.isfinite(ratio)) or np.any(ratio <= 0.0):
            raise ValueError(f"k={k:.2f}: EB/E0 contains non-positive/non-finite values.")

        curves[float(k)] = (t, ratio)
        rows.append((
            float(k), eh_eq, e0, float(eb[0]), float(ef[0]),
            float(ratio[0]), float(ratio[-1]),
            float(np.min(ratio)), float(np.max(ratio)),
        ))

        print(
            f"{k:4.2f}  {eh_eq:13.6e}  {e0:13.6e}  "
            f"{ratio[0]:13.6e}  {ratio[-1]:13.6e}"
        )

    # Match the original Standard-FEM panel geometry and axis limits.
    fig = plt.figure(figsize=(6.0, 2.05))
    ax = fig.add_axes([0.105, 0.235, 0.605, 0.615])

    for k in K_VALUES:
        t, ratio = curves[float(k)]
        ax.plot(
            t,
            ratio,
            linewidth=0.8,
            label=fr"$k={k:.2f}\ |\Omega_{{ce}}|/c$",
        )

    ax.set_xlim(0, 400)
    ax.set_ylim(1e-12, 1.0)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_yscale("log")
    ax.set_yticks([1e-12, 1e-9, 1e-6, 1e-3, 1.0])
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.tick_params(
        axis="both", which="minor",
        left=False, right=False, bottom=False, top=False,
    )
    ax.set_xlabel(r"$t|\Omega_{ce}|$")
    ax.set_ylabel(r"$E_{\tilde{B}}/E(0)$")
    ax.set_title("Wavenumber scan", pad=3)
    ax.grid(False)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.015, 0.50),
        frameon=True,
        framealpha=1.0,
        edgecolor="0.65",
        borderpad=0.30,
        labelspacing=0.22,
        handlelength=1.8,
        handletextpad=0.35,
    )

    png = FIGURES / "fig4_5_reproduction_run3.png"
    pdf = FIGURES / "fig4_5_reproduction_run3.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    summary_path = FIGURES / "fig4_5_energy_summary_run3.csv"
    np.savetxt(
        summary_path,
        np.asarray(rows, dtype=float),
        delimiter=",",
        header=(
            "k,Eh_eq,E0_corrected,EB0,en_f0,"
            "EB_over_E0_initial,EB_over_E0_final,ratio_min,ratio_max"
        ),
        comments="",
    )

    print()
    print("Saved:")
    print(" ", png)
    print(" ", pdf)
    print(" ", summary_path)
    print()
    print("No simulations were run. No historical Standard-FEM or StruPhy source files were changed.")


if __name__ == "__main__":
    main()
