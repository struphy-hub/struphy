#!/usr/bin/env python3
"""
Reproducible FEEC Run-1 post-processing for thesis counterparts of Figs. 4.1–4.3.

No simulation is run. Reads completed StruPhy HDF5 output only.

Run placement:
  Put this file in:
    examples/ColdPlasmaVlasov/anisotropic_maxwellian/
  then run from the StruPhy environment:
    python pproc_feec_run1_all.py

Run assignments are deliberate:
  Figs. 4.1, 4.2 -> thesis_feec_run1_cv_final (CV on)
  Fig. 4.3       -> thesis_feec_run1_nocv_final (CV off)

The thesis's historical Figs. 4.1–4.4 use standard FEM with CV on.
Therefore Fig. 4.3 from the CV-off FEEC run is a diagnostic counterpart,
not a strictly like-for-like CV comparison. This distinction is written
into the report and figure metadata.

Outputs:
  post_processing_FEEC_Run1/
    Figure_4_1_FEEC.{pdf,png}
    Figure_4_2_FEEC.{pdf,png}
    Figure_4_3_FEEC.{pdf,png}
    diagnostics.json
    figure_4_3_marginals.npz

Important:
  - Figure 4.3 marginals are calculated from the saved Cartesian
    f(v1,v2,v3) histogram using cylindrical annular averaging:
       f_parallel(v_parallel) = integral 2*pi*v_perp*f dv_perp
       f_perp(v_perp)         = integral f dv_parallel
  - Figure 4.2 plots angular frequency omega/|Omega_ce|, as in the thesis.
    The energy signal Bx^2 is expected at 2*omega_r, not omega_r.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import detrend, windows


# ---------------------------- configuration ----------------------------

# FEEC Maxwellian input density is nu_h=0.06. In ColdPlasmaVlasov,
# particle coupling and kinetic energy carry alpha_hot**2; with alpha=-2,
# the effective physical density is alpha**2 * nu_h = 0.24, matching the
# thesis normalization n_h = nu_h * Omega_pe**2 = 0.06 * 2**2.
N_HOT = 0.06
N_HOT_THESIS = 0.24
VTH_PERP = 0.53
VTH_PARALLEL = 0.20
HOT_ALPHA = -2.0  # Derived by the completed FEEC run's BaseUnits.
DOMAIN_VOLUME = np.pi  # 1 x 1 x pi cuboid.
N_HOT_PHYSICAL = HOT_ALPHA**2 * N_HOT

# The saved scalar/en_f is already the kinetic-energy contribution used by
# this run's model scalar. Do not add a second analytic Maxwellian energy to
# scalar/en_tot: the HDF5 record proves en_tot == en_B+en_E+en_J+en_f, and
# en_f(0) is itself approximately the analytic Maxwellian kinetic energy.
# The analytic value below is retained ONLY as an independent initialization
# cross-check, never as an additive term in the energy normalization.
E_HOT_EQUILIBRIUM_REFERENCE = (
    HOT_ALPHA**2 * 0.5 * N_HOT * DOMAIN_VOLUME
    * (VTH_PARALLEL**2 + 2.0 * VTH_PERP**2)
)

OMEGA_R_ANALYTICAL = 0.4742
GAMMA_ANALYTICAL = 0.0447
OMEGA_ENERGY_ANALYTICAL = 2.0 * OMEGA_R_ANALYTICAL
K_PARALLEL = 2.0
V_RESONANT = abs(OMEGA_R_ANALYTICAL - 1.0) / K_PARALLEL

# Match the thesis's visible plotting windows for Fig. 4.3.
VPAR_RANGE = (-1.0, 1.0)
VPERP_RANGE = (0.0, 2.0)
N_RADIAL_BINS = 64
SUBCELL_QUADRATURE = 16

# FEEC logical point used for the Bx time trace.
ETA_POINT = (0.5, 0.5, 0.25)
DOMAIN_LENGTH_Z = np.pi
INITIAL_BX_AMPLITUDE = 1.0e-4


def find_repo_root(script_path: Path, explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Repository root does not exist: {root}")
        return root
    for parent in script_path.resolve().parents:
        if (parent / "src" / "struphy").is_dir():
            return parent
    raise RuntimeError(
        "Could not locate StruPhy repository root (expected src/struphy). "
        "Pass --repo-root /path/to/struphy."
    )


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.15,
        "figure.dpi": 140,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_scalar_run(h5_path: Path) -> dict[str, np.ndarray]:
    required = (
        "time/value", "scalar/en_B", "scalar/en_E",
        "scalar/en_J", "scalar/en_f", "scalar/en_tot",
    )
    if not h5_path.is_file():
        raise FileNotFoundError(f"Required completed-run file not found: {h5_path}")
    with h5py.File(h5_path, "r") as h5:
        missing = [key for key in required if key not in h5]
        if missing:
            raise KeyError(f"{h5_path}: missing HDF5 datasets {missing}")
        out = {key: np.asarray(h5[key], dtype=float) for key in required}
    t = out["time/value"]
    if t.ndim != 1 or len(t) < 3 or not np.all(np.isfinite(t)):
        raise ValueError("Invalid saved time array.")
    if not np.all(np.diff(t) > 0):
        raise ValueError("Saved time array must be strictly increasing.")
    for key, arr in out.items():
        if key == "time/value":
            continue
        if arr.shape != t.shape or not np.all(np.isfinite(arr)):
            raise ValueError(f"Invalid shape or non-finite values in {key}.")
    return out


def savefig(fig, outbase: Path) -> None:
    fig.savefig(outbase.with_suffix(".pdf"))
    fig.savefig(outbase.with_suffix(".png"), dpi=400)
    plt.close(fig)


def plot_fig41(data: dict[str, np.ndarray], outdir: Path) -> dict:
    t = data["time/value"]
    en_B = data["scalar/en_B"]
    en_E = data["scalar/en_E"]
    en_J = data["scalar/en_J"]
    en_f_delta = data["scalar/en_f"]
    en_tot = data["scalar/en_tot"]

    # Plot the scalar components exactly as saved by the completed model.
    # The analytic Maxwellian value is a cross-check only; adding it to en_tot
    # would double-count the initial hot-particle kinetic energy in this run.
    component_sum = en_B + en_E + en_J + en_f_delta
    scale = max(float(np.max(np.abs(en_tot))),
                float(np.max(np.abs(component_sum))), 1e-30)
    decomp_rel = float(np.max(np.abs(en_tot - component_sum))) / scale
    if decomp_rel > 1e-9:
        raise ValueError(
            "Saved scalar/en_tot is inconsistent with en_B+en_E+en_J+en_f "
            f"(relative discrepancy {decomp_rel:.3e})."
        )

    E0_physical = float(en_tot[0])
    if not np.isfinite(E0_physical) or E0_physical <= 0:
        raise ValueError("Initial saved total-energy normalization is invalid.")
    equilibrium_reference_ratio = float(en_f_delta[0] / E_HOT_EQUILIBRIUM_REFERENCE)

    fig, ax = plt.subplots(figsize=(5.4, 3.55))
    series = [
        (en_B, r"$\widetilde{\mathcal{E}}_B$", "tab:orange", "-"),
        (en_E, r"$\widetilde{\mathcal{E}}_E$", "tab:purple", "-"),
        (en_J, r"$\mathcal{E}_c$", "tab:brown", "--"),
        (en_f_delta, r"$\mathcal{E}_{h,\delta f}$", "tab:blue", "-"),
    ]
    for values, label, color, ls in series:
        y = np.ma.masked_less_equal(values / E0_physical, 0.0)
        ax.semilogy(t, y, color=color, ls=ls, label=label)

    tg = np.linspace(0.0, 200.0, 300)
    # Energy grows at twice the field-amplitude growth rate.
    growth_guide = 9.0e-9 * np.exp(2.0 * GAMMA_ANALYTICAL * tg)
    ax.semilogy(tg, growth_guide, color="black", ls="--", lw=0.9,
                label=rf"Expected energy growth ($\gamma={GAMMA_ANALYTICAL:.4f}$ for field amplitude)")
    ax.set(xlim=(0, 200), xlabel=r"$t\,|\Omega_{ce}|$",
           ylabel=r"Energy / $\mathcal{E}_{\mathrm{total}}(0)$",
           title="Run 1: energy components (FEEC, CV on)")
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    savefig(fig, outdir / "Figure_4_1_FEEC")
    return {
        "normalization": "saved model total energy en_tot(0), with no added equilibrium constant",
        "plotted_hot_particle_component": "saved scalar/en_f exactly as recorded by StruPhy",
        "analytic_equilibrium_energy_reference_not_added": float(E_HOT_EQUILIBRIUM_REFERENCE),
        "initial_en_f_to_analytic_equilibrium_reference_ratio": equilibrium_reference_ratio,
        "initial_saved_total_energy_normalization": E0_physical,
        "saved_en_tot_decomposition_relative_error": decomp_rel,
        "final_component_values_over_initial_physical_total": {
            "magnetic": float(en_B[-1] / E0_physical),
            "electric": float(en_E[-1] / E0_physical),
            "cold_current": float(en_J[-1] / E0_physical),
            "hot_particle_energy": float(en_f_delta[-1] / E0_physical),
        },
    }


def reconstruct_bx_trace(h5_path: Path, repo_root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    # Import from the same installed/source StruPhy tree as the simulation.
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))
    from struphy.bsplines.evaluation_kernels_3d import evaluate_3d
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    with h5py.File(h5_path, "r") as h5:
        t = np.asarray(h5["time/value"], dtype=float)
        coeff = np.asarray(h5["feec/em_fields/b_field/1"], dtype=float)
    if coeff.shape[0] != len(t):
        raise ValueError("Saved Bx coefficient time dimension does not match time array.")

    derham = Derham(
        TensorProductGrid(num_elements=(1, 1, 32)),
        DerhamOptions(degree=(1, 1, 1), bcs=(None, None, None)),
    )
    tn1, tn2, tn3 = derham.V0fem.knots
    indN, indD = derham.indN, derham.indD
    p1, p2, p3 = derham.degree[0], derham.degree[1] - 1, derham.degree[2] - 1

    # H(div) Bx component has N x D x D tensor-product structure.
    args = (
        1, 2, 2, tn1, tn2, tn3, p1, p2, p3,
        indN[0], indD[1], indD[2],
    )
    eta1, eta2, eta3 = ETA_POINT
    bx = np.empty(len(t), dtype=float)
    for n in range(len(t)):
        bx_logical = evaluate_3d(*args, coeff[n], eta1, eta2, eta3)
        # H(div) Piola map for diag(1,1,pi): Bx_phys = Bx_ref / pi.
        bx[n] = float(bx_logical) / DOMAIN_LENGTH_Z

    z = DOMAIN_LENGTH_Z * eta3
    bx_expected = INITIAL_BX_AMPLITUDE * np.sin(K_PARALLEL * z)
    initial_relerr = abs(bx[0] - bx_expected) / max(abs(bx_expected), 1e-30)
    meta = {
        "logical_evaluation_point": list(ETA_POINT),
        "physical_z": z,
        "reconstructed_Bx_initial": float(bx[0]),
        "expected_continuous_Bx_initial": float(bx_expected),
        "initial_pointwise_relative_difference": float(initial_relerr),
        "note": "Pointwise comparison is diagnostic; p=1 FEEC projection need not reproduce the sine pointwise exactly.",
    }
    return t, bx, meta


def plot_fig42(h5_path: Path, repo_root: Path, outdir: Path) -> dict:
    t, bx, init_meta = reconstruct_bx_trace(h5_path, repo_root)
    signal = bx**2

    # Use the linear-growth interval used for the frequency diagnostic.
    mask = (t >= 20.0) & (t <= 100.0)
    tw, yw = t[mask], signal[mask]
    if len(tw) < 32:
        raise ValueError("Insufficient saved samples in 20 <= t <= 100 for FFT.")
    dtv = np.diff(tw)
    if not np.allclose(dtv, np.median(dtv), rtol=1e-5, atol=1e-12):
        raise ValueError("FFT requires uniformly sampled saved times in selected window.")
    dt = float(np.median(dtv))

    # Remove the slowly varying component, then apply a Hann window to
    # control leakage. Zero-padding interpolates the displayed spectrum;
    # it does not improve the true 1/T frequency resolution.
    residual = detrend(yw, type="linear")
    window = windows.hann(len(residual), sym=False)
    nfft = 8 * len(residual)
    spectrum = np.abs(np.fft.rfft(residual * window, n=nfft))
    freq = np.fft.rfftfreq(nfft, d=dt)  # cycles per normalized time
    omega = 2.0 * np.pi * freq         # |Omega_ce| units
    valid = omega >= 0.25
    if not np.any(valid):
        raise ValueError("No usable positive-frequency bins.")
    local = np.flatnonzero(valid)
    ip = local[int(np.argmax(spectrum[valid]))]
    # Three-point parabolic interpolation in log amplitude.
    peak_omega = float(omega[ip])
    if 0 < ip < len(spectrum) - 1:
        yy = np.log(np.maximum(spectrum[ip-1:ip+2], np.finfo(float).tiny))
        den = yy[0] - 2.0*yy[1] + yy[2]
        if den != 0:
            delta = 0.5 * (yy[0] - yy[2]) / den
            peak_omega += float(delta * (omega[1] - omega[0]))
    expected = OMEGA_ENERGY_ANALYTICAL
    relerr = abs(peak_omega - expected) / expected
    angular_resolution = 2.0 * np.pi / float(tw[-1] - tw[0])

    fig, ax = plt.subplots(figsize=(5.4, 3.55))
    ax.plot(omega[valid], spectrum[valid], color="tab:purple",
            label=r"FFT of $B_x^2(t)$")
    ax.axvline(expected, color="black", ls="--", lw=1.0,
               label=rf"Thesis reference $2\omega_r={expected:.4f}$")
    ax.axvline(peak_omega, color="tab:orange", ls=":",
               label=rf"Interpolated FEEC peak: $\omega={peak_omega:.4f}$")
    ax.text(0.02, 0.97,
            rf"Window resolution: $\Delta\omega\approx{angular_resolution:.4f}$" + "\n"
            + r"$\alpha_h^2 n_h=0.24$ (thesis-normalized)",
            transform=ax.transAxes, va="top", fontsize=7)
    ax.set(xlim=(0.25, 2.0), xlabel=r"Angular frequency $\omega/|\Omega_{ce}|$",
           ylabel=r"$|\widehat{B_x^2}|$",
           title="Run 1: magnetic-field energy spectrum (FEEC, CV on)")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    savefig(fig, outdir / "Figure_4_2_FEEC")

    return {
        **init_meta,
        "fft_window": [20.0, 100.0],
        "number_of_samples": int(len(tw)),
        "sample_interval": dt,
        "true_frequency_resolution_cycles_per_time": float(1.0 / (tw[-1] - tw[0])),
        "true_angular_frequency_resolution": angular_resolution,
        "peak_offset_is_below_true_resolution": bool(abs(peak_omega - expected) < angular_resolution),
        "expected_energy_angular_frequency_thesis_reference": expected,
        "feec_input_hot_density": N_HOT,
        "feec_effective_physical_hot_density_alpha_squared_times_n": N_HOT_PHYSICAL,
        "thesis_hot_density": N_HOT_THESIS,
        "effective_density_ratio_feec_over_thesis": N_HOT_PHYSICAL / N_HOT_THESIS,
        "frequency_reference_uses_matched_effective_density": True,
        "measured_energy_angular_frequency_interpolated": peak_omega,
        "relative_frequency_difference": float(relerr),
        "note": "The plotted signal is Bx^2, so its expected angular frequency is 2*omega_r. The reported peak is parabolically interpolated; zero-padding does not improve physical resolution. Treat offsets smaller than the window-limited angular resolution as unresolved, not as a precision error measurement.",
    }


def get_centres(ds, axis: int) -> np.ndarray:
    key = f"bin_centers_{axis}"
    if key not in ds.attrs:
        raise KeyError(f"{ds.name}: missing required histogram attribute {key}")
    c = np.asarray(ds.attrs[key], dtype=float)
    if c.ndim != 1 or len(c) < 2 or not np.all(np.diff(c) > 0):
        raise ValueError(f"Invalid velocity-bin centres in {key}")
    return c


def cell_widths(centres: np.ndarray) -> np.ndarray:
    edges = np.empty(len(centres) + 1)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0] = centres[0] - 0.5 * (centres[1] - centres[0])
    edges[-1] = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    return np.diff(edges)


def make_annular_weights(v1: np.ndarray, v2: np.ndarray,
                         radial_edges: np.ndarray, q: int) -> np.ndarray:
    dv1 = cell_widths(v1)
    dv2 = cell_widths(v2)
    if not np.allclose(dv1, dv1[0], rtol=1e-5) or not np.allclose(dv2, dv2[0], rtol=1e-5):
        raise ValueError("This histogram must have uniform v1/v2 bins for sub-cell quadrature.")
    offsets = (np.arange(q) + 0.5) / q - 0.5
    W = np.zeros((len(radial_edges)-1, len(v1), len(v2)), dtype=float)
    for i, x in enumerate(v1):
        xx = x + offsets * dv1[i]
        for j, y in enumerate(v2):
            yy = y + offsets * dv2[j]
            rr = np.sqrt(xx[:, None]**2 + yy[None, :]**2).ravel()
            ib = np.searchsorted(radial_edges, rr, side="right") - 1
            good = (ib >= 0) & (ib < len(radial_edges)-1)
            W[:, i, j] += np.bincount(
                ib[good], weights=np.full(np.count_nonzero(good), dv1[i]*dv2[j]/q**2),
                minlength=len(radial_edges)-1
            )
    return W


def plot_fig43(h5_path: Path, outdir: Path) -> dict:
    key = "kinetic/hot_elec/f/v1_v2_v3_density"
    if not h5_path.is_file():
        raise FileNotFoundError(f"Missing CV-off histogram run: {h5_path}")
    with h5py.File(h5_path, "r") as h5:
        if key not in h5:
            raise KeyError(f"Missing {key} in {h5_path}")
        ds = h5[key]
        t = np.asarray(h5["time/value"], dtype=float)
        v1, v2, v3 = get_centres(ds, 1), get_centres(ds, 2), get_centres(ds, 3)
        if ds.ndim != 4 or ds.shape != (len(t), len(v1), len(v2), len(v3)):
            raise ValueError(f"Unexpected histogram shape {ds.shape}")
        F0 = np.asarray(ds[0], dtype=float)
        Ff = np.asarray(ds[-1], dtype=float)

    if not np.all(np.isfinite(F0)) or not np.all(np.isfinite(Ff)):
        raise ValueError("Non-finite values in saved 3D velocity histogram.")
    if abs(float(t[0])) > 1e-10 or abs(float(t[-1]) - 200.0) > 1e-8:
        raise ValueError(
            f"Figure 4.3 requires initial/final snapshots at t=0 and t=200; "
            f"the histogram contains t={t[0]:.8g} to {t[-1]:.8g}. "
            "Do not label the first/last available snapshots as thesis endpoints."
        )
    if np.min(F0) < -1e-12 or np.min(Ff) < -1e-12:
        raise ValueError("Histogram contains materially negative values; cannot interpret as a density.")

    # Clip to thesis plotting domain. Integrals are then explicitly checked;
    # the script will refuse to silently accept a truncated density.
    radial_edges = np.linspace(VPERP_RANGE[0], VPERP_RANGE[1], N_RADIAL_BINS + 1)
    radial = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    W = make_annular_weights(v1, v2, radial_edges, SUBCELL_QUADRATURE)
    annulus_area = W.sum(axis=(1, 2))
    if np.any(annulus_area <= 0):
        raise ValueError("At least one radial annulus is not covered by the Cartesian histogram.")

    # Area-average f over the Cartesian samples belonging to each annulus.
    # F dimensions: (v1, v2, v_parallel).
    def radial_average(F):
        integral = np.einsum("rij,ijk->rk", W, F, optimize=True)
        return integral / annulus_area[:, None]

    fr0 = radial_average(F0)
    frf = radial_average(Ff)

    # Normalize each snapshot to the known hot-electron density. This
    # removes any histogram convention (probability density vs n_h*f)
    # while preserving the shape and makes the thesis dimensional scaling
    # explicit. The normalization is based on the represented plotting
    # domain; check that the histogram domain covers the full populated tail.
    dv3 = cell_widths(v3)
    dvr = np.diff(radial_edges)
    def normalize_histogram_density(fr):
        nrepresented = float(np.sum(fr * (2*np.pi*radial[:, None]) *
                                     dvr[:, None] * dv3[None, :]))
        if not np.isfinite(nrepresented) or nrepresented <= 0:
            raise ValueError("Invalid represented hot-electron density.")
        # StruPhy histogram output may be stored either as a physical
        # density (integral n_h) or as a unit-normalized PDF (integral 1).
        # Accept only these identifiable conventions; never force an
        # arbitrary integral to n_h, which could hide truncated velocity tails.
        if abs(nrepresented - N_HOT) / N_HOT <= 0.15:
            factor, convention = HOT_ALPHA**2, "FEEC-normalized distribution scaled by alpha_hot^2 to physical density"
        elif abs(nrepresented - 1.0) <= 0.15:
            factor, convention = N_HOT_PHYSICAL, "unit-normalized PDF scaled by effective physical density alpha_hot^2*n_h"
        else:
            raise ValueError(
                f"Velocity histogram integrates to {nrepresented:.8g} over "
                f"the thesis plotting domain; expected approximately FEEC input density={N_HOT} "
                "or 1.0. This may indicate truncated velocity coverage, a "
                "different histogram normalization, or incorrect dataset semantics. "
                "Do not renormalize blindly: inspect the histogram bounds/metadata."
            )
        return fr * factor, nrepresented, convention

    fr0, n0_before, norm0 = normalize_histogram_density(fr0)
    frf, nf_before, normf = normalize_histogram_density(frf)
    if norm0 != normf:
        raise ValueError(
            "Initial and final histograms use different normalization conventions: "
            f"{norm0!r} versus {normf!r}."
        )

    # Thesis marginals:
    # f_parallel(v_parallel) = integral_0^inf 2*pi*v_perp*f dv_perp
    # f_perp(v_perp)         = integral_-inf^inf f dv_parallel
    fpar0 = np.sum(fr0 * (2*np.pi*radial[:, None]) * dvr[:, None], axis=0)
    fparf = np.sum(frf * (2*np.pi*radial[:, None]) * dvr[:, None], axis=0)
    fperp0 = np.sum(fr0 * dv3[None, :], axis=1)
    fperpf = np.sum(frf * dv3[None, :], axis=1)

    # Plot only the thesis ranges; retain all histogram data for integrals.
    psel = (v3 >= VPAR_RANGE[0]) & (v3 <= VPAR_RANGE[1])
    rsel = (radial >= VPERP_RANGE[0]) & (radial <= VPERP_RANGE[1])
    if np.count_nonzero(psel) < 3 or np.count_nonzero(rsel) < 3:
        raise ValueError("Velocity histogram resolution does not cover thesis plot ranges.")

    dpar = fparf - fpar0
    dperp = fperpf - fperp0

    dv_parallel = float(np.median(dv3))
    parallel_resolution_warning = len(v3) < 96 or dv_parallel > 0.04
    if parallel_resolution_warning:
        print(
            "WARNING: the saved parallel-velocity histogram is coarse "
            f"({len(v3)} bins, median dv_parallel={dv_parallel:.6g}). "
            "It cannot resolve the resonant flattening with thesis-level "
            "confidence. The plotted marginal is a coarse diagnostic only."
        )

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))
    # Top-left: parallel marginal and resonant markers.
    ax[0, 0].plot(v3[psel], fpar0[psel], color="tab:orange", label=r"$t=0$")
    ax[0, 0].plot(v3[psel], fparf[psel], color="tab:purple", label=r"$t=200$")
    for sign in (-1, 1):
        ax[0, 0].axvline(sign*V_RESONANT, color="black", ls="--", lw=0.8)
    ax[0, 0].set(xlim=VPAR_RANGE, xlabel=r"$v_\parallel/c$",
                 ylabel=r"$f_h(v_\parallel)c^3/|\Omega_{ce}|$",
                 title="Parallel velocity marginal")
    ax[0, 0].legend(frameon=False)

    # Top-right: perpendicular marginal.
    ax[0, 1].plot(radial[rsel], fperp0[rsel], color="tab:orange", label=r"$t=0$")
    ax[0, 1].plot(radial[rsel], fperpf[rsel], color="tab:purple", label=r"$t=200$")
    ax[0, 1].set(xlim=VPERP_RANGE, xlabel=r"$v_\perp/c$",
                 ylabel=r"$f_h(v_\perp)c^2/|\Omega_{ce}|$",
                 title="Perpendicular velocity marginal")
    ax[0, 1].legend(frameon=False)

    # Bottom row: final minus initial, as described in thesis text.
    ax[1, 0].plot(v3[psel], dpar[psel], color="tab:blue")
    ax[1, 0].axhline(0, color="black", lw=0.7)
    for sign in (-1, 1):
        ax[1, 0].axvline(sign*V_RESONANT, color="black", ls="--", lw=0.8)
    ax[1, 0].set(xlim=VPAR_RANGE, xlabel=r"$v_\parallel/c$",
                 ylabel=r"$\Delta f_h(v_\parallel)c^3/|\Omega_{ce}|$",
                 title="Change in parallel marginal")

    ax[1, 1].plot(radial[rsel], dperp[rsel], color="tab:blue")
    ax[1, 1].axhline(0, color="black", lw=0.7)
    ax[1, 1].set(xlim=VPERP_RANGE, xlabel=r"$v_\perp/c$",
                 ylabel=r"$\Delta f_h(v_\perp)c^2/|\Omega_{ce}|$",
                 title="Change in perpendicular marginal")

    fig.suptitle(
        f"Run 1: hot-electron velocity marginals (FEEC, CV off; {len(v3)} parallel bins)",
        y=1.01,
    )
    fig.tight_layout()
    savefig(fig, outdir / "Figure_4_3_FEEC")

    # Moment/shape diagnostics from the represented radial distribution.
    # Integrate on the represented domain after density normalization.
    def moments(fr):
        mass = np.sum(fr * (2*np.pi*radial[:, None]) *
                      dvr[:, None] * dv3[None, :])
        mpar2 = np.sum(fr * (2*np.pi*radial[:, None]) *
                       dvr[:, None] * (v3[None, :]**2) * dv3[None, :]) / mass
        mperp2 = np.sum(fr * (2*np.pi*radial[:, None]) *
                        dvr[:, None] * (radial[:, None]**2) * dv3[None, :]) / mass
        return {"represented_density": float(mass),
                "mean_vparallel_squared": float(mpar2),
                "mean_vperp_squared": float(mperp2),
                "anisotropy_ratio_vperp2_over_2vparallel2":
                    float(mperp2 / (2.0*mpar2))}

    np.savez_compressed(
        outdir / "figure_4_3_marginals.npz",
        time=t, v_parallel=v3, v_perp=radial,
        f_parallel_initial=fpar0, f_parallel_final=fparf,
        f_perp_initial=fperp0, f_perp_final=fperpf,
        delta_f_parallel=dpar, delta_f_perp=dperp,
        radial_average_initial=fr0, radial_average_final=frf,
    )
    return {
        "run": "thesis_feec_run1_nocv_final",
        "control_variate": False,
        "snapshots_used": [float(t[0]), float(t[-1])],
        "feec_input_hot_density_nu_h": N_HOT,
        "feec_effective_physical_hot_density_alpha_squared_times_nu_h": N_HOT_PHYSICAL,
        "thesis_hot_density_nu_h_times_omega_pe_squared": N_HOT_THESIS,
        "effective_feec_to_thesis_density_ratio": N_HOT_PHYSICAL / N_HOT_THESIS,
        "pre_normalization_density_in_thesis_plot_window": {
            "initial": n0_before, "final": nf_before
        },
        "histogram_normalization_convention": norm0,
        "moments_initial": moments(fr0),
        "moments_final": moments(frf),
        "resonant_velocities": [-V_RESONANT, V_RESONANT],
        "parallel_velocity_resolution": {
            "number_of_bins": int(len(v3)),
            "median_bin_width": dv_parallel,
            "bins_from_zero_to_resonance": float(V_RESONANT / dv_parallel),
            "quantitative_resonance_resolution_warning": bool(parallel_resolution_warning),
        },
        "distribution_difference_convention": "final minus initial",
        "normalization_note": (
            "The histogram is accepted only if its represented integral is "
            "consistent with FEEC input density nu_h=0.06 or a unit-normalized PDF. "
            "FEEC-normalized histograms are multiplied by alpha_hot^2=4 to obtain physical n_h=0.24; unit PDFs are scaled directly by 0.24. "
            "Any other integral raises an error rather than hiding truncated tails."
        ),
    }



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None,
                        help="StruPhy repository root; auto-detected when omitted.")
    parser.add_argument("--cv-run", default="thesis_feec_run1_cv_final")
    parser.add_argument("--nocv-run", default="thesis_feec_run1_nocv_final")
    args = parser.parse_args()

    repo = find_repo_root(Path(__file__), args.repo_root)
    cv_h5 = repo / args.cv_run / "data" / "data_proc0.hdf5"
    nocv_h5 = repo / args.nocv_run / "data" / "data_proc0.hdf5"
    outdir = repo / "post_processing_FEEC_Run1"
    outdir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    print(f"Repository: {repo}")
    print(f"CV-on data: {cv_h5}")
    print(f"CV-off data: {nocv_h5}")
    print(f"Output: {outdir}")
    print("No simulation will be run.")

    cv_data = load_scalar_run(cv_h5)
    diagnostics = {
        "benchmark": "Holderied (2018), Run 1, current FEEC implementation",
        "figure_run_assignments": {
            "4.1": {"run": args.cv_run, "control_variate": True},
            "4.2": {"run": args.cv_run, "control_variate": True},
            "4.3": {"run": args.nocv_run, "control_variate": False},
        },
        "fig4_1": plot_fig41(cv_data, outdir),
        "fig4_2": plot_fig42(cv_h5, repo, outdir),
        "fig4_3": plot_fig43(nocv_h5, outdir),
        "interpretation_limits": [
            "The CV-off FEEC Fig. 4.3 is not a like-for-like control-variate comparison with historical standard-FEM Fig. 4.3, which uses CV on.",
            "Energy post-processing uses the saved scalar/en_tot directly. The analytic Maxwellian kinetic-energy estimate is an independent initialization cross-check only and is NOT added to en_tot; the HDF5 scalar identity en_tot=en_B+en_E+en_J+en_f is verified. No Np multiplier is applied.",
            "The FEEC Maxwellian input n=0.06 is the normalized nu_h parameter; the ColdPlasmaVlasov alpha_hot^2 coupling/energy normalization gives effective physical density 4*0.06=0.24, matching the thesis n_h=nu_h*Omega_pe^2. Fig. 4.3 histogram marginals are converted to this physical normalization.",
            "The FEEC Fig. 4.3 run is CV off whereas thesis Fig. 4.3 is CV on; it is not a like-for-like control-variate comparison, although the effective density normalization matches.",
            "The Fig. 4.3 parallel histogram has 32 bins in the current saved output; resonant flattening is under-resolved and cannot be recovered by interpolation.",
            "The Fig. 4.2 interpolated peak offset is below the finite-window angular-frequency resolution; report agreement within resolution, not a sub-percent measured accuracy.",
            "Velocity-marginal normalization reports the pre-normalization represented density; inspect it for truncation before interpreting moments.",
        ],
    }
    report = outdir / "diagnostics.json"
    report.write_text(json.dumps(diagnostics, indent=2))
    print("\nPOST-PROCESSING COMPLETE")
    for p in sorted(outdir.glob("Figure_4_*.pdf")):
        print(p)
    print(report)
    print("\nKey diagnostics:")
    print(json.dumps({
        "fig4_2": diagnostics["fig4_2"],
        "fig4_3": diagnostics["fig4_3"],
    }, indent=2))


if __name__ == "__main__":
    main()
