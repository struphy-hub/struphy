#!/usr/bin/env python3
"""
Figure 4.7 post-processing for CURRENT StruPhy FEEC output.

This is not the historical FEM postprocessor:
- reads StruPhy's FEEC HDF5 output;
- reconstructs Bx from the FEEC H(div) coefficient field;
- evaluates on unique periodic z points;
- computes the 2-D (k, omega) spectrum using FFT frequency axes;
- reproduces the thesis two-panel zoom/dispersion-branch layout.

Run from the StruPhy repository root:
    python examples/ColdPlasmaVlasov/anisotropic_maxwellian/pproc_current_feec_fig4_7.py

Optional:
    --h5 PATH
    --out PATH
"""

from pathlib import Path
import sys
import argparse
import json

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import ConnectionPatch

RUN = "thesis_feec_fig4_7_current_feec_fullf_noise"
NZ = 256
LZ = 80.0
DEGREE = (1, 1, 3)
C = 1.0
WCE = -1.0
WPE = 2.0


def reconstruct_bx(h5_path: Path, repo: Path):
    """Reconstruct physical Bx from current FEEC H(div) coefficients."""
    sys.path.insert(0, str(repo / "src"))
    from struphy.bsplines.evaluation_kernels_3d import evaluate_3d
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    with h5py.File(h5_path, "r") as h5:
        required = ("time/value", "feec/em_fields/b_field/1")
        for key in required:
            if key not in h5:
                raise KeyError(f"Missing FEEC HDF5 dataset: {key}")
        time = np.asarray(h5[required[0]], dtype=float)
        coeff = np.asarray(h5[required[1]], dtype=float)

    if coeff.shape[0] != len(time):
        raise ValueError(
            f"FEEC Bx time axis ({coeff.shape[0]}) does not match time samples ({len(time)})."
        )

    derham = Derham(
        TensorProductGrid(num_elements=(1, 1, NZ)),
        DerhamOptions(degree=DEGREE, bcs=(None, None, None)),
    )
    # Use the actual spline space of the first magnetic 2-form component.
    # Its basis is (N, D, D), with degrees (1, 0, 2).
    component_space = derham.V2fem.spaces[0]
    tn1, tn2, tn3 = component_space.knots
    indN, indD = derham.indN, derham.indD
    eval_args = (
        1, 2, 2, tn1, tn2, tn3,
        DEGREE[0], DEGREE[1] - 1, DEGREE[2] - 1,
        indN[0], indD[1], indD[2],
    )

    # The field is periodic in z. Never duplicate z=Lz in the FFT grid.
    z = np.arange(NZ, dtype=float) * LZ / NZ
    eta3 = z / LZ
    bx = np.empty((len(time), NZ), dtype=np.float64)

    for it in range(len(time)):
        if it % 500 == 0:
            print(f"FEEC reconstruction: {it}/{len(time)}")
        coeff_it = np.ascontiguousarray(coeff[it], dtype=np.float64)
        for iz, eta in enumerate(eta3):
            # Contravariant Piola map for the physical x-component.
            bx[it, iz] = float(
                evaluate_3d(*eval_args, coeff_it, 0.5, 0.5, float(eta))
            ) / LZ

    return time, z, bx


def cold_parallel_branches(k, repo: Path):
    """
    Use the thesis' own cold-plasma root solver for the analytical overlays.
    This imports only the dispersion solver: no historical FEM basis,
    coefficient reconstruction, or evaluation routine is used.
    """
    sys.path.insert(0, str(repo))
    import Utilitis_HybridCode as thesis_utils

    curves = [np.full(len(k), np.nan) for _ in range(3)]
    seeds = [0.0001, 1.5001, 2.5001]
    signs = [+1, -1, +1]

    for i, kval in enumerate(k):
        for j in range(3):
            guess = seeds[j] if i == 0 else curves[j][i - 1]
            root, *_ = thesis_utils.solveDispersionCold(
                float(kval), signs[j], C, WCE, WPE,
                float(guess), 1e-6, 100
            )
            curves[j][i] = root
    return curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--h5", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    repo = args.repo.resolve()
    h5 = args.h5.resolve() if args.h5 else (
        repo / RUN / "data" / "data_proc0.hdf5"
    )
    out = args.out.resolve() if args.out else repo / RUN / "figures"
    out.mkdir(parents=True, exist_ok=True)

    time, z, bx = reconstruct_bx(h5, repo)
    if len(time) < 4:
        raise ValueError("Insufficient saved time levels.")
    dtime = np.diff(time)
    dt = float(np.median(dtime))
    if not np.allclose(dtime, dt, rtol=1e-7, atol=1e-12):
        raise ValueError("Nonuniform time output: cannot use a standard 2-D FFT.")

    bx_rms = float(np.sqrt(np.mean(bx * bx)))
    bx_peak = float(np.max(np.abs(bx)))
    print(f"Bx min/max : {bx.min():.8e} / {bx.max():.8e}")
    print(f"Bx RMS     : {bx_rms:.8e}")

    # Critical: a zero field has no spectrum. Stop rather than draw a
    # misleading all-floor image and call it Figure 4.7.
    scale = max(float(np.max(np.abs(bx))), 1.0)
    if bx_peak <= 100.0 * np.finfo(float).eps * scale:
        raise RuntimeError(
            "FEEC Bx is identically zero (or at round-off). No Figure 4.7 "
            "spectrum can be formed from this run. This is a simulation/"
            "initialization issue, not a post-processing issue. Do not "
            "normalize zero FFT data or interpret an all-floor plot."
        )

    # Match the thesis convention: FFT of the reconstructed field, normalized
    # by its global maximum. No artificial perturbation is introduced here.
    signal = bx - np.mean(bx)
    spec = np.fft.fftshift(np.fft.fft2(signal))
    amp = np.abs(spec)
    amp_max = float(amp.max())
    if not np.isfinite(amp_max) or amp_max == 0.0:
        raise RuntimeError("FFT amplitude is zero/non-finite; refusing to plot.")

    amp_norm = amp / amp_max
    omega = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(time), d=dt))
    kvals = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(z), d=LZ / len(z)))
    K, W = np.meshgrid(kvals, omega)

    # Thesis analytical overlays, evaluated independently of historical FEM code.
    k_branch = np.linspace(0.1, 8.0, 400)
    branches = cold_parallel_branches(k_branch, repo)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 17,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    positive = amp_norm[amp_norm > 0]
    floor = max(float(positive.max()) * 1e-8, 1e-14)
    levels = np.logspace(np.log10(floor), 0.0, 60)
    norm = LogNorm(vmin=floor, vmax=1.0)

    # Thesis-style wide layout: equal-sized spectral panels and
    # an independently allocated colourbar column.
    fig = plt.figure(figsize=(13.0, 5.2))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.0, 1.0, 0.045],
        wspace=0.25,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    cf1 = ax1.contourf(
        K, W, np.maximum(amp_norm, floor),
        levels=levels, cmap="jet", norm=norm, extend="min"
    )
    ax1.plot(np.linspace(0, 8, 10), np.zeros(10), "k")
    ax1.plot(np.linspace(0, 8, 10), np.ones(10) * 6, "k")
    ax1.plot(np.zeros(10), np.linspace(0, 6, 10), "k")
    ax1.plot(np.ones(10) * 8, np.linspace(0, 6, 10), "k")
    ax1.set_xlim(float(kvals.min()), float(kvals.max()))
    ax1.set_ylim(-20, 20)
    ax1.set_xlabel(r"$kc/|\Omega_\mathrm{ce}|$")
    ax1.set_ylabel(r"$\omega_\mathrm{r}/|\Omega_\mathrm{ce}|$")
    ax1.set_title(
        r"$|\hat{B}_x(k,\omega_\mathrm{r})|/"
        r"|\hat{B}_x(k,\omega_\mathrm{r})|_\mathrm{max}$"
    )

    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    cf2 = ax2.contourf(
        K, W, np.maximum(amp_norm, floor),
        levels=levels, cmap="jet", norm=norm, extend="min"
    )
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 6)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for branch in branches:
        ax2.plot(k_branch, branch, "k--", linewidth=1.5)
    cb = fig.colorbar(
        cf2,
        cax=cax,
        ticks=np.logspace(-8, 0, 5),
    )
    cb.set_label("Normalized Fourier amplitude")

    # Connect the zoom rectangle to the expanded panel.
    for xyA, xyB in [
        ((0, 6), (0, 6)),
        ((0, 0), (0, 0)),
        ((0, 4.3), (8, 6)),
        ((0, 2.4), (8, 0)),
    ]:
        ax2.add_artist(ConnectionPatch(
            xyA=xyA, xyB=xyB, coordsA="data", coordsB="data",
            axesA=ax2, axesB=ax1, color="black", linewidth=1
        ))

    fig.tight_layout()
    png = out / "fig4_7_current_feec.png"
    pdf = out / "fig4_7_current_feec.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        out / "fig4_7_current_feec_spectrum.npz",
        time=time, z=z, Bx=bx, omega=omega, k=kvals,
        normalized_amplitude=amp_norm,
    )
    metadata = {
        "source_hdf5": str(h5),
        "time_samples": len(time),
        "space_samples": len(z),
        "dt": dt,
        "Tend": float(time[-1]),
        "Lz": LZ,
        "Nz": NZ,
        "Bx_min": float(bx.min()),
        "Bx_max": float(bx.max()),
        "Bx_rms": bx_rms,
        "normalization": "max(abs(fftshift(fft2(Bx-mean(Bx))))))",
        "branches": "thesis solveDispersionCold analytical branches; historical solver only, no historical FEM reconstruction",
    }
    (out / "fig4_7_current_feec_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("FIGURE 4.7 CURRENT FEEC COMPLETE")
    print("PNG:", png)
    print("PDF:", pdf)
    print("Data:", out / "fig4_7_current_feec_spectrum.npz")


if __name__ == "__main__":
    main()
