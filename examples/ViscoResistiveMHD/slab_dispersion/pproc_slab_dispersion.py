"""Post-processing for params_ViscoResistiveMHD.py.

Compares numerical Alfven/slow/fast wave speeds against the analytical
dispersion relation of a homogeneous MHD slab.
"""

import params_slab_dispersion as params

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI
from matplotlib import colors
from matplotlib import pyplot as plt

from struphy.diagnostics.diagn_tools import power_spectrum_2d


def stack_time_series(data: dict):
    """Stack time-keyed scalar plotting data into a time-first array."""
    times = sorted(data, key=float)
    values = xp.stack([data[t][0] for t in times], axis=0)
    return times, values


def analytical_speeds(B0x, B0y, B0z, beta, n0, gamma):
    """Alfven, slow and fast speeds for a homogeneous MHD slab."""
    Bsq = B0x**2 + B0y**2 + B0z**2
    p0 = beta * Bsq / 2.0

    vA = xp.sqrt(Bsq / n0)
    v_alfven = vA * B0z / xp.sqrt(Bsq)

    cS = xp.sqrt(gamma * p0 / n0)
    delta = 4.0 * B0z**2 * cS**2 * vA**2 / ((cS**2 + vA**2) ** 2 * Bsq)

    v_slow = xp.sqrt(0.5 * (cS**2 + vA**2) * (1.0 - xp.sqrt(1.0 - delta)))
    v_fast = xp.sqrt(0.5 * (cS**2 + vA**2) * (1.0 + xp.sqrt(1.0 - delta)))

    disp_params = dict(B0x=B0x, B0y=B0y, B0z=B0z, p0=p0, n0=n0, gamma=gamma)
    return float(v_alfven), float(v_slow), float(v_fast), disp_params


def fit_branch(omega, kvec, spectrum, expected_speed, competing_speed=None,
                search_bins=1, min_separation_bins=2):
    """Least-squares fit of omega = v*k near an expected branch speed."""
    domega = float(omega[1] - omega[0])
    k_fit, omega_fit = [], []

    for j in range(kvec.size // 8, kvec.size // 2):
        k = float(kvec[j])

        if competing_speed is not None:
            separation = abs(expected_speed - competing_speed) * k
            if separation < min_separation_bins * domega:
                continue

        center = int(xp.argmin(xp.abs(omega - expected_speed * k)))
        lower = max(1, center - search_bins)
        upper = min(omega.size - 1, center + search_bins + 1)
        if upper <= lower:
            continue

        peak = lower + int(xp.argmax(spectrum[lower:upper, j]))
        omega_peak = float(omega[peak])

        # sub-bin refinement via quadratic interpolation
        if 0 < peak < omega.size - 1:
            y0, y1, y2 = spectrum[peak - 1, j], spectrum[peak, j], spectrum[peak + 1, j]
            denom = y0 - 2 * y1 + y2
            if abs(denom) > 1e-30:
                offset = 0.5 * (y0 - y2) / denom
                if abs(offset) <= 1.0:
                    omega_peak += offset * domega

        k_fit.append(k)
        omega_fit.append(omega_peak)

    assert len(k_fit) >= 2, "Not enough spectrally resolved points to fit the branch."

    k_fit, omega_fit = xp.asarray(k_fit), xp.asarray(omega_fit)
    return float(xp.sum(k_fit * omega_fit) / xp.sum(k_fit**2))


def plot_alfven_spectrum(sim, disp_params):
    velocity_data = sim.spline_values.mhd.velocity_log.data

    _, _, _, coeffs = power_spectrum_2d(
        velocity_data,
        "velocity_log",
        grids=sim.grids_log,
        grids_mapped=sim.grids_phy,
        component=0,
        slice_at=[0, 0, None],
        do_plot=True,
        disp_name="MHDhomogenSlab",
        disp_params=disp_params,
        fit_branches=1,
        noise_level=0.5,
        extr_order=10,
        fit_degree=(1,),
    )
    return float(coeffs[0][0])


def reconstruct_pressure(sim, gamma):
    """Reconstruct physical pressure perturbation from density and entropy."""
    density_data = sim.spline_values.mhd.density_log.data
    entropy_data = sim.spline_values.mhd.entropy_log.data

    times, density_3form = stack_time_series(density_data)
    entropy_times, entropy_3form = stack_time_series(entropy_data)

    assert len(times) == len(entropy_times)
    assert all(xp.isclose(float(t1), float(t2)) for t1, t2 in zip(times, entropy_times))

    # logical 3-forms -> physical densities
    jacobian = sim.domain.jacobian_det(*sim.grids_log)
    rho = density_3form / jacobian[None, ...]
    entropy = entropy_3form / jacobian[None, ...]

    pressure = (gamma - 1.0) * rho**gamma * xp.exp(entropy / rho)

    assert xp.all(xp.isfinite(pressure)) and xp.min(pressure) > 0.0

    # perturbation, to match the evolved variable of LinearMHD
    return {t: [pressure[i] - pressure[0]] for i, t in enumerate(times)}


def plot_pressure_spectrum(kvec, omega, spectrum, v_slow, v_fast, v_slow_fit, v_fast_fit):
    K, W = xp.meshgrid(kvec, omega)
    power = spectrum**2
    power /= xp.max(power)

    fig, ax = plt.subplots(figsize=(10, 10))
    cf = ax.contourf(K, W, power, levels=xp.logspace(-15, -1, 27),
                      cmap="plasma", norm=colors.LogNorm())
    fig.colorbar(cf, ax=ax, ticks=[1e-12, 1e-9, 1e-6, 1e-3],
                 format="%.0e", label="normalized spectral power")

    ax.plot(kvec, v_slow_fit * kvec, "r:", lw=2, label=rf"slow fit: $v={v_slow_fit:.4f}$")
    ax.plot(kvec, v_fast_fit * kvec, "m:", lw=2, label=rf"fast fit: $v={v_fast_fit:.4f}$")
    ax.plot(kvec, v_slow * kvec, "c--", lw=2, label=rf"slow exact: $v={v_slow:.4f}$")
    ax.plot(kvec, v_fast * kvec, "g--", lw=2, label=rf"fast exact: $v={v_fast:.4f}$")

    ax.set(title="Reconstructed pressure, space-time power spectrum",
           xlabel=r"$k$", ylabel=r"$\omega$",
           xlim=(0.0, kvec[-1]), ylim=(0.0, 1.1 * v_fast * kvec[-1]))
    ax.legend()
    fig.tight_layout()
    plt.show()


def main(do_plot: bool = True):
    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    sim = params.sim
    sim.pproc()
    sim.load_plotting_data()

    v_alfven, v_slow, v_fast, disp_params = analytical_speeds(
        params.B0x, params.B0y, params.B0z, params.beta, params.n0, params.gamma,
    )

    # --- Alfven branch, from velocity spectrum ---
    v_alfven_fit = plot_alfven_spectrum(sim, disp_params) if do_plot else None
    print(f"v_alfven = {v_alfven:.4f}, fit = {v_alfven_fit}")

    # --- slow/fast branches, from reconstructed pressure spectrum ---
    pressure_data = reconstruct_pressure(sim, params.gamma)

    omega, kvec, spectrum, _ = power_spectrum_2d(
        pressure_data,
        "pressure_log",
        grids=sim.grids_log,
        grids_mapped=sim.grids_phy,
        component=0,
        slice_at=[0, 0, None],
        do_plot=False,
        disp_name="MHDhomogenSlab",
        disp_params=disp_params,
        fit_branches=0,
    )

    v_slow_fit = fit_branch(omega, kvec, spectrum, v_slow, competing_speed=v_alfven,
                             search_bins=1, min_separation_bins=2)
    v_fast_fit = fit_branch(omega, kvec, spectrum, v_fast, search_bins=2)

    print(f"v_slow = {v_slow:.4f}, fit = {v_slow_fit:.4f}")
    print(f"v_fast = {v_fast:.4f}, fit = {v_fast_fit:.4f}")

    if do_plot:
        plot_pressure_spectrum(kvec, omega, spectrum, v_slow, v_fast, v_slow_fit, v_fast_fit)


if __name__ == "__main__":
    main(do_plot=True)