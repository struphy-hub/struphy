import argparse

import cunumpy as xp
from matplotlib import pyplot as plt

from struphy import Output


def main(path_out="sim_data"):
    run = Output(path_out, time_units="normalized")
    time = run.time
    Tend = run.time_opts.Tend
    algo = run.time_opts.split_algo

    # ------------------
    # Gauss law violation
    # ------------------
    if run.model.measure_gauss_law:
        gauss_error = run.scalars.gauss_error

        fig, ax = plt.subplots(1, figsize=(10, 6))
        ax.plot(gauss_error.t, gauss_error)
        ax.set_xlim(0, Tend)
        ax.set_yscale("log")
        ax.set_xlabel("time")
        ax.set_ylabel("gauss error")
        ax.set_title("Gauss law violation as function of time")
        ax.grid()
        plt.tight_layout()
        plt.show()

    # ------------------
    # progression of EM-field energy along different directions
    # ------------------
    e_field = run.fields.em_fields.e_field
    b_field = run.fields.em_fields.b_field
    spatial = ("e1", "e2", "e3")
    unit_volume = xp.prod([1 / (e_field.sizes[dim] - 1) for dim in spatial])

    def field_energy(field):
        """Energy of each component over space, as array of shape (component, t)."""
        return (field**2).sum(spatial).transpose("component", "t").values * unit_volume / 2

    electric_energy = field_energy(e_field)
    magnetic_energy = field_energy(b_field)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), sharex=True)
    ax.plot(time, electric_energy[0], label=r"|$E_1|^2$/2", color="blue")
    ax.plot(time, electric_energy[1], label=r"|$E_2|^2$/2", color="green")
    ax.plot(time, magnetic_energy[2], label=r"|$B_3|^2$/2", color="red")

    # determine magnetic field growth rate
    exp_func = lambda x, m, b: 10 ** (m * x + b)

    ti = time[-1] // 5
    tf = time[-1] if ti == 0.0 else 2 * ti
    print(f"{ti = }, {tf = }")

    xi = xp.abs(time - ti).argmin() + 1
    xf = xp.abs(time - tf).argmin() + 1

    fitting = xp.polyfit(time[xi:xf], xp.log10(magnetic_energy[2][xi:xf]), deg=1)
    ax.plot(
        time,
        exp_func(time, *fitting),
        label="fitted growth rate\n" + rf"$10^{{{fitting[0]:.5f}x {fitting[1]:.0f}}}$",
        color="cyan",
    )
    ax.plot(
        time,
        exp_func(time, 0.02784, fitting[1]),
        label="analytical growth rate\n" + rf"$10^{{0.02784x {fitting[1]:.0f}}}$",
        color="cyan",
        ls="--",
        alpha=0.5,
    )

    ax.set_title("Energy in EM field")
    ax.set_ylabel("Energy [a.u.]")
    ax.set_xlabel("time")
    ax.set_ylim(1e-14, 1e0)
    ax.set_xlim(0, Tend)
    ax.legend(ncol=3)
    ax.set_yscale("log")
    ax.minorticks_on()

    fig.suptitle(f"VlasovMaxwellOneSpecies simulation:\n {algo=}")
    plt.tight_layout()
    plt.show()

    # ------------------
    # Binning distribution evolution
    # ------------------
    distributions = run.distributions.kinetic_ions
    for bin_name, x, y in (("e1_v1_density", "e1", "v1"), ("v1_v2_density", "v1", "v2")):
        for quantity in ("f", "delta_f"):
            getattr(getattr(distributions, bin_name), quantity).struphy.plot.panels(x=x, y=y, nrows=5, ncols=4).show()

    # ------------------
    # EM field at selected times
    # ------------------
    def plot_EM_state(time_step: float, n_dim=3):
        electric_field = e_field.sel(t=time_step, method="nearest").isel(e2=0, e3=0)
        magnetic_field = b_field.sel(t=time_step, method="nearest").isel(e2=0, e3=0)

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 6), sharex=True, sharey=True)
        for i in range(n_dim):
            axs[0, i].plot(electric_field.e1, electric_field.isel(component=i))
            axs[0, i].set_title(rf"$E_{i + 1}$")
            axs[1, i].plot(magnetic_field.e1, magnetic_field.isel(component=i))
            axs[1, i].set_title(rf"$B_{i + 1}$")

        axs[0, 0].set_ylabel(r"Electric field value")
        axs[1, 0].set_ylabel(r"Magnetic field value")
        for i in range(n_dim):
            axs[1, i].set_xlabel(r"$\eta_1$")
        axs[0, 0].set_ylim(-5e-3, 5e-3)
        axs[1, 0].set_ylim(-5e-3, 5e-3)

        fig.suptitle(f"EM-field at time step: {float(electric_field.t):.2f}")
        plt.show()
        plt.close()

    for t in xp.linspace(0, time[-1], 2):
        plot_EM_state(t)

    # ------------------
    # Current density evolution
    # ------------------
    def current_1D(time_step: float):
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(9, 9), sharey=True, sharex=True)
        for i in range(3):
            for j in range(3):
                current = getattr(distributions, f"e{i + 1}_current_{j + 1}").f
                current = current.sel(t=time_step, method="nearest")
                ax[i, j].axhline(color="red", alpha=0.5)
                ax[i, j].plot(current[f"e{i + 1}"], current)
            ax[i, 0].set_ylim(-0.01, 0.01)

        for i in range(3):
            ax[i, 0].set_ylabel(rf"$j_{i + 1}$")
        for j in range(3):
            ax[2, j].set_xlabel(rf"$\eta_{ {j + 1} }$")

        fig.suptitle(f"Current density at time {time_step:.2f}")
        plt.tight_layout()
        plt.show()

    for t in xp.linspace(0, time[-1], 2):
        current_1D(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a saved simulation run.")
    parser.add_argument("path_out", nargs="?", default="sim_data", help="Simulation output folder")
    args = parser.parse_args()
    main(args.path_out)
