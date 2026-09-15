import cunumpy as xp
import params_weak_Landau_damping as params


def E_exact(t):
    """Analytical electric energy of weak Landau damping, t in normalized units."""
    eps = params.perturbation.amps[0]
    r = 0.3677
    omega_r = 1.4156
    omega_i = -0.1533
    phi = 0.5362
    return (4 * eps * r * xp.exp(omega_i * t) * xp.cos(omega_r * t - phi)) ** 2 * xp.pi


def main():
    run = params.sim.output

    # electric field energy against the analytical damping
    energy = run.scalars.electric_energy.copy()
    energy.attrs["label"] = "numerical"
    analytical = energy.copy(data=E_exact(energy.t.values / run.sim.model.units.t))
    analytical.attrs["label"] = "analytical"
    run.plot.timeseries(energy, analytical, title="Electric energy").show()

    # full f and delta f in the e1-v1 plane at four times
    for quantity, title in (("f_binned", "full-$f$"), ("delta_f_binned", r"$\delta f$")):
        run.plot.panels(f"kinetic_ions/e1_v1_density/{quantity}", x="e1", y="v1", nrows=1, ncols=4, title=title).show()


if __name__ == "__main__":
    main()
