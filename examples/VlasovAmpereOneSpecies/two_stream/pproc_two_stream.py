import params_two_stream as params


def main():
    run = params.sim.output

    # table and figures of every scalar: post_processing/report/
    run.save_report()

    # electric field growth against the analytical rate (0.2845 in units of m/c)
    energy = run.scalars.electric_energy
    analytical = energy.copy(data=10 ** (0.2845 / run.sim.model.units.t * energy.t - 5.3))
    analytical.attrs["label"] = "analytical"
    run.plot.timeseries(energy, analytical, title="Electric energy").show()

    # phase space evolution
    f = "kinetic_ions/e1_v1_density/f_binned"
    run.plot.panels(f, x="e1", y="v1", nrows=3, ncols=4).show()

    # interactive alternative to dumping a frame sequence
    run.plot.viewer(f, x="e1", y="v1").show()


if __name__ == "__main__":
    main()
