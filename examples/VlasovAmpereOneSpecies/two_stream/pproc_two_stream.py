import params_two_stream as params


def main():
    run = params.sim.output

    # table and figures of every scalar: post_processing/report/
    run.save_report()

    # electric field growth against the analytical rate (0.2845 in units of m/c)
    energy = run.scalars.electric_energy
    analytical = energy.copy(data=10 ** (0.2845 * energy.t - 5.3))  # t is in Struphy units
    analytical.attrs["label"] = "analytical"
    energy.struphy.plot.timeseries(analytical, title="Electric energy").show()

    # phase space evolution
    f = run.kinetic_ions.e1_v1_density.f_binned
    f.struphy.plot.panels(x="e1", y="v1", nrows=3, ncols=4).show()

    # interactive alternative to dumping a frame sequence
    f.struphy.plot.viewer(x="e1", y="v1").show()


if __name__ == "__main__":
    main()
