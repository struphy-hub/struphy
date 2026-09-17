import argparse


from struphy import open_output


def main(path_out="sim_data"):
    run = open_output(path_out)

    # table and figures of every scalar: post_processing/report/
    run.save_report()

    # electric field growth against the analytical rate (0.2845 in units of m/c)
    energy = run.scalars.electric_energy
    analytical = energy.copy(data=10 ** (0.2845 * energy.t - 5.3))  # t is in Struphy units
    analytical.attrs["label"] = "analytical"
    energy.struphy.plot.timeseries(analytical, title="Electric energy").show()

    # phase space evolution
    f = run.kinetic_ions.e1_v1_density.f
    f.struphy.plot.panels(x="e1", y="v1", nrows=3, ncols=4).show()

    # interactive alternative to dumping a frame sequence
    f.struphy.plot.viewer(x="e1", y="v1").show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a saved simulation run.")
    parser.add_argument("path_out", nargs="?", default="sim_data", help="Simulation output folder")
    args = parser.parse_args()
    main(args.path_out)
