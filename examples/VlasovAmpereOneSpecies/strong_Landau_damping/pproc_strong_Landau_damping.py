import argparse


from struphy import Output


def main(path_out="sim_data"):
    run = Output(path_out)

    # electric field energy
    run.scalars.electric_energy.struphy.plot.timeseries(title="Electric energy").show()

    # full f in the e1-v1 plane
    run.kinetic_ions.e1_v1_density.f.struphy.plot.panels(x="e1", y="v1", nrows=3, ncols=4, title="full-$f$").show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a saved simulation run.")
    parser.add_argument("path_out", nargs="?", default="sim_data", help="Simulation output folder")
    args = parser.parse_args()
    main(args.path_out)
