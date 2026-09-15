import params_strong_Landau_damping as params


def main():
    run = params.sim.output

    # electric field energy
    run.plot.timeseries("electric_energy", title="Electric energy").show()

    # full f in the e1-v1 plane
    run.plot.panels("kinetic_ions/e1_v1_density/f_binned", x="e1", y="v1", nrows=3, ncols=4, title="full-$f$").show()


if __name__ == "__main__":
    main()
