import params_bump_on as params
from matplotlib import pyplot as plt


def main():
    run = params.sim.output

    # initial velocity distribution
    initial = run["kinetic_ions/v1_density/f_binned"].isel(t=0)
    ax = initial.plot()[0].axes
    ax.set(xlabel="velocity $v$", ylabel="distribution $f(v)$", title="Initial velocity distribution")
    plt.show()

    # electric field energy
    run.plot.timeseries("electric_energy", title="Electric energy").show()

    # full f in the e1-v1 plane
    run.plot.panels("kinetic_ions/e1_v1_density/f_binned", x="e1", y="v1", nrows=3, ncols=4, title="full-$f$").show()


if __name__ == "__main__":
    main()
