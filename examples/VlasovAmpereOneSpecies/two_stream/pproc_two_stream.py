import params_two_stream as params

from struphy.diagnostics.plotting import InteractiveSliceViewer, View, plot_panels, plot_timeseries


def main():
    run = params.sim.output

    # every scalar at every time step: post_processing/scalars/{scalars.csv,*.png}
    run.save_scalar_plots()

    # electric field growth against the analytical rate (0.2845 in units of m/c)
    energy = run.scalars["electric_energy"]
    analytical = energy.copy(data=10 ** (0.2845 / run.sim.model.units.t * energy.t - 5.3))
    analytical.attrs["label"] = "analytical"

    plot_timeseries(
        [energy, analytical],
        run_label=run.label,
        title="Electric energy",
    ).show()

    # phase space evolution
    f = run.distributions.kinetic_ions.e1_v1_density.f_binned
    view = View(x="e1", y="v1")

    plot_panels(f, view=view, nrows=3, ncols=4, shared_clim=True, run_label=run.label).show()

    # interactive alternative to dumping a frame sequence
    InteractiveSliceViewer(f, view=view, run_label=run.label).show()


if __name__ == "__main__":
    main()
