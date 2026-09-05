import cunumpy as xp
import params_two_stream as params

from struphy import PlottingData, PostProcessor
from struphy.diagnostics.plotting import PanelGridPlot, SliderPlot, TimeSeriesPlot
from struphy.post_processing.arrays import StruphyArray


def main():
    PostProcessor(sim=params.sim).process(force=False)

    pdata = PlottingData(sim=params.sim)
    pdata.load()

    # electric field growth against the analytical rate (0.2845 in units of m/c)
    energy = pdata.scalars["electric_energy"]
    t = energy.coord("t")
    analytical = StruphyArray(
        10 ** (0.2845 / pdata.units.t * t - 5.3),
        dims=("t",),
        coords={"t": t},
        label="analytical",
    ).with_coord_units(t="s")

    TimeSeriesPlot(
        [energy, analytical],
        params=pdata.params,
        title="Electric energy",
    ).show()

    # phase space evolution
    f = pdata.f.kinetic_ions["e1_v1_density"]["f_binned"]

    PanelGridPlot(f, nrows=3, ncols=4, shared_clim=True, params=pdata.params).show()

    # interactive alternative to dumping a frame sequence
    SliderPlot(f, equal_aspect=False, params=pdata.params).show()


if __name__ == "__main__":
    main()
