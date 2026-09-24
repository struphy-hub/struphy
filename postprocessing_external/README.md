# struphy-plots (staging)

This directory contains the optional plotting layer extracted from Struphy. It
is deliberately a standalone source package so it can be moved to its own
repository without changing the Struphy runtime package.

For development, install it with `pip install -e postprocessing_external`.
Import `struphy_plots` after installing it to register the optional
`xarray.DataArray.struphy` accessor on Struphy output arrays. For example,
`out.evaluate("em_fields/phi").struphy.plot.slice(x="e1", y="e2", t="last")`.
Direct plotting functions are available from
`struphy_plots.plotting`; analysis functions are in `struphy_plots.analysis`.

The accessor provides time-series, lineout, slice, panel, vector, comparison,
animation, and marker-trajectory plots. For three-dimensional scalar fields,
install the optional PyVista dependency (`pip install struphy-plots[pyvista]`) and
use `field.struphy.plot.volume(t=-1)`, then call `show()` on the returned plotter.
