"""``out.plot`` and ``out.analysis``: diagnostics of a whole run.

Plots and diagnostics of a single array live on the array itself, see
:class:`~struphy.post_processing.xarray_accessors.StruphyAccessor`. The methods here take a
product name as well (``"en_phi"``, ``"em_fields/phi_log"``; see :meth:`Output.__getitem__`),
and label arrays that carry no run with this run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

import struphy.post_processing.xarray_accessors  # noqa: F401  (registers array.struphy)

if TYPE_CHECKING:
    from struphy.post_processing.output import Output


class OutputPlots:
    """Standard plots of a run, as ``out.plot.<kind>(...)``.

    Plots return a rendered :class:`~struphy.diagnostics.plotting.PlotResult` with ``.show()``
    and ``.save(path)``, titled with the run's numerical parameters. The array-level methods are
    the accessor methods of that array: ``out.plot.slice("em_fields/phi_log", ...)`` is
    ``out.em_fields.phi_log.struphy.slice(...)``.
    """

    def __init__(self, output: "Output"):
        self._output = output

    def _array(self, data) -> xr.DataArray:
        array = self._output[data] if isinstance(data, str) else data
        if not array.attrs.get("run"):
            array = array.copy()
            array.attrs["run"] = self._output.label
            array.attrs["run_name"] = self._output.path_out.name
        return array

    def scalars(self, names=None, *, relative_to: str | None = None, logy: bool = False):
        """Overview of the scalar time series in one axes.

        Parameters
        ----------
        names:
            Scalars to show; all by default.
        relative_to:
            Show every scalar divided by this one.
        logy:
            Logarithmic value axis.
        """
        from struphy.diagnostics.plotting import plot_scalars

        return plot_scalars(self._output.scalars, names=names, relative_to=relative_to, logy=logy,
                            run_label=self._output.label)

    def timeseries(self, *data, **kwargs):
        """One or more time series; see the ``timeseries`` accessor method of an array."""
        if not data:
            raise TypeError("timeseries() needs at least one name or array")
        first, *others = (self._array(item) for item in data)
        return first.struphy.timeseries(*others, **kwargs)

    def slice(self, data, **kwargs):
        """A two-dimensional slice; see the ``slice`` accessor method of an array."""
        return self._array(data).struphy.slice(**kwargs)

    def panels(self, data, **kwargs):
        """Snapshots along a sweep; see the ``panels`` accessor method of an array."""
        return self._array(data).struphy.panels(**kwargs)

    def viewer(self, data, **kwargs):
        """An interactive viewer; see the ``viewer`` accessor method of an array."""
        return self._array(data).struphy.viewer(**kwargs)

    def animation(self, data, **kwargs):
        """An animation; see the ``animation`` accessor method of an array."""
        return self._array(data).struphy.animation(**kwargs)

    def frames(self, data, directory, **kwargs):
        """PNG files of a sweep; see the ``frames`` accessor method of an array."""
        return self._array(data).struphy.frames(directory, **kwargs)

    def orbits(self, species: str | None = None, **kwargs):
        """Marker trajectories of ``species``, the only species with saved markers by default."""
        available = tuple(self._output.orbits)
        if species is None:
            if len(available) != 1:
                raise ValueError(f"choose a species from {available}")
            species = available[0]
        return self._array(self._output.orbits[species]).struphy.trajectories(**kwargs)

    def equilibrium(self, ax=None):
        """Radial equilibrium profiles, from the geometry written at the start of the run."""
        from struphy.diagnostics.plotting import plot_equilibrium_profile

        return plot_equilibrium_profile(self._output.path_out, ax=ax)


class OutputAnalysis:
    """Quantitative diagnostics of a run, as ``out.analysis.<quantity>(...)``.

    Each method takes a product name or any array, and is the corresponding accessor method of
    that array: ``out.analysis.growth_rate("en_phi")`` is ``out["en_phi"].struphy.growth_rate()``.
    """

    def __init__(self, output: "Output"):
        self._output = output

    def _array(self, data) -> xr.DataArray:
        return self._output[data] if isinstance(data, str) else data

    def growth_rate(self, data, **kwargs):
        """Exponential growth rate; see the ``growth_rate`` accessor method of an array."""
        return self._array(data).struphy.growth_rate(**kwargs)

    def drift(self, data, **kwargs) -> xr.DataArray:
        """Deviation from the first sample; see the ``drift`` accessor method of an array."""
        return self._array(data).struphy.drift(**kwargs)

    def relative_error(self, data, **kwargs) -> xr.DataArray:
        """Relative deviation; see the ``relative_error`` accessor method of an array."""
        return self._array(data).struphy.relative_error(**kwargs)

    def dispersion(self, field, **kwargs):
        """Space-time spectrum; see the ``dispersion`` accessor method of an array.

        A field given by name is taken in normalized time, whatever this run's time units are.
        """
        if isinstance(field, str):
            output = self._output
            if output.time_units != "normalized":
                output = output.with_time_units("normalized")
            field = output[field]
        return field.struphy.dispersion(**kwargs)
