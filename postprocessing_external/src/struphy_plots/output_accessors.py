"""Optional plots that need a whole run.

Plots and diagnostics of a single array live on the array, see
:class:`~struphy.post_processing.xarray_accessors.StruphyAccessor`:
``out.em_fields.phi_log.struphy.plot.slice(...)``, or from a value returned by
``out.evaluate("em_fields/phi_log")``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import accessors  # noqa: F401  (registers array.struphy)

if TYPE_CHECKING:
    from struphy.post_processing.output import Output


class OutputPlots:
    """Plots of a whole run, constructed as ``OutputPlots(out)``.

    They return plotting-library objects with ``.show()``
    and ``.save(path)``, titled with the run's numerical parameters. Plots of one product are
    methods of that product, e.g. ``out.kinetic_ions.orbits.struphy.plot.trajectories()``.
    """

    def __init__(self, output: "Output"):
        self._output = output

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
        from .plotting import plot_scalars

        return plot_scalars(
            self._output.scalars, names=names, relative_to=relative_to, logy=logy, run_label=self._output.label
        )

    def equilibrium(self, ax=None):
        """Radial equilibrium profiles, from the geometry written at the start of the run."""
        from .plotting import plot_equilibrium_profile

        return plot_equilibrium_profile(self._output.path_out, ax=ax)

    def equilibrium_3d(self, *, scalars: str = "p0", cmap="viridis"):
        """Create a PyVista equilibrium view; call ``.show()`` on the returned plotter."""
        from .plotting import show_equilibrium

        return show_equilibrium(self._output.path_out, scalars=scalars, cmap=cmap)
