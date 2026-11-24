"""The binning plots of struphy.post_processing.particle_diagnostics at the standard snapshots."""

import numpy as np
import pytest
import xarray as xr

from struphy.post_processing.particle_diagnostics import save_equilibrium_diagnostics


class FakeOutput:
    """Stands in for Output: binned distributions (1d and 2d) at ``nt`` saved times."""

    def __init__(self, path_out, nt):
        t = np.arange(nt, dtype=float)
        v = np.linspace(-1, 1, 5)
        e = np.linspace(0, 1, 4)
        f1 = np.exp(-(v**2))[None, :] * (1 + 0.1 * t[:, None])
        f2 = np.exp(-(v**2))[None, :, None] * (1 + e)[None, None, :] * (1 + 0.1 * t[:, None, None])
        self.path_out = str(path_out)
        self.distribution_catalog = {
            "ions/v1_density/f": xr.DataArray(f1, dims=("t", "v1"), coords={"t": t, "v1": v}),
            "ions/v1_e1_density/f": xr.DataArray(f2, dims=("t", "v1", "eta1"), coords={"t": t, "v1": v, "eta1": e}),
        }
        self.scalars = xr.Dataset({"en": xr.DataArray(1 + 0.01 * t, dims=("t",), coords={"t": t})})

    def pproc(self):
        return self

    def standard_snapshots(self):
        last = len(self.distribution_catalog["ions/v1_density/f"].t) - 1
        return sorted({0, last // 2, last})


@pytest.mark.mpi_skip
@pytest.mark.parametrize("nt, panels", [(1, 2), (2, 3), (5, 4)])
def test_binning_plots_at_the_standard_snapshots(tmp_path, monkeypatch, nt, panels):
    from matplotlib import figure

    saved = {}
    original = figure.Figure.savefig

    def savefig(self, name, *args, **kwargs):
        saved[str(name)] = self
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(figure.Figure, "savefig", savefig)
    save_equilibrium_diagnostics(FakeOutput(tmp_path, nt), "ions", scalars=("en",))
    results = tmp_path / "results"
    assert {"binning_v1_density.png", "binning_v1_e1_density.png", "deviation.png", "scalars.png"} <= {
        f.name for f in results.iterdir()
    }
    # 2d: one map per standard snapshot and the difference between the last and the first
    fig2 = saved[str(results / "binning_v1_e1_density.png")]
    titles = [ax.get_title() for ax in fig2.axes if ax.get_title()]
    assert len(titles) == panels and titles[-1] == "difference"
    # 1d: one curve per standard snapshot
    fig1 = saved[str(results / "binning_v1_density.png")]
    assert len(fig1.axes[0].lines) == panels - 1
