"""Small real simulation exercising optional plasma-plots and the output CLI."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from struphy import DerhamOptions, EnvironmentOptions, Output, Simulation, Time, grids, perturbations
from struphy.models import Maxwell


def run_command(*args):
    """Include CLI diagnostics in pytest's failure report."""
    result = subprocess.run(
        list(map(str, args)),
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Command failed ({result.returncode}): {result.args}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result.stdout


@pytest.fixture(scope="module")
def raw_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("maxwell")
    model = Maxwell()
    model.em_fields.e_field.save_data = True
    model.em_fields.b_field.save_data = True
    model.em_fields.e_field.add_perturbation(
        perturbations.ModesCos(ls=(0,), ms=(0,), ns=(1,), amps=(0.1,), comp=0, given_in_basis="1")
    )
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(root), sim_folder="raw", save_step=1),
        time_opts=Time(dt=0.025, Tend=0.2),
        grid=grids.TensorProductGrid(num_elements=(1, 1, 8)),
        derham_opts=DerhamOptions(degree=(1, 1, 2)),
    )
    out = sim.run()
    assert (out.path_out / "run_metadata.json").is_file()
    assert not out.is_processed
    assert out.scalars.sizes["t"] >= 8
    assert np.isfinite(out.scalars.to_array()).all()
    out.close()
    return out.path_out


@pytest.fixture
def run_path(raw_run, tmp_path):
    # Each interface must create its own products from the original raw output.
    return Path(shutil.copytree(raw_run, tmp_path / "run"))


def assert_image(path):
    assert path.is_file() and path.stat().st_size > 100
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_python_api(run_path):
    # No explicit plasma_plots import: Output must register its accessors.
    with Output(run_path) as out:
        out.pproc(physical=True)
        assert out.is_processed
        assert "em_fields/e_field" in out.keys()
        field = out.evaluate("em_fields/e_field", representation="1")
        assert isinstance(field, xr.DataArray)
        assert np.isfinite(field).all()
        signal = field.isel(component=0, eta1=0, eta2=0)
        assert float(signal.std("t").max()) > 0

        spectrum = out.analysis.time_fft(signal)
        assert "omega" in spectrum.dims
        assert np.isfinite(spectrum.power).all()
        np.testing.assert_allclose(spectrum.power.sum("omega"), (signal**2).mean("t"), atol=1e-12)
        filtered = signal.plasma.analysis.filter_time()
        assert filtered.filtered.dims == signal.dims
        assert bool(filtered.spectrum.has_peak.all())
        assert np.isfinite(filtered.filtered).all()

        energies = run_path / "energies.png"
        out.plot.energies(parts=["electric_energy", "magnetic_energy"], total="total_energy").save(energies)
        profile = run_path / "electric-field.png"
        signal.plasma.plot.lineout(x="eta3", t=-1).save(profile)
        assert_image(energies)
        assert_image(profile)


def test_cli(run_path):
    cli = Path(sys.executable).parent / "struphy"

    def command(*args):
        return run_command(cli, "output", *args)

    command("pproc", run_path, "--physical")
    with Output(run_path) as out:
        assert out.is_processed
        assert np.isfinite(out.evaluate("em_fields/e_field_xyz")).all()
    assert "em_fields/e_field" in command("keys", run_path)
    assert "em_fields/e_field" in command("info", run_path)
    report_dir = run_path / "cli-report"
    command("report", run_path, "--format", "html", "--directory", report_dir)
    reports = list(report_dir.glob("*.html"))
    assert reports and "total_energy" in reports[0].read_text()

    def plasma_command(*args):
        return run_command(sys.executable, "-m", "plasma_plots", *args)

    assert "em_fields/e_field" in plasma_command("info", run_path)
    energy_plot = run_path / "cli-energies.png"
    plasma_command(
        "plot",
        run_path,
        ".",
        "energies",
        'parts=["electric_energy","magnetic_energy"]',
        "total=total_energy",
        "-o",
        energy_plot,
    )
    spectrum_plot = run_path / "cli-spectrum.png"
    plasma_command("plot", run_path, "em_fields/e_field", "power_spectrum", "component=0", "-o", spectrum_plot)
    assert_image(energy_plot)
    assert_image(spectrum_plot)
