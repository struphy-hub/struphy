"""Prescribed-field planar ion accelerator.

Run from the repository root with::

    python examples/IonOpticsElectrostatic/planar_accelerator/planar_accelerator.py

The example accelerates a thin proton beam in a linear electrostatic potential
and writes ``planar_accelerator.png`` next to this file.  It is intentionally a
vacuum test: no space-charge deposition or Poisson update is performed yet.
"""

import copy
from pathlib import Path

import cunumpy as xp
import h5py
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt
import numpy as np

from struphy import (
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    SavingParameters,
    Simulation,
    SortingParameters,
    Time,
    WeightsParameters,
    domains,
    grids,
    maxwellians,
)
from struphy.initial.base import Perturbation
from struphy.models import IonOpticsElectrostatic


class LinearPotential(Perturbation):
    """Potential decreasing linearly from ``voltage`` to zero along x."""

    def __init__(self, voltage: float, length: float):
        self.params = copy.deepcopy(locals())
        self.given_in_basis = "physical"
        self.comp = 0
        self.voltage = voltage
        self.length = length

    def __call__(self, x, y, z):
        return self.voltage * (1.0 - x / self.length)


def build_simulation(output_dir, dt=0.005, end_time=1.5, epsilon=1.0):
    """Build a serial, dimensionless constant-acceleration benchmark."""
    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError("Run this small trajectory example on one MPI rank.")
    length = 1.0
    voltage = 0.4
    initial_vx = 0.20
    initial_y = xp.linspace(0.2, 0.8, 17)
    initial_markers = tuple((0.05, float(y), 0.5, initial_vx, 0.04 * (float(y) - 0.5), 0.0) for y in initial_y)

    model = IonOpticsElectrostatic(epsilon=epsilon)
    model.em_fields.phi.save_data = True
    model.em_fields.e_field.save_data = True
    model.ions.var.save_data = True
    model.em_fields.phi.add_perturbation(LinearPotential(voltage, length))
    model.ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    model.ions.set_markers(
        loading_params=LoadingParameters(Np=len(initial_markers), specific_markers=initial_markers, seed=7),
        weights_params=WeightsParameters(control_variate=False),
        boundary_params=BoundaryParameters(bc=("remove", "reflect", "reflect")),
        sorting_params=SortingParameters(boxes_per_dim=(8, 1, 1), do_sort=True),
        saving_params=SavingParameters(n_markers=len(initial_markers)),
    )

    model.propagators.push_v.options = model.propagators.push_v.Options()
    model.propagators.push_eta.options = model.propagators.push_eta.Options()

    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(output_dir), sim_folder="planar_accelerator"),
        time_opts=Time(dt=dt, Tend=end_time, split_algo="Strang"),
        domain=domains.Cuboid(r1=length),
        grid=grids.TensorProductGrid(num_elements=(32, 1, 1)),
        derham_opts=DerhamOptions(degree=(3, 1, 1), bcs=(("free", "free"), ("free", "free"), ("free", "free"))),
    )
    return sim


def plot_results(sim, output):
    """Plot saved numerical trajectories and compare with constant acceleration."""
    with h5py.File(Path(sim.env.path_out) / "data" / "data_proc0.hdf5") as f:
        time = f["time/value"][:]
        history = f["kinetic/ions/markers"][:]
    # Saving is ID ordered, so each column is the same particle throughout.
    x = history[:, :, 0]
    y = history[:, :, 1]
    vx = history[:, :, 3]
    kinetic = 0.5 * np.sum(history[:, :, 3:6] ** 2, axis=-1)
    acceleration = 0.4 / sim.model.ions.equation_params.epsilon
    exact_x = x[0] + time[:, None] * vx[0] + 0.5 * acceleration * time[:, None] ** 2
    exact_vx = vx[0] + acceleration * time[:, None]
    np.testing.assert_allclose(x, exact_x, atol=2e-11, rtol=0)
    np.testing.assert_allclose(vx, exact_vx, atol=2e-11, rtol=0)
    gain = kinetic - kinetic[0]
    work = acceleration * (x - x[0])
    np.testing.assert_allclose(gain, work, atol=2e-11, rtol=0)
    print(f"Maximum trajectory error: {np.max(np.abs(x - exact_x)):.3e}")

    x_line = np.linspace(0.0, 1.0, 200)
    phi_line = np.asarray(sim.model.em_fields.phi.spline(x_line, np.array([0.5]), np.array([0.5]))).ravel()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle("Planar ion accelerator — prescribed field, normalized units")
    axes[0, 0].plot(x_line, phi_line)
    axes[0, 0].set(xlabel="x", ylabel=r"$\phi$", title="Projected electrostatic potential")

    axes[0, 1].plot(x, y, linewidth=1)
    axes[0, 1].set(xlabel="x", ylabel="y", title="Numerical beam trajectories")

    axes[1, 0].plot(time, gain[:, 8], label="Numerical kinetic-energy gain")
    axes[1, 0].plot(time[::15], work[::15, 8], "o", fillstyle="none", label=r"$(\phi_0-\phi)/\varepsilon$")
    axes[1, 0].set(xlabel="Time", ylabel="Energy per unit mass", title="Work–energy check (central ion)")
    axes[1, 0].legend()

    for index, label in [(0, "Initial"), (-1, "Final")]:
        axes[1, 1].scatter(y[index], history[index, :, 4] / vx[index], label=label)
    axes[1, 1].set(xlabel="y", ylabel=r"$y'=v_y/v_x$", title="Transverse phase space")
    axes[1, 1].legend()

    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote {output}")


def main():
    sim = build_simulation(Path(__file__).parent / "output")
    sim.run()
    plot_results(sim, Path(__file__).with_name("planar_accelerator.png"))


if __name__ == "__main__":
    main()
