"""Space-charge benchmark: planar ion diode and the Child–Langmuir law.

Run from the repository root with::

    python examples/IonOpticsElectrostatic/child_langmuir_diode/child_langmuir_diode.py

Protons are emitted almost at rest from an emitter plate (x = 0, potential U) and
accelerated to a collector (x = d, potential 0). With ``IonOpticsUnits(length=d,
voltage=U)`` the self-consistent steady state satisfies

    -phi'' = J / v(x),   v = sqrt(v0² + 2 (1 - phi)),   phi(0) = 1, phi(1) = 0,

and the space-charge-limited (Child–Langmuir) current density is J_CL = 4 sqrt(2) / 9.
For J < J_CL, ``solve_bvp`` gives the reference potential; at J_CL, phi = 1 - x^(4/3).
Above J_CL, a potential maximum (virtual anode) in front of the emitter
reflects part of the beam back to the emitter, and the transmitted current
is limited to about J_CL.

Writes ``child_langmuir_diode.png`` next to this file.
"""

from pathlib import Path

import h5py
import numpy as np
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt
from scipy.integrate import solve_bvp

from struphy import (
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    SavingParameters,
    Simulation,
    Time,
    WeightsParameters,
    domains,
    grids,
    maxwellians,
)
from struphy.initial.perturbations import PiecewiseLinearPotential
from struphy.models import IonOpticsElectrostatic
from struphy.physics.ion_optics_units import IonOpticsUnits
from struphy.pic.ion_beams import LossTag, PlaneSource

J_CL = 4.0 * np.sqrt(2.0) / 9.0
GAP = 10e-3  # m
VOLTAGE = 10e3  # V
UNITS = IonOpticsUnits(length=GAP, voltage=VOLTAGE)
V0 = 0.02  # emission speed (normalized): 2 V of kinetic energy, negligible against U = 10 kV
TAGS = ("collector", "emitter")


def build_simulation(
    output_dir,
    current_fraction,
    num_elements=64,
    degree=3,
    dt=0.01,
    end_time=8.0,
    markers_per_time=1000.0,
    name=None,
):
    """Diode with injected current density ``current_fraction * J_CL`` (cross-section 1 x 1)."""
    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError("Run this example on one MPI rank.")
    source = PlaneSource(
        rate=markers_per_time,
        current=current_fraction * J_CL,
        axis=0,
        eta_plane=0.0,
        velocity=(V0, 0.0, 0.0),
        seed=3,
    )
    model = IonOpticsElectrostatic(
        base_units=UNITS.base_units(),
        electrode_faces=((True, True), (False, False), (False, False)),
        source=source,
        loss_tags=(LossTag("collector", axis=0, side=1), LossTag("emitter", axis=0, side=0)),
        space_charge=True,
    )
    model.em_fields.phi.add_perturbation(PiecewiseLinearPotential((0.0, 1.0), (1.0, 0.0), coordinate=0))
    model.em_fields.e_field.save_data = False
    model.ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    capacity = int(2.5 * markers_per_time * end_time)
    model.ions.set_markers(
        loading_params=LoadingParameters(Np=1, specific_markers=((0.5, 0.5, 0.5, 0.0, 0.0, 0.0),)),
        weights_params=WeightsParameters(control_variate=False),
        boundary_params=BoundaryParameters(bc=("remove", "periodic", "periodic")),
        saving_params=SavingParameters(n_markers=1),
        bufsize=float(capacity),
    )
    for prop in model.prop_list:
        prop.options = prop.Options()

    return Simulation(
        model=model,
        env=EnvironmentOptions(
            out_folders=str(output_dir),
            sim_folder=name or f"diode_{current_fraction:g}",
        ),
        time_opts=Time(dt=dt, Tend=end_time, split_algo="Strang"),
        domain=domains.Cuboid(),
        grid=grids.TensorProductGrid(num_elements=(num_elements, 1, 1)),
        derham_opts=DerhamOptions(degree=(degree, 1, 1), bcs=(("free", "free"),) * 3),
    )


def reference_potential(current_fraction, x):
    """Steady planar-diode potential for J = current_fraction * J_CL (J <= J_CL)."""
    j = current_fraction * J_CL
    if np.isclose(current_fraction, 1.0) and V0 == 0.0:
        return 1.0 - x ** (4 / 3)

    def rhs(s, y):
        return np.vstack([y[1], -j / np.sqrt(V0**2 + 2.0 * np.maximum(1.0 - y[0], 0.0))])

    def bc(ya, yb):
        return np.array([ya[0] - 1.0, yb[0]])

    mesh = np.linspace(0.0, 1.0, 401)
    guess = np.vstack([1.0 - mesh ** (4 / 3), -(4 / 3) * mesh ** (1 / 3)])
    sol = solve_bvp(rhs, bc, mesh, guess, tol=1e-8, max_nodes=100000)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.sol(x)[0]


def steady_state(sim, t_start):
    """Time-averaged potential on the axis and current fractions over ``t >= t_start``."""
    model = sim.model
    model.update_ledger()
    with h5py.File(Path(sim.env.path_out) / "data" / "data_proc0.hdf5") as f:
        time = f["time/value"][:]
        phi_coeffs = f["feec/em_fields/phi"][:]
        charges = {key: f["scalar"][key][:] for key in f["scalar"]}
    window = time >= t_start
    vector = model.em_fields.phi.spline.vector
    vector._data[:] = phi_coeffs[window].mean(axis=0)
    x = np.linspace(0.0, 1.0, 201)
    phi = np.asarray(model.em_fields.phi.spline(x, np.array([0.5]), np.array([0.5]))).ravel()

    def gain(key):
        series = charges[key][window]
        return series[-1] - series[0]

    injected = gain("injected_charge")
    fractions = {name: gain(f"lost_charge_{name}") / injected for name in TAGS}
    duration = time[window][-1] - time[window][0]
    currents = {name: gain(f"lost_charge_{name}") / duration for name in TAGS}
    return x, phi, fractions, currents, time, charges


CASES = (0.25, 0.5, 0.9, 2.0)


def run_cases(output_dir, cases=CASES, t_start=6.0, **kwargs):
    results = {}
    for fraction in cases:
        sim = build_simulation(output_dir, fraction, **kwargs)
        sim.run()
        results[fraction] = steady_state(sim, t_start)
        x, phi, fractions, currents, *_ = results[fraction]
        line = f"J = {fraction:4.2f} J_CL: collector current {currents['collector'] / J_CL:.3f} J_CL"
        if fraction <= 1.0:
            line += f", max |phi - phi_ref| = {np.max(np.abs(phi - reference_potential(fraction, x))):.2e}"
        print(line + f", reflected to emitter {100 * fractions['emitter']:.1f} %")
    return results


def plot_results(results, output, t_start=6.0):
    fig, (ax_phi, ax_err, ax_current) = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig.suptitle(
        f"Planar proton diode with space charge: d = {GAP * 1e3:g} mm, U = {VOLTAGE / 1e3:g} kV, "
        f"J_CL = {J_CL * UNITS.current_density / 10:.3g} mA/cm²"
    )
    x_ref = np.linspace(0.0, 1.0, 201)
    ax_phi.plot(x_ref, 1.0 - x_ref, "k:", label="Vacuum")
    ax_phi.plot(x_ref, 1.0 - x_ref ** (4 / 3), "k--", label="Child–Langmuir limit")
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for colour, (fraction, (x, phi, _, _, time, charges)) in zip(colours, results.items()):
        ax_phi.plot(x, phi, color=colour, label=f"J = {fraction:g} J_CL")
        if fraction <= 1.0:
            reference = reference_potential(fraction, x)
            ax_phi.plot(x[::10], reference[::10], "o", color=colour, fillstyle="none")
            ax_err.plot(x, phi - reference, color=colour, label=f"J = {fraction:g} J_CL")
        t_ns = time * UNITS.time * 1e9
        injected = np.gradient(charges["injected_charge"], time)
        collected = np.gradient(charges["lost_charge_collector"], time)
        smooth = np.convolve(collected, np.ones(50) / 50, mode="valid")
        ax_current.plot(t_ns[25 : 25 + len(smooth)], smooth / J_CL, color=colour, label=f"J = {fraction:g} J_CL")
        ax_current.axhline(injected[-1] / J_CL, color=colour, linestyle=":", linewidth=0.8)
    ax_phi.set(xlabel="x / d", ylabel="φ / U", title="Time-averaged potential (circles: steady-state ODE)")
    ax_phi.legend(fontsize=8)
    ax_err.set(xlabel="x / d", ylabel="φ − φ_ref", title="Potential error against the steady-state ODE")
    ax_err.legend(fontsize=8)
    ax_current.axhline(1.0, color="k", linestyle="--", linewidth=1, label="J_CL")
    ax_current.axvspan(t_start * UNITS.time * 1e9, t_ns[-1], color="0.92", zorder=-1)
    ax_current.set(
        xlabel="t (ns)",
        ylabel="Collector current / J_CL (50-step average)",
        title="Collector current (dotted: injected)",
        ylim=(0, 2.2),
    )
    ax_current.legend(fontsize=8)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


def main():
    results = run_cases(Path(__file__).parent / "output")
    plot_results(results, Path(__file__).with_name("child_langmuir_diode.png"))


if __name__ == "__main__":
    main()
