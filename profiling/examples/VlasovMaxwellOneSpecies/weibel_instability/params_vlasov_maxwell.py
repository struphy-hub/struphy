# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Vlasov-Maxwell Weibel instability"
description = """
Weibel instability with the VlasovMaxwellOneSpecies model, the setup of Kraus et al., J. Plasma Phys. 83 (2017),
Sec. 6.2, as in the verification test in src/struphy/models/tests/verification: periodic box of length 2 pi / 1.25,
anisotropic Maxwellian electrons (vth1 = vth3 = 0.02 / sqrt(2), vth2 = sqrt(12) vth1), seed field
B3 = -1e-4 cos(k x), no density perturbation, control-variate weights, cubic splines in x. The magnetic field energy
grows at the rate 0.02784. alpha = 1 and epsilon = -1 (electrons, time in units of the inverse plasma frequency,
velocities in units of c).

Runs on the NumPy (CPU) or CuPy (GPU) backend chosen with `struphy backend set` (the profiling driver runs it in
the job script). All timing comes from scope-profiler: the run activates
the profiling session of Simulation.run, and on the CuPy backend the session records CUDA-event device time per
region (GPUOptions.timing) and synchronizes the device at region ends, so the CPU time of a region includes its
kernels. The post-processing writes the scope-profiler HTML report and a summary of the cost per step and marker
into the run's results folder.
"""

import argparse
import logging

from struphy import set_logging_level

set_logging_level(logging.INFO)

import math

import cunumpy as xp

# ------------------
# Import Struphy API
# ------------------
from struphy import (
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    ProfilingOptions,
    SavingParameters,
    Simulation,
    SortingParameters,
    Time,
    WeightsParameters,
    domains,
    grids,
    maxwellians,
    perturbations,
)

# --------------------------
# Command line parameters
# --------------------------
# `--id` distinguishes runs that share a rank count but differ in something else; the
# profiling driver passes its launch counter (see `ProfilingJob.build_commands`).
# Unknown flags are ignored so the driver can forward other parameters as well.
parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=0, help="Run id, used to name the output folder.")
parser.add_argument("--num-elements", type=int, default=32, help="Cells in x (1 in y and z).")
parser.add_argument("--ppc", type=int, default=1000, help="Markers per cell.")
parser.add_argument("--steps", type=int, default=20, help="Number of time steps.")
args, _ = parser.parse_known_args()

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import VlasovMaxwellOneSpecies

model = VlasovMaxwellOneSpecies(alpha=1.0, epsilon=-1.0)

# List all variables and decide whether to save their data
model.em_fields.e_field.save_data = True
model.em_fields.b_field.save_data = True
model.kinetic_ions.var.save_data = False

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(
    sim_folder=f"sim_{args.id:02d}",
    restart=False,
    save_restart=False,  # restart data would dominate the run time for many markers
)

# Time stepping
dt = 0.05
time_opts = Time(dt=dt, Tend=round(args.steps * dt, 14))

# Geometry
Lx = 2 * math.pi / 1.25  # k = 1.25
domain = domains.Cuboid(r1=Lx)

# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Grid
grid = grids.TensorProductGrid(num_elements=(args.num_elements, 1, 1))

# Derham options
derham_opts = DerhamOptions(degree=(3, 1, 1))

# Profiling options: CUDA-event device times per region on the CuPy backend.
# cunumpy's fake CuPy (CUNUMPY_FAKE_CUPY=1, no GPU) has CUDA events without elapsed time, so GPU timing needs a GPU.
from cunumpy.kernel_testing import fake_cupy_active

gpu_timing = xp.get_backend() == "cupy" and not fake_cupy_active()
profiling_opts = ProfilingOptions(
    label=f"{name} ({xp.get_backend()})",
    use_gpu_timing=gpu_timing,
    gpu_timing_backend="cupy",
    gpu_sync_on_exit=gpu_timing,
)

# Simulation object
sim = Simulation(
    model=model,
    name=name,
    description=description,
    params_path=__file__,
    env=env,
    time_opts=time_opts,
    domain=domain,
    equil=equil,
    grid=grid,
    derham_opts=derham_opts,
    profiling_opts=profiling_opts,
)

# -------------------
# Particle parameters
# -------------------

loading_params = LoadingParameters(ppc=args.ppc, seed=1234)
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters()
# sorting boxes are off: SortingBoxes does not work on the CuPy backend yet (struphy-hub/struphy#726)
sorting_params = SortingParameters()
saving_params = SavingParameters(n_markers=0)  # profile the physics, not the marker output

model.kinetic_ions.set_markers(
    loading_params=loading_params,
    weights_params=weights_params,
    boundary_params=boundary_params,
    sorting_params=sorting_params,
    saving_params=saving_params,
    bufsize=0.4,
)

# ------------------
# Propagator options
# ------------------

for prop in model.propagators.__dict__.values():
    prop.options = prop.Options()
model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# For kinetic species the background is mandatory; without add_initial_condition() it is the initial condition.

vth1 = 0.02 / math.sqrt(2.0)
background = maxwellians.Maxwellian3D(
    n=(1.0, None), vth1=(vth1, None), vth2=(math.sqrt(12.0) * vth1, None), vth3=(vth1, None)
)
model.kinetic_ions.var.add_background(background)

# B3 = -1e-4 cos(k x) in physical coordinates: kx = 2 pi l / Lx, so Lx must be the box length
seed = perturbations.ModesCos(ls=(1,), amps=(-1e-4,), Lx=Lx, given_in_basis="physical", comp=2)
model.em_fields.b_field.add_perturbation(seed)


if __name__ == "__main__":
    import contextlib

    # Without a GPU, `struphy backend set cupy` and CUNUMPY_FAKE_CUPY=1 run the CuPy code path on cunumpy's fake CuPy with every
    # CUDA launch emulated on the CPU: a check that the model runs on the device backend, not a timing. On a GPU
    # (or on NumPy) the context does nothing.
    from struphy.models.tests.gpu_e2e import device_context

    with device_context() if fake_cupy_active() else contextlib.nullcontext():
        out = sim.run(profiling_activated=True)

    if sim.rank == 0:
        # `out.path_out` is the run's (absolute) output folder. The profiling packaging picks the files
        # written to `results` up from here and uploads them next to the run's profiling_data.h5.
        import json
        import statistics

        import numpy as np
        from scope_profiler import create_html_report

        results = out.profile.results  # the scope-profiler results of the run, all ranks merged
        results_dir = out.path_out / "results"
        results_dir.mkdir(exist_ok=True)

        # the full scope-profiler report: call tree, charts, per-rank statistics
        create_html_report(results, results_dir / "report.html")

        # cost per step and per marker, from the `model.integrate` region (the maximum over the ranks per step);
        # the first step includes one-time work (CUDA module loads, allocations of work arrays)
        steps = results["model.integrate"]
        per_rank = [steps[rank].durations for rank in steps.ranks]
        n_calls = min(len(durations) for durations in per_rank)
        step_times = np.max([durations[:n_calls] for durations in per_rank], axis=0)
        step_median = float(statistics.median(step_times[1:] if len(step_times) > 1 else step_times))
        n_markers = args.ppc * args.num_elements
        summary = {
            "backend": xp.get_backend(),
            "gpu_timing": gpu_timing,
            "mpi_ranks": results.num_ranks,
            "num_elements": args.num_elements,
            "ppc": args.ppc,
            "n_markers": n_markers,
            "steps": args.steps,
            "dt": dt,
            "step_times": step_times.tolist(),
            "step_median": step_median,
            "ns_per_marker_step": 1e9 * step_median / n_markers,
        }
        (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(
            f"\nstep: median {1e3 * step_median:.3f} ms (first step {1e3 * step_times[0]:.3f} ms), "
            f"{summary['ns_per_marker_step']:.1f} ns per marker and step ({n_markers:,} markers)"
        )
        # the propagators and kernels of the run; on the CuPy backend with GPU columns
        results.print_summary(include=["^prop: ", "^kernel: "], sort="total", title="propagators and kernels")
