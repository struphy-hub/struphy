# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Vlasov-Maxwell weak Landau damping"
description = """
Weak Landau damping with the VlasovMaxwellOneSpecies model, the setup of the verification test in
src/struphy/models/tests/verification: periodic box of length 4 pi (k = 0.5), Maxwellian electrons with a density
perturbation of 1e-3 cos(k x), control-variate weights, cubic splines in x. The electric field energy decays at the
rate -0.1533. alpha = 1 and epsilon = -1 (electrons, time in units of the inverse plasma frequency, velocities in
units of c).

Runs on the NumPy (CPU) or CuPy (GPU) backend, chosen by the environment variable CUNUMPY_BACKEND (the profiling
driver exports it in the job script). All timing comes from scope-profiler: the run activates
the profiling session of Simulation.run, and on the CuPy backend the session records CUDA-event device time per
region (GPUOptions.timing) and synchronizes the device at region ends, so the CPU time of a region includes its
kernels. The post-processing writes the scope-profiler HTML report and a summary of the cost per step and marker
into the run's results folder.
"""

import argparse
import logging
from pathlib import Path

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
# `--out-folder` and `--sim-folder` set the output folder; the profiling driver passes its own
# (see `ProfilingCase.build_commands`).
# Unknown flags are ignored so the driver can forward other parameters as well.
# no -h here when imported by the pproc script, which has its own help
parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_1", help="Name of the output folder.")
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
    out_folders=args.out_folder,
    sim_folder=args.sim_folder,
    restart=False,
    save_restart=False,  # restart data would dominate the run time for many markers
)

# Time stepping
dt = 0.05
steps = 20
time_opts = Time(dt=dt, Tend=round(steps * dt, 14))

# Geometry
Lx = 4 * math.pi  # k = 0.5
domain = domains.Cuboid(r1=Lx)

# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Grid
num_elements = 32  # cells in x (1 in y and z)
grid = grids.TensorProductGrid(num_elements=(num_elements, 1, 1))

# Derham options
derham_opts = DerhamOptions(degree=(3, 1, 1))

# Profiling options: CUDA-event device times per region on the CuPy backend.
# cunumpy's fake CuPy (CUNUMPY_FAKE_CUPY=1, no GPU) has CUDA events without elapsed time, so GPU timing needs a GPU.
from cunumpy.kernel_testing import fake_cupy_active

gpu_timing = xp.get_backend() == "cupy" and not fake_cupy_active()

# scope-profiler run label, the name of the run in the comparison plots of the profiling-data page: the runs of
# one case differ in backend and rank count only, so the label names those two (e.g. "cupy, 4 ranks").
from maybempi import MPI, SerialMPI

num_ranks = 1 if isinstance(MPI, SerialMPI) else MPI.COMM_WORLD.Get_size()
profiling_label = f"{xp.get_backend()}, {num_ranks} rank" + ("s" if num_ranks != 1 else "")

profiling_opts = ProfilingOptions(
    label=profiling_label,
    use_gpu_timing=gpu_timing,
    gpu_timing_backend="cupy",
    # synchronize the device at region ends (needs a scope-profiler newer than struphy's pin, GPU timing only)
    **({"gpu_sync_on_exit": True} if gpu_timing else {}),
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

ppc = 1000
loading_params = LoadingParameters(ppc=ppc, seed=1234)
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
# For kinetic species the background is mandatory; the perturbation is added to the density.

background = maxwellians.Maxwellian3D(n=(1.0, None))
model.kinetic_ions.var.add_background(background)

perturbation = perturbations.ModesCos(ls=(1,), amps=(1e-3,))
model.kinetic_ions.var.add_initial_condition(maxwellians.Maxwellian3D(n=(1.0, perturbation)))


if __name__ == "__main__":
    # post-processing and plots: pproc_vlasov_maxwell.py
    import contextlib

    # Without a GPU, CUNUMPY_FAKE_CUPY=1 CUNUMPY_BACKEND=cupy runs the CuPy code path on cunumpy's fake CuPy with every
    # CUDA launch emulated on the CPU: a check that the model runs on the device backend, not a timing. On a GPU
    # (or on NumPy) the context does nothing.
    from struphy.models.tests.gpu_e2e import device_context

    with device_context() if fake_cupy_active() else contextlib.nullcontext():
        sim.run(profiling_activated=True)
