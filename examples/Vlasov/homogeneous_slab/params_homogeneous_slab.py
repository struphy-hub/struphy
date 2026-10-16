"""Particle pushing in a homogeneous slab (Vlasov model, no field solve): a CPU vs GPU benchmark. The pproc writes
the scope-profiler report and the cost per step and marker."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Vlasov in a homogeneous slab: 28 CPU ranks vs 1 GPU"
description = """
The Vlasov model (markers pushed in a static, homogeneous background magnetic field, no field solve): a pure
particle-pushing benchmark. Unit cube, periodic boundaries, HomogenSlab equilibrium (constant B), a Maxwellian
with a cosine density perturbation plus a second, uniform Maxwellian, 16 million markers, 20 steps.
Each step runs the PushVxB (rotation of v about B) and PushEta (position update, explicit Runge-Kutta)
propagators, followed by the MPI exchange of markers that left their process domain.

Runs on the NumPy (CPU) or CuPy (GPU) backend, chosen by the environment variable CUNUMPY_BACKEND (the profiling
driver exports it in the job script). All timing comes from scope-profiler; on the CuPy backend it records
CUDA-event device time per region. The post-processing (pproc_homogeneous_slab.py) writes the scope-profiler HTML
report and a summary of the cost per step and marker into the run's results folder.
"""

# One thread per MPI rank on both backends (CPU: 28 ranks x 1 thread, GPU: 1 rank x 1 thread): OMP_NUM_THREADS=1, and
# the same for OpenBLAS/MKL, which would otherwise start one thread per core in every rank. Set unconditionally, before
# NumPy is imported (OpenBLAS reads it when it is loaded).
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

# Column-major marker buffer on the GPU (no effect on the NumPy backend): both pushers of this case have CUDA versions
# that prepare the marker rows themselves, see struphy.pic.base.markers_column_major
os.environ.setdefault("STRUPHY_MARKERS_COLUMN_MAJOR", "1")

import argparse
import logging
from pathlib import Path

from struphy import set_logging_level

set_logging_level(logging.INFO)

import cunumpy as xp

# ------------------
# Import Struphy API
# ------------------
from struphy import (
    BaseUnits,
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
    equils,
    grids,
    maxwellians,
    perturbations,
)

# --------------------------
# Command line parameters
# --------------------------
# `--out-folder` and `--sim-folder` set the output folder; the profiling driver passes its own
# (see `ProfilingCase.build_commands`). pproc_homogeneous_slab.py takes the same flags.
# Unknown flags are ignored so the driver can forward other parameters as well.
# no -h here when imported by the pproc script, which has its own help
parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_1", help="Name of the output folder.")
parser.add_argument("--n-markers", type=int, default=16_000_000, help="Total number of markers.")
parser.add_argument("--steps", type=int, default=20, help="Number of time steps.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="10,000 markers and 2 steps, for the tests (test_examples.py).",
)
args, _ = parser.parse_known_args()
if args.test_run:
    args.n_markers = 10_000
    args.steps = 2

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import Vlasov

# Units
base_units = BaseUnits()

# Model instance
model = Vlasov(base_units=base_units)

# List all variables and decide whether to save their data
model.kinetic_ions.var.save_data = False  # profile the pushing, not the marker output

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
dt = 0.01
steps = args.steps
time_opts = Time(dt=dt, Tend=round(steps * dt, 14))

# Geometry
domain = domains.Cuboid()

# Fluid equilibrium: constant background magnetic field
equil = equils.HomogenSlab()

# Grid
# The grid only carries the projected background field and defines the MPI domains of the markers. 28 x 16 x 16
# cells, decomposed in all directions: 28 ranks get a 7 x 2 x 2 process grid, i.e. a 4 x 8 x 8 block of cells each.
grid = grids.TensorProductGrid(num_elements=(28, 16, 16), mpi_dims_mask=(True, True, True))

# Derham options
derham_opts = DerhamOptions()

# Profiling options: CUDA-event device times per region on the CuPy backend.
# cunumpy's fake CuPy (CUNUMPY_FAKE_CUPY=1, no GPU) has CUDA events without elapsed time, so GPU timing needs a GPU.
from cunumpy.kernel_testing import fake_cupy_active

gpu_timing = xp.get_backend() == "cupy" and not fake_cupy_active()

# scope-profiler run label, the name of the run in the reports and comparison plots, e.g.
# "CPU (numpy): 28 MPI ranks x 1 thread" and "GPU (cupy): 1 MPI rank x 1 thread, 1 GPU"
from maybempi import MPI, SerialMPI

num_ranks = 1 if isinstance(MPI, SerialMPI) else MPI.COMM_WORLD.Get_size()
num_threads = int(os.environ["OMP_NUM_THREADS"])
profiling_label = (
    f"{'GPU' if xp.get_backend() == 'cupy' else 'CPU'} ({xp.get_backend()}): "
    f"{num_ranks} MPI rank{'s' if num_ranks != 1 else ''} x {num_threads} thread{'s' if num_threads != 1 else ''}"
    + (f", {num_ranks} GPU{'s' if num_ranks != 1 else ''}" if xp.get_backend() == "cupy" else "")
)

profiling_opts = ProfilingOptions(
    label=profiling_label,
    use_gpu_timing=gpu_timing,
    gpu_timing_backend="cupy",
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

# 16 million markers in total: 4 GB of marker data on the GPU, ~570k markers per rank on 28 CPU ranks.
# pseudo_random loading is fast on both backends (sobol_standard is a Python loop, ~1 s per 100k markers per rank);
# NumPy and CuPy draw different markers from the same seed, so the two runs agree statistically, not value by value.
n_markers = args.n_markers
loading_params = LoadingParameters(Np=n_markers, seed=1234)
weights_params = WeightsParameters()
boundary_params = BoundaryParameters()
# sorting boxes are off: SortingBoxes does not work on the CuPy backend yet (struphy-hub/struphy#726)
sorting_params = SortingParameters()
saving_params = SavingParameters(n_markers=0)  # profile the pushing, not the marker output

model.kinetic_ions.set_markers(
    loading_params=loading_params,
    weights_params=weights_params,
    boundary_params=boundary_params,
    sorting_params=sorting_params,
    saving_params=saving_params,
    # room for markers arriving from other ranks: 25% of the markers (default 100%). Every marker kernel and column
    # operation runs over all rows of the buffer, holes included, so a smaller buffer makes each step cheaper on both
    # backends; with 28 ranks a rank's domain is 1/7 of the box in x and markers move ~0.01 per step.
    bufsize=0.25,
)

# ------------------
# Propagator options
# ------------------

model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
model.propagators.push_eta.options = model.propagators.push_eta.Options()

# ------------------
# Initial conditions
# ------------------
# For kinetic species the background is mandatory; the perturbation is added to the density.

maxwellian_1 = maxwellians.Maxwellian3D(n=(1.0, None))
maxwellian_2 = maxwellians.Maxwellian3D(n=(0.1, None))
model.kinetic_ions.var.add_background(maxwellian_1 + maxwellian_2)

perturbation = perturbations.TorusModesCos()
model.kinetic_ions.var.add_initial_condition(maxwellians.Maxwellian3D(n=(1.0, perturbation)) + maxwellian_2)


if __name__ == "__main__":
    import contextlib

    # Without a GPU, CUNUMPY_FAKE_CUPY=1 CUNUMPY_BACKEND=cupy runs the CuPy code path on cunumpy's fake CuPy with every
    # CUDA launch emulated on the CPU: a check that the model runs on the device backend, not a timing. On a GPU
    # (or on NumPy) the context does nothing.
    from struphy.models.tests.gpu_e2e import device_context

    # post-processing (profiling report and cost summary): pproc_homogeneous_slab.py
    with device_context() if fake_cupy_active() else contextlib.nullcontext():
        sim.run(profiling_activated=True)
