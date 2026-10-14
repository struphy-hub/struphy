# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

import argparse
from pathlib import Path

# Command line options
# `--out-folder` and `--sim-folder` set the output folder; the profiling driver passes its own
# (see `ProfilingCase.build_commands`). `--setup profiling` selects the profiling setup of
# submit_vlasov_ampere_landau_damping.py; pproc_weak_Landau_damping.py takes the same flags.
# Unknown flags are ignored so the driver can forward other parameters as well.
# no -h here when imported by the pproc script, which has its own help
parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_data", help="Name of the output folder.")
parser.add_argument(
    "--setup",
    default="example",
    choices=["example", "profiling"],
    help=(
        "example: 32 cells, 400 steps, binning plots; profiling: strong scaling setup profiled with scope-profiler "
        "(1024 cells, 20 steps, no marker output, no sorting boxes, NumPy or CuPy backend)."
    ),
)
args, _ = parser.parse_known_args()
profiling = args.setup == "profiling"

if profiling:
    name = "Vlasov-Ampere weak Landau damping"
    description = """
Weak Landau damping with the VlasovAmpereOneSpecies model, the setup of the verification test in
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
else:
    name = ""
    description = """
Weak Landau damping: A linear test case for the VlasovAmpereOneSpecies model.
This test involves a small amplitude electrostatic perturbation in a uniform, collisionless plasma.
The perturbation is damped due to phase mixing effects (Landau damping) as particles interact with 
the self-consistent electric field. This benchmark validates the numerical discretization of the 
Vlasov-Ampère system and the accuracy of particle-in-cell methods.
"""

# ------------------
# Import Struphy API
# ------------------

import logging
import math

import cunumpy as xp

# For particles:
from struphy import (
    BaseUnits,
    BinningPlot,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    KernelDensityPlot,
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
    set_logging_level,
)
from struphy.models import VlasovAmpereOneSpecies

if profiling:
    set_logging_level(logging.INFO)

# ---------------------
# Instance of the model
# ---------------------


# Units
base_units = BaseUnits()

# Model instance
model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)

# List all variables and decide whether to save their data
model.em_fields.e_field.save_data = True
model.em_fields.phi.save_data = True
model.kinetic_ions.var.save_data = not profiling

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(
    out_folders=args.out_folder,
    sim_folder=args.sim_folder,
    # restart data would dominate the run time of the profiling setup for many markers
    save_restart=not profiling,
)

# Time stepping
dt = 0.05
if profiling:
    steps = 20
    time_opts = Time(dt=dt, Tend=round(steps * dt, 14))
else:
    time_opts = Time(dt=dt, Tend=20.0, split_algo="LieTrotter")

# Geometry
if profiling:
    domain = domains.Cuboid(r1=4 * math.pi)  # k = 0.5
else:
    domain = domains.Cuboid(r1=12.56)  # r1 -> pi * 4 -> k = 0.5

# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Grid
if profiling:
    # cells in x (1 in y and z). Only x is decomposed, and every rank needs at least `degree` = 3 cells of its own,
    # so 1024 cells run on up to 256 ranks (4 cells per rank), one full CPU node.
    num_elements = 1024
    grid = grids.TensorProductGrid(num_elements=(num_elements, 1, 1), mpi_dims_mask=(True, False, False))
else:
    num_elements = 32
    grid = grids.TensorProductGrid(num_elements=(num_elements, 1, 1))

# Derham options
derham_opts = DerhamOptions(degree=(3, 1, 1))

# Profiling options
if profiling:
    # CUDA-event device times per region on the CuPy backend.
    # cunumpy's fake CuPy (CUNUMPY_FAKE_CUPY=1, no GPU) has CUDA events without elapsed time, so GPU timing needs a GPU.
    from cunumpy.kernel_testing import fake_cupy_active
    from maybempi import MPI, SerialMPI

    gpu_timing = xp.get_backend() == "cupy" and not fake_cupy_active()

    # scope-profiler run label, the name of the run in the comparison plots of the profiling-data page: the runs of
    # one case differ in backend and rank count only, so the label names those two (e.g. "cupy, 4 ranks").
    num_ranks = 1 if isinstance(MPI, SerialMPI) else MPI.COMM_WORLD.Get_size()
    profiling_label = f"{xp.get_backend()}, {num_ranks} rank" + ("s" if num_ranks != 1 else "")

    profiling_opts = ProfilingOptions(
        label=profiling_label,
        use_gpu_timing=gpu_timing,
        gpu_timing_backend="cupy",
        # synchronize the device at region ends (needs a scope-profiler newer than struphy's pin, GPU timing only)
        **({"gpu_sync_on_exit": True} if gpu_timing else {}),
    )
else:
    profiling_opts = ProfilingOptions()

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
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters()
if profiling:
    loading_params = LoadingParameters(ppc=ppc, seed=1234)
    # sorting boxes are off: SortingBoxes does not work on the CuPy backend yet (struphy-hub/struphy#726)
    sorting_params = SortingParameters()
    saving_params = SavingParameters(n_markers=0)  # profile the physics, not the marker output
else:
    loading_params = LoadingParameters(ppc=ppc)
    sorting_params = SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True)
    binplot = BinningPlot(slice="e1_v1", n_bins=(128, 128), ranges=((0.0, 1.0), (-5.0, 5.0)))
    saving_params = SavingParameters(binning_plots=(binplot,))

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

model.propagators.push_eta.options = model.propagators.push_eta.Options()
if model.with_B0:
    model.propagators.push_vxb.options = model.propagators.push_vxb.Options()

model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# For kinetic species the background is mandatory.
# For kinetic species, if add_initial_condition() is not called, the background is taken as the kinetic initial condition.
# For kinetic species the perturbations are added to the moments of the distribution function (defined as tuples).

# Background for kinetic species
background = maxwellians.Maxwellian3D(n=(1.0, None))
model.kinetic_ions.var.add_background(background)

# Perturbations for (some) kinetic species
perturbation = perturbations.ModesCos(amps=(0.001,), ls=(1,))
init = maxwellians.Maxwellian3D(n=(1.0, perturbation))
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    # post-processing and plots: pproc_weak_Landau_damping.py
    if profiling:
        import contextlib

        # Without a GPU, CUNUMPY_FAKE_CUPY=1 CUNUMPY_BACKEND=cupy runs the CuPy code path on cunumpy's fake CuPy with
        # every CUDA launch emulated on the CPU: a check that the model runs on the device backend, not a timing. On a
        # GPU (or on NumPy) the context does nothing.
        from struphy.models.tests.gpu_e2e import device_context

        with device_context() if fake_cupy_active() else contextlib.nullcontext():
            sim.run(profiling_activated=True)
    else:
        sim.run()
