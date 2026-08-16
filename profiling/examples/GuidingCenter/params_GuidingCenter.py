# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "GuidingCenter NumPy vs CuPy"
description = """
Guiding-centre (5D drift-kinetic) test particles in a homogeneous slab, used as the
NumPy-vs-CuPy backend comparison case.

This model is chosen for the backend comparison because its entire propagator stack
(PushGuidingCenterBxEstar, PushGuidingCenterParallel) is backed by kernels that have a
hand-written CUDA implementation, and it carries no FEEC field solve. The measured
wall-clock time is therefore dominated by the particle kernels themselves, which is what
the GPU port is meant to accelerate -- unlike e.g. LinearMHDDriftkineticCC, whose runtime
is dominated by MHD field propagators and one-off setup, so that particle-kernel speedups
are invisible in the total.

Both propagators are run with algo="explicit"; the default
("discrete_gradient_1st_order") is also CUDA-ported, but the explicit scheme keeps the
comparison to a single kernel call per stage and avoids the outer Picard loop, whose
iteration count can differ between runs.

Measured at the defaults below (Np=200000, 100 steps, single rank, one A100):

    backend   total (setup to finalize)
    numpy     126.2 s
    cupy        9.7 s          -> 13.0x

    region                                    numpy      cupy    speedup
    prop: PushGuidingCenterParallel          46.87 s   0.685 s     68x
    prop: PushGuidingCenterBxEstar           42.16 s   0.556 s     76x
    kernel: push_gc_Bstar_explicit_multistage 33.48 s  0.050 s    665x
    kernel: push_gc_bxEstar_explicit_multistage 29.66 s 0.051 s   581x
"""

import argparse
import os

parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    "--backend",
    choices=("numpy", "cupy"),
    default="numpy",
    help="Array backend to run the simulation with (default: numpy).",
)
# `--id` distinguishes runs that share a rank count but differ in something else (here:
# the array backend); the profiling driver passes its launch counter and looks for the
# output under `sim_<id>` (see `ProfilingCase.build_commands` / `package_run`).
# Unknown flags are ignored so the driver can forward other parameters as well.
parser.add_argument("--id", type=int, default=0, help="Run id, used to name the output folder.")
parser.add_argument("--Np", type=int, default=None, help="Number of markers (overrides the default).")
parser.add_argument("--Tend", type=float, default=None, help="End time (overrides the default).")
args, _ = parser.parse_known_args()

# Must be set before struphy (and therefore cunumpy) is imported.
os.environ["ARRAY_BACKEND"] = args.backend

import logging

from struphy import set_logging_level

set_logging_level(logging.WARNING)

# ------------------
# Import Struphy API
# ------------------

from struphy import (
    BaseUnits,
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
    equils,
    grids,
    maxwellians,
    perturbations,
)

# ---------------------
# Instance of the model
# ---------------------

from struphy.models import GuidingCenter

# Units
base_units = BaseUnits()

# Model instance
model = GuidingCenter(base_units=base_units)

# List all variables and decide whether to save their data
model.kinetic_ions.var.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

name = f"GuidingCenter ({args.backend})"

# Environment options
env = EnvironmentOptions(
    sim_folder=f"sim_{args.id:02d}",
    profiling_activated=True,
)

# Time stepping. Enough steps that the per-step particle work, not the one-off
# setup (which includes the CUDA RawKernel JIT compile), dominates the total.
time_opts = Time(dt=0.01, Tend=args.Tend if args.Tend is not None else 1.0)

# Geometry
domain = domains.Cuboid()

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab()

# Grid
grid = grids.TensorProductGrid(num_elements=(16, 16, 16))

# Derham options
derham_opts = DerhamOptions()

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
)

# -------------------
# Particle parameters
# -------------------

# Marker count is the knob that decides how particle-dominated the run is.
loading_params = LoadingParameters(Np=args.Np if args.Np is not None else 200000)
weights_params = WeightsParameters()
boundary_params = BoundaryParameters()
sorting_params = SortingParameters()
saving_params = SavingParameters()
model.kinetic_ions.set_markers(
    loading_params=loading_params,
    weights_params=weights_params,
    boundary_params=boundary_params,
    sorting_params=sorting_params,
    saving_params=saving_params,
)

# ------------------
# Propagator options
# ------------------

# algo="explicit" selects push_gc_bxEstar_explicit_multistage /
# push_gc_Bstar_explicit_multistage, both CUDA-ported.
model.propagators.push_bxe.options = model.propagators.push_bxe.Options(algo="explicit")
model.propagators.push_parallel.options = model.propagators.push_parallel.Options(algo="explicit")

# ------------------
# Initial conditions
# ------------------

# Background for kinetic species
maxwellian_1 = maxwellians.GyroMaxwellian2Dvperp(n=(1.0, None), equil=equil)
maxwellian_2 = maxwellians.GyroMaxwellian2Dvperp(n=(0.1, None), equil=equil)
background = maxwellian_1 + maxwellian_2
model.kinetic_ions.var.add_background(background)

# Perturbations for (some) kinetic species
perturbation = perturbations.TorusModesCos()
maxwellian_1pt = maxwellians.GyroMaxwellian2Dvperp(n=(1.0, perturbation), equil=equil)
init = maxwellian_1pt + maxwellian_2
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    sim.run()
