# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "GuidingCenter CuPy multi-GPU scaling"
description = """
Guiding-centre (5D drift-kinetic) test particles in a homogeneous slab, used as the CuPy
multi-GPU/multi-rank strong-scaling case. Np is much larger than params_GuidingCenter.py's
(50,000,000 vs 200,000) so there's enough per-rank compute between marker-exchange calls
for scaling to actually pay off, rather than being dominated by mpi_sort_markers.
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

if args.backend == "cupy":
    import cunumpy

    # Under CuPy with more than one MPI rank per node, every rank must bind to its own GPU
    # -- cupy defaults to device 0, so without this every rank on a node would contend for
    # the same GPU instead of getting one each. SLURM_LOCALID (the rank's index within its
    # node) is set by srun before this process even starts, so it works without MPI being
    # initialized yet. Falls back to device 0 outside SLURM (e.g. a single-GPU login node).
    cunumpy.set_device(int(os.environ.get("SLURM_LOCALID", 0)))

    # feectools.ddm.mpi disables MPI by default on the CuPy backend (see the comment
    # there): every rank falls back to a MockComm reporting rank 0/size 1, so with more
    # than one rank every process independently creates the same output directory/HDF5
    # dataset and the survivors deadlock in the next collective. This file is specifically
    # meant to run multi-rank/multi-GPU, so opt back in; a single-GPU run pays only a
    # no-op collective for it.
    os.environ.setdefault("FEECTOOLS_ENABLE_MPI", "1")

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

name = f"GuidingCenter scaling ({args.backend})"

# Environment options
env = EnvironmentOptions(
    sim_folder=f"sim_{args.id:02d}",
    profiling_activated=True,
    save_restart=False,
)

# Time stepping. Enough steps that the per-step particle work (and its MPI exchange),
# not the one-off setup (marker loading scales with Np, plus the CUDA RawKernel JIT
# compile), dominates the total -- see the setup-vs-loop measurement in the description.
time_opts = Time(dt=0.01, Tend=args.Tend if args.Tend is not None else 1.0)

# Geometry
domain = domains.Cuboid()

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab()

# Grid. Coarser than this relative to Np/rank count means the fixed-width ghost/halo
# padding around each rank's local sub-grid is a bigger fraction of its data -- see the
# description for why this was raised from (16, 16, 16).
grid = grids.TensorProductGrid(num_elements=(32, 32, 32))

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

# Marker count is the knob that decides how particle-dominated (vs. communication-bound)
# the run is -- see the description for why this file defaults to 50,000,000 rather than
# params_GuidingCenter.py's 200,000.
loading_params = LoadingParameters(Np=args.Np if args.Np is not None else 50_000_000)
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
