# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "PressureLessSPH CuPy multi-GPU scaling"
description = """
SPH test particles in a homogeneous cube, used as a third CuPy multi-GPU/multi-rank
strong-scaling case alongside GuidingCenter and VlasovAmpereOneSpecies -- this one has no
FEEC field solve at all, the low-per-marker-compute end of the three.
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
    perturbations,
)

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import PressureLessSPH

# Units
base_units = BaseUnits()

# Model instance
model = PressureLessSPH(base_units=base_units)

# List all variables and decide whether to save their data
model.cold_fluid.var.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

name = f"PressureLessSPH scaling ({args.backend})"

# Environment options
env = EnvironmentOptions(
    sim_folder=f"sim_{args.id:02d}",
    save_restart=False,
)

# Time stepping. Same dt as the other two scaling cases; enough steps that the
# per-step particle work, not one-off setup, dominates the total.
time_opts = Time(dt=0.01, Tend=args.Tend if args.Tend is not None else 1.0)

# Geometry -- same unit cube as the other two scaling cases, for comparability.
domain = domains.Cuboid()

# Fluid equilibrium: PushVinEfield pushes against equil.p0 (see below).
equil = equils.HomogenSlab()

# Grid -- same resolution as the other two scaling cases.
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

# Smaller default than the other two scaling cases' 50,000,000 -- see the description
# for why (first run of this case at scale).
loading_params = LoadingParameters(Np=args.Np if args.Np is not None else 10_000_000, seed=1234)
weights_params = WeightsParameters()
boundary_params = BoundaryParameters()
sorting_params = SortingParameters()
saving_params = SavingParameters()
model.cold_fluid.set_markers(
    loading_params=loading_params,
    weights_params=weights_params,
    boundary_params=boundary_params,
    sorting_params=sorting_params,
    saving_params=saving_params,
)

# ------------------
# Propagator options
# ------------------

model.propagators.push_eta.options = model.propagators.push_eta.Options()
phi = equil.p0
model.propagators.push_v.phi = phi
model.propagators.push_v.options = model.propagators.push_v.Options()

# ------------------
# Initial conditions
# ------------------

# Background for (some) sph variables -- uniform, no perturbation: this case
# measures scaling behaviour, not physical accuracy (same spirit as the other two
# scaling cases).
background = equils.ConstantVelocity()
model.cold_fluid.var.add_background(background)

if __name__ == "__main__":
    sim.run(profiling_activated=True)
