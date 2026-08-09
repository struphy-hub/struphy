import os

# -----------------------------
# Description of the simulation
# -----------------------------

description = """
Periodic-slab variant of the ToyDrift model (the model behind
examples/ToyGyrokinetic/diocotron_instability, which uses a physically non-periodic
HollowCylinder domain -- radial confinement is inherent to the diocotron instability). This case
swaps in a periodic Cuboid domain instead: unlike PoissonAdiabaticGyrokinetic (used by
DriftKineticElectrostaticAdiabatic), ToyDrift's field solve is a plain PoissonSolve with no
geometry-coupled averaging, so it works correctly on a periodic domain out of the box.

Unlike VlasovAmpereOneSpecies (which only solves Poisson once, as an initial condition), this
model's gc_poisson runs as a *regular per-step propagator* -- exactly the repeated-solve pattern
where PETSc's algebraic multigrid preconditioner shows a genuine win (see
struphy.linear_algebra.petsc_examples_benchmark's module docstring for the general story).
"""

# ------------------
# Import Struphy API
# ------------------

from struphy import (
    BaseUnits,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
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
from struphy.linear_algebra.solver import SolverParameters

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import ToyDrift

# Units
base_units = BaseUnits(kBT=1.0)

# Model instance
model = ToyDrift(base_units=base_units)

# List all variables and decide whether to save their data
model.em_fields.phi.save_data = True
model.kinetic_ions.var.save_data = False

# --------------------------
# Instance of the simulation
# --------------------------

# `--id` distinguishes runs that share a rank count but differ in something else; the
# profiling driver passes its launch counter (see `ProfilingJob.build_commands`).
# Unknown flags are ignored so the driver can forward other parameters as well.
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=0, help="Run id, used to name the output folder.")
parser.add_argument(
    "--solver", type=str, default="pcg", choices=["pcg", "petsc"], help="Solver for the Poisson-type solve."
)
args, _ = parser.parse_known_args()

# Environment options
env = EnvironmentOptions(
    sim_folder=f"sim_{args.id:02d}",
    out_folders=os.environ.get("STRUPHY_PROFILING_OUT_FOLDERS", os.getcwd()),
    profiling_activated=True,
    profiling_trace=True,
)

# Time stepping
time_opts = Time(dt=0.05, Tend=0.05, split_algo="LieTrotter")

# Geometry
domain = domains.Cuboid()

# Fluid equilibrium: straight B field, homogeneous density (default n0=1.0)
equil = equils.HomogenSlab(B0z=1.0, n0=1.0)

# Grid
grid = grids.TensorProductGrid(num_elements=(24, 24, 24))

# Derham options -- fully periodic (required: PETScSolver cannot (yet) assemble
# DirectionalDerivativeOperator along a non-periodic axis, see
# struphy.linear_algebra.petsc_solver._directional_derivative_to_stencil_matrix)
derham_opts = DerhamOptions(degree=(3, 3, 3), bcs=(None, None, None))

# Simulation object
sim = Simulation(
    model=model,
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

loading_params = LoadingParameters(ppc=5, seed=42)
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters()
sorting_params = SortingParameters(boxes_per_dim=(4, 4, 4), do_sort=True)

model.kinetic_ions.set_markers(
    loading_params=loading_params,
    weights_params=weights_params,
    boundary_params=boundary_params,
    sorting_params=sorting_params,
    bufsize=0.4,
)

# ------------------
# Propagator options
# ------------------

model.propagators.gc_poisson.options.solver = args.solver
model.propagators.gc_poisson.options.solver_params = SolverParameters(tol=1e-10, maxiter=20_000)
model.propagators.push_gc_bxe.options = model.propagators.push_gc_bxe.Options(
    algo="explicit",
    evaluate_e_field=True,
)

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).

# Background for kinetic species
background = maxwellians.GyroMaxwellian2D(
    n=(1.0, None),
    vth_para=(1.0, None),
    vth_perp=(1.0, None),
    equil=equil,
)
model.kinetic_ions.var.add_background(background)

# Perturbation, matching the Landau-damping style used elsewhere in this benchmark suite
perturbation = perturbations.ModesCos(amps=(0.5,), ls=(1,))
init = maxwellians.GyroMaxwellian2D(
    n=(1.0, perturbation),
    vth_para=(1.0, None),
    vth_perp=(1.0, None),
    equil=equil,
)
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    # one_time_step=True isolates the (still per-step, unlike VlasovAmpereOneSpecies)
    # gc_poisson solve for a single-step timing snapshot.
    sim.run(one_time_step=True)
