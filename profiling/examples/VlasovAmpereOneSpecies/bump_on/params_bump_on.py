import os

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

description = """
Nonlinear bump-on-tail instability: A kinetic plasma instability test case for the Vlasov-Ampère model.
This test features a "bump" (localized excess) in the high-velocity tail of the electron velocity distribution.
The bump-on-tail configuration is unstable to the generation of Langmuir waves, leading to energy transfer
from the hot electron population to the growing wave field. This nonlinear process exhibits complex dynamics
including mode coupling and particle trapping in the wave potential.
This benchmark validates the particle-in-cell treatment of velocity-space instabilities and wave-particle interactions.
"""

# ------------------
# Import Struphy API
# ------------------

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
from struphy.models import VlasovAmpereOneSpecies

# Units
base_units = BaseUnits()

# Model instance
model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)

# List all variables and decide whether to save their data
model.em_fields.e_field.save_data = True
model.em_fields.phi.save_data = True
model.kinetic_ions.var.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
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

env = EnvironmentOptions(
    sim_folder=f"sim_{args.id:02d}",
    out_folders=os.environ.get("STRUPHY_PROFILING_OUT_FOLDERS", os.getcwd()),
    profiling_activated=True,
    profiling_trace=True,
)

# Time stepping
time_opts = Time(dt=0.1, Tend=60.0, split_algo="LieTrotter")

# Geometry
domain = domains.Cuboid(r1=62.83)

# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Grid
grid = grids.TensorProductGrid(num_elements=(16, 16, 16))

# Derham options
derham_opts = DerhamOptions(degree=(3, 1, 1))

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

loading_params = LoadingParameters(ppc=20, seed=42, moments=(0.0, 0.0, 0.0, 3.0, 1.0, 1.0))
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters()
sorting_params = SortingParameters(boxes_per_dim=(4, 4, 4), do_sort=True)

binplot_1 = BinningPlot(
    slice="e1_v1", n_bins=(128, 128), ranges=((0.0, 1.0), (-10.0, 10.0))
)  # for initial velocity distribution
binplot_2 = BinningPlot(
    slice="v1", n_bins=128, ranges=(-10.0, 10.0)
)  # for progression of velocity and space distribution
saving_params = SavingParameters(binning_plots=(binplot_1, binplot_2))

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
model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0", solver=args.solver)

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# For kinetic species the background is mandatory.
# For kinetic species, if add_initial_condition() is not called, the background is taken as the kinetic initial condition.
# For kinetic species the perturbations are added to the moments of the distribution function (defined as tuples).

# Background for kinetic species
maxwellian_1 = maxwellians.Maxwellian3D(n=(9 / 10, None), u1=(3.0, None))
maxwellian_2 = maxwellians.Maxwellian3D(n=(1 / 10, None), u1=(-4.5, None), vth1=(0.5, None))
background = maxwellian_1 + maxwellian_2
model.kinetic_ions.var.add_background(background)

# Perturbations for (some) kinetic species
perturbation = perturbations.ModesCos(amps=(0.05,), ls=(1,))
init1 = maxwellians.Maxwellian3D(n=(9 / 10, None), u1=(3.0, None))
init2 = maxwellians.Maxwellian3D(n=(1 / 10, perturbation), u1=(-4.5, None), vth1=(0.5, None))
init = init1 + init2
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    # one_time_step=True isolates the initial Poisson solve (the only part of this model that
    # solver= affects -- the field then evolves via VlasovAmpereCoupling, unrelated to
    # PETScSolver) from the many identical transport steps a full run would otherwise dilute
    # the comparison with.
    sim.run(one_time_step=True)
