import os

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

description = """
Weibel instability: A linear test case for the VlasovMaxwellOneSpecies model. This test considers
a plasma with an anisotropic velocity distribution, where temperature differs between directions.
Small magnetic perturbations grow due to the anisotropy, leading to the generation of transverse
magnetic fields.

Like VlasovAmpereOneSpecies (see strong_Landau_damping/weak_Landau_damping/two_stream/bump_on in
this same profiling suite), VlasovMaxwellOneSpecies exposes its Poisson solve as a one-time
`model.initial_poisson` (not a regular per-step propagator: the fields then evolve via
MaxwellWeakAmpere, PushVxB and VlasovAmpereCoupling instead), so `solver=` only affects this
initial solve -- see `struphy.linear_algebra.petsc_examples_benchmark`'s module docstring for why
that benchmark re-invokes it directly rather than relying on a single sim.run().

Plain copy of examples/VlasovMaxwellOneSpecies/weibel_instability, with `num_elements` scaled up
from the original's tiny, highly-anisotropic 1D-style default of `(32, 1, 1)` cells to a proper
`(16, 16, 16)` 3D grid (PETSc's advantage only shows up above roughly 5,000 dofs -- see
struphy.linear_algebra.petsc_poisson_benchmark's module docstring), particle count reduced to
match, and a fixed seed (the original doesn't fix ppc/seed the same way) so pcg/petsc draw the
same particles.
"""

# ------------------
# Import Struphy API
# ------------------

from struphy import (
    BaseUnits,
    BinningPlot,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
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
from struphy.models import VlasovMaxwellOneSpecies

# Units
base_units = BaseUnits()

# Model instance
model = VlasovMaxwellOneSpecies(
    base_units=base_units,
    alpha=1.0,
    epsilon=-1.0,
    measure_gauss_law=True,
)

# ---------------------
# Parameters setup
# ---------------------

import cunumpy as xp

k = 1.25
B_pert_amp = -1e-4

vth1_background_val = 0.02 / xp.sqrt(2)
vth2_background_val = vth1_background_val * xp.sqrt(12)

# List all variables and decide whether to save their data
model.em_fields.e_field.save_data = True
model.em_fields.phi.save_data = True
model.kinetic_ions.var.save_data = True

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

# scope-profiler label: distinguishes solver/rank-count combinations in post-processing
# (chart legends, `scope-profiler inspect`), see EnvironmentOptions.profiling_label.
from feectools.ddm.mpi import mpi as MPI

_comm = MPI.COMM_WORLD
_num_ranks = _comm.Get_size() if _comm is not None else 1
_profiling_label = f"{args.solver}, {_num_ranks} rank" + ("s" if _num_ranks != 1 else "")

# Environment options
env = EnvironmentOptions(
    sim_folder=f"sim_{args.id:02d}",
    out_folders=os.environ.get("STRUPHY_PROFILING_OUT_FOLDERS", os.getcwd()),
    profiling_activated=True,
    profiling_trace=True,
    profiling_label=_profiling_label,
)

# Time stepping
time_opts = Time(dt=0.05, Tend=400, split_algo="LieTrotter")

# Geometry
domain = domains.Cuboid(r1=2 * xp.pi / k)

# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Grid
grid = grids.TensorProductGrid(num_elements=(16, 16, 16))

# Derham options
derham_opts = DerhamOptions()

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

loading_params = LoadingParameters(
    ppc=20,
    set_zero_velocity=(False, False, True),
    moments=(0.0, 0.0, 0.0, vth1_background_val, vth2_background_val, 1.0),
    seed=42,
)
weights_params = WeightsParameters(control_variate=False)
boundary_params = BoundaryParameters()
sorting_params = SortingParameters(boxes_per_dim=(4, 4, 4), do_sort=True)

binplot_dens = BinningPlot(slice="e1_v1", n_bins=(128, 128), ranges=((0.0, 1.0), (-0.1, 0.1)))
binplot_velocity = BinningPlot(slice="v1_v2", n_bins=(128, 128), ranges=((-0.1, 0.1), (-0.1, 0.1)))
binplot_current = tuple(
    BinningPlot(slice=f"e{i}", n_bins=32, ranges=(0.0, 1.0), output_quantity=f"current_{j}")
    for j in range(1, 4)
    for i in range(1, 4)
)
saving_params = SavingParameters(binning_plots=(binplot_dens, binplot_velocity, *binplot_current))

model.kinetic_ions.set_markers(
    loading_params=loading_params,
    weights_params=weights_params,
    boundary_params=boundary_params,
    sorting_params=sorting_params,
    saving_params=saving_params,
    bufsize=2.0,
)

# ------------------
# Propagator options
# ------------------

model.propagators.maxwell.options = model.propagators.maxwell.Options()
model.propagators.push_eta.options = model.propagators.push_eta.Options()
model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0", solver=args.solver)

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

maxwellian = maxwellians.Maxwellian3D(
    vth1=(vth1_background_val, None),
    vth2=(vth2_background_val, None),
)
model.kinetic_ions.var.add_background(maxwellian)

# Perturbation of initial magnetic field
model.em_fields.b_field.add_perturbation(
    perturbation=perturbations.ModesCos(amps=(B_pert_amp,), ls=(1,), comp=2),  # Initial Bz depending on x-axis
)

if __name__ == "__main__":
    # one_time_step=True isolates the initial Poisson solve (the only part of this model that
    # solver= affects -- the fields then evolve via MaxwellWeakAmpere/PushVxB/VlasovAmpereCoupling,
    # unrelated to PETScSolver) from the ~8000 identical transport steps a full run would
    # otherwise dilute the comparison with.
    sim.run(one_time_step=True)
