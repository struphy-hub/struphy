"""Weibel instability with the VlasovMaxwellOneSpecies model: electrons with an anisotropic temperature in
a 1D periodic box drive the growth of a transverse magnetic field. `--setup profiling` selects the setup of
Kraus et al. (2017) used for profiling."""

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
# submit_weibel_instability.py; pproc_weibel_instability.py takes the same flags.
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
        "example: 1e5 markers without control variate, 8000 steps, binning plots; profiling: the setup of Kraus "
        "et al. (control variate, vth3 = vth1) profiled with scope-profiler (1000 markers per cell, 20 steps, "
        "no marker output, no sorting boxes, NumPy or CuPy backend)."
    ),
)
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Few markers and three saved time steps of the example setup, for the tests (test_examples.py).",
)
args, _ = parser.parse_known_args()
profiling = args.setup == "profiling"

if profiling:
    name = "Vlasov-Maxwell Weibel instability"
    description = """
Weibel instability with the VlasovMaxwellOneSpecies model, the setup of Kraus et al., J. Plasma Phys. 83 (2017),
Sec. 6.2, as in the verification test in src/struphy/models/tests/verification: periodic box of length 2 pi / 1.25,
anisotropic Maxwellian electrons (vth1 = vth3 = 0.02 / sqrt(2), vth2 = sqrt(12) vth1), seed field
B3 = -1e-4 cos(k x), no density perturbation, control-variate weights, cubic splines in x. The magnetic field energy
grows at the rate 0.02784. alpha = 1 and epsilon = -1 (electrons, time in units of the inverse plasma frequency,
velocities in units of c).

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
Weibel instability: A linear test case for the VlasovMaxwellOneSpecies model.
This test considers a plasma with an anisotropic velocity distribution, where 
temperature differs between directions. Small magnetic perturbations grow due to the anisotropy, 
leading to the generation of transverse magnetic fields. 

Note that for this simulaiton, the control_variate is set to False as it violates the Gauss law, 
and create growth in electric field.
"""

# ------------------
# Import Struphy API
# ------------------

import logging
import math

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

set_logging_level(logging.INFO if profiling else logging.WARNING)

# ---------------------
# Instance of the model
# ---------------------

from struphy.models import VlasovMaxwellOneSpecies

# Units
base_units = BaseUnits()

# Model instance
model = VlasovMaxwellOneSpecies(base_units=base_units, alpha=1.0, epsilon=-1.0, measure_gauss_law=not profiling)

# ---------------------
# Parameters setup
# ---------------------

import cunumpy as xp

k = 1.25
B_pert_amp = -1e-4

vth1_background_val = 0.02 / math.sqrt(2)  # plain floats, also on the CuPy backend
vth2_background_val = vth1_background_val * math.sqrt(12)

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
    # --test-run: the initial, a middle and a final saved time step
    time_opts = Time(dt=dt, Tend=2 * dt if args.test_run else 400, split_algo="LieTrotter")

# Geometry
Lx = 2 * math.pi / k
domain = domains.Cuboid(r1=Lx)

# Fluid equilibrium (can be used as part of initial conditions)
equil = None

# Grid
num_elements = 32  # cells in x (1 in y and z)
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

# Siumlation object
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

boundary_params = BoundaryParameters()
if profiling:
    ppc = 1000
    loading_params = LoadingParameters(ppc=ppc, seed=1234)
    weights_params = WeightsParameters(control_variate=True)
    # sorting boxes are off: SortingBoxes does not work on the CuPy backend yet (struphy-hub/struphy#726)
    sorting_params = SortingParameters()
    saving_params = SavingParameters(n_markers=0)  # profile the physics, not the marker output
    bufsize = 0.4
else:
    loading_params = LoadingParameters(
        Np=2000 if args.test_run else 100000,
        set_zero_velocity=(False, False, True),
        moments=(0.0, 0.0, 0.0, vth1_background_val, vth2_background_val, 1.0),
        seed=1234,
    )
    weights_params = WeightsParameters(control_variate=False)
    sorting_params = SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True)

    binplot_dens = BinningPlot(slice="e1_v1", n_bins=(128, 128), ranges=((0.0, 1.0), (-0.1, 0.1)))
    binplot_velocity = BinningPlot(slice="v1_v2", n_bins=(128, 128), ranges=((-0.1, 0.1), (-0.1, 0.1)))
    binplot_current = tuple(
        [
            BinningPlot(slice=f"e{i}", n_bins=32, ranges=(0.0, 1.0), output_quantity=f"current_{j}")
            for j in range(1, 4)
            for i in range(1, 4)
        ]
    )
    saving_params = SavingParameters(binning_plots=(binplot_dens, binplot_velocity, *binplot_current))
    bufsize = 2.0

model.kinetic_ions.set_markers(
    loading_params=loading_params,
    weights_params=weights_params,
    boundary_params=boundary_params,
    sorting_params=sorting_params,
    saving_params=saving_params,
    bufsize=bufsize,
)

# ------------------
# Propagator options
# ------------------

model.propagators.maxwell.options = model.propagators.maxwell.Options()
model.propagators.push_eta.options = model.propagators.push_eta.Options()
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

if profiling:
    # Kraus et al.: vth3 = vth1
    maxwellian = maxwellians.Maxwellian3D(
        n=(1.0, None),
        vth1=(vth1_background_val, None),
        vth2=(vth2_background_val, None),
        vth3=(vth1_background_val, None),
    )
else:
    maxwellian = maxwellians.Maxwellian3D(vth1=(vth1_background_val, None), vth2=(vth2_background_val, None))
model.kinetic_ions.var.add_background(maxwellian)

# Perturbations of initial magnetic field
if profiling:
    # B3 = -1e-4 cos(k x) in physical coordinates: kx = 2 pi l / Lx, so Lx must be the box length
    seed = perturbations.ModesCos(ls=(1,), amps=(B_pert_amp,), Lx=Lx, given_in_basis="physical", comp=2)
else:
    seed = perturbations.ModesCos(amps=(B_pert_amp,), ls=(1,), comp=2)  # Initial Bz depending on x-axis
model.em_fields.b_field.add_perturbation(perturbation=seed)

if __name__ == "__main__":
    # post-processing and plots: pproc_weibel_instability.py
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
