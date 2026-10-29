"""Weak (linear) Landau damping with the VlasovAmpereOneSpecies model: a small density perturbation in a 1D
periodic box is damped by phase mixing. The pproc compares the electric field energy with the analytical
damping; `--setup profiling` selects the setup used for profiling."""

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
# submit_weak_Landau_damping.py; pproc_weak_Landau_damping.py takes the same flags.
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
        "example: 32 cells, 400 steps, binning plots; profiling: setup of the profiling runs profiled with scope-profiler "
        "(1024 cells, 20 steps, no marker output, no sorting boxes, NumPy or CuPy backend)."
    ),
)
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Few markers and three saved time steps of the example setup, for the tests (test_examples.py).",
)
args, _ = parser.parse_known_args()
profiling = args.setup == "profiling"

name = "Vlasov-Ampere weak Landau damping"

# the parts of the description common to both setups
_summary = """
Weak (linear) Landau damping with the VlasovAmpereOneSpecies model: electrostatic 1D-3V kinetic electrons (PIC)
in a periodic box with a neutralizing ion background. A small density perturbation of a Maxwellian launches an
electron plasma wave that is damped without collisions, by the resonant electrons (Landau damping).
"""
_physics = """
Physics. With alpha = 1 and epsilon = -1 (electrons, time in units of the inverse plasma frequency 1/omega_p,
with vth = 1 lengths are in Debye lengths) and no background magnetic field
(equil = None) the model solves
    df/dt + v . grad_x f - E . grad_v f = 0,
    dE/dt = int v f d^3v       (Ampere's law; the current density of the electrons is -int v f d^3v),
where the initial field follows from Poisson's equation div E = n0 - n with n = int f d^3v and the ion density
n0 = int f0 d^3v of the background f0; Gauss's law is then kept by the Ampere equation. Linearized around a
Maxwellian with vth = 1, perturbations exp(i (k x - omega t)) obey the dispersion relation
    1 + 1/k^2 [1 + xi Z(xi)] = 0,   xi = omega / (sqrt(2) k),
with the plasma dispersion function Z (Landau's analytic continuation). For small k the solution is the Langmuir
wave omega^2 = 1 + 3 k^2 (Bohm-Gross) with a small negative imaginary part. Electrons with velocities close to the
phase velocity omega/k exchange energy with the wave; in a Maxwellian there are more slower than faster ones, so
the wave loses energy to them. Equivalently, the perturbation of f is sheared in phase space (filamentation in v,
phase mixing) and its density moment, which drives the field, decays. The perturbation is small enough that
trapping of the resonant electrons plays no role during the run.
"""
_expected = """
Expected result. For k = 0.5 the dispersion relation gives omega = 1.4156 - 0.1533 i: the electric field oscillates
with the frequency 1.4156 and its amplitude decays at the rate 0.1533 (the energy at twice the rate, with
oscillations at twice the frequency).
"""
_references = """
References:
- L. D. Landau, "On the vibrations of the electronic plasma", J. Phys. USSR 10, 25 (1946).
- E. Sonnendruecker, J. Roche, P. Bertrand and A. Ghizzo, "The semi-Lagrangian method for the numerical
  resolution of the Vlasov equation", J. Comput. Phys. 149, 201 (1999).
"""

if profiling:
    description = (
        _summary
        + _physics
        + """
Setup (profiling). The setup of the verification test in src/struphy/models/tests/verification: periodic box of
length 4 pi (k = 0.5), 1024 cells and cubic splines in x (only x is decomposed, up to 256 ranks), Maxwellian
electrons with vth = 1 and the density n = 1 + 1e-3 cos(k x), control-variate (delta-f) weights with 1000 markers
per cell (seed 1234), no sorting boxes, no marker output; 20 steps of dt = 0.05.

Runs on the NumPy (CPU) or CuPy (GPU) backend, chosen by the environment variable CUNUMPY_BACKEND (the profiling
driver exports it in the job script). All timing comes from scope-profiler: the run activates
the profiling session of Simulation.run, and on the CuPy backend the session records CUDA-event device time per
region (GPUOptions.timing) and synchronizes the device at region ends, so the CPU time of a region includes its
kernels.
"""
        + _expected
        + """The 20 steps (t = 1) are too short to see the damping; the post-processing writes the scope-profiler
HTML report and a summary of the cost per step and marker into the run's results folder.
"""
        + _references
    )
else:
    description = (
        _summary
        + _physics
        + """
Setup. Periodic box of length L = 12.56 (about 4 pi, so k = 2 pi / L = 0.5), 32 cells and cubic splines in x.
Maxwellian electrons with vth = 1 and the density n = 1 + 1e-3 cos(k x). Control-variate (delta-f) weights with
1000 markers per cell; time step dt = 0.05 up to t = 20 (Lie-Trotter splitting).
"""
        + _expected
        + """The pproc compares the electric energy with the analytical curve
    W_E(t) = pi (4 eps r exp(-0.1533 t) cos(1.4156 t - phi))^2,   eps = 1e-3, r = 0.3677, phi = 0.5362,
and shows the full f and delta f in the (x, v1) plane at four times (filamentation in v) and the fields along x
at the first, middle and last saved time.
"""
        + _references
    )

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
model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0)

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
    time_opts = Time(dt=dt, Tend=2 * dt if args.test_run else 20.0, split_algo="LieTrotter")

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

ppc = 50 if args.test_run and not profiling else 1000
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
