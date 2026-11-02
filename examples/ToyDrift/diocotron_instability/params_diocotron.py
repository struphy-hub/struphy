"""Diocotron instability with the ToyDrift model: a density ring in a hollow cylinder with a uniform
axial magnetic field is unstable due to the shear of its E x B rotation (Kelvin-Helmholtz-like), see Davidson,
"Physics of Nonneutral Plasmas", and Crouseilles et al. (2014). The pproc fits the growth rate of the
electrostatic energy; `--setup profiling` selects the setup used for profiling."""

# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "Diocotron instability (ToyDrift)"
description = """
Diocotron instability of an annular non-neutral plasma (a ring of charge of a single sign) in a uniform axial
magnetic field, between two conducting cylinders, with the ToyDrift model: guiding centers that move with the
E x B drift in their own electrostatic field.

Physics. The guiding-center distribution f(X, v_par, mu, t) is advected by the E x B drift only,
    df/dt + (E x b0 / B*_par) . grad_X f = 0,   E = -grad phi,
and the potential is given by the charge density through the Poisson equation
    -Laplace(phi) = int f B*_par dv_par dmu
(with control variate: f - f0 on the right-hand side). In the uniform field B0 = e_z, the drift velocity
v = E x e_z is divergence free and its vorticity is proportional to the density, so the density is advected like
the vorticity of a 2D incompressible fluid (guiding-center plasma = 2D Euler equations). The ring rotates
differentially (shear of the E x B rotation). Its inner and outer edges carry surface waves that propagate in
opposite directions relative to the local flow; for azimuthal mode numbers l below a cutoff they phase-lock and
grow, as in the Kelvin-Helmholtz instability of a shear layer. For a step profile between conducting walls the
linear dispersion relation is a quadratic in the frequency omega for each l (see Davidson); in the nonlinear
phase the ring rolls up into l vortices, which later merge.

Setup. HollowCylinder with r in [1, 10] and length Lz = 10 (one element along z, i.e. a 2D problem), uniform
field B0 = e_z (HomogenSlab), epsilon = alpha = 1, phi = 0 on both walls (Dirichlet). The density is n = 1 in
the ring 4 <= r < 5 and zero elsewhere, perturbed by 1e-6 cos(l theta) with l = 4 inside the ring. Markers are
loaded uniformly on the disc with weights rejected below 1e-4 (background f0 = 0); the E x B pusher is the
first-order discrete gradient method (Newton iteration). Grid 64 x 128 x 1 elements (example) or 128 x 1024 x 1
(profiling, one time step), splines of degree (3, 3, 1). Time step dt = 0.01, end time Tend = 51.

Expected result. Linear theory of this step profile with conducting walls at r = 1 and r = 10 gives for l = 4 the
growth rate gamma = 0.19 (time unit of the model; l = 3: 0.18, l = 5: 0.16), i.e. the energy of the perturbation
grows like exp(2 gamma t) = exp(0.38 t). The pproc fits an exponential to en_phi in 0 <= t <= 42 (en_phi is the
total electrostatic energy, including the constant energy of the equilibrium field), and shows the equilibrium
profiles and the density, its perturbation and phi at the first, middle and last saved time (roll-up of the ring
into 4 vortices).

References:
- R. C. Davidson, "Physics of Nonneutral Plasmas", Addison-Wesley, Redwood City (1990).
- N. Crouseilles, P. Glanc, S. A. Hirstoaga, E. Madaule, M. Mehrenberger and J. Petri, "A new fully
  two-dimensional conservative semi-Lagrangian method: applications on polar grids, from diocotron instability to
  ITG turbulence", Eur. Phys. J. D 68, 252 (2014).
"""

import argparse
from pathlib import Path

# Command line options
# `--out-folder` and `--sim-folder` set the output folder; the profiling driver passes its own
# (see `ProfilingCase.build_commands`). `--setup profiling` selects the profiling setup of
# submit_diocotron_instability.py; pproc_diocotron.py takes the same flags.
# Unknown flags are ignored so the driver can forward other parameters as well.
# no -h here when imported by the pproc script, which has its own help
parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_1", help="Name of the output folder.")
parser.add_argument(
    "--setup",
    default="example",
    choices=["example", "profiling"],
    help=(
        "example: 64 x 128 cells, 16 x 16 sorting boxes; profiling: one time step with 128 x 1024 cells and "
        "256 x 256 sorting boxes."
    ),
)
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small resolution (16 x 32 cells), few markers and three saved time steps, for the tests (test_examples.py).",
)
args, _ = parser.parse_known_args()
profiling = args.setup == "profiling"

# ------------------
# Import Struphy API
# ------------------


import logging

from struphy import set_logging_level

set_logging_level(logging.INFO)

# For particles:
import cunumpy as xp

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
from struphy.models import ToyDrift

base_units = BaseUnits(kBT=1.0)
model = ToyDrift(
    epsilon=1.0,
    alpha=1.0,
    base_units=base_units,
)

# List all variables and decide whether to save their data
model.em_fields.phi.save_data = True
model.kinetic_ions.var.save_data = False

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder, restart=False)

# Time stepping
dt = 0.01
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=dt, Tend=2 * dt if args.test_run else 51.0, split_algo="LieTrotter")

# Geometry
domain = domains.HollowCylinder(a1=1.0, a2=10.0, Lz=10.0)

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab()

# Grid
num_elements = (128, 1024, 1) if profiling else (16, 32, 1) if args.test_run else (64, 128, 1)
grid = grids.TensorProductGrid(num_elements=num_elements, mpi_dims_mask=(False, True, False))

# Derham options
derham_opts = DerhamOptions(
    degree=(3, 3, 1),
    bcs=(("dirichlet", "dirichlet"), None, None),
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
)

# -------------------
# Particle parameters
# -------------------

ppc = 10 if args.test_run else 100  # run with 1000 minimum
loading_params = LoadingParameters(ppc=ppc, loading="sobol_standard", spatial="disc")
weights_params = WeightsParameters(control_variate=True, reject_weights=True, threshold=0.0001)
boundary_params = BoundaryParameters()
boxes_per_dim = (256, 256, 1) if profiling else (4, 4, 1) if args.test_run else (16, 16, 1)
sorting_params = SortingParameters(boxes_per_dim=boxes_per_dim, do_sort=True, sorting_frequency=5)

# density binning
eta_bin = BinningPlot(slice="e1_e2", n_bins=(128, 128), ranges=((0.0, 1.0), (0.0, 1.0)))
saving_params = SavingParameters(binning_plots=(eta_bin,))

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

model.propagators.gc_poisson.options = model.propagators.gc_poisson.Options()
model.propagators.push_gc_bxe.options = model.propagators.push_gc_bxe.Options(
    algo="discrete_gradient_1st_order_newton", evaluate_e_field=True
)

# ------------------
# Initial conditions
# ------------------
# Initial conditions are the sum of the background(s) and the perturbation(s).
# If backgrounds or perturbations are not specified, they are assumed to be zero.

# For kinetic species the background is mandatory.
# For kinetic species, if add_initial_condition() is not called, the background is taken as the kinetic initial condition.
# For kinetic species the perturbations are added to the moments of the distribution function (defined as tuples).

# piecewise function for initial condition of density
r_minus, r_plus = 4.0, 5.0
ms = 4


def n_init(etas, r_minus=r_minus, r_plus=r_plus):

    # transform logical coordinate to polar
    a1, a2 = domain.params["a1"], domain.params["a2"]
    radial = a1 + (a2 - a1) * etas[0]

    return 1.0 * ((r_minus <= radial) & (radial < r_plus))


# Background for kinetic species
background = maxwellians.GyroMaxwellian2D(
    n=(0.0, None),
)  # B0=equil.absB0)
model.kinetic_ions.var.add_background(background)


eta_minus = (r_minus - domain.params["a1"]) / (domain.params["a2"] - domain.params["a1"])
eta_plus = (r_plus - domain.params["a1"]) / (domain.params["a2"] - domain.params["a1"])

# Perturbations for (some) kinetic species

# for non linear case amps = (0.5,)
perturbation = perturbations.ModesCos(amps=(1e-6,), ms=(ms,), perb_domain=((eta_minus, eta_plus), None, None))
init = maxwellians.GyroMaxwellian2D(
    n=(n_init, perturbation),
)  # B0=equil.absB0)
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    # post-processing and plots: pproc_diocotron.py
    sim.run(profiling_activated=True, one_time_step=profiling)
