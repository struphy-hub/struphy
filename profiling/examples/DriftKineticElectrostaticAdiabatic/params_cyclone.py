# -----------------------------
# Description of the simulation
# -----------------------------
# Please fill in a verbal description of the simulation.
# It will be printed at the beginning of the simulation and can be used to keep track of the different runs.

name = "DriftKineticElectrostaticAdiabatic Cyclone NumPy vs CuPy"
description = """
Cyclone-instability ITG turbulence case for DriftKineticElectrostaticAdiabatic (see
examples/DriftKineticElectrostaticAdiabatic/cyclone/params_cyclone.py, the physics case
this profiling params file is adapted from), used as the NumPy-vs-CuPy backend
comparison case for a real gyrokinetic model rather than a toy one.

Unlike GuidingCenter (profiling/examples/GuidingCenter), this model carries a real FEEC
field solve every step (PoissonAdiabaticGyrokinetic, an ImplicitDiffusion subclass: a PCG
solve against the H1 mass/stiffness matrices in toroidal geometry with a Fourier filter),
on top of the two CUDA-ported guiding-center pushers (PushGuidingCenterBxEstar,
PushGuidingCenterParallel). The geometry is a toroidal HollowTorus (not a Cuboid slab),
and particle weights use the control-variate method. This is the case that answers
whether the CUDA port helps a real ITG run end to end, including the parts (Poisson
solve, control variates, sorting, HollowTorus mapping evaluation) that were not
individually targeted by the port.

Getting this to run under CuPy at all required fixing several real device-portability
bugs found while validating this case (not toy-model artifacts): a NumPy-vs-CuPy
scalar-typing trap in the `AdhocTorus` equilibrium (`xp.sqrt` on a plain float silently
returns a 0-d CuPy array, which then hit scipy's `UnivariateSpline`/`quad`, both
host-only), the Sobol marker-loading sequence (a scalar bit-manipulation algorithm with
nothing to vectorize, now forced onto plain NumPy instead of CuPy), a full-slice
NumPy->CuPy assignment in `AverageOperator` that does not auto-convert the way
boolean/fancy indexing does, CuPy's `einsum` not accepting `out=` at all (unlike
NumPy's), and (for the `--solver direct` path below) several `tosparse()`/`toarray()`
stubs and CuPy/NumPy mixing bugs in `feectools` that had never been exercised under
CuPy before. See the commits touching `fields_background/equils.py`, `pic/sobol_seq.py`,
`feec/mass.py`, `feec/linear_operators.py`, `pic/accumulation/filter.py`,
`feectools/feec/derivatives.py`, and `feectools/linalg/{stencil,solvers,direct_solvers}.py`.

Once it ran under the original iterative solver (`--solver pcg`), the result was a
genuine (non-toy) finding: unlike GuidingCenter, this model was *slower end to end* on
CuPy than on NumPy, because the part the CUDA port never touched -- the per-step
`PoissonAdiabaticGyrokinetic` PCG solve -- dominated the runtime and was itself slower on
the device (with tol=1e-12 it runs close to the full maxiter=3000 on both backends: the
operator is not well conditioned enough to converge much faster, and each of those
iterations needs a matvec plus a reduction whose result has to be read back to host
before the next iteration can even start -- a hard sequential dependency that is cheap
per iteration on CPU but not on GPU, where several small kernel launches plus the sync
add up to real cost a matrix this size cannot amortize away). Measured at
num_elements=(16, 64, 4), ppc=5, 1 step (dt=0.001), single rank, one H100:

    backend   total (setup to finalize)   PCG solve/call (model.integrate)   push_gc_bxe    push_gc_para
    numpy     62.7 s                      6.90 s                             0.099 s        0.095 s
    cupy      91.4 s                      14.54 s (2.1x SLOWER)              0.035 s (2.9x)  0.019 s (5.0x)

The default solver is now `direct` (`feectools.linalg.solvers.DirectSolver`), not `pcg`:
the LHS operator this propagator solves against is *constant* across every time step
(`divide_by_dt=False`, fixed `epsilon`/`Z` -- see `ImplicitDiffusion.__call__`), so
re-running an iterative solve close to `maxiter` every single call was pure waste on
either backend. `DirectSolver` factorizes once (lazily, on the first `solve()` call) via
a cached sparse LU (`feectools.linalg.direct_solvers.SparseSolver`) and reuses that
factorization for every later call, which needs only a triangular solve -- a small, fixed
number of kernel launches regardless of the operator's conditioning, which is exactly
what removes the GPU sync penalty above. Measured at the same configuration, 5 steps
(dt=0.001, Tend=0.005), single rank, one H100 -- `direct`'s first call pays the one-time
factorization, every later call is the pure solve:

    backend   solver   total (setup to finalize)   solve/call: 1st (factorize)   solve/call: later (solve only)
    numpy     pcg      108.7 s                      n/a (iterative every call)   6.9 s (unchanged)
    numpy     direct   52.5 s                        3.4 s                       0.008 s   (~860x vs pcg)
    cupy      pcg      n/a (>500s for 1 step alone, node-contended -- see below)
    cupy      direct   75.7 s                         2.4 s                       0.0076 s (~1900x vs the clean
                                                                                    14.54 s/solve pcg number above)

Both `pcg` and `direct` were verified to produce matching physics (`en_phi`/`en_tot`/
`phi_integral` scalars agree to ~1e-10, consistent with `pcg` simply not having fully
converged at `maxiter` while `direct` solves exactly). The `cupy pcg` 5-step number above
is intentionally omitted rather than reported: a rerun on this shared login-node GPU took
174.55 s for a *single* step (vs the clean, isolated 14.54 s/solve measured earlier the
same session), evidently due to GPU contention from other jobs -- worth knowing if you
reproduce this yourself, but not a number to trust as `pcg`'s true cost.

Whether a larger, more production-scale grid (num_elements=(32, 135, 5), the physics
case's own default) changes any of this is open -- `direct`'s one-time factorization cost
should grow with problem size (the setup-phase matrix assembly alone took 226 s on NumPy
at that size when this case was first validated), so whether it stays worth it at scale,
and by how much, needs its own longer-budget run; pass a larger --num-elements (with a
longer SLURM walltime) to check.

`num_elements`/`ppc`/`Tend` default to the validated, quick configuration above (a
speed/coverage tradeoff against the physics case's own (32, 135, 5)/50/0.01); override
with --num-elements/--ppc/--Tend for a larger run.
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
parser.add_argument("--ppc", type=int, default=None, help="Markers per cell (overrides the default, 5).")
parser.add_argument("--Tend", type=float, default=None, help="End time (overrides the default, 0.001 -> 1 step).")
parser.add_argument(
    "--solver",
    choices=("pcg", "direct"),
    default="direct",
    help=(
        "Symmetric solver for the PoissonAdiabaticGyrokinetic field solve (default: "
        "direct). 'direct' uses feectools.linalg.solvers.DirectSolver, a cached sparse "
        "LU factorization -- valid here because the LHS operator is constant across "
        "time steps (divide_by_dt=False, fixed epsilon/Z), so one factorization serves "
        "every step instead of a fresh (near-maxiter, since tol=1e-12 barely converges) "
        "PCG solve each time; see this file's docstring for the measured ~1900x "
        "per-solve speedup on CuPy. 'pcg' reproduces the original, much slower baseline."
    ),
)
parser.add_argument(
    "--num-elements",
    type=int,
    nargs=3,
    default=None,
    help="Grid resolution (overrides the default, 16 64 4).",
)
args, _ = parser.parse_known_args()

# Must be set before struphy (and therefore cunumpy) is imported.
os.environ["ARRAY_BACKEND"] = args.backend

if args.backend == "cupy":
    import cunumpy

    # Under CuPy with more than one MPI rank per node, every rank must bind to its own
    # GPU -- cupy defaults to device 0, so without this every rank on a node would
    # contend for the same GPU instead of getting one each. SLURM_LOCALID (the rank's
    # index within its node) is set by srun before this process even starts, so it works
    # without MPI being initialized yet. Falls back to device 0 outside SLURM (e.g. a
    # single-GPU login node).
    cunumpy.set_device(int(os.environ.get("SLURM_LOCALID", 0)))

    # feectools.ddm.mpi disables MPI by default on the CuPy backend (see the comment
    # there): every rank falls back to a MockComm reporting rank 0/size 1, so with more
    # than one rank every process independently creates the same output directory/HDF5
    # dataset and the survivors deadlock in the next collective.
    os.environ.setdefault("FEECTOOLS_ENABLE_MPI", "1")

import logging

from struphy import set_logging_level

set_logging_level(logging.WARNING)

# ------------------
# Import Struphy API
# ------------------

import cunumpy as xp

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
)
from struphy.initial.base import GenericPerturbation
from struphy.linear_algebra.solver import SolverParameters

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import DriftKineticElectrostaticAdiabatic
from struphy.pic.accumulation.filter import FilterParameters

# provides the correct value for epsilon = 1.4142e-3 = 0.36/(180*sqrt(2)) from the
# cyclone paper (10.1140/epjd/e2014-50180-9)
base_units = BaseUnits(kBT=0.1916)
model = DriftKineticElectrostaticAdiabatic(
    base_units=base_units,
    use_diagnostic_poisson=True,
)

# List all variables and decide whether to save their data
model.em_fields.phi.save_data = True
model.kinetic_ions.var.save_data = False

# --------------------------
# Instance of the simulation
# --------------------------

name = f"DriftKineticElectrostaticAdiabatic Cyclone ({args.backend})"

# Environment options
env = EnvironmentOptions(
    sim_folder=f"sim_{args.id:02d}",
    profiling_activated=True,
    save_restart=False,
)

# Time stepping. Short by default: enough steps to warm past one-off setup (Poisson
# assembly, particle loading, CUDA RawKernel JIT compile) without a long profiling run.
time_opts = Time(dt=0.001, Tend=args.Tend if args.Tend is not None else 0.001, split_algo="LieTrotter")

a, r_min, R0 = 0.36, 0.01, 1.0
num_elements = tuple(args.num_elements) if args.num_elements is not None else (16, 64, 4)
degree = (3, 3, 3)

# Fluid equilibrium (can be used as part of initial conditions)
equil = equils.AdhocTorus(a=a, R0=R0, B0=1.0, q_kind=2, q0=0.86, q1=2.52 + 0.86, l=-0.16, psi_k=5, psi_nel=200)

# Geometry
domain = domains.HollowTorus(a1=r_min, a2=a, R0=R0, sfl=True, pol_period=1, tor_period=19)

# Grid
grid = grids.TensorProductGrid(num_elements=num_elements, mpi_dims_mask=(True, True, False))

# Derham options
derham_opts = DerhamOptions(
    degree=degree,
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

ppc = args.ppc if args.ppc is not None else 5
loading_params = LoadingParameters(ppc=ppc, loading="sobol_standard", spatial="uniform", moments=(0, 0, 4, 4))
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters(bc=("remove", "periodic", "periodic"))
sorting_params = SortingParameters(boxes_per_dim=(12, 12, 6), do_sort=True, sorting_frequency=5)

saving_params = SavingParameters(n_markers=100)

model.kinetic_ions.set_markers(
    loading_params=loading_params,
    weights_params=weights_params,
    boundary_params=boundary_params,
    sorting_params=sorting_params,
    saving_params=saving_params,
    bufsize=1.0,
)

# ------------------
# Propagator options
# ------------------

model.propagators.gc_poisson.options = model.propagators.gc_poisson.Options(
    which_geometry="toroidal",
    solver=args.solver,
    solver_params=SolverParameters(tol=1e-12, maxiter=3000, recycle=False),
    filter_params={model.kinetic_ions.var: FilterParameters("fourier_in_tor", (1,), repeat=1)},
)
model.propagators.push_gc_bxe.options = model.propagators.push_gc_bxe.Options(
    algo="explicit",
    evaluate_e_field=True,
    maxiter=100,
)
model.propagators.push_gc_para.options = model.propagators.push_gc_para.Options(
    algo="explicit",
    evaluate_e_field=True,
    maxiter=100,
)

# ------------------
# Initial conditions
# ------------------

ns = 1
ms = 27
amps = 1.0e-6
kappa_n = 2.23
kappa_Ti = 6.96
Delta_n = Delta_Ti = 0.3
delta_r = 0.02
r0 = 0.5 * a
n0 = 1.0
Ti0 = 1.0


def n_r(r):
    return n0 * xp.exp(-kappa_n * a * Delta_n * xp.tanh((r - r0) / (Delta_n * a)))


def n_init(*etas):
    if len(etas) == 1:
        eta1 = etas[0][:, 0]
    else:
        eta1 = etas[0]
    r = r_min + (a - r_min) * eta1
    return n_r(r)


def Ti_r(r):
    return Ti0 * xp.exp(-kappa_Ti * a * Delta_Ti * xp.tanh((r - r0) / (Delta_Ti * a)))


def vth_init(*etas):
    if len(etas) == 1:
        eta1 = etas[0][:, 0]
    else:
        eta1 = etas[0]
    r = r_min + (a - r_min) * eta1
    return xp.sqrt(Ti_r(r))


def n_xyz(x, y, z):
    r = xp.sqrt((xp.sqrt(x**2 + y**2) - R0) ** 2 + z**2)
    return n_r(r)


def p_xyz(x, y, z):
    r = xp.sqrt((xp.sqrt(x**2 + y**2) - R0) ** 2 + z**2)
    return n_r(r) * Ti_r(r)


equil.p_xyz = p_xyz
equil.n_xyz = n_xyz


def pert_func(*etas):
    if len(etas) == 1:
        e1, e2, e3 = etas[0][:, 0], etas[0][:, 1], etas[0][:, 2]
    else:
        e1, e2, e3 = etas[0], etas[1], etas[2]
    r = (a - r_min) * e1 + r_min
    teta = 2 * xp.arctan(xp.sqrt((R0 + r) / (R0 - r)) * xp.tan(xp.pi * e2))
    phi = 2 * xp.pi * e3
    return n_r(r) * amps * xp.exp(-((r - r0) ** 2) / delta_r**2) * xp.cos(ms * teta - ns * phi)


# Background for kinetic species
background = maxwellians.GyroMaxwellian2D(n=(n_init, None), vth_para=(vth_init, None), vth_perp=(vth_init, None))
model.kinetic_ions.var.add_background(background)

perturbation = GenericPerturbation(pert_func, given_in_basis="0")
init = maxwellians.GyroMaxwellian2D(n=(n_init, perturbation), vth_para=(vth_init, None), vth_perp=(vth_init, None))
model.kinetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    sim.run()
