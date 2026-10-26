"""ITPA benchmark TAE (n = 6, m = 10, 11) driven by drift-kinetic energetic deuterons (LinearMHDDriftkineticCC):
toroidal Fourier filter and the canonical Maxwellian with the B_max correction (exact equilibrium)."""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_fourier", help="Name of the output folder.")
parser.add_argument("--num-elements", type=int, nargs=3, default=[24, 96, 16], help="Number of elements per direction.")
parser.add_argument("--ppc", type=int, default=100, help="Markers per cell of the energetic ions.")
parser.add_argument("--dt", type=float, default=0.25, help="Time step.")
parser.add_argument("--tend", type=float, default=1000.0, help="End time.")
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Small grid (8, 32, 4), ppc = 2 and a short run with three saved time steps, for the tests.",
)
args, _ = parser.parse_known_args()

if args.test_run:
    args.num_elements = [8, 32, 4]
    args.ppc = 2

# -----------------------------
# Description of the simulation
# -----------------------------

name = "ITPA TAE (LinearMHDDriftkineticCC), Fourier filter, B_max canonical Maxwellian"
description = """
ITPA benchmark TAE (n = 6, m = 10, 11, Koenies et al., Nucl. Fusion 58, 126027 (2018)) driven by energetic deuterons.
Toroidal Fourier filter, canonical Maxwellian with correction_B="Bmax" (exact equilibrium), J0 x b term off.
"""

import numpy as np
from maybempi import MPI

# ------------------
# Import Struphy API
# ------------------
from struphy import (
    BaseUnits,
    BinningPlot,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
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
)

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import LinearMHDDriftkineticCC
from struphy.pic.accumulation.filter import FilterParameters

base_units = BaseUnits(n=0.2)
model = LinearMHDDriftkineticCC(base_units=base_units, hot_mass_number=2.0)

# Variables to save
model.em_fields.b_field.save_data = True
model.mhd.density.save_data = True
model.mhd.pressure.save_data = True
model.mhd.velocity.save_data = True
model.energetic_ions.var.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(out_folders=args.out_folder, sim_folder=args.sim_folder, save_step=8, max_runtime=24 * 60)

# Time stepping
# --test-run: the initial, a middle and a final saved time step
time_opts = Time(dt=args.dt, Tend=2 * env.save_step * args.dt if args.test_run else args.tend, split_algo="Strang")

# Geometry
domain = domains.HollowTorus(a1=0.1, a2=1.0, R0=10.0, sfl=False, tor_period=6)

# Fluid equilibrium
equil = equils.AdhocTorus(
    a=1.0,
    R0=10.0,
    B0=3.0,
    q_kind=0,
    q0=1.71,
    q1=1.87,
    n1=0.0,
    n2=0.0,
    na=1.0,
    p_kind=1,
    p0=1.0,
    p1=0.95,
    p2=0.05,
    beta=0.0018,
)

# Grid
# (no domain decomposition along eta3, needed by the toroidal Fourier filter)
grid = grids.TensorProductGrid(num_elements=tuple(args.num_elements), mpi_dims_mask=(True, True, False))

# Derham options: u.n = 0 and b.n = 0 at the walls
derham_opts = DerhamOptions(degree=(3, 3, 3), bcs=(("dirichlet", "dirichlet"), None, None))

# Profiling options
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

vth = 0.89740839

loading_params = LoadingParameters(ppc=args.ppc, moments=(0.0, 0.0, vth, vth), seed=1234, B0=equil.params["B0"])
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters(bc=("remove", "periodic", "periodic"))
sorting_params = SortingParameters()

vpar_max = 5 * vth
binplot_r = BinningPlot(slice="e1", n_bins=64, ranges=(0.0, 1.0))
binplot_r_vpar = BinningPlot(slice="e1_v1", n_bins=(64, 64), ranges=((0.0, 1.0), (-vpar_max, vpar_max)))
saving_params = SavingParameters(n_markers=0, binning_plots=(binplot_r, binplot_r_vpar))

model.energetic_ions.set_markers(
    loading_params=loading_params,
    weights_params=weights_params,
    boundary_params=boundary_params,
    sorting_params=sorting_params,
    saving_params=saving_params,
)

# ------------------
# Propagator options
# ------------------

ep_scale = 2.0
filter_params = FilterParameters(use_filter="fourier_in_tor")

props = model.propagators
props.push_bxe.options = props.push_bxe.Options(algo="explicit")
props.push_parallel.options = props.push_parallel.Options(algo="explicit")
props.shearalfen_cc5d.options = props.shearalfen_cc5d.Options(
    ep_scale=ep_scale, filter_params=filter_params, nonlinear=True
)
props.magnetosonic.options = props.magnetosonic.Options(with_jxb=False)
props.cc5d_density.options = props.cc5d_density.Options(ep_scale=ep_scale, filter_params=filter_params)
props.cc5d_gradb.options = props.cc5d_gradb.Options(ep_scale=ep_scale, algo="explicit", filter_params=filter_params)
props.cc5d_curlb.options = props.cc5d_curlb.Options(ep_scale=ep_scale, filter_params=filter_params)

# ------------------
# Initial conditions
# ------------------

ms = (10, 11)
amp = 0.001
gap = (0.44444444, 0.1)
model.mhd.velocity.add_perturbation(
    perturbations.TorusModesSin(
        given_in_basis="2",
        amps=(amp, amp),
        ms=ms,
        ns=(-1, -1),
        pfuns=("exp", "exp"),
        pfun_params=(gap, gap),
        comp=0,
    )
)
model.mhd.velocity.add_perturbation(
    perturbations.TorusModesCos(
        given_in_basis="2",
        amps=tuple(amp / (2 * np.pi * m) for m in ms),
        ms=ms,
        ns=(-1, -1),
        pfuns=("d_exp", "d_exp"),
        pfun_params=(gap, gap),
        comp=1,
    )
)

itpa = perturbations.ITPA_density(n0=0.00720652, c=(0.50626, 0.18156, 0.13012, 0.54286))
psi_axis, psi_edge = equil.psi_range

def n_hot(psi_c):
    s = (psi_c - psi_axis) / (psi_edge - psi_axis)
    r_c = np.sign(s) * np.sqrt(np.abs(s))
    return itpa(r_c)

n_cells = int(np.prod(args.num_elements))
cache_size = int(2 * args.ppc * n_cells / MPI.COMM_WORLD.Get_size())

model.energetic_ions.var.add_background(
    maxwellians.CanonicalMaxwellian2D(
        n=(n_hot, None),
        vth=(vth, None),
        equil=equil,
        volume_form=True,
        epsilon=model.energetic_ions.equation_params.epsilon,
        correction_B="Bmax",
        cache_size=cache_size,
    )
)

if __name__ == "__main__":
    sim.run()
