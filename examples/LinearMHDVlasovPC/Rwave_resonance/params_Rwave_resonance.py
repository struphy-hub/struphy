"""Resonance between the R wave and the gyromotion of full-orbit energetic ions in a homogeneous slab, pressure-coupling scheme."""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
parser.add_argument(
    "--out-folder",
    default=str(Path(__file__).resolve().parent),
    help="Folder in which the output folder is created (default: beside this file).",
)
parser.add_argument("--sim-folder", default="sim_data", help="Name of the output folder.")
parser.add_argument(
    "--num-clones", type=int, default=1, help="Number of domain clones (must divide the number of MPI ranks)."
)
parser.add_argument(
    "--test-run",
    action="store_true",
    help="Few markers (ppc = 100) and a short run with three saved time steps, for the tests.",
)
args, _ = parser.parse_known_args()

# -----------------------------
# Description of the simulation
# -----------------------------

name = "LinearMHDVlasovPC R-wave resonance"
description = """
Resonance between the R wave (k = 0.8 along z) and the gyromotion of full-orbit energetic ions in a homogeneous slab, pressure-coupling scheme. The energetic ions, a shifted Maxwellian (u_par = 2.5 v_A, density 0.02 of the bulk), resonate with the wave at v_par = (omega + Omega_c) / k and drive it unstable.
Then the magnetic energy is expected to grow exponentially and saturates by flattening f(v_par) at the vicinity of the resonance velocity.
"""

import numpy as np

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
)

# ---------------------
# Instance of the model
# ---------------------
from struphy.models import LinearMHDVlasovPC
from struphy.pic.accumulation.filter import FilterParameters

# units: n chosen such that the length unit 1 m = v_A/Omega_ci (time unit 1/Omega_ci)
base_units = BaseUnits(n=0.0005185219355)

# Model instance
model = LinearMHDVlasovPC(base_units=base_units)

# Variables to save
model.em_fields.b_field.save_data = True
model.mhd.density.save_data = True
model.mhd.velocity.save_data = True
model.mhd.pressure.save_data = True
model.energetic_ions.var.save_data = True

# --------------------------
# Instance of the simulation
# --------------------------

# Environment options
env = EnvironmentOptions(
    out_folders=args.out_folder,
    sim_folder=args.sim_folder,
    save_step=10,
    max_runtime=710,
    num_clones=args.num_clones,
)

epsilon = 1.0

# Time stepping
# --test-run: the initial, a middle and a final saved time step
dt = 0.32 * epsilon
time_opts = Time(dt=dt, Tend=2 * env.save_step * dt if args.test_run else 200 * epsilon, split_algo="Strang")

# Geometry
k = 0.8 / epsilon
domain = domains.Cuboid(r1=1.0 * epsilon, r2=1.0 * epsilon, r3=2 * np.pi / k)

# Fluid equilibrium
equil = equils.HomogenSlab()

# Grid
grid = grids.TensorProductGrid(
    num_elements=(2, 2, 16),
    mpi_dims_mask=(True, True, False),
)

# Derham options
derham_opts = DerhamOptions(
    degree=(1, 1, 3),
    nquads=(2, 2, 6),
    nquads_proj=(2, 2, 6),
)

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

# hot-ion Maxwellian, exp(-(v - u)^2 / (2 vth^2))
u_par = 2.5
vth = 1.0 / np.sqrt(2.0)

loading_params = LoadingParameters(
    ppc=100 if args.test_run else 4000,
    moments=(0.0, 0.0, u_par, vth, vth, vth),
    seed=1111,
)
weights_params = WeightsParameters(control_variate=False)
boundary_params = BoundaryParameters()
sorting_params = SortingParameters(boxes_per_dim=None)

# f(z, v_par) and f(v_par), saved every save_step (pproc: f(v_par) at t = 0 and saturation, and f(v_par, t))
binplot = BinningPlot(slice="e3_v3", n_bins=(128, 128), ranges=((0.0, 1.0), (0.0, 5.0)))
binplot_v3 = BinningPlot(slice="v3", n_bins=256, ranges=(0.0, 5.0))
saving_params = SavingParameters(binning_plots=(binplot, binplot_v3))

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

filter_params = FilterParameters(use_filter="fourier_in_tor")

model.propagators.push_eta_pc.options = model.propagators.push_eta_pc.Options(use_perp_model=True)
model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
model.propagators.pc6d.options = model.propagators.pc6d.Options(use_perp_model=True, filter_params=filter_params)
model.propagators.shear_alf.options = model.propagators.shear_alf.Options()
model.propagators.bg_current.options = model.propagators.bg_current.Options()
model.propagators.sound.options = model.propagators.sound.Options()
model.propagators.bg_pressure.options = model.propagators.bg_pressure.Options()

# ------------------
# Initial conditions
# ------------------

maxwellian = maxwellians.Maxwellian3D(
    n=(0.02, None),
    u3=(u_par, None),
    vth1=(vth, None),
    vth2=(vth, None),
    vth3=(vth, None),
)
model.energetic_ions.var.add_background(maxwellian)

if __name__ == "__main__":
    sim.run()
