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
parser.add_argument("--ppc", type=int, default=None, help="Markers per cell (overrides the default, 200).")
parser.add_argument("--Tend", type=float, default=None, help="End time (overrides the default, 0.01 -> 10 steps).")
parser.add_argument(
    "--solver",
    choices=("pcg",),
    default="pcg",
    help="Symmetric solver for the PoissonAdiabaticGyrokinetic field solve (default: pcg).",
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
    BinningPlot,
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
    # The physics case (examples/.../cyclone/params_cyclone.py) enables this, but it
    # wires up a *second*, completely separate solve every step
    # (ImplicitDiffusion.__call__'s `if self.diagnostic is not None: ... proj.solve(rhs)`,
    # a fresh L2Projector with the default "pcg" solver). It only feeds
    # `self.diagnostics.rho`, an extra saved diagnostic field that nothing else in this
    # model reads back (the `phi_integral` scalar uses `phi` directly) -- disabled here
    # so this profiling case measures the model's actual per-step cost.
    use_diagnostic_poisson=False,
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
time_opts = Time(dt=0.001, Tend=args.Tend if args.Tend is not None else 0.01, split_algo="LieTrotter")

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

ppc = args.ppc if args.ppc is not None else 200
loading_params = LoadingParameters(ppc=ppc, loading="sobol_standard", spatial="uniform", moments=(0, 0, 4, 4))
weights_params = WeightsParameters(control_variate=True)
boundary_params = BoundaryParameters(bc=("remove", "periodic", "periodic"))
sorting_params = SortingParameters(boxes_per_dim=(16, 16, 6), do_sort=True, sorting_frequency=5)

# density binning, needed for the e1_e2 density slice generated by the pproc block below
# (matches examples/DriftKineticElectrostaticAdiabatic/cyclone/params_cyclone.py's own setup)
eta_bin = BinningPlot(slice="e1_e2", n_bins=(64, 64), ranges=((0.01, 0.99), (0.0, 1.0)))
saving_params = SavingParameters(n_markers=100, binning_plots=(eta_bin,))

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
    sim.pproc(parallel_pproc=True, physical=True)

    # Static, non-interactive figures for this profiling run -- adapted from
    # examples/DriftKineticElectrostaticAdiabatic/cyclone/pproc_cyclone.py (which is
    # meant for local, interactive use: matplotlib Slider widgets, plt.show()) into
    # fixed-time-step, save-to-file plots, following the same results-directory
    # convention as profiling/examples/Poisson/cube_strong_scaling/params_poisson.py.
    if sim.rank == 0:
        import os

        import h5py
        import numpy as np
        from matplotlib import pyplot as plt

        sim.load_plotting_data()

        # `path_out` is the run's output folder; `sim_folder` alone is a bare name
        # resolved against the CWD. The profiling packaging picks these files up from
        # here and uploads them as `results-run<id>`.
        results_dir = os.path.join(sim.env.path_out, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Deliberately plain NumPy from here on, not xp/cunumpy, for data that really is
        # already host-side (h5py reads). spline_values/grids_phy/PlottingData attributes
        # are NOT host-only under ARRAY_BACKEND=cupy despite being "post-processed" --
        # they stay on-device, so those are pulled to host explicitly via xp.to_numpy()
        # below rather than np.asarray(), which CuPy refuses as an implicit conversion.

        # -------------------------------------------------------------------
        # phi_integral evolution + exponential growth-rate fit (the ITG-test
        # diagnostic scalar; see pproc_cyclone.py's plot_energy_fit).
        # -------------------------------------------------------------------
        data_path = os.path.join(sim.env.path_out, "data")
        with h5py.File(os.path.join(data_path, "data_proc0.hdf5"), "r") as f:
            t_scalar = np.asarray(f["time"]["value"][()])
            phi_integral = np.asarray(f["scalar"]["phi_integral"][()])

        fig_energy, ax_energy = plt.subplots()
        ax_energy.plot(t_scalar, phi_integral, label="phi_integral")
        ax_energy.set_xlabel("time")
        ax_energy.set_ylabel("phi_integral")
        ax_energy.set_title("Evolution of phi_integral")

        positive = np.isfinite(phi_integral) & (phi_integral > 0.0)
        gamma = None
        if int(np.count_nonzero(positive)) >= 2:
            idx = np.nonzero(positive)[0]
            i0, i1 = int(idx[0]), int(idx[-1]) + 1
            fit_time = t_scalar[i0:i1]
            fit_signal = np.log(np.sqrt(phi_integral[i0:i1]))
            gamma, b = np.polyfit(fit_time, fit_signal, 1)
            fit_curve = np.exp(2.0 * (gamma * fit_time + b))
            ax_energy.plot(fit_time, fit_curve, "--", label=f"fit: gamma={float(gamma):.4e}")
            print(f"phi_integral growth rate: gamma = {float(gamma):.8e}")
        ax_energy.legend()
        fig_energy.tight_layout()

        # -------------------------------------------------------------------
        # Electric potential phi, poloidal (R, Z) slice at the last saved time
        # step, toroidal index 0 (see pproc_cyclone.py's plot_field_slider).
        # -------------------------------------------------------------------
        # float(): dict keys of `.data` are plain Python floats, and t_grid may be a
        # 0-d CuPy array here (ARRAY_BACKEND=cupy), which is unhashable.
        Tend_saved = float(sim.t_grid[-1])
        # xp.to_numpy(), not np.asarray(): unlike the PlottingData read above,
        # spline_values/grids_phy stay on-device under ARRAY_BACKEND=cupy, and CuPy
        # arrays refuse implicit conversion via np.asarray().
        phi_phy = xp.to_numpy(sim.spline_values.em_fields.phi_phy.data[Tend_saved][0])
        X, Y, Z = (xp.to_numpy(g) for g in sim.grids_phy)
        R = np.sqrt(X**2 + Y**2)

        toroidal_index = 0
        fig_phi, ax_phi = plt.subplots()
        pcm_phi = ax_phi.pcolormesh(
            R[:, :, toroidal_index],
            Z[:, :, toroidal_index],
            phi_phy[:, :, toroidal_index],
            shading="auto",
        )
        fig_phi.colorbar(pcm_phi, ax=ax_phi)
        ax_phi.set_aspect("equal", adjustable="box")
        ax_phi.set_xlabel("R")
        ax_phi.set_ylabel("Z")
        ax_phi.set_title(f"Electric potential phi at t = {Tend_saved:.4e}")
        fig_phi.tight_layout()

        # -------------------------------------------------------------------
        # Density perturbation, e1-e2 binned, at the last saved time step, in
        # logical (eta) space (see pproc_cyclone.py's plot_binned_quantity_slider).
        # -------------------------------------------------------------------
        density_data = sim.f.kinetic_ions.e1_e2_density
        # xp.to_numpy(): same on-device-under-cupy issue as phi_phy/grids_phy above --
        # the "PlottingData is host-only regardless of backend" premise noted at the top
        # of this function does not hold for these attributes.
        delta_f_final = xp.to_numpy(density_data.delta_f_binned)[-1]
        eta1_grid, eta2_grid = np.meshgrid(
            xp.to_numpy(density_data.grid_e1),
            xp.to_numpy(density_data.grid_e2),
            indexing="ij",
        )

        fig_density, ax_density = plt.subplots()
        pcm_density = ax_density.pcolormesh(eta1_grid, eta2_grid, delta_f_final, shading="auto")
        fig_density.colorbar(pcm_density, ax=ax_density)
        ax_density.set_xlabel("eta1")
        ax_density.set_ylabel("eta2")
        ax_density.set_title(f"delta_f (eta1, eta2) at t = {Tend_saved:.4e}")
        fig_density.tight_layout()

        # -------------------------------------------------------------------
        # Save everything into results_dir, matching params_poisson.py's convention.
        # -------------------------------------------------------------------
        if gamma is not None:
            np.save(os.path.join(results_dir, "phi_integral_growth_rate.npy"), float(gamma))
        np.save(os.path.join(results_dir, "resolution.npy"), np.asarray(num_elements))
        np.save(os.path.join(results_dir, "spline_degree.npy"), np.asarray(degree))

        fig_energy.savefig(os.path.join(results_dir, "phi_integral_evolution.png"))
        fig_phi.savefig(os.path.join(results_dir, "phi_slice.png"))
        fig_density.savefig(os.path.join(results_dir, "density_e1e2.png"))
