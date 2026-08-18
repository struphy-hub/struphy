from numpy import pi, cos, sin, zeros_like, ones_like
from struphy.io.options import EnvironmentOptions, BaseUnits, Time
from struphy.geometry import domains
from struphy.fields_background import equils
from struphy.topology import grids
from struphy.io.options import DerhamOptions
from struphy.initial import perturbations
from struphy.initial.base import GenericPerturbation
from struphy import Simulation
from struphy.linear_algebra.solver import SolverParameters
import logging
logging.getLogger("struphy").setLevel(logging.DEBUG)

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI

from struphy.models.two_fluid_quasi_neutral_compressible import TwoFluidQuasiNeutral

# ------------------ args ------------------
parser = argparse.ArgumentParser()
parser.add_argument("bc", choices=[
    "periodic",
    "dirichlet_hom",
    "dirichlet_inhom_essential",
    "dirichlet_inhom_natural",
    "dirichlet_inhom_mixed",
    "poly",
])
args = parser.parse_args()
BC = args.bc

name = f"runs/sim_2D_hcurl_{BC}"

# ------------------ setup ------------------
env = EnvironmentOptions(sim_folder=name)

B0 = 0
nu = 10.0
nu_e = 1.0
mu = 1.0
Nel = (20, 20, 1)
p = (2, 2, 1)
epsilon = 1.0
dt = 1
Tend = 1
tol = 1e-5

time_opts = Time(dt=dt, Tend=Tend)
domain = domains.Cuboid()
equil = equils.HomogenSlab(B0x=0, B0y=0, B0z=B0, beta=0, n0=0)
grid = grids.TensorProductGrid(num_elements=Nel)

# ------------------ boundary conditions ------------------

if BC == "periodic":
    derham_opts = DerhamOptions(degree=p, bcs=(None, None, None))

elif BC == "dirichlet_hom":
    derham_opts = DerhamOptions(degree=p, bcs=(("dirichlet", "dirichlet"), ("dirichlet", "dirichlet"), None))

elif BC == "dirichlet_inhom_essential":
    derham_opts = DerhamOptions(degree=p, bcs=(("dirichlet", "dirichlet"), ("dirichlet", "dirichlet"), None))

    # tangential lifting: g_i x n enforced strongly on H(curl)
    lifting_function_u = [
        GenericPerturbation(lambda x, y, z: -np.sin(2*pi*x)*np.sin(2*pi*y), comp=0, given_in_basis="physical"),
        GenericPerturbation(lambda x, y, z: -np.sin(2*pi*x)*np.cos(2*pi*y), comp=1, given_in_basis="physical"),
    ]
    lifting_function_ue = [
        GenericPerturbation(lambda x, y, z: -np.sin(2*pi*x)*np.sin(2*pi*y), comp=0, given_in_basis="physical"),
        GenericPerturbation(lambda x, y, z: -np.sin(2*pi*x)*np.cos(2*pi*y), comp=1, given_in_basis="physical"),
    ]

elif BC == "dirichlet_inhom_mixed":
    derham_opts = DerhamOptions(degree=p, bcs=(("dirichlet", "dirichlet"), ("dirichlet", "dirichlet"), None))

    lifting_function_u = [
        GenericPerturbation(lambda x, y, z: -np.sin(2*pi*x)*np.sin(2*pi*y), comp=0, given_in_basis="physical"),
        GenericPerturbation(lambda x, y, z: -np.cos(2*pi*x)*np.cos(2*pi*y), comp=1, given_in_basis="physical"),
    ]
    lifting_function_ue = [
        GenericPerturbation(lambda x, y, z: -np.sin(4*pi*x)*np.sin(4*pi*y), comp=0, given_in_basis="physical"),
        GenericPerturbation(lambda x, y, z: -np.cos(4*pi*x)*np.cos(4*pi*y), comp=1, given_in_basis="physical"),
    ]

elif BC == "poly":
    derham_opts = DerhamOptions(degree=p, bcs=(("dirichlet", "dirichlet"), ("dirichlet", "dirichlet"), None))

    lifting_function_u = [
        GenericPerturbation(lambda x, y, z: x**2 * y, comp=0, given_in_basis="physical"),
        GenericPerturbation(lambda x, y, z: -x * y**2, comp=1, given_in_basis="physical"),
    ]
    lifting_function_ue = [
        GenericPerturbation(lambda x, y, z: x**2 * y, comp=0, given_in_basis="physical"),
        GenericPerturbation(lambda x, y, z: -x * y**2, comp=1, given_in_basis="physical"),
    ]

elif BC == "dirichlet_inhom_natural":
    derham_opts = DerhamOptions(degree=p, bcs=(("dirichlet", "dirichlet"), ("dirichlet", "dirichlet"), None))

    # tangential lifting for the essential part
    lifting_function_u = [
        GenericPerturbation(lambda x, y, z: -np.sin(2*pi*x)*np.sin(2*pi*y), comp=0, given_in_basis="physical"),
        GenericPerturbation(lambda x, y, z: -np.cos(2*pi*x)*np.sin(2*pi*y), comp=1, given_in_basis="physical"),
    ]
    lifting_function_ue = [
        GenericPerturbation(lambda x, y, z: -np.sin(2*pi*x)*np.sin(2*pi*y), comp=0, given_in_basis="physical"),
        GenericPerturbation(lambda x, y, z: -np.cos(2*pi*x)*np.sin(2*pi*y), comp=1, given_in_basis="physical"),
    ]

# defaults for BCs without natural lifting
if BC in ("periodic", "dirichlet_hom"):
    lifting_function_u = None
    lifting_function_ue = None

# ------------------ manufactured solutions ------------------
if BC == "periodic":
    def mms_phi(x, y, z):
        return np.cos(2*pi*x) + np.sin(2*pi*y), np.zeros_like(x), np.zeros_like(x)

    def mms_ion_u(x, y, z):
        return -np.sin(2*pi*x)*np.sin(2*pi*y), -np.sin(2*pi*x)*np.sin(2*pi*y), np.zeros_like(x)

    def mms_electron_u(x, y, z):
        return -np.sin(2*pi*x)*np.sin(2*pi*y), -np.sin(2*pi*x)*np.sin(2*pi*y), np.zeros_like(x)

elif BC == "dirichlet_hom":
    def mms_phi(x, y, z):
        return np.cos(2*pi*x) + np.sin(2*pi*y), np.zeros_like(x), np.zeros_like(x)

    def mms_ion_u(x, y, z):
        return -np.sin(2*pi*x)*np.sin(2*pi*y), -np.sin(2*pi*x)*np.sin(2*pi*y), np.zeros_like(x)

    def mms_electron_u(x, y, z):
        return -np.sin(2*pi*x)*np.sin(2*pi*y), -np.sin(2*pi*x)*np.sin(2*pi*y), np.zeros_like(x)

elif BC == "dirichlet_inhom_essential":
    def mms_phi(x, y, z):
        return np.cos(2*pi*x) + np.sin(2*pi*y), np.zeros_like(x), np.zeros_like(x)

    def mms_ion_u(x, y, z):
        return -np.sin(2*pi*x)*np.sin(2*pi*y), -np.sin(2*pi*x)*np.cos(2*pi*y), np.zeros_like(x)

    def mms_electron_u(x, y, z):
        return -np.sin(2*pi*x)*np.sin(2*pi*y), -np.sin(2*pi*x)*np.cos(2*pi*y), np.zeros_like(x)

elif BC == "dirichlet_inhom_mixed":
    def mms_phi(x, y, z):
        return np.cos(2*pi*x) + np.sin(2*pi*y), np.zeros_like(x), np.zeros_like(x)

    def mms_ion_u(x, y, z):
        return -np.sin(2*pi*x)*np.sin(2*pi*y), -np.cos(2*pi*x)*np.cos(2*pi*y), np.zeros_like(x)

    def mms_electron_u(x, y, z):
        return -np.sin(4*pi*x)*np.sin(4*pi*y), -np.cos(4*pi*x)*np.cos(4*pi*y), np.zeros_like(x)

elif BC == "poly":
    def mms_phi(x, y, z):
        return x**2 + y**2, np.zeros_like(x), np.zeros_like(x)

    def mms_ion_u(x, y, z):
        return x**2 * y, -x * y**2, np.zeros_like(x)

    def mms_electron_u(x, y, z):
        return x**2 * y, -x * y**2, np.zeros_like(x)

elif BC == "dirichlet_inhom_natural":
    def mms_phi(x, y, z):
        return np.cos(2*pi*x) + np.sin(2*pi*y), np.zeros_like(x), np.zeros_like(x)

    def mms_ion_u(x, y, z):
        return -np.sin(2*pi*x)*np.sin(2*pi*y), -np.cos(2*pi*x)*np.sin(2*pi*y), np.zeros_like(x)

    def mms_electron_u(x, y, z):
        return -np.sin(2*pi*x)*np.sin(2*pi*y), -np.cos(2*pi*x)*np.sin(2*pi*y), np.zeros_like(x)

# ------------------ source terms ------------------

if BC == "periodic":
    def source_function_u(x, y, z):
        fx = (
            -2*pi*np.sin(2*pi*x)
            - B0/epsilon * np.sin(2*pi*x)*np.sin(2*pi*y)
            - nu*8*pi**2 * np.sin(2*pi*x)*np.sin(2*pi*y)
        )
        fy = (
            2*pi*np.cos(2*pi*y)
            + B0/epsilon * np.sin(2*pi*x)*np.sin(2*pi*y)
            - nu*8*pi**2 * np.sin(2*pi*x)*np.sin(2*pi*y)
        )
        return fx, fy, zeros_like(x)

    def source_function_ue(x, y, z):
        fx = (
            2*pi*np.sin(2*pi*x)
            - B0/epsilon * np.sin(2*pi*x)*np.sin(2*pi*y)
            - nu_e*8*pi**2 * np.sin(2*pi*x)*np.sin(2*pi*y)
        )
        fy = (
            -2*pi*np.cos(2*pi*y)
            - B0/epsilon * np.sin(2*pi*x)*np.sin(2*pi*y)
            - nu_e*8*pi**2 * np.sin(2*pi*x)*np.sin(2*pi*y)
        )
        return fx, fy, zeros_like(x)

elif BC == "dirichlet_hom":
    def source_function_u(x, y, z):
        fx = (
            -2*pi*np.sin(2*pi*x)
            - B0/epsilon * np.sin(2*pi*x)*np.sin(2*pi*y)
            - nu*8*pi**2 * np.sin(2*pi*x)*np.sin(2*pi*y)
        )
        fy = (
            2*pi*np.cos(2*pi*y)
            + B0/epsilon * np.sin(2*pi*x)*np.sin(2*pi*y)
            - nu*8*pi**2 * np.sin(2*pi*x)*np.sin(2*pi*y)
        )
        return fx, fy, zeros_like(x)

    def source_function_ue(x, y, z):
        fx = (
            2*pi*np.sin(2*pi*x)
            - B0/epsilon * np.sin(2*pi*x)*np.sin(2*pi*y)
            - nu_e*8*pi**2 * np.sin(2*pi*x)*np.sin(2*pi*y)
        )
        fy = (
            -2*pi*np.cos(2*pi*y)
            - B0/epsilon * np.sin(2*pi*x)*np.sin(2*pi*y)
            - nu_e*8*pi**2 * np.sin(2*pi*x)*np.sin(2*pi*y)
        )
        return fx, fy, zeros_like(x)
    
elif BC == "dirichlet_inhom_essential":
    def source_function_u(x, y, z):
        fx = (
            -2*pi*np.sin(2*pi*x)
            - B0/epsilon * np.sin(2*pi*x)*np.cos(2*pi*y)
            - nu*8*pi**2 * np.sin(2*pi*x)*np.sin(2*pi*y)
        )
        fy = (
            2*pi*np.cos(2*pi*y)
            + B0/epsilon * np.sin(2*pi*x)*np.sin(2*pi*y)
            - nu*8*pi**2 * np.sin(2*pi*x)*np.cos(2*pi*y)
        )
        return fx, fy, zeros_like(x)

    def source_function_ue(x, y, z):
        fx = (
            2*pi*np.sin(2*pi*x)
            + B0/epsilon * np.sin(2*pi*x)*np.cos(2*pi*y)
            - nu_e*8*pi**2 * np.sin(2*pi*x)*np.sin(2*pi*y)
        )
        fy = (
            -2*pi*np.cos(2*pi*y)
            - B0/epsilon * np.sin(2*pi*x)*np.sin(2*pi*y)
            - nu_e*8*pi**2 * np.sin(2*pi*x)*np.cos(2*pi*y)
        )
        return fx, fy, zeros_like(x)

elif BC == "dirichlet_inhom_mixed":
    def source_function_u(x, y, z):
        fx = (
            -2*pi*np.sin(2*pi*x)
            + B0/epsilon * np.cos(2*pi*x)*np.cos(2*pi*y)
            - nu*8*pi**2 * np.sin(2*pi*x)*np.sin(2*pi*y)
        )
        fy = (
            2*pi*np.cos(2*pi*y)
            - B0/epsilon * np.sin(2*pi*x)*np.sin(2*pi*y)
            - nu*8*pi**2 * np.cos(2*pi*x)*np.cos(2*pi*y)
        )
        return fx, fy, zeros_like(x)

    def source_function_ue(x, y, z):
        fx = (
            2*pi*np.sin(2*pi*x)
            - B0/epsilon * np.cos(4*pi*x)*np.cos(4*pi*y)
            - nu_e*32*pi**2 * np.sin(4*pi*x)*np.sin(4*pi*y)
        )
        fy = (
            -2*pi*np.cos(2*pi*y)
            + B0/epsilon * np.sin(4*pi*x)*np.sin(4*pi*y)
            - nu_e*32*pi**2 * np.cos(4*pi*x)*np.cos(4*pi*y)
        )
        return fx, fy, zeros_like(x)

elif BC == "poly":
    def source_function_u(x, y, z):
        fx = 2*x + B0/epsilon * x*y**2 + nu*2*y
        fy = 2*y - B0/epsilon * x**2*y - nu*2*x
        return fx, fy, zeros_like(x)

    def source_function_ue(x, y, z):
        fx = -2*x + B0/epsilon * x*y**2 + nu_e*2*y
        fy = -2*y - B0/epsilon * x**2*y - nu_e*2*x
        return fx, fy, zeros_like(x)

elif BC == "dirichlet_inhom_natural":
    def source_function_u(x, y, z):
        fx = (
            -2*pi*np.sin(2*pi*x)
            - B0/epsilon * np.cos(2*pi*x)*np.sin(2*pi*y)
            - nu*8*pi**2 * np.sin(2*pi*x)*np.sin(2*pi*y)
        )
        fy = (
            2*pi*np.cos(2*pi*y)
            + B0/epsilon * np.sin(2*pi*x)*np.sin(2*pi*y)
            - nu*8*pi**2 * np.cos(2*pi*x)*np.sin(2*pi*y)
        )
        return fx, fy, zeros_like(x)

    def source_function_ue(x, y, z):
        fx = (
            2*pi*np.sin(2*pi*x)
            + B0/epsilon * np.cos(2*pi*x)*np.sin(2*pi*y)
            - nu_e*8*pi**2 * np.sin(2*pi*x)*np.sin(2*pi*y)
        )
        fy = (
            -2*pi*np.cos(2*pi*y)
            - B0/epsilon * np.sin(2*pi*x)*np.sin(2*pi*y)
            - nu_e*8*pi**2 * np.cos(2*pi*x)*np.sin(2*pi*y)
        )
        return fx, fy, zeros_like(x)


# ------------------ model ------------------
model = TwoFluidQuasiNeutral()

model.propagators.qn_comp.options = model.propagators.qn_comp.Options(
    nu=nu,
    nu_e=nu_e,
    mu=mu,
    eps_norm=epsilon,
    source_u=source_function_u,
    source_ue=source_function_ue,
    essential_u=lifting_function_u,
    essential_ue=lifting_function_ue,
    solver="gmres",
    solver_params=SolverParameters(info=True, tol=tol),
)

if BC in ("dirichlet_inhom_essential", "dirichlet_inhom_mixed", "dirichlet_inhom_natural", "poly"):
    model.ions.u.lifting_function = lifting_function_u
    model.electrons.u.lifting_function = lifting_function_ue

# ------------------ simulation ------------------
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

# ------------------ run ------------------
if __name__ == "__main__":
    sim.run()

    if MPI.COMM_WORLD.Get_rank() == 0:
        sim.pproc()
        sim.load_plotting_data()

        simdata = sim.plotting_data

        n1_vals = simdata.grids_log[0]
        n2_vals = simdata.grids_log[1]
        X, Y = np.meshgrid(n1_vals, n2_vals, indexing="ij")

        os.makedirs(f"{name}/plots", exist_ok=True)
        for f in glob.glob(f"{name}/plots/*.png"):
            os.remove(f)

        def save_plot(numerical, analytical_fn, title, fname, t):
            analytical = analytical_fn(X, Y, 0 * X)
            diff = numerical - analytical
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            im0 = axes[0].contourf(X, Y, numerical, levels=50)
            axes[0].set_title("numerical")
            plt.colorbar(im0, ax=axes[0])
            im1 = axes[1].contourf(X, Y, analytical, levels=50)
            axes[1].set_title("manufactured")
            plt.colorbar(im1, ax=axes[1])
            im2 = axes[2].contourf(X, Y, diff, levels=50)
            axes[2].set_title("difference")
            plt.colorbar(im2, ax=axes[2])
            fig.suptitle(f"{title} at t={t:.3f}")
            plt.savefig(f"{name}/plots/{fname}_{t:.3f}.png", dpi=300)
            plt.close(fig)

        for t in simdata.spline_values.ions.u_log.data.keys():
            u_ions      = simdata.spline_values.ions.u_log.data[t]
            u_electrons = simdata.spline_values.electrons.u_log.data[t]
            phi         = simdata.spline_values.em_fields.phi_log.data[t]

            phi_plot = phi[0][:, :, 0]
            uix_plot = u_ions[0][:, :, 0]
            uiy_plot = u_ions[1][:, :, 0]
            uex_plot = u_electrons[0][:, :, 0]
            uey_plot = u_electrons[1][:, :, 0]

            if BC in ("dirichlet_inhom_essential", "dirichlet_inhom_mixed", "dirichlet_inhom_natural", "poly"):
                e1 = np.array(n1_vals)
                e2 = np.array(n2_vals)
                e3 = np.array([0.5])
                lift_u  = model.ions.u.spline_lift(e1, e2, e3, squeeze_out=True)
                lift_ue = model.electrons.u.spline_lift(e1, e2, e3, squeeze_out=True)
                uix_plot = uix_plot + lift_u[0]
                uiy_plot = uiy_plot + lift_u[1]
                uex_plot = uex_plot + lift_ue[0]
                uey_plot = uey_plot + lift_ue[1]

                for label, zero_bc, lift, comp in [
                    ("ion_ux",      u_ions[0][:, :, 0],      lift_u[0],  0),
                    ("ion_uy",      u_ions[1][:, :, 0],      lift_u[1],  1),
                    ("electron_ux", u_electrons[0][:, :, 0], lift_ue[0], 0),
                    ("electron_uy", u_electrons[1][:, :, 0], lift_ue[1], 1),
                ]:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                    for ax, data, ttl in zip(
                        axes,
                        [zero_bc + lift, zero_bc, lift],
                        ["postprocessed + lift (full)", "postprocessed (zero-BC)", "lift"],
                    ):
                        im = ax.contourf(X, Y, data, levels=50)
                        ax.set_title(f"{label}: {ttl}")
                        plt.colorbar(im, ax=ax)
                    out = f"{name}/plots/lifting_{label}_{t:.3f}.png"
                    plt.savefig(out, dpi=300)
                    plt.close(fig)
                    print(f"  -> saved {out}")

            save_plot(phi_plot,  lambda x, y, z: mms_phi(x, y, z)[0],         "φ",    "plot_phi", t)
            save_plot(uix_plot,  lambda x, y, z: mms_ion_u(x, y, z)[0],       "u_ix", "plot_uix", t)
            save_plot(uiy_plot,  lambda x, y, z: mms_ion_u(x, y, z)[1],       "u_iy", "plot_uiy", t)
            save_plot(uex_plot,  lambda x, y, z: mms_electron_u(x, y, z)[0],  "u_ex", "plot_uex", t)
            save_plot(uey_plot,  lambda x, y, z: mms_electron_u(x, y, z)[1],  "u_ey", "plot_uey", t)

        # ---- source diagnostics ----
        prop = model.propagators.qn_comp
        e1 = np.linspace(0, 1, 80)
        e2 = np.linspace(0, 1, 80)
        e3 = np.array([0.5])
        E1, E2 = np.meshgrid(e1, e2, indexing="ij")
        zeros_E = np.zeros_like(E1)

        for label, spline, src_fn, comp in [
            ("ion_source_x",      prop._src_u,  prop.options.source_u,  0),
            ("ion_source_y",      prop._src_u,  prop.options.source_u,  1),
            ("electron_source_x", prop._src_ue, prop.options.source_ue, 0),
            ("electron_source_y", prop._src_ue, prop.options.source_ue, 1),
        ]:
            if spline is None:
                print(f"  {label}: None, skipping")
                continue

            vals_proj = spline(e1, e2, e3, squeeze_out=True)[comp]
            vals_ref  = src_fn(E1, E2, zeros_E)[comp]

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            im0 = axes[0].contourf(E1, E2, vals_proj, levels=50)
            axes[0].set_title("projected (FE)")
            plt.colorbar(im0, ax=axes[0])
            im1 = axes[1].contourf(E1, E2, vals_ref, levels=50)
            axes[1].set_title("reference (analytical)")
            plt.colorbar(im1, ax=axes[1])
            fig.suptitle(label)
            out = f"{name}/plots/source_{label}.png"
            plt.savefig(out, dpi=300)
            plt.close(fig)
            print(f"  -> saved {out}")

        if BC in ("dirichlet_inhom_essential", "dirichlet_inhom_mixed", "dirichlet_inhom_natural", "poly"):
            y_check = np.linspace(0, 1, 80)
            x_check = np.linspace(0, 1, 80)
            z_check = np.array([0.5])

            for x_bnd, label in [(0.0, "x=0"), (1.0, "x=1")]:
                x_bnd_arr = np.array([x_bnd])
                mms_vals  = mms_ion_u(x_bnd_arr, y_check, z_check)[0]
                lift_vals = model.ions.u.boundary_spline(x_bnd_arr, y_check, z_check, squeeze_out=True)[0]
                print(f"ion ux tangential trace diff at {label}: max={np.max(np.abs(mms_vals - lift_vals)):.3e}")

                mms_vals  = mms_electron_u(x_bnd_arr, y_check, z_check)[0]
                lift_vals = model.electrons.u.boundary_spline(x_bnd_arr, y_check, z_check, squeeze_out=True)[0]
                print(f"elec ux tangential trace diff at {label}: max={np.max(np.abs(mms_vals - lift_vals)):.3e}")

            for y_bnd, label in [(0.0, "y=0"), (1.0, "y=1")]:
                y_bnd_arr = np.array([y_bnd])
                mms_vals  = mms_ion_u(x_check, y_bnd_arr, z_check)[1]
                lift_vals = model.ions.u.boundary_spline(x_check, y_bnd_arr, z_check, squeeze_out=True)[1]
                print(f"ion uy tangential trace diff at {label}: max={np.max(np.abs(mms_vals - lift_vals)):.3e}")

                mms_vals  = mms_electron_u(x_check, y_bnd_arr, z_check)[1]
                lift_vals = model.electrons.u.boundary_spline(x_check, y_bnd_arr, z_check, squeeze_out=True)[1]
                print(f"elec uy tangential trace diff at {label}: max={np.max(np.abs(mms_vals - lift_vals)):.3e}")