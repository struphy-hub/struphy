from cunumpy import pi, cos, sin, zeros_like, ones_like
from struphy.io.options import EnvironmentOptions, BaseUnits, Time
from struphy.geometry import domains
from struphy.fields_background import equils
from struphy.topology import grids
from struphy.io.options import DerhamOptions
from struphy.initial import perturbations
from struphy.initial.base import GenericPerturbation
from struphy import Simulation
from struphy.linear_algebra.solver import SolverParameters

import argparse
import os
import glob
import cunumpy as xp
import matplotlib.pyplot as plt

from mpi4py import MPI

from struphy.models.two_fluid_quasi_neutral_toy import TwoFluidQuasiNeutralToy

# ------------------ args ------------------
parser = argparse.ArgumentParser()
parser.add_argument("bc", choices=["periodic", "dirichlet_hom", "dirichlet_inhom"])
args = parser.parse_args()
BC = args.bc

name = f"runs/sim_2D_{BC}"

# ------------------ setup ------------------
env = EnvironmentOptions(sim_folder=name)

B0 = 1
nu = 10.0
nu_e = 1.0
Nel = (8, 8, 1)
p = (2, 2, 1)
epsilon = 1.0
dt = 1
Tend = 1
sigma = 0
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
    # derham_opts = DerhamOptions(degree=p, bcs=(None, None, None))


elif BC == "dirichlet_inhom":
    derham_opts = DerhamOptions(degree=p, bcs=(("dirichlet", "dirichlet"), ("dirichlet", "dirichlet"), None))
    # derham_opts = DerhamOptions(degree=p, bcs=(None, None, None))

    lifting_function_u = [
        GenericPerturbation(lambda x, y, z: -xp.sin(2*pi*x)*xp.sin(2*pi*y), comp=0, given_in_basis="physical"),
        GenericPerturbation(lambda x, y, z: -xp.cos(2*pi*x)*xp.cos(2*pi*y), comp=1, given_in_basis="physical"),
    ]
    lifting_function_ue = [
        GenericPerturbation(lambda x, y, z: -xp.sin(4*pi*x)*xp.sin(4*pi*y), comp=0, given_in_basis="physical"),
        GenericPerturbation(lambda x, y, z: -xp.cos(4*pi*x)*xp.cos(4*pi*y), comp=1, given_in_basis="physical"),
    ]


# ------------------ manufactured solutions ------------------
if BC == "periodic":

    def mms_phi(x, y, z):
        return xp.cos(2 * pi * x) + xp.sin(2 * pi * y), xp.zeros_like(x), xp.zeros_like(x)

    def mms_ion_u(x, y, z):
        return -xp.sin(2 * pi * x) * xp.sin(2 * pi * y), -xp.cos(2 * pi * x) * xp.cos(2 * pi * y), xp.zeros_like(x)

    def mms_electron_u(x, y, z):
        return -xp.sin(4 * pi * x) * xp.sin(4 * pi * y), -xp.cos(4 * pi * x) * xp.cos(4 * pi * y), xp.zeros_like(x)



elif BC == "dirichlet_hom":

    def mms_phi(x, y, z):
        return xp.cos(2 * pi * x) + xp.sin(2 * pi * y), xp.zeros_like(x), xp.zeros_like(x)

    def mms_ion_u(x, y, z):
        return -xp.sin(2 * pi * x) * xp.cos(2 * pi * y), xp.cos(2 * pi * x) * xp.sin(2 * pi * y), xp.zeros_like(x)

    def mms_electron_u(x, y, z):
        return -xp.sin(4 * pi * x) * xp.cos(4 * pi * y), xp.cos(4 * pi * x) * xp.sin(4 * pi * y), xp.zeros_like(x)


elif BC == "dirichlet_inhom":

    def mms_phi(x, y, z):
        return xp.cos(2*pi*x) + xp.sin(2*pi*y), xp.zeros_like(x), xp.zeros_like(x)

    def mms_ion_u(x, y, z):
        return -xp.sin(2*pi*x)*xp.sin(2*pi*y), -xp.cos(2*pi*x)*xp.cos(2*pi*y), xp.zeros_like(x)

    def mms_electron_u(x, y, z):
        return -xp.sin(4*pi*x)*xp.sin(4*pi*y), -xp.cos(4*pi*x)*xp.cos(4*pi*y), xp.zeros_like(x)



# ------------------ source terms ------------------
if BC == "periodic":

    def source_function_u(x, y, z):
        fx = (
            -2 * pi * xp.sin(2 * pi * x)
            + B0 / epsilon * xp.cos(2 * pi * x) * xp.cos(2 * pi * y)
            - nu * 8 * pi**2 * xp.sin(2 * pi * x) * xp.sin(2 * pi * y)
        )
        fy = (
            2 * pi * xp.cos(2 * pi * y)
            - B0 / epsilon * xp.sin(2 * pi * x) * xp.sin(2 * pi * y)
            - nu * 8 * pi**2 * xp.cos(2 * pi * x) * xp.cos(2 * pi * y)
        )
        return fx, fy, zeros_like(x)

    def source_function_ue(x, y, z):
        fx = (
            2 * pi * xp.sin(2 * pi * x)
            - B0 / epsilon * xp.cos(4 * pi * x) * xp.cos(4 * pi * y)
            - nu_e * 32 * pi**2 * xp.sin(4 * pi * x) * xp.sin(4 * pi * y)
            + sigma * xp.sin(4 * pi * x) * xp.sin(4 * pi * y)
        )
        fy = (
            -2 * pi * xp.cos(2 * pi * y)
            + B0 / epsilon * xp.sin(4 * pi * x) * xp.sin(4 * pi * y)
            - nu_e * 32 * pi**2 * xp.cos(4 * pi * x) * xp.cos(4 * pi * y)
            + sigma * xp.cos(4 * pi * x) * xp.cos(4 * pi * y)
        )
        return fx, fy, zeros_like(x)


elif BC == "dirichlet_hom":

    def source_function_u(x, y, z):
        fx = (
            -2 * pi * xp.sin(2 * pi * x)
            - B0 / epsilon * xp.cos(2 * pi * x) * xp.sin(2 * pi * y)
            - nu * 8 * pi**2 * xp.sin(2 * pi * x) * xp.cos(2 * pi * y)
        )
        fy = (
            2 * pi * xp.cos(2 * pi * y)
            - B0 / epsilon * xp.sin(2 * pi * x) * xp.cos(2 * pi * y)
            + nu * 8 * pi**2 * xp.cos(2 * pi * x) * xp.sin(2 * pi * y)
        )
        return fx, fy, zeros_like(x)

    def source_function_ue(x, y, z):
        fx = (
            2 * pi * xp.sin(2 * pi * x)
            + B0 / epsilon * xp.cos(4 * pi * x) * xp.sin(4 * pi * y)
            - nu_e * 32 * pi**2 * xp.sin(4 * pi * x) * xp.cos(4 * pi * y)
            + sigma * xp.sin(4 * pi * x) * xp.cos(4 * pi * y)
        )
        fy = (
            -2 * pi * xp.cos(2 * pi * y)
            + B0 / epsilon * xp.sin(4 * pi * x) * xp.cos(4 * pi * y)
            + nu_e * 32 * pi**2 * xp.cos(4 * pi * x) * xp.sin(4 * pi * y)
            - sigma * xp.cos(4 * pi * x) * xp.sin(4 * pi * y)
        )
        return fx, fy, zeros_like(x)


elif BC == "dirichlet_inhom":
    def source_function_u(x, y, z):
        fx = (
            -2*pi*xp.sin(2*pi*x)
            + B0/epsilon * xp.cos(2*pi*x) * xp.cos(2*pi*y)
            - nu*8*pi**2 * xp.sin(2*pi*x) * xp.sin(2*pi*y)
        )
        fy = (
            2*pi*xp.cos(2*pi*y)
            - B0/epsilon * xp.sin(2*pi*x) * xp.sin(2*pi*y)
            - nu*8*pi**2 * xp.cos(2*pi*x) * xp.cos(2*pi*y)
        )
        return fx, fy, zeros_like(x)

    def source_function_ue(x, y, z):
        fx = (
            2*pi*xp.sin(2*pi*x)
            - B0/epsilon * xp.cos(4*pi*x) * xp.cos(4*pi*y)
            - nu_e*32*pi**2 * xp.sin(4*pi*x) * xp.sin(4*pi*y)
            + sigma * xp.sin(4*pi*x) * xp.sin(4*pi*y)
        )
        fy = (
            -2*pi*xp.cos(2*pi*y)
            + B0/epsilon * xp.sin(4*pi*x) * xp.sin(4*pi*y)
            - nu_e*32*pi**2 * xp.cos(4*pi*x) * xp.cos(4*pi*y)
            + sigma * xp.cos(4*pi*x) * xp.cos(4*pi*y)
        )
        return fx, fy, zeros_like(x)



class MMSIonVelocity(perturbations.Perturbation):
    def __init__(self, comp=0):
        self.comp = comp
        self.given_in_basis = "physical"

    def __call__(self, x, y, z):
        return mms_ion_u(x, y, z)[self.comp]


class MMSElectronVelocity(perturbations.Perturbation):
    def __init__(self, comp=0):
        self.comp = comp
        self.given_in_basis = "physical"

    def __call__(self, x, y, z):
        return mms_electron_u(x, y, z)[self.comp]


class MMSPotential(perturbations.Perturbation):
    def __init__(self):
        self.given_in_basis = "physical"

    def __call__(self, x, y, z):
        return mms_phi(x, y, z)[0]


# ------------------ model ------------------
model = TwoFluidQuasiNeutralToy()

model.propagators.qn_full.options = model.propagators.qn_full.Options(
    nu=nu,
    nu_e=nu_e,
    eps_norm=epsilon,
    stab_sigma=sigma,
    source_u=source_function_u,
    source_ue=source_function_ue,
    solver="uzawa",
    solver_params=SolverParameters(info=True, tol=tol),
)

if BC == "dirichlet_inhom":
    model.ions.u.lifting_function = lifting_function_u
    model.electrons.u.lifting_function = lifting_function_ue

# model.ions.u.add_perturbation(MMSIonVelocity(comp=0))
# model.ions.u.add_perturbation(MMSIonVelocity(comp=1))
# model.electrons.u.add_perturbation(MMSElectronVelocity(comp=0))
# model.electrons.u.add_perturbation(MMSElectronVelocity(comp=1))
# model.em_fields.phi.add_perturbation(MMSPotential())

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
        X, Y = xp.meshgrid(n1_vals, n2_vals, indexing="ij")

        x = xp.linspace(0, 1, 100)
        y = xp.linspace(0, 1, 100)
        Xf, Yf = xp.meshgrid(x, y, indexing="ij")

        os.makedirs(f"{name}/plots", exist_ok=True)
        for f in glob.glob(f"{name}/plots/*.png"):
            os.remove(f)

        def save_plot(numerical, analytical, title, fname, t):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            im0 = axes[0].contourf(X, Y, numerical, levels=50)
            axes[0].set_title("numerical")
            plt.colorbar(im0, ax=axes[0])
            im1 = axes[1].contourf(Xf, Yf, analytical, levels=50)
            axes[1].set_title("manufactured")
            plt.colorbar(im1, ax=axes[1])
            fig.suptitle(f"{title} at t={t:.3f}")
            plt.savefig(f"{name}/plots/{fname}_{t:.3f}.png", dpi=300)
            plt.close(fig)

        for t in simdata.spline_values.ions.u_log.data.keys():
            u_ions      = simdata.spline_values.ions.u_log.data[t]
            u_electrons = simdata.spline_values.electrons.u_log.data[t]
            phi         = simdata.spline_values.em_fields.phi_log.data[t]

            if BC == "dirichlet_inhom":
                e1 = xp.array(n1_vals)
                e2 = xp.array(n2_vals)
                e3 = xp.array([0.5])
                full_u  = model.ions.u.spline_full(e1, e2, e3, squeeze_out=True)
                full_ue = model.electrons.u.spline_full(e1, e2, e3, squeeze_out=True)
                phi_plot = phi[0][:, :, 0]
                uix_plot = full_u[0]
                uiy_plot = full_u[1]
                uex_plot = full_ue[0]
                uey_plot = full_ue[1]
            else:
                phi_plot = phi[0][:, :, 0]
                uix_plot = u_ions[0][:, :, 0]
                uiy_plot = u_ions[1][:, :, 0]
                uex_plot = u_electrons[0][:, :, 0]
                uey_plot = u_electrons[1][:, :, 0]

            mms_phi_x, _, _           = mms_phi(Xf, Yf, 0 * Xf)
            mms_ion_ux, mms_ion_uy, _ = mms_ion_u(Xf, Yf, 0 * Xf)
            mms_el_ux, mms_el_uy, _   = mms_electron_u(Xf, Yf, 0 * Xf)

            save_plot(phi_plot,  mms_phi_x,  "φ",    "plot_phi", t)
            save_plot(uix_plot,  mms_ion_ux, "u_ix", "plot_uix", t)
            save_plot(uiy_plot,  mms_ion_uy, "u_iy", "plot_uiy", t)
            save_plot(uex_plot,  mms_el_ux,  "u_ex", "plot_uex", t)
            save_plot(uey_plot,  mms_el_uy,  "u_ey", "plot_uey", t)

        # ---- lifting diagnostics (dirichlet_inhom only) ----
        if BC == "dirichlet_inhom":
            e1 = xp.linspace(0, 1, 80)
            e2 = xp.linspace(0, 1, 80)
            e3 = xp.array([0.5])
            E1, E2 = xp.meshgrid(e1, e2, indexing="ij")

            for label, var, comp in [
                ("ion_ux", model.ions.u, 0),
                ("ion_uy", model.ions.u, 1),
                ("electron_ux", model.electrons.u, 0),
                ("electron_uy", model.electrons.u, 1),
            ]:
                if var.spline_lift is None:
                    print(f"  {label}: spline_lift is None, skipping")
                    continue

                def _eval(fn):
                    return fn(e1, e2, e3, squeeze_out=True)[comp]

                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                for ax, fn, ttl in zip(
                    axes,
                    [var.spline_lift, var.spline_0, var.boundary_spline],
                    ["lifting", "zero-BC part", "boundary spline"],
                ):
                    im = ax.contourf(E1, E2, _eval(fn), levels=50)
                    ax.set_title(f"{label}: {ttl}")
                    plt.colorbar(im, ax=ax)
                out = f"{name}/plots/lifting_{label}.png"
                plt.savefig(out, dpi=300)
                plt.close(fig)
                print(f"  -> saved {out}")

        # ---- source diagnostics ----
        prop = model.propagators.qn_full
        e1 = xp.linspace(0, 1, 80)
        e2 = xp.linspace(0, 1, 80)
        e3 = xp.array([0.5])
        E1, E2 = xp.meshgrid(e1, e2, indexing="ij")
        zeros_E = xp.zeros_like(E1)

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

        if BC == "dirichlet_inhom":
            y_check = xp.linspace(0, 1, 80)
            x_check = xp.linspace(0, 1, 80)
            z_check = xp.array([0.5])

            for x_bnd, label in [(0.0, "x=0"), (1.0, "x=1")]:
                x_bnd_arr = xp.array([x_bnd])
                mms_vals  = mms_ion_u(x_bnd_arr, y_check, z_check)[0]
                lift_vals = model.ions.u.spline_lift(x_bnd_arr, y_check, z_check, squeeze_out=True)[0]
                print(f"ion ux normal trace diff at {label}: max={xp.max(xp.abs(mms_vals - lift_vals)):.3e}")

                mms_vals  = mms_electron_u(x_bnd_arr, y_check, z_check)[0]
                lift_vals = model.electrons.u.spline_lift(x_bnd_arr, y_check, z_check, squeeze_out=True)[0]
                print(f"elec ux normal trace diff at {label}: max={xp.max(xp.abs(mms_vals - lift_vals)):.3e}")

            for y_bnd, label in [(0.0, "y=0"), (1.0, "y=1")]:
                y_bnd_arr = xp.array([y_bnd])
                mms_vals  = mms_ion_u(x_check, y_bnd_arr, z_check)[1]
                lift_vals = model.ions.u.spline_lift(x_check, y_bnd_arr, z_check, squeeze_out=True)[1]
                print(f"ion uy normal trace diff at {label}: max={xp.max(xp.abs(mms_vals - lift_vals)):.3e}")

                mms_vals  = mms_electron_u(x_check, y_bnd_arr, z_check)[1]
                lift_vals = model.electrons.u.spline_lift(x_check, y_bnd_arr, z_check, squeeze_out=True)[1]
                print(f"elec uy normal trace diff at {label}: max={xp.max(xp.abs(mms_vals - lift_vals)):.3e}")
