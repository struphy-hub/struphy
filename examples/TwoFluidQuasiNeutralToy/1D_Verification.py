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
import logging
logging.getLogger("struphy").setLevel(logging.DEBUG)

import argparse
import os
import glob
import cunumpy as xp
import matplotlib.pyplot as plt

from struphy.models.two_fluid_quasi_neutral_toy import TwoFluidQuasiNeutralToy

parser = argparse.ArgumentParser()
parser.add_argument("bc", choices=["periodic", "dirichlet_hom", "dirichlet_inhom"])
args = parser.parse_args()
BC = args.bc

name = f"runs/sim_1D_{BC}"

env = EnvironmentOptions(sim_folder=name)

B0 = 0
nu = 10.0
nu_e = 1.0
Nel = (32, 1, 1)
p = (1, 1, 1)
epsilon = 1.0
dt = 1
Tend = 1
sigma = 0
tol = 1e-5

time_opts = Time(dt=dt, Tend=Tend)
domain = domains.Cuboid()
equil = equils.HomogenSlab(B0x=0, B0y=0, B0z=B0, beta=0, n0=0)
grid = grids.TensorProductGrid(num_elements=Nel)

# ---- boundary conditions ----
if BC == "periodic":
    derham_opts = DerhamOptions(degree=p, bcs=(None, None, None))

elif BC == "dirichlet_hom":
    derham_opts = DerhamOptions(degree=p, bcs=(("dirichlet", "dirichlet"), None, None))

elif BC == "dirichlet_inhom":
    derham_opts = DerhamOptions(degree=p, bcs=(("dirichlet", "dirichlet"), None, None))
    lifting_function_u = GenericPerturbation(lambda x, y, z: x + 1, comp=0, given_in_basis="physical")
    lifting_function_ue = GenericPerturbation(lambda x, y, z: x, comp=0, given_in_basis="physical")

# ---- manufactured solutions ----
if BC == "periodic":

    def mms_phi(x, y, z):
        return xp.sin(2 * xp.pi * x), xp.zeros_like(x), xp.zeros_like(x)

    def mms_ion_u(x, y, z):
        return xp.sin(2 * xp.pi * x) + 1, xp.zeros_like(x), xp.zeros_like(x)

    def mms_electron_u(x, y, z):
        return xp.sin(2 * xp.pi * x), xp.zeros_like(x), xp.zeros_like(x)

elif BC == "dirichlet_hom":

    def mms_phi(x, y, z):
        return xp.sin(2 * xp.pi * x), xp.zeros_like(x), xp.zeros_like(x)

    def mms_ion_u(x, y, z):
        return xp.sin(2 * xp.pi * x), xp.zeros_like(x), xp.zeros_like(x)

    def mms_electron_u(x, y, z):
        return xp.sin(2 * xp.pi * x), xp.zeros_like(x), xp.zeros_like(x)

elif BC == "dirichlet_inhom":

    def mms_phi(x, y, z):
        return xp.sin(2 * xp.pi * x), xp.zeros_like(x), xp.zeros_like(x)

    def mms_ion_u(x, y, z):
        return xp.sin(2 * xp.pi * x) + x + 1, xp.zeros_like(x), xp.zeros_like(x)

    def mms_electron_u(x, y, z):
        return xp.sin(2 * xp.pi * x) + x, xp.zeros_like(x), xp.zeros_like(x)


# ---- source terms ----
if BC == "periodic":

    def source_function_u(x, y, z):
        fx = 2.0 * pi * (cos(2 * pi * x) + 2 * nu * pi * sin(2 * pi * x))
        fy = zeros_like(x)
        fz = zeros_like(x)
        return fx, fy, fz

    def source_function_ue(x, y, z):
        fx = -2.0 * pi * cos(2 * pi * x) + nu_e * 4.0 * pi**2 * sin(2 * pi * x) - sigma * sin(2 * pi * x)
        fy = zeros_like(x)
        fz = zeros_like(x)
        return fx, fy, fz

elif BC == "dirichlet_hom":

    def source_function_u(x, y, z):
        fx = 2.0 * pi * (cos(2 * pi * x) + 2 * nu * pi * sin(2 * pi * x))
        fy = zeros_like(x)
        fz = zeros_like(x)
        return fx, fy, fz

    def source_function_ue(x, y, z):
        fx = -2.0 * pi * cos(2 * pi * x) + nu_e * 4.0 * pi**2 * sin(2 * pi * x) - sigma * sin(2 * pi * x)
        fy = zeros_like(x)
        fz = zeros_like(x)
        return fx, fy, fz

elif BC == "dirichlet_inhom":

    def source_function_u(x, y, z):
        fx = 2.0 * pi * (cos(2 * pi * x) + 2 * nu * pi * sin(2 * pi * x))
        fy = zeros_like(x)
        fz = zeros_like(x)
        return fx, fy, fz

    def source_function_ue(x, y, z):
        fx = -2.0 * pi * cos(2 * pi * x) + (4.0 * nu_e * pi**2 - sigma) * sin(2 * pi * x) - sigma * x
        fy = zeros_like(x)
        fz = zeros_like(x)
        return fx, fy, fz


# ---- perturbation classes for MMS initial conditions ----
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


# ---- model ----
model = TwoFluidQuasiNeutralToy()

model.propagators.qn_full.options = model.propagators.qn_full.Options(
    nu=nu,
    nu_e=nu_e,
    eps_norm=epsilon,
    stab_sigma=sigma,
    source_u=source_function_u,
    source_ue=source_function_ue,
    solver="gmres",
    solver_params=SolverParameters(info=True, tol=tol),
)

if BC == "dirichlet_inhom":
    model.ions.u.lifting_function = lifting_function_u
    model.electrons.u.lifting_function = lifting_function_ue

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

if __name__ == "__main__":
    sim.run()
    sim.pproc()
    sim.load_plotting_data()

    simdata = sim.plotting_data
    n1_vals = simdata.grids_log[0]
    x = xp.linspace(0, 1, 100)

    os.makedirs(f"{name}/plots", exist_ok=True)
    for f in glob.glob(f"{name}/plots/*.png"):
        os.remove(f)

    def save_plot(n1_vals, numerical, analytical, ylabel, title, fname, t):
        plt.plot(n1_vals, numerical, label="numerical")
        plt.plot(x, analytical, "--", label="manufactured")
        plt.plot(n1_vals, numerical, "k.", markersize=4, label="n1 points")
        plt.xlabel("x")
        plt.ylabel(ylabel)
        plt.title(f"{title} at t={t:.3f}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{name}/plots/{fname}_{t:.3f}.png", dpi=300)
        plt.clf()

    for t in list(simdata.spline_values.ions.u_log.data.keys()):
        u_ions      = simdata.spline_values.ions.u_log.data[t]
        u_electrons = simdata.spline_values.electrons.u_log.data[t]
        phi         = simdata.spline_values.em_fields.phi_log.data[t]

        u_ions_x      = u_ions[0][:, 0, 0]
        u_electrons_x = u_electrons[0][:, 0, 0]

        if BC == "dirichlet_inhom":
            e1 = xp.array(n1_vals)
            e2 = xp.array([0.5])
            e3 = xp.array([0.5])
            lift_u  = model.ions.u.boundary_spline(e1, e2, e3, squeeze_out=True)
            lift_ue = model.electrons.u.boundary_spline(e1, e2, e3, squeeze_out=True)
            u_ions_x      = u_ions_x      + lift_u[0]
            u_electrons_x = u_electrons_x + lift_ue[0]

            # ---- lifting diagnostics ----
            for label, zero_bc, lift in [
                ("ion",      u_ions[0][:, 0, 0],      lift_u[0]),
                ("electron", u_electrons[0][:, 0, 0], lift_ue[0]),
            ]:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].plot(n1_vals, zero_bc + lift)
                axes[0].set_title(f"{label}: postprocessed + lift (full)")
                axes[1].plot(n1_vals, zero_bc)
                axes[1].set_title(f"{label}: postprocessed (zero-BC)")
                axes[2].plot(n1_vals, lift)
                axes[2].set_title(f"{label}: lift")
                for ax in axes:
                    ax.set_xlabel("x")
                    ax.grid(True)
                plt.tight_layout()
                plt.savefig(f"{name}/plots/lifting_{label}_{t:.3f}.png", dpi=300)
                plt.clf()

        mms_phi_x, _, _ = mms_phi(x, x * 0, x * 0)
        mms_ion_ux, _, _ = mms_ion_u(x, x * 0, x * 0)
        mms_el_ux, _, _  = mms_electron_u(x, x * 0, x * 0)

        save_plot(n1_vals, phi[0][:, 0, 0], mms_phi_x,  "φ",   "Potential φ",       "plot_potential",   t)
        save_plot(n1_vals, u_ions_x,         mms_ion_ux, "u_x", "Ion velocity u_x",  "plot_ion_ux",      t)
        save_plot(n1_vals, u_electrons_x,    mms_el_ux,  "u_x", "Electron velocity", "plot_electron_ux", t)

    # ---- source diagnostics ----
    prop = model.propagators.qn_full
    e1 = xp.linspace(0, 1, 200)
    e2 = xp.array([0.5])
    e3 = xp.array([0.5])
    zeros_e = xp.zeros_like(e1)

    for label, spline, src_fn, comp in [
        ("ion_source_x",      prop._src_u,  prop.options.source_u,  0),
        ("electron_source_x", prop._src_ue, prop.options.source_ue, 0),
    ]:
        if spline is None:
            print(f"  {label}: None, skipping")
            continue
        vals_proj = spline(e1, e2, e3, squeeze_out=True)[comp]
        vals_ref  = src_fn(e1, zeros_e, zeros_e)[comp]
        plt.figure(figsize=(8, 4))
        plt.plot(e1, vals_ref,  "--", label="analytical")
        plt.plot(e1, vals_proj, "-",  label="projected (FE)")
        plt.xlabel("x")
        plt.title(f"{label}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{name}/plots/source_{label}.png", dpi=300)
        plt.close()
        print(f"  -> saved {name}/plots/source_{label}.png")