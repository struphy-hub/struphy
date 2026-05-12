import os
import glob
import cunumpy as xp
import matplotlib.pyplot as plt
from mpi4py import MPI

from struphy.io.options import EnvironmentOptions, Time
from struphy.geometry import domains
from struphy.fields_background import equils
from struphy.topology import grids
from struphy.io.options import DerhamOptions
from struphy.initial.base import GenericPerturbation
from struphy import Simulation
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.two_fluid_quasi_neutral_toy import TwoFluidQuasiNeutralToy

# ------------------ parameters ------------------
name = "runs/sim_restelli"

R0    = 2.0
a     = 1.0
ain   = 0.1
B0    = 10.0
Bp    = 12.5
nu    = 1.0
nu_e  = 0.01
alpha = 0.1
beta  = 1.0
eps   = 1.0
sigma = 1e-6
dt    = 1
Tend  = 1
Nel   = (8, 8, 1)
p     = (1, 1, 1)
tol   = 1e-5

env       = EnvironmentOptions(sim_folder=name)
time_opts = Time(dt=dt, Tend=Tend)

# ------------------ domain & equilibrium ------------------
domain = domains.HollowTorus(a1=ain, a2=a, R0=R0)
equil  = equils.CircularTokamak(a=a, R0=R0, B0=B0, Bp=Bp)
grid   = grids.TensorProductGrid(num_elements=Nel)

derham_opts = DerhamOptions(
    degree=p,
    bcs=(("dirichlet", "dirichlet"), None, None)
)

# ------------------ manufactured solution ------------------

def _cylindrical(x, y, z):
    R   = xp.sqrt(x**2 + y**2)
    R   = xp.where(R == 0.0, 1e-9, R)
    phi = xp.arctan2(-y, x)
    Z   = z
    return R, phi, Z

def mms_u_cartesian(x, y, z):
    R, phi, Z = _cylindrical(x, y, z)
    u_R   = alpha/R * (-Z) / (a*R0/R) + beta * Bp/B0 * R0/(a*R) * Z
    u_Z   = alpha/R * (R - R0) / (a*R0/R) + beta * Bp/B0 * R0/(a*R) * (-(R - R0))
    u_phi = beta * Bp/B0 * R0/(a*R) * B0*Bp*a / (Bp*R0) 

    u_R   = alpha * R / (a*R0) * (-Z)       + beta * Bp/B0 * R0/(a*R) * Z
    u_Z   = alpha * R / (a*R0) * (R - R0)   + beta * Bp/B0 * R0/(a*R) * (-(R - R0))
    u_phi = beta * Bp/B0 * R0/(a*R) * (B0/Bp * a)

    # Transform to Cartesian via eq (5.14)
    ux = xp.cos(phi) * u_R - R * xp.sin(phi) * u_phi
    uy = -xp.sin(phi) * u_R - R * xp.cos(phi) * u_phi
    uz = u_Z
    return ux, uy, uz

def mms_phi_cartesian(x, y, z):
    R, phi, Z = _cylindrical(x, y, z)
    # eq (5.22): phi_hat = 0.5 * a * B0 * alpha * ((R-R0)^2 + Z^2)/a^2 - 2/3)
    phi_val = 0.5 * a * B0 * alpha * (((R - R0)**2 + Z**2) / a**2 - 2.0/3.0)
    return phi_val

# ------------------ source terms ------------------

def _omega_cartesian(x, y, z):
    R, phi, Z = _cylindrical(x, y, z)
    omega_Z = alpha * (R0 - 4*R) / (a*R0*R) - beta * Bp/B0 * R0**2 / (a*R**3)
    ox = xp.zeros_like(x)
    oy = xp.zeros_like(x)
    oz = omega_Z
    return ox, oy, oz

def source_function_u(x, y, z):
    ox, oy, oz = _omega_cartesian(x, y, z)
    return nu * ox, nu * oy, nu * oz

def source_function_ue(x, y, z):
    ox, oy, oz = _omega_cartesian(x, y, z)
    return nu_e * ox, nu_e * oy, nu_e * oz

# ------------------ lifting (inhomogeneous Dirichlet on radial boundary) ------------------
lifting_u = [
    GenericPerturbation(lambda x, y, z: mms_u_cartesian(x, y, z)[0], comp=0, given_in_basis="physical"),
    GenericPerturbation(lambda x, y, z: mms_u_cartesian(x, y, z)[1], comp=1, given_in_basis="physical"),
    GenericPerturbation(lambda x, y, z: mms_u_cartesian(x, y, z)[2], comp=2, given_in_basis="physical"),
]

lifting_ue = [
    GenericPerturbation(lambda x, y, z: mms_u_cartesian(x, y, z)[0], comp=0, given_in_basis="physical"),
    GenericPerturbation(lambda x, y, z: mms_u_cartesian(x, y, z)[1], comp=1, given_in_basis="physical"),
    GenericPerturbation(lambda x, y, z: mms_u_cartesian(x, y, z)[2], comp=2, given_in_basis="physical"),
]

# ------------------ model ------------------
model = TwoFluidQuasiNeutralToy()

model.propagators.qn_full.options = model.propagators.qn_full.Options(
    nu=nu,
    nu_e=nu_e,
    eps_norm=eps,
    stab_sigma=sigma,
    source_u=source_function_u,
    source_ue=source_function_ue,
    solver='gmres',
    solver_params=SolverParameters(verbose=True, info=True, tol=tol),
)

model.ions.u.lifting_function      = lifting_u
model.electrons.u.lifting_function = lifting_ue

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
    # sim.run(verbose=True)

    if MPI.COMM_WORLD.Get_rank() == 0:
        sim.pproc(verbose=True)
        sim.load_plotting_data(verbose=True)

        simdata = sim.plotting_data
        os.makedirs(f'{name}/plots', exist_ok=True)
        for f in glob.glob(f'{name}/plots/*.png'):
            os.remove(f)

        e1 = xp.linspace(0, 1, 40)
        e2 = xp.linspace(0, 1, 40)
        e3 = xp.array([0.5])
        x_phys, y_phys, z_phys = domain(e1, e2, e3, squeeze_out=True)

        theta_bnd = xp.linspace(0, 1, 200)
        x_inner, y_inner, _ = domain(xp.array([0.0]), theta_bnd, xp.array([0.5]), squeeze_out=True)
        x_outer, y_outer, _ = domain(xp.array([1.0]), theta_bnd, xp.array([0.5]), squeeze_out=True)

        def _add_domain_boundary(ax):
            ax.plot(x_inner, y_inner, 'w-', linewidth=0.8)
            ax.plot(x_outer, y_outer, 'w-', linewidth=0.8)

        prop = model.propagators.qn_full

        for label, spline, src_fn, comp in [
            ('ion_source_x',      prop._src_u,  prop.options.source_u,  0),
            ('ion_source_y',      prop._src_u,  prop.options.source_u,  1),
            ('ion_source_z',      prop._src_u,  prop.options.source_u,  2),
            ('electron_source_x', prop._src_ue, prop.options.source_ue, 0),
            ('electron_source_y', prop._src_ue, prop.options.source_ue, 1),
            ('electron_source_z', prop._src_ue, prop.options.source_ue, 2),
        ]:
            if spline is None:
                print(f"  {label}: None, skipping")
                continue

            vals_proj = spline(e1, e2, e3, squeeze_out=True)[comp]
            vals_ref  = src_fn(x_phys, y_phys, z_phys)[comp]

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            im0 = axes[0].contourf(x_phys, y_phys, vals_proj, levels=50); axes[0].set_title('projected (FE)');        plt.colorbar(im0, ax=axes[0]); _add_domain_boundary(axes[0])
            im1 = axes[1].contourf(x_phys, y_phys, vals_ref,  levels=50); axes[1].set_title('reference (analytical)'); plt.colorbar(im1, ax=axes[1]); _add_domain_boundary(axes[1])
            fig.suptitle(label)
            out = f'{name}/plots/source_{label}.png'
            plt.savefig(out, dpi=300)
            plt.close(fig)
            print(f"  -> saved {out}")

        for t in simdata.spline_values.ions.u_log.data.keys():
            u_ions      = simdata.spline_values.ions.u_log.data[t]
            u_electrons = simdata.spline_values.electrons.u_log.data[t]
            phi_num     = simdata.spline_values.em_fields.phi_log.data[t]

            mms_ux, mms_uy, mms_uz = mms_u_cartesian(x_phys, y_phys, z_phys)
            mms_phi_val = mms_phi_cartesian(x_phys, y_phys, z_phys)

            for num, mms, lbl in [
                (u_ions[0][:, :, 0],      mms_ux,      'u_ix'),
                (u_ions[1][:, :, 0],      mms_uy,      'u_iy'),
                (u_ions[2][:, :, 0],      mms_uz,      'u_iz'),
                (u_electrons[0][:, :, 0], mms_ux,      'u_ex'),
                (u_electrons[1][:, :, 0], mms_uy,      'u_ey'),
                (u_electrons[2][:, :, 0], mms_uz,      'u_ez'),
                (phi_num[0][:, :, 0],     mms_phi_val, 'phi'),
            ]:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                im0 = axes[0].contourf(x_phys, y_phys, num,       levels=50); axes[0].set_title('numerical');  plt.colorbar(im0, ax=axes[0]); _add_domain_boundary(axes[0])
                im1 = axes[1].contourf(x_phys, y_phys, mms,       levels=50); axes[1].set_title('MMS');        plt.colorbar(im1, ax=axes[1]); _add_domain_boundary(axes[1])
                im2 = axes[2].contourf(x_phys, y_phys, num - mms, levels=50); axes[2].set_title('difference'); plt.colorbar(im2, ax=axes[2]); _add_domain_boundary(axes[2])
                fig.suptitle(f'{lbl} at t={t:.4f}')
                out = f'{name}/plots/{lbl}_{t:.4f}.png'
                plt.savefig(out, dpi=300)
                plt.close(fig)