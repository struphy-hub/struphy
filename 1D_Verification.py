from cunumpy import pi, cos, sin, zeros_like, ones_like
from struphy.io.options import EnvironmentOptions, BaseUnits, Time
from struphy.geometry import domains
from struphy.fields_background import equils
from struphy.topology import grids
from struphy.io.options import DerhamOptions
from struphy.initial import perturbations
from struphy import main

import os
import glob
import cunumpy as xp
import matplotlib.pyplot as plt

from struphy.models.rework_model import TwoFluidQuasiNeutralToy

import warnings
# warnings.filterwarnings("error")


BC = 'dirichlet_hom'  # 'periodic' | 'dirichlet_hom' | 'dirichlet_inhom'

name = f"runs/sim_1D_{BC}"

env        = EnvironmentOptions(sim_folder=name)
base_units = BaseUnits(kBT=1.0)

B0      = 1.0
nu      = 10.0
nu_e    = 1.0
Nel     = (32, 1, 1)
p       = (2, 1, 1)
epsilon = 1.0
dt      = 1
Tend    = 1
sigma   = 1

time_opts = Time(dt=dt, Tend=Tend)
domain    = domains.Cuboid()
equil     = equils.HomogenSlab(B0x=0, B0y=0, B0z=B0, beta=0, n0=0)
grid      = grids.TensorProductGrid(Nel=Nel)

# ---- boundary conditions ----
if BC == 'periodic':
    spl_kind     = (True, True, True)
    dirichlet_bc = ((False, False), (False, False), (False, False))

    bcs_u = bcs_ue = {
        (0, -1): "periodic", (0, 1): "periodic",
        (1, -1): "periodic", (1, 1): "periodic",
        (2, -1): "periodic", (2, 1): "periodic",
    }
    boundary_data_u = boundary_data_ue = None

elif BC == 'dirichlet_hom':
    spl_kind     = (False, True, True)
    dirichlet_bc = ((True, True), (False, False), (False, False))

    bcs_u = bcs_ue = {
        (0, -1): "dirichlet", (0, 1): "dirichlet",
        (1, -1): "periodic",  (1, 1): "periodic",
        (2, -1): "periodic",  (2, 1): "periodic",
    }
    boundary_data_u = boundary_data_ue = None

elif BC == 'dirichlet_inhom':
    spl_kind     = (False, True, True)
    dirichlet_bc = ((False, False), (False, False), (False, False))

    bcs_u = bcs_ue = {
        (0, -1): "dirichlet", (0, 1): "dirichlet",
        (1, -1): "periodic",  (1, 1): "periodic",
        (2, -1): "periodic",  (2, 1): "periodic",
    }
    boundary_data_u = {
        (0, -1): lambda x, y, z: (zeros_like(x) + 1, zeros_like(x), zeros_like(x)),
        (0,  1): lambda x, y, z: (zeros_like(x) + 2, zeros_like(x), zeros_like(x)),
    }
    boundary_data_ue = {
        (0, -1): lambda x, y, z: (zeros_like(x) + 1, zeros_like(x), zeros_like(x)),
        (0,  1): lambda x, y, z: (zeros_like(x) + 2, zeros_like(x), zeros_like(x)),
    }

derham_opts = DerhamOptions(
    p=p,
    spl_kind=spl_kind,
    dirichlet_bc=dirichlet_bc,
)

# ---- manufactured solutions ----
if BC == 'periodic':
    def mms_phi(x, y, z):
        return xp.sin(2 * xp.pi * x), xp.zeros_like(x), xp.zeros_like(x)

    def mms_ion_u(x, y, z):
        return xp.sin(2 * xp.pi * x) + 1, xp.zeros_like(x), xp.zeros_like(x)

    def mms_electron_u(x, y, z):
        return xp.sin(2 * xp.pi * x), xp.zeros_like(x), xp.zeros_like(x)

elif BC == 'dirichlet_hom':
    def mms_phi(x, y, z):
        return xp.sin(2 * xp.pi * x), xp.zeros_like(x), xp.zeros_like(x)

    def mms_ion_u(x, y, z):
        return xp.sin(2 * xp.pi * x), xp.zeros_like(x), xp.zeros_like(x)

    def mms_electron_u(x, y, z):
        return xp.sin(2 * xp.pi * x), xp.zeros_like(x), xp.zeros_like(x)

elif BC == 'dirichlet_inhom':
    def mms_phi(x, y, z):
        return x + 1, xp.zeros_like(x), xp.zeros_like(x)

    def mms_ion_u(x, y, z):
        return x + 1, xp.zeros_like(x), xp.zeros_like(x)

    def mms_electron_u(x, y, z):
        return x + 1, xp.zeros_like(x), xp.zeros_like(x)

# ---- source terms ----
if BC == 'periodic':
    def source_function_u(x, y, z):
        fx = 2.0 * pi * cos(2 * pi * x) + nu * 4.0 * pi**2 * sin(2 * pi * x)
        fy = (sin(2 * pi * x) + 1.0) * B0 / epsilon
        fz = zeros_like(x)
        return fx, fy, fz

    def source_function_ue(x, y, z):
        fx = -2.0 * pi * cos(2 * pi * x) + nu_e * 4.0 * pi**2 * sin(2 * pi * x) - sigma * sin(2 * pi * x)
        fy = -sin(2 * pi * x) * B0 / epsilon
        fz = zeros_like(x)
        return fx, fy, fz

elif BC == 'dirichlet_hom':
    def source_function_u(x, y, z):
        fx = 2.0 * pi * cos(2 * pi * x) + nu * 4.0 * pi**2 * sin(2 * pi * x)
        fy = B0 * sin(2 * pi * x) / epsilon
        fz = zeros_like(x)
        return fx, fy, fz

    def source_function_ue(x, y, z):
        fx = -2.0 * pi * cos(2 * pi * x) + nu_e * 4.0 * pi**2 * sin(2 * pi * x) - sigma * sin(2 * pi * x)
        fy = -sin(2 * pi * x) * B0 / epsilon
        fz = zeros_like(x)
        return fx, fy, fz

elif BC == 'dirichlet_inhom':
    def source_function_u(x, y, z):
        fx = ones_like(x)
        fy = B0 * (1 + x) / epsilon
        fz = zeros_like(x)
        return fx, fy, fz

    def source_function_ue(x, y, z):
        fx = -ones_like(x) - sigma * (1 + x)
        fy = -B0 * (1 + x) / epsilon
        fz = zeros_like(x)
        return fx, fy, fz

# ---- model ----
model = TwoFluidQuasiNeutralToy()
model.ions.set_phys_params()
model.electrons.set_phys_params()

model.propagators.qn_full.options = model.propagators.qn_full.Options(
    nu=nu,
    nu_e=nu_e,
    eps_norm=epsilon,
    stab_sigma=sigma,
    source_u=source_function_u,
    source_ue=source_function_ue,
    solver='gmres',
    boundary_conditions_u=bcs_u,
    boundary_conditions_ue=bcs_ue,
    boundary_data_u=boundary_data_u,
    boundary_data_ue=boundary_data_ue,
)

if __name__ == "__main__":
    main.run(model,
             params_path=__file__,
             env=env,
             base_units=base_units,
             time_opts=time_opts,
             domain=domain,
             equil=equil,
             grid=grid,
             derham_opts=derham_opts,
             verbose=True,
             )

    path    = os.path.join(os.getcwd(), name)
    main.pproc(path)
    simdata = main.load_data(path)

    n1_vals = simdata.grids_log[0]
    x       = xp.linspace(0, 1, 100)

    os.makedirs(f'{name}/plots', exist_ok=True)
    for f in glob.glob(f'{name}/plots/*.png'):
        os.remove(f)

    def save_plot(n1_vals, numerical, analytical, ylabel, title, fname, t):
        plt.plot(n1_vals, numerical, label='numerical')
        plt.plot(x, analytical, '--', label='manufactured')
        plt.plot(n1_vals, numerical, 'k.', markersize=4, label='n1 points')
        plt.xlabel('n1 (radial)')
        plt.ylabel(ylabel)
        plt.title(f'{title} at t={t:.3f}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{name}/plots/{fname}_{t:.3f}.png', dpi=300)
        plt.clf()

    for t in list(simdata.spline_values['ions']['u_log'].keys()):

        u_ions      = simdata.spline_values['ions']['u_log'][t]
        u_electrons = simdata.spline_values['electrons']['u_log'][t]
        phi         = simdata.spline_values['em_fields']['phi_log'][t]

        mms_phi_x,  _,          _ = mms_phi(x, x*0, x*0)
        mms_ion_ux, mms_ion_uy, _ = mms_ion_u(x, x*0, x*0)
        mms_el_ux,  mms_el_uy,  _ = mms_electron_u(x, x*0, x*0)

        save_plot(n1_vals, phi[0][:, 0, 0],         mms_phi_x,  'φ',   'Electrostatic potential φ', 'plot_potential',   t)
        save_plot(n1_vals, u_ions[0][:, 0, 0],      mms_ion_ux, 'u_x', 'Ion velocity (u_x)',        'plot_ion_ux',      t)
        save_plot(n1_vals, u_electrons[0][:, 0, 0], mms_el_ux,  'u_x', 'Electron velocity (u_x)',   'plot_electron_ux', t)