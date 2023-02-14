import pickle

import yaml, h5py, sys, glob

import numpy as np
import matplotlib.pyplot as plt

from struphy.post_processing.post_processing_tools import create_femfields
from struphy.psydac_api.psydac_derham import Derham
from struphy.geometry import domains
from struphy.fields_background.mhd_equil import equils
from struphy.psydac_api.fields import Field
from struphy.diagnostics.continuous_spectra import get_mhd_continua_2d
from struphy.dispersion_relations.analytic import MhdContinousSpectraCylinder
from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space

sim_path = sys.argv[1]

# read in parameters
with open(sim_path + '/parameters.yml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
    
Nel = params['grid']['Nel']
p = params['grid']['p']
spl_kind = params['grid']['spl_kind']
nq_el = params['grid']['nq_el']
bc = params['grid']['bc']
polar_ck = params['grid']['polar_ck']
    
# load domain and evaluate mapped grid
dom_type = params['geometry']['type']
dom_params = params['geometry'][dom_type]

domain_class = getattr(domains, dom_type)
domain = domain_class(dom_params)

etaplot = [np.linspace(0., 1., 101),
           np.linspace(0., 1., 101),
           np.linspace(0., 1.,  21)]

xplot = domain(*etaplot)

# load MHD equilibrium
mhd_name = params['mhd_equilibrium']['type']
mhd_params = params['mhd_equilibrium'][mhd_name]

mhd_class = getattr(equils, mhd_name)
mhd_equil = mhd_class(mhd_params)

if params['mhd_equilibrium']['use_equil_domain']:
    assert mhd_equil.domain is not None
    domain = mhd_equil.domain
else:
    mhd_equil.domain = domain

# field names, grid info and energies
file = h5py.File(sim_path + '/data_proc0.hdf5', 'r')

names = list(file['feec'].keys())

t  = file['scalar']['time'][:]
eU = file['scalar']['en_U'][:]
eB = file['scalar']['en_B'][:]

file.close()

# FEM fields at t=0
fields, space_ids, code = create_femfields(sim_path + '/', snapshots=[0])

# perform continuous spectra diagnostics
spec_path = glob.glob(sim_path + '/*.npy')[0]
n_tor = int(spec_path[-6:-4])

fem_1d_1 = Spline_space_1d(Nel[0], p[0], spl_kind[0], nq_el[0], bc[0])
fem_1d_2 = Spline_space_1d(Nel[1], p[1], spl_kind[1], nq_el[1], bc[1])

fem_2d = Tensor_spline_space([fem_1d_1, fem_1d_2], polar_ck, domain.cx[:, :, 0], domain.cy[:, :, 0], n_tor=n_tor, basis_tor='i')

# load and analyze .npy spectrum
omega2, U2_eig = np.split(np.load(spec_path), [1], axis=1)
omega2 = omega2.flatten()

omegaA = mhd_params['B0']/mhd_params['R0']
A, S = get_mhd_continua_2d(fem_2d, domain, omega2, U2_eig, [0, 4], omegaA, 0.03, 3)

# plot some results
fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(14)

f_size = 16
plt.rcParams.update({'font.size': f_size})

# plot safety factor
plt.subplot(2, 2, 1)
r = np.linspace(0., 1., 101)
plt.plot(r, mhd_equil.q(r))
plt.xlabel('r [m]')
plt.ylabel('safety factor')

# plot shear Alfvén continuous spectra for m = [2, 3, 4]

# analytical continuous spectra
spec_calc = MhdContinousSpectraCylinder(R0=mhd_params['R0'], Bz=lambda r : mhd_params['B0'] - 0*r, q=mhd_equil.q, rho=mhd_equil.nr, p=mhd_equil.pr, gamma=5/3)

plt.subplot(2, 2, 2)
for m in range(2, 4 + 1):  
    plt.plot(0.1 + 0.9*A[m][0], A[m][1]/omegaA**2, '+', label='m = ' + str(m))
    plt.plot(domain(etaplot[0], 0., 0.)[0] - mhd_params['R0'], spec_calc(domain(etaplot[0], 0., 0.)[0] - mhd_params['R0'], m, -2)['shear_Alfvén']**2/omegaA**2, 'k--', linewidth=0.5)

plt.xlabel('$r$ [m]')
plt.ylabel('$\omega^2/\omega_\mathrm{A}^2$')
plt.xlim((0., 1.))
plt.ylim((0.05, omegaA**2))
plt.legend()
plt.title('Shear Alfvén continuum ($n=-2$)', pad=10, fontsize=f_size)
plt.xticks([0., 0.5, 1.])
plt.arrow(0.44, 0.5, 0., -0.30, head_width=.02)
plt.text(0.39, 0.55, 'TAE')
plt.plot(0.1*np.ones(11), np.linspace(0., 1., 11), 'k--')

# plot U2_1(t=0) on mapped grid
plt.subplot(2, 2, 3)
plt.contourf(xplot[0][:, :, 0], xplot[2][:, :, 0], fields[0][names[3]](*etaplot)[0][:, :, 6], levels=51, cmap='coolwarm')
plt.axis('square')
plt.colorbar()
plt.title('$U^2_1(t=0)$', pad=10, fontsize=f_size)
plt.xlabel('x [m]')
plt.ylabel('y [m]')

# plot energie time series
plt.subplot(2, 2, 4)
plt.plot(t, eU, label='$\epsilon_U$')
plt.plot(t, eB, label='$\epsilon_B$')
plt.xlabel('$t$ [Alfvén times]')
plt.ylabel('energies')
plt.legend()

plt.subplots_adjust(wspace=0.3, hspace=0.5)

plt.show()