import yaml, h5py, sys
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

from struphy.geometry import domains
from struphy.fields_background.mhd_equil import analytical

sim_path = sys.argv[1]

# load parameters and markers
with open(sim_path + 'parameters.yml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

# load domain
dom_type = params['geometry']['type']
dom_params = params['geometry'][dom_type]

domain_class = getattr(domains, dom_type)
domain = domain_class(dom_params)

# load MHD equilibrium
equil_params = params['fields']['mhd_equilibrium']
mhd_equil_class = getattr(analytical, equil_params['type'])
mhd_equil = mhd_equil_class(equil_params[equil_params['type']], domain)

file = h5py.File(sim_path + 'data_proc0.hdf5', 'r')
grid_info = file['scalar'].attrs['grid_info']
file.close()
    
Nt = int(params['time']['Tend']/params['time']['dt'])
Np = params['kinetic']['ions']['save_data']['n_markers']

log_Nt = int(np.log10(Nt)) + 1

pos = np.zeros((Nt + 1, Np, 3), dtype=float)

print('Load ion orbits ...')
for n in tqdm(range(Nt + 1)):
    
    # load x, y, z coordinates
    pos[n] = np.loadtxt(sim_path + 'kinetic_data/ions/orbits/' + 'ions_{0:0{1}d}.txt'.format(n, log_Nt), delimiter=',')[:, 1:]
    
    # convert to R, y, z, coordinates
    pos[n, :, 0] = np.sqrt(pos[n, :, 0]**2 + pos[n, :, 2]**2)
    
fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(12)

# plot safety factor
plt.subplot(2, 2, 1)
r = np.linspace(0., 1., 101)
plt.plot(r, mhd_equil.q(r))
plt.xlabel('r [m]')
plt.ylabel('safety factor')

# plot absolute value of magnetic field in poloidal plane
plt.subplot(2, 2, 2)
e1 = np.linspace(0., 1., 101)
e2 = np.linspace(0., 1., 101)
X = domain(e1, e2, 0.)

plt.contourf(X[0], X[1], mhd_equil.b(X[0], X[1], X[2]), levels=50, cmap='inferno')

plt.xlabel('x [m]')
plt.ylabel('y [m]')

plt.axis('square')
plt.colorbar()
plt.title(r'$|\mathbf{B}|$ [T]')

# plot particle orbits
orbits_plt = plt.subplot(2, 2, 3)
orbits_plt.set_title('Passing, co-passing and trapped particle')
domain.show(grid_info=grid_info, markers=pos, marker_coords='phy')