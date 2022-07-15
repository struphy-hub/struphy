from struphy.diagnostics.post_processing import create_femfields, eval_femfields
from struphy.diagnostics.diagn_tools import fourier_1d

import sys
import yaml

path = sys.argv[1] 

fields, spaces, domain, code = create_femfields(path)

values_log, values_phys, grids_log, grids_phys = eval_femfields(
    fields, spaces, domain, npts_per_cell=1)  # evaluation at cell boundaries for fft

# read in parameters for analytical dispersion relation
with open(path + '/parameters.yml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
    
B0x = params['fields']['mhd_equilibrium']['HomogenSlab']['B0x']
B0y = params['fields']['mhd_equilibrium']['HomogenSlab']['B0y']
B0z = params['fields']['mhd_equilibrium']['HomogenSlab']['B0z']

p0 = (2*params['fields']['mhd_equilibrium']['HomogenSlab']['beta']/100)/(B0x**2 + B0y**2 + B0z**2)
n0 = 1.

gamma = 5/3

disp_params = {'B0x': B0x, 'B0y': B0y,'B0z': B0z, 'p0': p0, 'n0': n0, 'gamma': 5/3}

# fft in (t, z) of first component of u_field on physical grid
fourier_1d(values_log['uv'], code, grids_log['uv'],
           grids_phys=grids_phys['uv'], component=0, slice_at=[0, 0, None], plot=True, disp_name='Mhd1D', disp_params=disp_params)

# fft in (t, z) of pressure on physical grid
fourier_1d(values_log['p3'], code, grids_log['p3'],
           grids_phys=grids_phys['p3'], component=0, slice_at=[0, 0, None], plot=True, disp_name='Mhd1D', disp_params=disp_params)
