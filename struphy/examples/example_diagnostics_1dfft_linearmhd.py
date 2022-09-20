from struphy.diagnostics.diagn_tools import fourier_1d

import pickle
import sys
import yaml

path = sys.argv[1] 

# read in parameters for analytical dispersion relation
with open(path + '/parameters.yml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
    
B0x = params['fields']['mhd_equilibrium']['HomogenSlab']['B0x']
B0y = params['fields']['mhd_equilibrium']['HomogenSlab']['B0y']
B0z = params['fields']['mhd_equilibrium']['HomogenSlab']['B0z']

p0 = (2*params['fields']['mhd_equilibrium']['HomogenSlab']['beta']/100)/(B0x**2 + B0y**2 + B0z**2)
n0 = params['fields']['mhd_equilibrium']['HomogenSlab']['n0']

gamma = 5/3

disp_params = {'B0x': B0x, 'B0y': B0y,'B0z': B0z, 'p0': p0, 'n0': n0, 'gamma': 5/3}

with open(path + '/meta.txt', 'r') as f:
        lines = f.readlines()

# code name
code = lines[-2].split()[-1]
# number of processes
mpi_size = int(lines[-1].split()[-1])

with open(path + '/MODEL_names.bin', 'rb') as handle:
    li = pickle.load(handle)
    names = li[0]
    space_ids = li[1]
    

# load grids
with open(path + '/eval_fields/grids.bin', 'rb') as handle:
    grids = pickle.load(handle)

with open(path + '/eval_fields/grids_mapped.bin', 'rb') as handle:
    grids_mapped = pickle.load(handle)

# load data dicts for u_field
name = names[1]

with open(path + '/eval_fields/' + name + '_logical.bin', 'rb') as handle:
    point_data_log = pickle.load(handle)

with open(path + '/eval_fields/' + name + '_physical.bin', 'rb') as handle:
    point_data_phys = pickle.load(handle)
    
with open(path + '/eval_fields/masks.bin', 'rb') as handle:
    masks = pickle.load(handle)

# fft in (t, z) of first component of u_field on physical grid
fourier_1d(point_data_log, name, code, grids, masks,
           grids_mapped=grids_mapped, component=0, slice_at=[0, 0, None], plot=True, disp_name='Mhd1D', disp_params=disp_params)

# load data dicts for pressure
name = names[2]

with open(path + '/eval_fields/' + name + '_logical.bin', 'rb') as handle:
    point_data_log = pickle.load(handle)

with open(path + '/eval_fields/' + name + '_physical.bin', 'rb') as handle:
    point_data_phys = pickle.load(handle)


# fft in (t, z) of pressure on physical grid
fourier_1d(point_data_log, name, code, grids, masks,
           grids_mapped=grids_mapped, component=0, slice_at=[0, 0, None], plot=True, disp_name='Mhd1D', disp_params=disp_params)
