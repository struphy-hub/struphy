from struphy.diagnostics.diagn_tools import fourier_1d

import sys
import pickle

path = sys.argv[1] 

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

# load data dicts for e_field
name = names[0]

with open(path + '/eval_fields/' + name + '_logical.bin', 'rb') as handle:
    point_data_log = pickle.load(handle)

with open(path + '/eval_fields/' + name + '_physical.bin', 'rb') as handle:
    point_data_phys = pickle.load(handle)

# load grids
with open(path + '/eval_fields/grids.bin', 'rb') as handle:
    grids = pickle.load(handle)

with open(path + '/eval_fields/grids_mapped.bin', 'rb') as handle:
    grids_mapped = pickle.load(handle)

with open(path + '/eval_fields/masks.bin', 'rb') as handle:
    masks = pickle.load(handle)

# fft in (t, z) of first component of e_field on physical grid
fourier_1d(point_data_log, name, code, grids, masks,
           grids_mapped=grids_mapped, component=0, slice_at=[0, 0, None], plot=True, disp_name='Maxwell1D')
