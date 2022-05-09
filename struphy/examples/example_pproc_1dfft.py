from struphy.diagnostics.post_processing import create_femfields, eval_femfields
from struphy.diagnostics.diagn_tools import fourier_1d

import sys

path = sys.argv[1] 
print(path)

fields, DOMAIN, code = create_femfields(path)

values_t, grids, grids_phys = eval_femfields(
    fields, DOMAIN, npts_per_cell=1)  # evaluation at cell boundaries for fft

# fft in (t, z) of first component of e_field on physical grid
fourier_1d(values_t['e_field'], code, grids['e_field'],
           grids_phys=grids_phys['e_field'], component=0, slice_at=[0, 0, None], plot=True)
