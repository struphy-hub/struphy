from struphy.diagnostics.post_processing import create_femfields, eval_femfields
from struphy.diagnostics.diagn_tools import fourier_1d

import sys

path = sys.argv[1] 

fields, spaces, domain, code = create_femfields(path)

values_log, values_phys, grids_log, grids_phys = eval_femfields(
    fields, spaces, domain, npts_per_cell=1)  # evaluation at cell boundaries for fft

# fft in (t, z) of first component of e_field on physical grid
fourier_1d(values_log['e_field'], code, grids_log['e_field'],
           grids_phys=grids_phys['e_field'], component=0, slice_at=[0, 0, None], plot=True, disp_name='Maxwell1D')
