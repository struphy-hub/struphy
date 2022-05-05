from struphy.diagnostics.post_processing import create_femfields, eval_femfields
from struphy.diagnostics.diagn_tools import fourier_1d

from psydac.fem.vector import ProductFemSpace

import sysconfig
import matplotlib.pyplot as plt
import numpy as np

path = sysconfig.get_path("platlib") + '/struphy/io/out/sim_1/'
#path = 'struphy/io/out/sim_1/'

fields, DOMAIN, code = create_femfields(path, snapshots=[0])

values_t, grids, grids_phys = eval_femfields(fields, DOMAIN, npts_per_cell=1)

for key, values in values_t.items():

    for t, snapshot in values.items():

        print(f'Plotting {key} at time {t}.')

        for n, value in enumerate(snapshot):
            plt.figure()
            plt.plot(grids[key][2].flatten(), value[0, 0, :])
            plt.title(key + ', component ' + str(n + 1) + ' at t=' + str(t))

plt.show()
