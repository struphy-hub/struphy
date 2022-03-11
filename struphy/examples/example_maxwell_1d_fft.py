
import sysconfig
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from struphy.diagnostics.diagn_tools import load_values_from_path, get_dispersion
from struphy.diagnostics.post_processing import Post_processing

path_out        = sysconfig.get_path("platlib") + '/struphy/io/out/examples/maxwell_1d_fft/'
params_name     = 'parameters_example_1d_fft.yml'
data_name       = 'data.hdf5'
eval_data_name  = 'eval_data.hdf5'

# Post processing data #
params, data    = load_values_from_path(path_out, params_name=params_name, data_name=data_name)
PPROC           = Post_processing(params, data)
PPROC.construct_FEEC_spaces()

quantities      = {'1_form_1':'electric field', 
                   '1_form_2':'electric field', 
                   '1_form_3':'electric field'}
PPROC.save_evaluated_data(quantities, path_out=path_out, data_name=eval_data_name)

# Fourier transformation of the data #
params, data        = load_values_from_path(path_out, params_name=params_name, data_name=eval_data_name)
k, w, disp_num      = get_dispersion(params, data, 'electric field x')

# Plot the data with a default plot function #
direction       = params['initialization']['params_noise']['direction']
Nx, Ny, Nz      = params['grid']['Nel']
domain_type     = params['geometry']['type']
dx              = (params['geometry']['params_' + domain_type]['e1'] - params['geometry']['params_' + domain_type]['b1']) / Nx
dy              = (params['geometry']['params_' + domain_type]['e2'] - params['geometry']['params_' + domain_type]['b2']) / Ny
dz              = (params['geometry']['params_' + domain_type]['e3'] - params['geometry']['params_' + domain_type]['b3']) / Nz

if direction == 'x': h = dx
if direction == 'y': h = dy
if direction == 'z': h = dz

fig, ax     = plt.subplots(1,1, figsize=(10,10))
colormap    = 'YlOrRd'
K, W        = np.meshgrid(k*h,w*h)
# sort_disp   = np.sort(disp_num.copy().flatten())
# levels      = np.linspace(sort_disp[10], sort_disp[-5], 20)
CX          = ax.contourf(K, W, disp_num**2/ (disp_num**2).max(), cmap=colormap, norm=colors.LogNorm())
# cbarx       = fig.colorbar(CX, ax=ax)
title = 'Nel=' + str(params['grid']['Nel']) + ', degrees=' + str(params['grid']['p']) + ', $\Delta t =$' + str(params['time']['dt'])
ax.set_title(title)
ax.set_aspect('equal', 'box')
ax.set_xlim(0, k[-1]*h)
ax.set_ylim(0, k[-1]*h)
# ax.set_xlabel('$c_0 k/\Omega_{ce}$')
ax.set_xlabel('$k \Delta_{x}$')
ax.set_ylabel('$\omega$')
# cbarx.ax.set_ylabel('Amplitude')
plt.savefig(path_out + 'example_plot.png')
plt.show()