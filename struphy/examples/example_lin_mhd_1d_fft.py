import sysconfig
from struphy.diagnostics.diagn_tools import load_values_from_path, get_dispersion, plot_dispersion
from struphy.diagnostics.post_processing import Post_processing

path_out        = sysconfig.get_path("platlib") + '/struphy/io/out/examples/lin_mhd_1d_fft/'
params_name     = 'parameters_example_1d_fft.yml'
data_name       = 'data.hdf5'
eval_data_name  = 'eval_data.hdf5'

params, data    = load_values_from_path(path_out, params_name=params_name, data_name=data_name)
PPROC           = Post_processing(params, data)
PPROC.construct_FEEC_spaces()

quantities      = {'2_form_1':'magnetic field', 
                   '2_form_2':'magnetic field', 
                   '2_form_3':'magnetic field',
                   '3_form': 'pressure'}
PPROC.save_evaluated_data(quantities, path_out=path_out, data_name=eval_data_name)

# # Initialize pproc 
# PPROC = Post_processing(params, data)
# PPROC.construct_FEEC_spaces()

# # Specify quantities to evaluate
# quantities  = {'magnetic field':'2_form_1', 'pressure':'3_form'}
# PPROC.save_evaluated_data(quantities, path_out=path_out)