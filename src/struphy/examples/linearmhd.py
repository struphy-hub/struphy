def run(n_procs):
    """
    Run an example for the model "LinearMHD", including post-processing.
    
    Parameters
    ----------
    n_procs : int
        Number of MPI processes to run the model.
    """
    
    import subprocess
    
    # name of simulation output folder
    out_name = 'sim_example_linearmhd'
    
    # run the model (could also call main() directly, but to enable parallel runs, use Struphy's command line interface)
    subprocess.run(['struphy', 
                    'run', 
                    'LinearMHD',
                    '-i',
                    'examples/params_linearmhd.yml',
                    '-o',
                    out_name,
                    '--mpi',
                    str(n_procs)], check=True)
    
    # perform post-processing
    subprocess.run(['struphy',
                    'pproc',
                    '-d',
                    out_name], check=True)
    

def diagnostics():
    """
    Perform diagnostics and plot results for the example run.
    """
    
    import os, yaml, h5py, pickle
    
    from struphy.diagnostics.diagn_tools import fourier_1d
    
    import struphy

    libpath = struphy.__path__[0]
    
    out_name = 'sim_example_linearmhd'
    out_path = os.path.join(libpath, 'io/out', out_name)
    
    # read in parameters for analytical dispersion relation
    with open(os.path.join(out_path, 'parameters.yml')) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # parameters for dispersion relation
    B0x = params['mhd_equilibrium']['HomogenSlab']['B0x']
    B0y = params['mhd_equilibrium']['HomogenSlab']['B0y']
    B0z = params['mhd_equilibrium']['HomogenSlab']['B0z']

    p0 = (2*params['mhd_equilibrium']['HomogenSlab']
          ['beta']/100)/(B0x**2 + B0y**2 + B0z**2)
    n0 = params['mhd_equilibrium']['HomogenSlab']['n0']

    disp_params = {'B0x': B0x, 
                   'B0y': B0y,
                   'B0z': B0z, 
                   'p0': p0, 
                   'n0': n0, 
                   'gamma': 5/3}

    # code name
    with open(os.path.join(out_path, 'parameters.yml')) as f:
        lines = f.readlines()

    code = lines[-2].split()[-1]

    # field names
    file = h5py.File(os.path.join(out_path, 'data/', 'data_proc0.hdf5'), 'r')
    names = list(file['feec'].keys())
    file.close()

    # load grids
    with open(os.path.join(out_path, 'post_processing/fields_data/grids_log.bin'), 'rb') as handle:
        grids_log = pickle.load(handle)

    with open(os.path.join(out_path, 'post_processing/fields_data/grids_phy.bin'), 'rb') as handle:
        grids_phy = pickle.load(handle)

    print(names)

    # load data dicts for u_field
    with open(os.path.join(out_path, 'post_processing/fields_data/mhd/uv_log.bin'), 'rb') as handle:
        point_data_log = pickle.load(handle)

    # fft in (t, z) of first component of u_field on physical grid
    fourier_1d(point_data_log,
               names[3],
               code,
               grids_log,
               grids_mapped=grids_phy, 
               component=0,
               slice_at=[0, 0, None],
               do_plot=True, 
               disp_name='Mhd1D', 
               disp_params=disp_params)

    # load data dicts for pressure
    with open(os.path.join(out_path, 'post_processing/fields_data/mhd/p3_log.bin'), 'rb') as handle:
        point_data_log = pickle.load(handle)

    # fft in (t, z) of pressure on physical grid
    fourier_1d(point_data_log,
               names[2], 
               code, 
               grids_log,
               grids_mapped=grids_phy,
               component=0,
               slice_at=[0, 0, None],
               do_plot=True, 
               disp_name='Mhd1D', 
               disp_params=disp_params)
    
    
if __name__ == '__main__':
    
    import argparse
    
    # get number of MPI processes
    parser = argparse.ArgumentParser(description='Run an example for the model "LinearMHD".')
    
    parser.add_argument('--mpi',
                        type=int,
                        metavar='N',
                        help='number of MPI processes used to run the model (default=1)',
                        default=1)
    
    parser.add_argument('-d', '--diagnostics',
                        help='run diagnostics only, if output folder of example already exists',
                        action='store_true')
    
    args = parser.parse_args()
    
    if not args.diagnostics:
        run(args.mpi)
        
    diagnostics()