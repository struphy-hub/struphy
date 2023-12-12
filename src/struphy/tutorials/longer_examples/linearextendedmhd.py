def run(n_procs):
    """
    Run an example for the model "LinearExtendedMHD", including post-processing.
    
    Parameters
    ----------
    n_procs : int
        Number of MPI processes to run the model.
    """
    
    import os, subprocess
    import struphy
    
    libpath = struphy.__path__[0]
    
    # name of simulation output folder
    out_name = 'sim_example_linearextendedmhd'
    
    # run the model (could also call main() directly, but to enable parallel runs, use Struphy's command line interface)
    subprocess.run(['struphy', 
                    'run', 
                    'LinearExtendedMHD',
                    '-i',
                    os.path.join(libpath, 'io/inp/longer_examples/params_linearextendedmhd.yml'),
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
    
    from struphy.diagnostics.diagn_tools import power_spectrum_2d
    
    import struphy

    libpath = struphy.__path__[0]
    
    with open(os.path.join(libpath, 'state.yml')) as f:
        state = yaml.load(f, Loader=yaml.FullLoader)

    o_path = state['o_path']
    
    out_name = 'sim_example_linearextendedmhd'
    out_path = os.path.join(o_path, out_name)
    
    # read in parameters for analytical dispersion relation
    with open(os.path.join(out_path, 'parameters.yml')) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    
    # parameters for dispersion relation
    B0x = params['mhd_equilibrium']['HomogenSlab']['B0x']
    B0y = params['mhd_equilibrium']['HomogenSlab']['B0y']
    B0z = params['mhd_equilibrium']['HomogenSlab']['B0z']
    Bu = params['units']['B']
    xu = params['units']['x']
    nu = params['units']['n']
    mH = 1.67262192369e-27  # proton mass (kg)
    MU = 1.25663706212e-6 #vacuum permeability (N / A^2)
    A = params['fluid']['mhd']['phys_params']['A']
    Z = params['fluid']['mhd']['phys_params']['Z']
    tu= (xu/Bu) * (MU*mH*A*nu)**(0.5) *(10.0**10.0)
    p0 = (2*params['mhd_equilibrium']['HomogenSlab']
          ['beta'])/(B0x**2 + B0y**2 + B0z**2)
    n0 = params['mhd_equilibrium']['HomogenSlab']['n0']

    disp_params = {'B0x': B0x, 
                   'B0y': B0y,
                   'B0z': B0z, 
                   'p0': p0, 
                   'n0': n0, 
                   'gamma': 5/3,
                   'Bu': Bu,
                   'tu': tu,
                   'A': A,
                   'Z': Z}

    # code name
    with open(os.path.join(out_path, 'parameters.yml')) as f:
        lines = f.readlines()

    code = lines[-2].split()[-1]

    # field names
    file = h5py.File(os.path.join(out_path, 'data', 'data_proc0.hdf5'), 'r')
    names = list(file['feec'].keys())
    file.close()

    # load grids
    with open(os.path.join(out_path, 'post_processing/fields_data/grids_log.bin'), 'rb') as handle:
        grids_log = pickle.load(handle)

    with open(os.path.join(out_path, 'post_processing/fields_data/grids_phy.bin'), 'rb') as handle:
        grids_phy = pickle.load(handle)

    # load data dicts for u_field
    with open(os.path.join(out_path, 'post_processing/fields_data/mhd/pi3_log.bin'), 'rb') as handle:
        point_data_log = pickle.load(handle)

    # fft in (t, z) of first component of u_field on physical grid
    power_spectrum_2d(point_data_log,
               names[3],
               code,
               grids_log,
               grids_mapped=grids_phy, 
               component=0,
               slice_at=[0, 0, None],
               do_plot=True, 
               disp_name='ExtendedMhd1D', 
               disp_params=disp_params)

    # load data dicts for u_field
    with open(os.path.join(out_path, 'post_processing/fields_data/mhd/pe3_log.bin'), 'rb') as handle:
        point_data_log = pickle.load(handle)


    # fft in (t, z) of pressure on physical grid
    power_spectrum_2d(point_data_log,
               names[2], 
               code, 
               grids_log,
               grids_mapped=grids_phy,
               component=0,
               slice_at=[0, 0, None],
               do_plot=True, 
               disp_name='ExtendedMhd1D', 
               disp_params=disp_params)
    
    # load data dicts for u_field
    with open(os.path.join(out_path, 'post_processing/fields_data/mhd/n3_log.bin'), 'rb') as handle:
        point_data_log = pickle.load(handle)


    # fft in (t, z) of pressure on physical grid
    power_spectrum_2d(point_data_log,
               names[1], 
               code, 
               grids_log,
               grids_mapped=grids_phy,
               component=0,
               slice_at=[0, 0, None],
               do_plot=True, 
               disp_name='ExtendedMhd1D', 
               disp_params=disp_params)
    
    # load data dicts for u_field
    with open(os.path.join(out_path, 'post_processing/fields_data/em_fields/b1_log.bin'), 'rb') as handle:
        point_data_log = pickle.load(handle)
    print(point_data_log)

    # fft in (t, z) of pressure on physical grid
    power_spectrum_2d(point_data_log,
               names[0], 
               code, 
               grids_log,
               grids_mapped=grids_phy,
               component=0,
               slice_at=[0, 0, None],
               do_plot=True, 
               disp_name='ExtendedMhd1D', 
               disp_params=disp_params)
    
    # load data dicts for u_field
    with open(os.path.join(out_path, 'post_processing/fields_data/mhd/u2_log.bin'), 'rb') as handle:
        point_data_log = pickle.load(handle)


    # fft in (t, z) of pressure on physical grid
    power_spectrum_2d(point_data_log,
               names[4], 
               code, 
               grids_log,
               grids_mapped=grids_phy,
               component=0,
               slice_at=[0, 0, None],
               do_plot=True, 
               disp_name='ExtendedMhd1D', 
               disp_params=disp_params)
    
    
if __name__ == '__main__':
    
    import argparse
    
    # get number of MPI processes
    parser = argparse.ArgumentParser(description='Run an example for the model "LinearExtendedMHD".')
    
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