def run(n_procs):
    """
    Run an example for the model "Maxwell", including post-processing.
    
    Parameters
    ----------
    n_procs : int
        Number of MPI processes to run the model.
    """
    
    import subprocess
       
    # name of simulation output folder
    out_name = 'sim_example_maxwell'
    
    # run the model (could also call main() directly, but to enable parallel runs, use Struphy's command line interface)
    subprocess.run(['struphy', 
                    'run', 
                    'Maxwell',
                    '-i',
                    'examples/params_maxwell.yml',
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
    
    # output path
    out_name = 'sim_example_maxwell'
    out_path = os.path.join(libpath, 'io/out', out_name) 
    
    # code name
    with open(os.path.join(out_path, 'meta.txt'), 'r') as f:
        lines = f.readlines()

    code = lines[3].split()[-1]

    # field names
    file = h5py.File(os.path.join(out_path,'data/', 'data_proc0.hdf5'), 'r')
    names = list(file['feec'].keys())
    file.close()

    # load data dicts for e_field
    with open(os.path.join(out_path, 'post_processing/fields_data/em_fields/e1_log.bin'), 'rb') as handle:
        point_data_log = pickle.load(handle)

    # load grids
    with open(os.path.join(out_path, 'post_processing/fields_data/grids_log.bin'), 'rb') as handle:
        grids_log = pickle.load(handle)

    with open(os.path.join(out_path, 'post_processing/fields_data/grids_phy.bin'), 'rb') as handle:
        grids_phy = pickle.load(handle)

    # fft in (t, z) of first component of e_field on physical grid
    fourier_1d(point_data_log, 
               names[1], 
               code, 
               grids_log,
               grids_mapped=grids_phy,
               component=0,
               slice_at=[0, 0, None],
               do_plot=True, 
               disp_name='Maxwell1D')
    
    
if __name__ == '__main__':
    
    import argparse
    
    # get number of MPI processes
    parser = argparse.ArgumentParser(description='Run an example for the model "Maxwell".')
    
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
