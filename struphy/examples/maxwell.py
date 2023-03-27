def main(mpi=1):
    """
    Run an example for the model "Maxwell", including post-processing, diagnostics and plots.
    
    Parameters
    ----------
    mpi : int
        Number of MPI processes to run the model.
    """
    
    from struphy.diagnostics.diagn_tools import fourier_1d
    
    import os, yaml, h5py, pickle
    import subprocess
    import struphy

    libpath = struphy.__path__[0]
    
    # output path
    out_name = 'sim_example_maxwell'
    out_path = os.path.join(libpath, 'io/out', out_name)    
    
    # run the model (could also call main() directly, but to enable parallel runs, use Struphy's command line interface)
    subprocess.run(['struphy', 
                    'run', 
                    'Maxwell',
                    '-i',
                    'examples/params_maxwell.yml',
                    '-o',
                    out_name,
                    '--mpi',
                    str(mpi)], check=True)
    
    # perform post-processing
    subprocess.run(['struphy',
                    'pproc',
                    out_name], check=True)
    
    # code name
    with open(os.path.join(out_path, 'meta.txt'), 'r') as f:
        lines = f.readlines()

    code = lines[-2].split()[-1]

    # field names
    file = h5py.File(os.path.join(out_path, 'data_proc0.hdf5'), 'r')
    names = list(file['feec'].keys())
    file.close()

    # load data dicts for e_field
    with open(os.path.join(out_path, 'eval_fields', names[1] + '_log.bin'), 'rb') as handle:
        point_data_log = pickle.load(handle)

    with open(os.path.join(out_path, 'eval_fields', names[1] + '_phy.bin'), 'rb') as handle:
        point_data_phy = pickle.load(handle)

    # load grids
    with open(os.path.join(out_path, 'eval_fields/grids_log.bin'), 'rb') as handle:
        grids_log = pickle.load(handle)

    with open(os.path.join(out_path, 'eval_fields/grids_phy.bin'), 'rb') as handle:
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
    main()