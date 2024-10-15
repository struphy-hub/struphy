def run(n_procs):
    """
    Run an example for the model "LinearMHDVlasovCC", including post-processing.
    
    Parameters
    ----------
    n_procs : int
        Number of MPI processes to run the model.
    """
    
    import os, subprocess
    import struphy
    
    libpath = struphy.__path__[0]
    
    # name of simulation output folder
    out_name = 'sim_example_linearmhdvlasovcc'
    
    # run the model
    subprocess.run(['struphy', 
                    'run', 
                    'LinearMHDVlasovCC',
                    '-i',
                    os.path.join(libpath, 'io/inp/longer_examples/params_hybridmhdvlasovcc.yml'),
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
    
    import os, h5py, pickle, yaml
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    import struphy

    libpath = struphy.__path__[0]
    
    with open(os.path.join(libpath, 'state.yml')) as f:
        state = yaml.load(f, Loader=yaml.FullLoader)

    o_path = state['o_path']
    
    out_name = 'sim_example_linearmhdvlasovcc'
    out_path = os.path.join(o_path, out_name)

    # load data
    file = h5py.File(os.path.join(out_path, 'data/', 'data_proc0.hdf5'), 'r')

    t  = file['time/value'][:]
    eu = file['scalar/en_U'][:]
    eb = file['scalar/en_B'][:]
    ef = file['scalar/en_f'][:]

    field_names = list(file['feec'].keys())

    file.close()

    # load grid
    with open(os.path.join(out_path, 'post_processing/fields_data/grids_phy.bin'), 'rb') as handle:
        grids_phy = pickle.load(handle)

    Lz = grids_phy[2][0, 0, -1]

    # load distriution function
    f  = np.load(os.path.join(out_path, 'post_processing/kinetic_data/energetic_ions/distribution_function/v3/f_binned.npy'))
    vz = np.load(os.path.join(out_path, 'post_processing/kinetic_data/energetic_ions/distribution_function/v3/grid_v3.npy'))

    fig = plt.figure()
    fig.set_figheight(3.5)
    fig.set_figwidth(12)

    plt.subplot(1, 2, 1)

    gamma = 0.0805

    plt.semilogy(t, (eu + eb)/2)
    plt.semilogy(t, 1.3e-6*np.exp(2*gamma*t), 'k--', linewidth=0.5)
    plt.ylim((1e-5, 1e-1))
    plt.xlim((0., 120.))
    plt.xlabel('$t$')
    plt.ylabel('magnetic energy + bulk kinetic energy')
    plt.title('Initialization with pure EP statistical noise')
    plt.plot(np.ones(11)*67, np.linspace(1e-6, 1e-1, 11), 'k--')

    plt.text(15, 2.5e-2, 'analytical growth')
    plt.arrow(51, 2.8e-2, 7., 0., head_width=.01, head_length=.5000)
    plt.text(15, 2e-3, 'linear phase')
    plt.text(80, 1e-4, 'nonlinear phase')

    plt.subplot(1, 2, 2)
    plt.plot(vz, f[0], label='$t=0$')
    plt.plot(vz, f[300], label='$t=60$')
    plt.xlabel('$v_z$')
    plt.ylabel('$f_{v_z}$')
    plt.title('EP distribution function')
    plt.text(3.5, 0.03, 'resonance velocity')
    plt.arrow(3.3, 0.03, -0.5, 0., head_width=.002, head_length=.15)
    plt.legend(loc='upper left')

    vR = 1 + 1/(2*np.pi/Lz)

    plt.plot(np.ones(11)*vR, np.linspace(0.01, 0.04, 11), 'k--')

    plt.show()
    

if __name__ == '__main__':
    
    import argparse
    
    # get number of MPI processes
    parser = argparse.ArgumentParser(description='Run an example for the model "LinearMHDVlasovCC".')
    
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