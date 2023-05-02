def run(n_procs):
    """
    Run an example for the model "Vlasov", including post-processing.
    
    Parameters
    ----------
    n_procs : int
        Number of MPI processes to run the model.
    """
    
    import subprocess
    
    # name of simulation output folder
    out_name = 'sim_example_orbits_tokamak' 
    
    # run the model
    subprocess.run(['struphy', 
                    'run', 
                    'Vlasov',
                    '-i',
                    'examples/params_orbits_tokamak.yml',
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
    
    import os, yaml, h5py
    from tqdm import tqdm
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    from struphy.models.utilities import setup_domain_mhd
    from struphy.fields_background.mhd_equil.equils import EQDSKequilibrium
    
    import struphy
    libpath = struphy.__path__[0]
    
    # output path
    out_name = 'sim_example_orbits_tokamak'
    out_path = os.path.join(libpath, 'io/out', out_name)
    
    # load simulation parameters
    with open(os.path.join(out_path, 'parameters.yml')) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # create domain and MHD equilibrium
    domain, mhd_equil = setup_domain_mhd(params)
    domain_name = domain.__class__.__name__
    
    # load grid_info (domain decomposition)
    file = h5py.File(os.path.join(out_path, 'data_proc0.hdf5'), 'r')
    grid_info = file['scalar'].attrs['grid_info']
    file.close()

    # load time grid info
    tgrid = np.load(os.path.join(out_path, 'post_processing/t_grid.npy'))
    Nt = len(tgrid) - 1
    
    log_Nt = int(np.log10(Nt)) + 1

    # load orbits
    Np = params['kinetic']['ions']['save_data']['n_markers']
    
    pos = np.zeros((Nt + 1, Np, 3), dtype=float)
    print('Load ion orbits ...')
    for n in tqdm(range(Nt + 1)):

        # load x, y, z coordinates
        orbits_path = os.path.join(out_path, 'post_processing/kinetic_data/ions/orbits', 'ions_{0:0{1}d}.txt'.format(n, log_Nt))
        pos[n] = np.loadtxt(orbits_path, delimiter=',')[:, 1:]

        # convert to R, y, z, coordinates
        pos[n, :, 0] = np.sqrt(pos[n, :, 0]**2 + pos[n, :, 1]**2)

    # plot results
    fig = plt.figure(figsize=(16, 8))

    # plot absolute value of magnetic field in poloidal plane
    plt.subplot(1, 2, 1)
    e1 = np.linspace(0., 1., 101)
    e2 = np.linspace(0., 1., 101)
    e1[0] += 1e-5
    X = domain(e1, e2, 0.)

    plt.contourf(X[0], X[2], mhd_equil.absB0(e1, e2, 0.), levels=51)
    
    for i in range(grid_info.shape[0]):

        e1 = np.linspace(grid_info[i, 0], grid_info[i, 1],
                         int(grid_info[i, 2]) + 1)
        e2 = np.linspace(grid_info[i, 3], grid_info[i, 4],
                         int(grid_info[i, 5]) + 1)
        X = domain(e1, e2, 0.)

        # plot xz-plane for torus mappings, xy-plane else
        if 'Torus' in domain_name or domain_name == 'GVECunit' or domain_name == 'Tokamak':
            co1, co2 = 0, 2
        else:
            co1, co2 = 0, 1

        # eta1-isolines
        for j in range(e1.size):
            if j == 0:
                if i == 0:
                    plt.plot(X[co1, j, :], X[co2, j, :], color='k', label='domain decomposition for ' + str(grid_info.shape[0]) + ' MPI processes')
                else:
                    plt.plot(X[co1, j, :], X[co2, j, :], color='k')
            elif j == e1.size - 1:
                plt.plot(X[co1, j, :], X[co2, j, :], color='k')
            else:
                plt.plot(X[co1, j, :], X[co2, j, :], color='tab:blue', alpha=.25)

        # eta2-isolines
        for k in range(e2.size):
            if k == 0:
                plt.plot(X[co1, :, k], X[co2, :, k], color='k')
            elif k == e2.size - 1:
                plt.plot(X[co1, :, k], X[co2, :, k], color='k')
            else:
                plt.plot(X[co1, :, k], X[co2, :, k], color='tab:blue', alpha=.25)
    
    if isinstance(mhd_equil, EQDSKequilibrium):
        plt.plot(mhd_equil.limiter_pts_R, mhd_equil.limiter_pts_Z, 'tab:orange')
    
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.legend()
    plt.axis('equal')
    plt.colorbar()
    plt.title(r'$|\mathbf{B}|$ [T]')

    # plot domain and particle orbits
    plt.subplot(1, 2, 2)
    
    for i in range(grid_info.shape[0]):

        e1 = np.linspace(grid_info[i, 0], grid_info[i, 1], 
                         int(grid_info[i, 2]) + 1)
        e2 = np.linspace(grid_info[i, 3], grid_info[i, 4], 
                         int(grid_info[i, 5]) + 1)

        X = domain(e1, e2, 0.)

        # plot xz-plane for torus mappings, xy-plane else
        if 'Torus' in domain_name or domain_name == 'GVECunit' or domain_name == 'Tokamak':
            co1, co2 = 0, 2
        else:
            co1, co2 = 0, 1

        # eta1-isolines
        for j in range(e1.size):
            if e1[j] == 0.:
                plt.plot(X[co1, j, :], X[co2, j, :], color='k')
            elif e1[j] == 1.:
                plt.plot(X[co1, j, :], X[co2, j, :], color='k')
            else:
                plt.plot(X[co1, j, :], X[co2, j, :], color='tab:blue', alpha=.25)

        # eta2-isolines
        for k in range(e2.size):
            plt.plot(X[co1, :, k], X[co2, :, k], color='tab:blue', alpha=.25)
    
    for i in range(pos.shape[1]):
        plt.scatter(pos[:, i, 0], pos[:, i, 2], s=2)
    
    if isinstance(mhd_equil, EQDSKequilibrium):
        plt.plot(mhd_equil.limiter_pts_R, mhd_equil.limiter_pts_Z, 'tab:orange')
    
    plt.axis('equal')
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('Passing and trapped particles (co- and counter direction)')
    
    plt.show()
    
    
if __name__ == '__main__':
    
    import argparse
    
    # get number of MPI processes
    parser = argparse.ArgumentParser(description='Run an example for the model "Vlasov".')
    
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