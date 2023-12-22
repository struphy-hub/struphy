def run(n_procs):
    """
    Run a TAE example for the model "LinearMHD", including post-processing.
    
    Parameters
    ----------
    n_procs : int
        Number of MPI processes to run the model.
    """
    
    import os
    import subprocess
    import struphy
    import yaml

    libpath = struphy.__path__[0]
    
    with open(os.path.join(libpath, 'state.yml')) as f:
        state = yaml.load(f, Loader=yaml.FullLoader)

    o_path = state['o_path']

    # name of simulation output folder
    out_name = 'sim_example_TAE_tokamak'
    
    # run MHD eigenvalue solver
    subprocess.run(['python3',
                    os.path.join(libpath, 'eigenvalue_solvers/mhd_axisymmetric_main.py'),
                    '-1',
                    '--input-abs',
                    os.path.join(libpath, 'io/inp/longer_examples/params_TAE_tokamak.yml'),
                    '-o',
                    out_name], 
                    check=True)
    
    # make the .npy eigenspectrum smaller (just for testing)
    subprocess.run(['python3',
                    os.path.join(libpath, 'eigenvalue_solvers/mhd_axisymmetric_pproc.py'),
                    '-n',
                    '-1',
                    '-i',
                    out_name,
                    '0.1',
                    '0.2'], 
                    check=True, cwd=libpath)
    
    # run the model
    subprocess.run(['struphy', 
                    'run', 
                    'LinearMHD',
                    '--input-abs',
                    os.path.join(o_path, out_name, 'parameters.yml'),
                    '-o',
                    out_name,
                    '--mpi',
                    str(n_procs)], check=True)
    
    # perform post-processing
    subprocess.run(['struphy',
                    'pproc',
                    '-d',
                    out_name,
                    '-s',
                    '500',
                    '--celldivide',
                    '5'], check=True)
    
    
def diagnostics():
    """
    Perform diagnostics and plot results for the example run.
    """
    
    import os, yaml, h5py, pickle
    
    import numpy as np
    import matplotlib.pyplot as plt

    from struphy.io.setup import setup_domain_mhd
    from struphy.diagnostics.continuous_spectra import get_mhd_continua_2d
    from struphy.dispersion_relations.analytic import MhdContinousSpectraCylinder
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    
    import struphy

    libpath = struphy.__path__[0]
    
    with open(os.path.join(libpath, 'state.yml')) as f:
        state = yaml.load(f, Loader=yaml.FullLoader)

    o_path = state['o_path']
    
    out_name = 'sim_example_TAE_tokamak'
    out_path = os.path.join(o_path, out_name)

    # load simulation parameters
    with open(os.path.join(out_path, 'parameters.yml')) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # load grid data
    Nel = params['grid']['Nel']
    p = params['grid']['p']
    spl_kind = params['grid']['spl_kind']
    nq_el = params['grid']['nq_el']
    dirichlet_bc = params['grid']['dirichlet_bc']
    polar_ck = params['grid']['polar_ck']
    
    # create domain and MHD equilibrium
    domain, mhd_equil = setup_domain_mhd(params)
    
    # get MHD equilibrium parameters
    mhd_params = params['mhd_equilibrium'][params['mhd_equilibrium']['type']]

    # field names, grid info and energies
    file = h5py.File(os.path.join(out_path, 'data/', 'data_proc0.hdf5'), 'r')

    names = list(file['feec'].keys())

    t  = file['time/value'][:]
    eU = file['scalar']['en_U'][:]
    eB = file['scalar']['en_B'][:]

    file.close()
    
    # load logical and physical grids
    with open(os.path.join(out_path, 'post_processing/fields_data/grids_log.bin'), 'rb') as handle:
        grids_log = pickle.load(handle)

    with open(os.path.join(out_path, 'post_processing/fields_data/grids_phy.bin'), 'rb') as handle:
        grids_phy = pickle.load(handle)

    # load data dicts for logical u_field
    with open(os.path.join(out_path, 'post_processing/fields_data/mhd/u2_log.bin'), 'rb') as handle:
        u_field_log = pickle.load(handle)

    # perform continuous spectra diagnostics
    spec_path = os.path.join(out_path, 'spec_n_-1.npy')
    n_tor = int(spec_path[-6:-4])

    bc = ['f', 'f']
    if dirichlet_bc[0]:
        bc[0] = 'd'
        
    if dirichlet_bc[1]:
        bc[1] = 'd'

    fem_1d_1 = Spline_space_1d(Nel[0], p[0], spl_kind[0], nq_el[0], bc[0])
    fem_1d_2 = Spline_space_1d(Nel[1], p[1], spl_kind[1], nq_el[1], bc[1])

    fem_2d = Tensor_spline_space([fem_1d_1, fem_1d_2], polar_ck,
                                 domain.cx[:, :, 0], domain.cy[:, :, 0],
                                 n_tor=n_tor, basis_tor='i')

    # load and analyze .npy spectrum
    omega2, U2_eig = np.split(np.load(spec_path), [1], axis=0)
    omega2 = omega2.flatten()

    omegaA = mhd_params['B0']/mhd_params['R0']
    A, S = get_mhd_continua_2d(fem_2d, domain, omega2, U2_eig, [
                               0, 4], omegaA, 0.03, 3)

    # plot some results
    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(14)

    f_size = 16
    plt.rcParams.update({'font.size': f_size})

    # plot safety factor
    plt.subplot(2, 2, 1)
    r = np.linspace(0., 1., 101)
    plt.plot(r, mhd_equil.q_r(r))
    plt.xlabel('r [m]')
    plt.ylabel('safety factor')

    # plot shear Alfvén continuous spectra for m = [2, 3, 4]

    # analytical continuous spectra
    spec_calc = MhdContinousSpectraCylinder(R0=mhd_params['R0'],
                                            Bz=lambda r: mhd_params['B0'] - 0*r,
                                            q=mhd_equil.q_r, rho=mhd_equil.n_r,
                                            p=mhd_equil.p_r, gamma=5/3)

    plt.subplot(2, 2, 2)
    for m in range(2, 4 + 1):
        plt.plot(0.1 + 0.9*A[m][0], A[m][1]/omegaA **2,
                 '+', label='m = ' + str(m))
        plt.plot(domain(grids_log[0], 0., 0.)[0] - mhd_params['R0'],
                 spec_calc(domain(grids_log[0], 0., 0.)[0] \
                 - mhd_params['R0'], m, -2)['shear_Alfvén']**2/omegaA**2,
                 'k--', linewidth=0.5)

    plt.xlabel('$r$ [m]')
    plt.ylabel('$\omega^2/\omega_\mathrm{A}^2$')
    plt.xlim((0., 1.))
    plt.ylim((0.05, omegaA**2))
    plt.legend()
    plt.title('Shear Alfvén continuum ($n=-2$)', pad=10, fontsize=f_size)
    plt.xticks([0., 0.5, 1.])
    plt.arrow(0.44, 0.5, 0., -0.30, head_width=.02)
    plt.text(0.39, 0.55, 'TAE')
    plt.plot(0.1*np.ones(11), np.linspace(0., 1., 11), 'k--')

    # plot U2_1(t=0) on mapped grid
    plt.subplot(2, 2, 3)
    
    plt.contourf(grids_phy[0][:, :, 0], grids_phy[2][:, :, 0],
                 u_field_log[0.][0][:, :, 8], levels=51, cmap='coolwarm')
    
    plt.axis('square')
    plt.colorbar()
    plt.title('$U^2_1(t=0)$', pad=10, fontsize=f_size)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')

    # plot energie time series
    plt.subplot(2, 2, 4)
    plt.plot(t, eU, label='$\epsilon_U$')
    plt.plot(t, eB, label='$\epsilon_B$')
    plt.xlabel('$t$ [Alfvén times]')
    plt.ylabel('energies')
    plt.legend()

    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    plt.show()
       
    
if __name__ == '__main__':
    
    import argparse
    
    # get number of MPI processes
    parser = argparse.ArgumentParser(description='Run a TAE (Toroidal Alfvén eigenmode) example for the model "LinearMHD".')
    
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