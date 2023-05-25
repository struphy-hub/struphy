import numpy as np


def derive_units(Z_bulk=1, A_bulk=1., x=1., B=1., n=1., time_scale='alfvén'):
    """
    Computes derived physics units of Struphy quantities.

    Parameters
    ---------
    Z_bulk : int
        Charge number of bulkd species.

    A_bulk : int
        Mass number of bulk species.

    x : float
        Unit of length (in meters).

    B : float
        Unit of magnetic field (in Tesla).

    n : float
        Unit of particle number density (in 1e20 per cubic meter).

    time_scale : str
        Time scale to be used (determined by some characteristic velocity: "alfvén", "cyclotron" or "light").

    Returns
    -------
    units_basic : dict
        Basic units for time, length, mass and magnetic field.

    units_der : dict
        Derived units for velocity, pressure, mass density and particle density.

    units_dimless :  dict
        Some dimensionless quantities:
            * alpha = omega_p / omega_c, ratio of bulk plasma to bulk cyclotron frequency.
    """

    # physics constants
    e = 1.602176634e-19  # elementary charge (C)
    mH = 1.67262192369e-27  # proton mass (kg)
    mu0 = 1.25663706212e-6  # magnetic constant (N/A^2)
    eps0 = 8.8541878128e-12  # vacuum permittivity (F/m)
    kB = 1.380649e-23  # Boltzmann constant (J/K)
    c = 299792458  # speed of light (m/s)

    # prescribed units
    x_unit = x * 1
    B_unit = B * 1
    n_unit = n * 1e20

    # basic units in SI units (time, length, particle density and magnetic field)
    units_basic = {}

    if time_scale == 'light':
        v_unit = 1*c
    elif time_scale == 'alfvén':
        v_unit = B_unit / np.sqrt(n_unit * A_bulk * mH * mu0)
    elif time_scale == 'cyclotron':
        v_unit = Z_bulk * e * B_unit / (A_bulk * mH) / (2*np.pi) * x_unit

    units_basic['t'] = x_unit / v_unit
    units_basic['x'] = x_unit
    units_basic['m'] = A_bulk * mH * n_unit * x_unit**3
    units_basic['B'] = B_unit

    # derived units
    units_der = {}

    units_der['v'] = units_basic['x'] / units_basic['t']
    units_der['p'] = units_basic['m'] / \
        (units_basic['x'] * units_basic['t']**2)
    units_der['rho'] = units_basic['m'] / units_basic['x']**3
    units_der['n'] = units_basic['m'] / units_basic['x']**3 / (A_bulk * mH)

    # relevant dimensionless quantities
    units_dimless = {}

    # unit of bulk plasma frequency
    omega_p = np.sqrt(n_unit * (Z_bulk * e)**2 / (eps0 * A_bulk * mH))

    # unit of bulk cyclotron frequency
    omega_c = Z_bulk * e * B_unit / (A_bulk * mH)

    # relevant unit parameters
    units_dimless['alpha'] = omega_p / omega_c
    units_dimless['kappa'] = omega_c * units_basic['t']

    return units_basic, units_der, units_dimless


def setup_domain_mhd(params):
    """
    Creates the domain object and MHD equilibrium for a given parameter file.

    Parameters
    ----------
    params : dict
        The full simulation parameter dictionary.

    Returns
    -------
    domain : struphy.geometry.base.Domain
        The Struphy domain object for evaluating the mapping F : [0, 1]^3 --> R^3 and the corresponding metric coefficients.

    mhd : struphy.fields_background.base.MHDequilibrium
        The ideal MHD equilibrium object.
    """

    from struphy.geometry import domains
    from struphy.fields_background.mhd_equil import equils
    from struphy.fields_background.mhd_equil.base import LogicalMHDequilibrium

    # MHD equilibrium given (load equilibrium first, then set domain)
    if 'mhd_equilibrium' in params:

        mhd_type = params['mhd_equilibrium']['type']
        mhd_class = getattr(equils, mhd_type)
        mhd = mhd_class(**params['mhd_equilibrium'][mhd_type])

        # for logical MHD equilibria, the domain comes with the equilibrium
        if isinstance(mhd, LogicalMHDequilibrium):
            domain = mhd.domain

        # for cartesian MHD equilibria, the domain can be chosen idependently
        else:
            dom_type = params['geometry']['type']
            dom_class = getattr(domains, dom_type)

            if dom_type == 'Tokamak':
                domain = dom_class(
                    **params['geometry'][dom_type], equilibrium=mhd)
            else:
                domain = dom_class(**params['geometry'][dom_type])

            # set domain attribute in mhd object
            mhd.domain = domain

    # no MHD equilibrium (load domain)
    else:

        dom_type = params['geometry']['type']
        dom_class = getattr(domains, dom_type)
        domain = dom_class(**params['geometry'][dom_type])

        mhd = None

    return domain, mhd


def setup_electric_background(params, domain):
    """
    Creates an electric background field for a given parameter file.

    Parameters
    ----------
    params : dict
        The full simulation parameter dictionary.

    domain : struphy.geometry.base.Domain
        The Struphy domain object for evaluating the mapping F : [0, 1]^3 --> R^3 and the corresponding metric coefficients.

    Returns
    -------
    electric_background : struphy.fields_background.electric_equil.base.EquilibriumElectric
        The electric background object.
    """

    from struphy.fields_background.electric_equil import analytical

    if 'electric_equilibrium' in params:

        electric_type = params['electric_equilibrium']['type']
        electric_class = getattr(analytical, electric_type)
        electric_background = electric_class(
            params['electric_equilibrium'][electric_type], domain)

    else:
        electric_background = None

    return electric_background


def setup_derham(params_grid, comm, domain=None, mpi_dims_mask=None):
    """
    Creates the 3d derham sequence for given grid parameters.

    Parameters
    ----------
    params_grid : dict
        Grid parameters dictionary.

    comm : mpi4py.MPI.Intracomm
        MPI communicator used for parallelization.

    domain : struphy.geometry.base.Domain, optional
        The Struphy domain object for evaluating the mapping F : [0, 1]^3 --> R^3 and the corresponding metric coefficients.

    mpi_dims_mask: list of bool
        True if the dimension is to be used in the domain decomposition (=default for each dimension). 
        If mpi_dims_mask[i]=False, the i-th dimension will not be decomposed.

    Returns
    -------
    derham : struphy.psydac_api.psydac_derham.Derham
        Discrete de Rham sequence on the logical unit cube.
    """

    from struphy.psydac_api.psydac_derham import Derham

    # number of grid cells
    Nel = params_grid['Nel']
    # spline degrees
    p = params_grid['p']
    # spline types (clamped vs. periodic)
    spl_kind = params_grid['spl_kind']
    # boundary conditions (Homogeneous Dirichlet or None)
    bc = params_grid['bc']
    # Number of quadrature points per histopolation cell
    nq_pr = params_grid['nq_pr']
    # Number of quadrature points per grid cell for L^2
    nq_el = params_grid['nq_el']
    # C^k smoothness at eta_1=0 for polar domains
    polar_ck = params_grid['polar_ck']

    quad_order = [nq_el[0] - 1,
                  nq_el[1] - 1,
                  nq_el[2] - 1]

    derham = Derham(Nel, p, spl_kind, bc,
                    quad_order=quad_order,
                    nq_pr=nq_pr,
                    comm=comm,
                    mpi_dims_mask=mpi_dims_mask,
                    with_projectors=True,
                    polar_ck=polar_ck,
                    domain=domain)

    if comm.Get_rank() == 0:
        print('MPI processes per direction:',
              derham.domain_decomposition.nprocs)
        print('')

    return derham


def pre_processing(model_name, parameters, path_out, restart, max_sim_time, mpi_rank, mpi_size):
    """
    Prepares simulation parameters, output folder and prints some information of the run to the screen. 

    Parameters
    ----------
    model_name : str
        The name of the model to run.

    parameters : dict | str
        The simulation parameters. Can either be a dictionary OR a string (path of .yml parameter file)

    path_out : str
        The output directory. Will create a folder if it does not exist OR cleans the folder for new runs.

    restart : bool
        Whether to restart a run.

    max_sim_time : int
        Maximum run time of simulation in minutes. Will finish the time integration once this limit is reached.

    mpi_rank : int
        The rank of the calling process.

    mpi_size : int
        Total number of MPI processes of the run.
    """

    import os
    import shutil
    import datetime
    import sysconfig
    import glob
    import yaml
    from struphy.models import models

    # prepare output folder
    if mpi_rank == 0:
        print('')

        # create output folder if it does not exit
        if not os.path.exists(path_out):
            os.mkdir(path_out)
            print('\nCreated folder ' + path_out)
            os.mkdir(os.path.join(path_out, 'data/'))
            print('\nCreated folder ' + os.path.join(path_out, 'data/'))

        # clean output folder if it already exists
        else:

            # remove post_processing folder
            folder = os.path.join(path_out, 'post_processing')
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print('Removed folder ' + folder)

            # remove data folder and create new one
            folder = os.path.join(path_out, 'data/')
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.mkdir(folder)
            print('(Re)-created folder ' + folder)

            # remove meta file
            file = os.path.join(path_out, 'meta.txt')
            if os.path.exists(file):
                os.remove(file)
                print('Removed file ' + file)

            # remove profiling file
            file = os.path.join(path_out, 'profile_tmp')
            if os.path.exists(file):
                os.remove(file)
                print('Removed file ' + file)

            # remove .png files (if NOT a restart)
            if not restart:
                files = glob.glob(os.path.join(path_out, '*.png'))
                for n, file in enumerate(files):
                    os.remove(file)
                    if n < 10:  # print only forty statements in case of many processes
                        print('Removed file ' + file)

    # save "parameters" dictionary as .yml file
    if isinstance(parameters, dict):
        parameters_path = os.path.join(path_out, 'parameters.yml')

        # write parameters to file and save it in output folder
        if mpi_rank == 0:
            params_file = open(parameters_path, 'w')
            yaml.dump(parameters, params_file)
            params_file.close()

        params = parameters

    # OR load parameters if "parameters" is a string (path)
    else:
        parameters_path = parameters

        with open(parameters) as file:
            params = yaml.load(file, Loader=yaml.FullLoader)

    if mpi_rank == 0:

        # copy parameter file to output folder
        if parameters_path != os.path.join(path_out, 'parameters.yml'):
            shutil.copy2(parameters_path, os.path.join(
                path_out, 'parameters.yml'))

        # print simulation info
        print('')
        print('model:'.ljust(30), model_name)
        print('parameter file:'.ljust(30), parameters_path)
        print('output folder:'.ljust(30), path_out)
        print('restart:'.ljust(30), restart)
        print('max wall-clock time [min]:'.ljust(30), max_sim_time)
        print('number of MPI processes:'.ljust(30), mpi_size)

        # print domain info
        print('\nDOMAIN parameters:')
        print(f'domain type :', params['geometry']['type'])
        print(f'domain parameters :')
        for key, val in params['geometry'][params['geometry']['type']].items():
            if key not in {'cx', 'cy', 'cz'}:
                print(key, ':', val)

        # print grid info
        print('\nGRID parameters:')
        print(f'number of elements  :', params['grid']['Nel'])
        print(f'spline degrees      :', params['grid']['p'])
        print(f'periodic bcs        :', params['grid']['spl_kind'])
        print(f'hom. Dirichlet bc   :', params['grid']['bc'])
        print(f'GL quad pts (L2)    :', params['grid']['nq_el'])
        print(f'GL quad pts (hist)  :', params['grid']['nq_pr'])

        # print units info
        getattr(models, model_name).model_units(params, verbose=True)
        print('')

        # write meta data to output folder
        with open(path_out + '/meta.txt', 'w') as f:
            f.write('date of simulation: '.ljust(30) +
                    str(datetime.datetime.now()) + '\n')
            f.write('platform: '.ljust(30) + sysconfig.get_platform() + '\n')
            f.write('python version: '.ljust(30) +
                    sysconfig.get_python_version() + '\n')
            f.write('model_name: '.ljust(30) + model_name + '\n')
            f.write('processes: '.ljust(30) + str(mpi_size) + '\n')
            f.write('output folder:'.ljust(30) + path_out + '\n')
            f.write('restart:'.ljust(30) + str(restart) + '\n')
            f.write(
                'max wall-clock time [min]:'.ljust(30) + str(max_sim_time) + '\n')

    return params
