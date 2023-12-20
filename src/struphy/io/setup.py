import numpy as np


def derive_units(Z_bulk=1, A_bulk=1., x=1., B=1., n=1., velocity_scale='alfvén'):
    """ Computes Struphy units used in Struphy model implementations.

    Input units from parameter file:

        * Length (m)
        * Magnetic field (T)
        * number density (10^20 1/m^3)

    Velocity unit must be defined in each model as one of "light", "alfvén" or "cyclotron":

        * Velocity (m/s)

    Derived units using mass and charge number of bulk species:

        * Time (s)
        * Mass density (kg/m^3)
        * Pressure (Pa)

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

    velocity_scale : str
        Velocity scale to be used ("alfvén", "cyclotron" or "light").

    Returns
    -------
    units : dict
        Basic units for time, length, mass and magnetic field.

    units : dict
        Derived units for velocity, pressure, mass density and particle density.
    """

    # physics constants
    e = 1.602176634e-19  # elementary charge (C)
    mH = 1.67262192369e-27  # proton mass (kg)
    mu0 = 1.25663706212e-6  # magnetic constant (N/A^2)
    eps0 = 8.8541878128e-12  # vacuum permittivity (F/m)
    kB = 1.380649e-23  # Boltzmann constant (J/K)
    c = 299792458  # speed of light (m/s)

    units = {}

    # length (m)
    units['x'] = x
    # magnetic field (T)
    units['B'] = B
    # number density (1/m^3)
    units['n'] = n * 1e20
    # velocity (m/s)
    if velocity_scale == 'light':
        units['v'] = 1*c
    elif velocity_scale == 'alfvén':
        units['v'] = units['B'] / np.sqrt(units['n'] * A_bulk * mH * mu0)
    elif velocity_scale == 'cyclotron':
        units['v'] = Z_bulk * e * units['B'] / \
            (A_bulk * mH) / (2*np.pi) * units['x']
    # time (s)
    units['t'] = units['x'] / units['v']
    # pressure (Pa)
    units['p'] = A_bulk * mH * units['n'] * units['x']**3 / \
        (units['x'] * units['t']**2)  # this is equal to B^2/(mu0*n) if velocity_scale='alfvén'
    # mass density (kg/m^3)
    units['rho'] = A_bulk * mH * units['n']

    return units


def setup_domain_mhd(params, units=None):
    """
    Creates the domain object and MHD equilibrium for a given parameter file.

    Parameters
    ----------
    params : dict
        The full simulation parameter dictionary.

    units : dict
        All Struphy units.

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

        if mhd_type == 'EQDSKequilibrium':
            mhd = mhd_class(units=units, **params['mhd_equilibrium'][mhd_type])
        else:
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
    derham : struphy.feec.psydac_derham.Derham
        Discrete de Rham sequence on the logical unit cube.
    """

    from struphy.feec.psydac_derham import Derham

    # number of grid cells
    Nel = params_grid['Nel']
    # spline degrees
    p = params_grid['p']
    # spline types (clamped vs. periodic)
    spl_kind = params_grid['spl_kind']
    # boundary conditions (Homogeneous Dirichlet or None)
    dirichlet_bc = params_grid['dirichlet_bc']
    # Number of quadrature points per histopolation cell
    nq_pr = params_grid['nq_pr']
    # Number of quadrature points per grid cell for L^2
    nq_el = params_grid['nq_el']
    # C^k smoothness at eta_1=0 for polar domains
    polar_ck = params_grid['polar_ck']

    derham = Derham(Nel, 
                    p, 
                    spl_kind, 
                    dirichlet_bc,
                    nquads=nq_el,
                    nq_pr=nq_pr,
                    comm=comm,
                    mpi_dims_mask=mpi_dims_mask,
                    with_projectors=True,
                    polar_ck=polar_ck,
                    domain=domain)

    if comm.Get_rank() == 0:
        print('\nDERHAM:')
        print(f'number of elements:'.ljust(25), Nel)
        print(f'spline degrees:'.ljust(25), p)
        print(f'periodic bcs:'.ljust(25), spl_kind)
        print(f'hom. Dirichlet bc:'.ljust(25), dirichlet_bc)
        print(f'GL quad pts (L2):'.ljust(25), nq_el)
        print(f'GL quad pts (hist):'.ljust(25), nq_pr)
        print('MPI proc. per dir.:'.ljust(25),
              derham.domain_decomposition.nprocs)
        print('use polar splines:'.ljust(25), derham.polar_ck == 1)
        print('domain on process 0:'.ljust(25), derham.domain_array[0])

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

    Returns
    -------
    params : dict
        The simulation parameters.
    """

    import os
    import shutil
    import datetime
    import sysconfig
    import glob
    import yaml
    from struphy.models import fluid, kinetic, hybrid, toy

    # prepare output folder
    if mpi_rank == 0:
        print('\nPREPARATION AND CLEAN-UP:')

        # create output folder if it does not exit
        if not os.path.exists(path_out):
            os.mkdir(path_out)
            print('Created folder ' + path_out)

        # create data folder in output folder if it does not exist
        if not os.path.exists(os.path.join(path_out, 'data/')):
            os.mkdir(os.path.join(path_out, 'data/'))
            print('Created folder ' + os.path.join(path_out, 'data/'))

        # clean output folder if it already exists
        else:

            # remove post_processing folder
            folder = os.path.join(path_out, 'post_processing')
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print('Removed existing folder ' + folder)

            # remove meta file
            file = os.path.join(path_out, 'meta.txt')
            if os.path.exists(file):
                os.remove(file)
                print('Removed existing file ' + file)

            # remove profiling file
            file = os.path.join(path_out, 'profile_tmp')
            if os.path.exists(file):
                os.remove(file)
                print('Removed existing file ' + file)

            # remove .png files (if NOT a restart)
            if not restart:
                files = glob.glob(os.path.join(path_out, '*.png'))
                for n, file in enumerate(files):
                    os.remove(file)
                    if n < 10:  # print only ten statements in case of many processes
                        print('Removed existing file ' + file)

                files = glob.glob(os.path.join(path_out, 'data', '*.hdf5'))
                for n, file in enumerate(files):
                    os.remove(file)
                    if n < 10:  # print only ten statements in case of many processes
                        print('Removed existing file ' + file)

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
        print('\nMETADATA:')
        print('platform:'.ljust(25), sysconfig.get_platform())
        print('python version:'.ljust(25), sysconfig.get_python_version())
        print('model:'.ljust(25), model_name)
        print('MPI processes:'.ljust(25), mpi_size)
        print('parameter file:'.ljust(25), parameters_path)
        print('output folder:'.ljust(25), path_out)
        print('restart:'.ljust(25), restart)
        print('max wall-clock [min]:'.ljust(25), max_sim_time)

        # print time info
        print('\nTIME:')
        print(f'time step:'.ljust(25), params['time']['dt'])
        print(f'final time:'.ljust(25), params['time']['Tend'])
        print(f'splitting algo:'.ljust(25), params['time']['split_algo'])

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


def descend_options_dict(d, out, d_default=None, d_opts=None, keys=None, depth=0, pop_again=False):
    '''Prepare parameter sub-dicts from model options dict.

    If d_default=None, will return the default parameter dict of a model
    (takes first list entries of options dict).

    Otherwise, will go through all sub-dicts of the options dict recursively 
    and check whether a value is a list (i.e. different options are available). 
    If True, creates one parameter dict for each value in the list,
    with all other parameters set to their defaults.

    Parameters
    ----------
    d : dict
        The (sub)-dict to investigate.

    out : list or dict
        The ouptut, must be passed as empty list. During recursion, if
        list: Holds one parameter dict for each option. If dict: the default parameters. 
        
    d_default : dict
        The default parameter dict of the model. 
        If passed as None, the default parameter dict will be returned.
        
    d_opts : dict
        A copy of "d" created at first call (when d_opts is None).

    keys : list
        The keys to the options in the options dict. The last entry is the lowest-level key.
        This list is filled automatically during recursion.
        
    depth : int
        The length of d from the previous recursion. 
        
    pop_again : bool
        Whether to pop one more time from keys; this is automatically set to True when depth is reached during recursion.'''

    import copy

    # set d_opts, keys and depth at first call
    if d_opts is None:
        assert out == []
        d_opts = d.copy()
        keys = []
        depth = len(d)

        if d_default is None:
            out = copy.deepcopy(d)

    count = 0
    for key, val in d.items():
        count += 1

        if isinstance(val, list):

            # create default parameter dict "out"
            if d_default is None:
                if len(keys) == 0:
                    out[key] = val[0]
                elif len(keys) == 1:
                    out[keys[0]][key] = val[0]
                elif len(keys) == 2:
                    out[keys[0]][keys[1]][key] = val[0]
                else:
                    raise ValueError(
                        f'Depth of options dictionary must not exceed 3, but is {len(keys) + 1}.')

            # add one parameter dict for each option in the list
            else:
                out_sublist = []
                for param in val:
                    
                    # exclude solvers without preconditioner
                    if isinstance(param, tuple):
                        if param[1] is None:
                            continue
                    
                    d_copy = copy.deepcopy(d_default)
                    if len(keys) == 0:
                        d_copy[key] = param
                    elif len(keys) == 1:
                        d_copy[keys[0]][key] = param
                    elif len(keys) == 2:
                        d_copy[keys[0]][keys[1]][key] = param
                    else:
                        raise ValueError(
                            f'Depth of options dictionary must not exceed 3, but is {len(keys) + 1}.')
                    out_sublist += [d_copy]
                out += [out_sublist]

        # recurse if necessary
        elif isinstance(val, dict):
            if count == depth and len(keys) > 0:
                pop_again = True
            keys += [key]
            descend_options_dict(val, out, d_opts=d_opts,
                                 keys=keys, depth=len(val), pop_again=pop_again, d_default=d_default)

        else:
            pass
        
    if len(keys) > 0:    
        keys.pop()
        if pop_again:
            keys.pop()
            pop_again = False

    if d_default is None:
        return out
