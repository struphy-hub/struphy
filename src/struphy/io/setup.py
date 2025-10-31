from dataclasses import dataclass

from psydac.ddm.mpi import mpi as MPI

from struphy.utils.arrays import xp as np
from struphy.utils.utils import dict_to_yaml


def derive_units(
    Z_bulk: int = None,
    A_bulk: int = None,
    x: float = 1.0,
    B: float = 1.0,
    n: float = 1.0,
    kBT: float = None,
    velocity_scale: str = "alfvén",
):
    """Computes units used in Struphy model's :ref:`normalization`.

    Input units from parameter file:

    * Length (m)
    * Magnetic field (T)
    * Number density (10^20 1/m^3)
    * Thermal energy (keV), optional

    Velocity unit is defined here:

    * Velocity (m/s)

    Derived units using mass and charge number of bulk species:

    * Time (s)
    * Pressure (Pa)
    * Mass density (kg/m^3)
    * Current density (A/m^2)

    Parameters
    ---------
    Z_bulk : int
        Charge number of bulk species.

    A_bulk : int
        Mass number of bulk species.

    x : float
        Unit of length (in meters).

    B : float
        Unit of magnetic field (in Tesla).

    n : float
        Unit of particle number density (in 1e20 per cubic meter).

    kBT : float
        Unit of internal energy (in keV). Only in effect if the velocity scale is set to 'thermal'.

    velocity_scale : str
        Velocity scale to be used ("alfvén", "cyclotron", "light" or "thermal").

    Returns
    -------
    units : dict
        The Struphy units defined above and some Physics constants.
    """

    units = {}

    # physics constants
    units["elementary charge"] = 1.602176634e-19  # elementary charge (C)
    units["proton mass"] = 1.67262192369e-27  # proton mass (kg)
    units["mu0"] = 1.25663706212e-6  # magnetic constant (N/A^2)
    units["eps0"] = 8.8541878128e-12  # vacuum permittivity (F/m)
    units["kB"] = 1.380649e-23  # Boltzmann constant (J/K)
    units["speed of light"] = 299792458  # speed of light (m/s)

    e = units["elementary charge"]
    mH = units["proton mass"]
    mu0 = units["mu0"]
    eps0 = units["eps0"]
    kB = units["kB"]
    c = units["speed of light"]

    # length (m)
    units["x"] = x
    # magnetic field (T)
    units["B"] = B
    # number density (1/m^3)
    units["n"] = n * 1e20

    # velocity (m/s)
    if velocity_scale is None:
        units["v"] = 1.0

    elif velocity_scale == "light":
        units["v"] = 1.0 * c

    elif velocity_scale == "alfvén":
        assert A_bulk is not None, 'Need bulk species to choose velocity scale "alfvén".'
        units["v"] = units["B"] / np.sqrt(units["n"] * A_bulk * mH * mu0)

    elif velocity_scale == "cyclotron":
        assert Z_bulk is not None, 'Need bulk species to choose velocity scale "cyclotron".'
        assert A_bulk is not None, 'Need bulk species to choose velocity scale "cyclotron".'
        units["v"] = Z_bulk * e * units["B"] / (A_bulk * mH) * units["x"]

    elif velocity_scale == "thermal":
        assert A_bulk is not None, 'Need bulk species to choose velocity scale "thermal".'
        assert kBT is not None
        units["v"] = np.sqrt(kBT * 1000 * e / (mH * A_bulk))

    # time (s)
    units["t"] = units["x"] / units["v"]
    if A_bulk is None:
        return units

    # pressure (Pa), equal to B^2/mu0 if velocity_scale='alfvén'
    units["p"] = A_bulk * mH * units["n"] * units["v"] ** 2

    # mass density (kg/m^3)
    units["rho"] = A_bulk * mH * units["n"]

    # current density (A/m^2)
    units["j"] = e * units["n"] * units["v"]

    return units


def setup_domain_and_equil(params: dict, units: dict = None):
    """
    Creates the domain object and equilibrium for a given parameter file.

    Parameters
    ----------
    params : dict
        The full simulation parameter dictionary.

    units : dict
        All Struphy units.

    Returns
    -------
    domain : Domain
        The Struphy domain object for evaluating the mapping F : [0, 1]^3 --> R^3 and the corresponding metric coefficients.

    equil : FluidEquilibrium
        The equilibrium object.
    """

    from struphy.fields_background import equils
    from struphy.fields_background.base import (
        NumericalFluidEquilibrium,
        NumericalFluidEquilibriumWithB,
        NumericalMHDequilibrium,
    )
    from struphy.geometry import domains

    if "fluid_background" in params:
        for eq_type, eq_params in params["fluid_background"].items():
            eq_class = getattr(equils, eq_type)
            if eq_type in ("EQDSKequilibrium", "GVECequilibrium", "DESCequilibrium"):
                equil = eq_class(**eq_params, units=units)
            else:
                equil = eq_class(**eq_params)

        # for numerical equilibria, the domain comes from the equilibrium
        if isinstance(equil, (NumericalMHDequilibrium, NumericalFluidEquilibrium, NumericalFluidEquilibriumWithB)):
            domain = equil.domain
        # for all other equilibria, the domain can be chosen idependently
        else:
            dom_type = params["geometry"]["type"]
            dom_class = getattr(domains, dom_type)

            if dom_type == "Tokamak":
                domain = dom_class(**params["geometry"][dom_type], equilibrium=equil)
            else:
                domain = dom_class(**params["geometry"][dom_type])

            # set domain attribute in mhd object
            equil.domain = domain

    # no equilibrium (just load domain)
    else:
        dom_type = params["geometry"]["type"]
        dom_class = getattr(domains, dom_type)
        domain = dom_class(**params["geometry"][dom_type])

        equil = None

    return domain, equil


def setup_derham(
    params_grid,
    comm=None,
    domain=None,
    mpi_dims_mask=None,
    verbose=False,
):
    """
    Creates the 3d derham sequence for given grid parameters.

    Parameters
    ----------
    params_grid : dict
        Grid parameters dictionary.

    comm: Intracomm
        MPI communicator (sub_comm if clones are used).

    domain : struphy.geometry.base.Domain, optional
        The Struphy domain object for evaluating the mapping F : [0, 1]^3 --> R^3 and the corresponding metric coefficients.

    mpi_dims_mask: list of bool
        True if the dimension is to be used in the domain decomposition (=default for each dimension).
        If mpi_dims_mask[i]=False, the i-th dimension will not be decomposed.

    verbose : bool
        Show info on screen.

    Returns
    -------
    derham : struphy.feec.psydac_derham.Derham
        Discrete de Rham sequence on the logical unit cube.
    """

    from struphy.feec.psydac_derham import Derham

    # number of grid cells
    Nel = params_grid["Nel"]
    # spline degrees
    p = params_grid["p"]
    # spline types (clamped vs. periodic)
    spl_kind = params_grid["spl_kind"]
    # boundary conditions (Homogeneous Dirichlet or None)
    dirichlet_bc = params_grid["dirichlet_bc"]
    # Number of quadrature points per histopolation cell
    nq_pr = params_grid["nq_pr"]
    # Number of quadrature points per grid cell for L^2
    nq_el = params_grid["nq_el"]
    # C^k smoothness at eta_1=0 for polar domains
    polar_ck = params_grid["polar_ck"]
    # local commuting projectors
    local_projectors = params_grid["local_projectors"]

    derham = Derham(
        Nel,
        p,
        spl_kind,
        dirichlet_bc=dirichlet_bc,
        nquads=nq_el,
        nq_pr=nq_pr,
        comm=comm,
        mpi_dims_mask=mpi_dims_mask,
        with_projectors=True,
        polar_ck=polar_ck,
        domain=domain,
        local_projectors=local_projectors,
    )

    if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
        print("\nDERHAM:")
        print(f"number of elements:".ljust(25), Nel)
        print(f"spline degrees:".ljust(25), p)
        print(f"periodic bcs:".ljust(25), spl_kind)
        print(f"hom. Dirichlet bc:".ljust(25), dirichlet_bc)
        print(f"GL quad pts (L2):".ljust(25), nq_el)
        print(f"GL quad pts (hist):".ljust(25), nq_pr)
        print(
            "MPI proc. per dir.:".ljust(25),
            derham.domain_decomposition.nprocs,
        )
        print("use polar splines:".ljust(25), derham.polar_ck == 1)
        print("domain on process 0:".ljust(25), derham.domain_array[0])

    return derham


def pre_processing(
    model_name: str,
    parameters: dict | str,
    path_out: str,
    restart: bool,
    max_sim_time: int,
    save_step: int,
    mpi_rank: int,
    mpi_size: int,
    use_mpi: bool,
    num_clones: int,
    verbose: bool = False,
):
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

    save_step : int
        When to save data output: every time step (save_step=1), every second time step (save_step=2).

    mpi_rank : int
        The rank of the calling process.

    mpi_size : int
        Total number of MPI processes of the run.

    use_mpi: bool
        True if MPI.COMM_WORLD is not None.

    num_clones: int
        Number of domain clones.

    verbose : bool
        Show full screen output.

    Returns
    -------
    params : dict
        The simulation parameters.
    """

    import datetime
    import glob
    import os
    import shutil
    import sysconfig

    import yaml

    from struphy.models import fluid, hybrid, kinetic, toy

    # prepare output folder
    if mpi_rank == 0:
        if verbose:
            print("\nPREPARATION AND CLEAN-UP:")

        # create output folder if it does not exit
        if not os.path.exists(path_out):
            os.makedirs(path_out, exist_ok=True)
            if verbose:
                print("Created folder " + path_out)

        # create data folder in output folder if it does not exist
        if not os.path.exists(os.path.join(path_out, "data/")):
            os.mkdir(os.path.join(path_out, "data/"))
            if verbose:
                print("Created folder " + os.path.join(path_out, "data/"))

        # clean output folder if it already exists
        else:
            # remove post_processing folder
            folder = os.path.join(path_out, "post_processing")
            if os.path.exists(folder):
                shutil.rmtree(folder)
                if verbose:
                    print("Removed existing folder " + folder)

            # remove meta file
            file = os.path.join(path_out, "meta.txt")
            if os.path.exists(file):
                os.remove(file)
                if verbose:
                    print("Removed existing file " + file)

            # remove profiling file
            file = os.path.join(path_out, "profile_tmp")
            if os.path.exists(file):
                os.remove(file)
                if verbose:
                    print("Removed existing file " + file)

            # remove .png files (if NOT a restart)
            if not restart:
                files = glob.glob(os.path.join(path_out, "*.png"))
                for n, file in enumerate(files):
                    os.remove(file)
                    if verbose and n < 10:  # print only ten statements in case of many processes
                        print("Removed existing file " + file)

                files = glob.glob(os.path.join(path_out, "data", "*.hdf5"))
                for n, file in enumerate(files):
                    os.remove(file)
                    if verbose and n < 10:  # print only ten statements in case of many processes
                        print("Removed existing file " + file)

    # save "parameters" dictionary as .yml file
    if isinstance(parameters, dict):
        parameters_path = os.path.join(path_out, "parameters.yml")

        # write parameters to file and save it in output folder
        if mpi_rank == 0:
            dict_to_yaml(parameters, parameters_path)

        params = parameters

    # OR load parameters if "parameters" is a string (path)
    else:
        parameters_path = parameters

        with open(parameters) as file:
            params = yaml.load(file, Loader=yaml.FullLoader)

    if model_name is None:
        assert "model" in params, "If model is not specified, then model: MODEL must be specified in the params!"
        model_name = params["model"]

    if mpi_rank == 0:
        # copy parameter file to output folder
        if parameters_path != os.path.join(path_out, "parameters.yml"):
            shutil.copy2(
                parameters_path,
                os.path.join(
                    path_out,
                    "parameters.yml",
                ),
            )

        # print simulation info
        print("\nMETADATA:")
        print("platform:".ljust(25), sysconfig.get_platform())
        print("python version:".ljust(25), sysconfig.get_python_version())
        print("model:".ljust(25), model_name)
        print("MPI processes:".ljust(25), mpi_size)
        print("use MPI.COMM_WORLD:".ljust(25), use_mpi)
        print("number of domain clones:".ljust(25), num_clones)
        print("parameter file:".ljust(25), parameters_path)
        print("output folder:".ljust(25), path_out)
        print("restart:".ljust(25), restart)
        print("max wall-clock [min]:".ljust(25), max_sim_time)
        print("save interval [steps]:".ljust(25), save_step)

        # write meta data to output folder
        with open(path_out + "/meta.txt", "w") as f:
            f.write(
                "date of simulation: ".ljust(
                    30,
                )
                + str(datetime.datetime.now())
                + "\n",
            )
            f.write("platform: ".ljust(30) + sysconfig.get_platform() + "\n")
            f.write(
                "python version: ".ljust(
                    30,
                )
                + sysconfig.get_python_version()
                + "\n",
            )
            f.write("model_name: ".ljust(30) + model_name + "\n")
            f.write("processes: ".ljust(30) + str(mpi_size) + "\n")
            f.write("use MPI.COMM_WORLD: ".ljust(30) + str(use_mpi) + "\n")
            f.write("output folder:".ljust(30) + path_out + "\n")
            f.write("restart:".ljust(30) + str(restart) + "\n")
            f.write(
                "max wall-clock time [min]:".ljust(30) + str(max_sim_time) + "\n",
            )
            f.write("save interval (steps):".ljust(30) + str(save_step) + "\n")

    return params


def descend_options_dict(
    d: dict,
    out: list | dict,
    *,
    d_default: dict = None,
    d_opts: dict = None,
    keys: list = None,
    depth: int = 0,
    pop_again: bool = False,
    verbose: bool = False,
):
    """Create all possible parameter dicts from a model options dict,
    by looping through options.

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
        Whether to pop one more time from keys; this is automatically set to True when depth is reached during recursion.

    verbose : bool
        Show some output on screen.
    """

    import copy

    # set d_opts, keys and depth at first call
    if d_opts is None:
        assert out == []
        d_opts = d.copy()
        keys = []
        depth = len(d)

        if d_default is None:
            out = copy.deepcopy(d)

    if verbose:
        print(f"{d = }")
        print(f"{out = }")
        print(f"{d_default = }")
        print(f"{d_opts = }")
        print(f"{keys = }")
        print(f"{depth = }")
        print(f"{pop_again = }")

    if verbose:
        print(f"{d = }")
        print(f"{out = }")
        print(f"{d_default = }")
        print(f"{d_opts = }")
        print(f"{keys = }")
        print(f"{depth = }")
        print(f"{pop_again = }")

    count = 0
    for key, val in d.items():
        count += 1

        if verbose:
            print(f"\n{keys = } | {key = }, {type(val) = }, {count = }\n")

        if isinstance(val, list):
            # create default parameter dict "out"

            if verbose:
                print(f"{val = }")

            if d_default is None:
                if len(keys) == 0:
                    out[key] = val[0]
                elif len(keys) == 1:
                    out[keys[0]][key] = val[0]
                elif len(keys) == 2:
                    out[keys[0]][keys[1]][key] = val[0]
                else:
                    raise ValueError(
                        f"Depth of options dictionary must not exceed 3, but is {len(keys) + 1}.",
                    )

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
                            f"Depth of options dictionary must not exceed 3, but is {len(keys) + 1}.",
                        )
                    out_sublist += [d_copy]
                out += [out_sublist]

            if verbose:
                print(f"{out = }")

            if verbose:
                print(f"{out = }")

        # recurse if necessary
        elif isinstance(val, dict):
            if count == depth and len(keys) > 0:
                pop_again = True
            keys += [key]
            descend_options_dict(
                val,
                out,
                d_opts=d_opts,
                keys=keys,
                depth=len(val),
                pop_again=pop_again,
                d_default=d_default,
                verbose=verbose,
            )

        else:
            pass

    if len(keys) > 0:
        keys.pop()
        if pop_again:
            keys.pop()
            pop_again = False

    if d_default is None:
        return out
