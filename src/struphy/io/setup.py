import importlib.util
import sys
import glob
import os
import shutil
import yaml

from mpi4py import MPI

from struphy.io.options import DerhamOptions
from struphy.io.parameters import StruphyParameters
from struphy.topology.grids import TensorProductGrid
from struphy.utils.utils import dict_to_yaml, read_state
from struphy.geometry.base import Domain
from struphy.geometry.domains import Cuboid
from struphy.io.options import Units, Time


def import_parameters_py(params_path: str):
    """Import a .py parameter file under the module name 'parameters' and return it."""
    assert ".py" in params_path
    spec = importlib.util.spec_from_file_location("parameters", params_path)
    params_in = importlib.util.module_from_spec(spec)
    sys.modules["parameters"] = params_in
    spec.loader.exec_module(params_in)
    return params_in


def setup_folders(
    path_out: str,
    restart: bool,
    verbose: bool = False,
):
    """
    Setup output folders.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
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


def setup_parameters(
    params_path: str,
    path_out: str,
    verbose: bool = False,
):
    """
    Prepare simulation parameters from .yml or .py file and save to output folder.

    Parameters
    ----------
    model_name : str
        The name of the model to run.

    params_path : str
        Path to .py parameter file.

    path_out : str
        The output directory. Will create a folder if it does not exist OR cleans the folder for new runs.

    verbose : bool
        Show full screen output.

    Returns
    -------
    params : StruphyParameters
        The simulation parameters.
    """

    if ".yml" in params_path or ".yaml" in params_path:
        with open(params_path) as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
    elif ".py" in params_path:
        # print(f'{params_path = }')
        # Read struphy state file
        # state = read_state()
        # i_path = state["i_path"]
        # load parameter.py
        params_in = import_parameters_py(params_path)

        if not hasattr(params_in, "model"):
            params_in.model = None

        if not hasattr(params_in, "domain"):
            params_in.domain = Cuboid()

        if not hasattr(params_in, "grid"):
            params_in.grid = None

        if not hasattr(params_in, "equil"):
            params_in.equil = None

        if not hasattr(params_in, "units"):
            params_in.units = Units()

        if not hasattr(params_in, "time"):
            params_in.time = Time()

        if not hasattr(params_in, "derham"):
            params_in.derham = None

        params = StruphyParameters(
            model=params_in.model,
            units=params_in.units,
            domain=params_in.domain,
            equil=params_in.equil,
            time=params_in.time,
            grid=params_in.grid,
            derham=params_in.derham,
            verbose=verbose,
        )

    if MPI.COMM_WORLD.Get_rank() == 0:
        # copy parameter file to output folder
        filename = params_path.split("/")[-1]
        ext = filename.split(".")[-1]
        shutil.copy2(
            params_path,
            os.path.join(path_out, "parameters." + ext),
        )

    return params


def setup_derham(
    grid: TensorProductGrid,
    options: DerhamOptions,
    comm: MPI.Intracomm = None,
    domain: Domain = None,
    verbose=False,
):
    """
    Creates the 3d derham sequence for given grid parameters.

    Parameters
    ----------
    grid : TensorProductGrid
        The FEEC grid.

    comm: Intracomm
        MPI communicator (sub_comm if clones are used).

    domain : Domain, optional
        The Struphy domain object for evaluating the mapping F : [0, 1]^3 --> R^3 and the corresponding metric coefficients.

    verbose : bool
        Show info on screen.

    Returns
    -------
    derham : struphy.feec.psydac_derham.Derham
        Discrete de Rham sequence on the logical unit cube.
    """

    from struphy.feec.psydac_derham import Derham

    # number of grid cells
    Nel = grid.Nel
    # mpi
    mpi_dims_mask = grid.mpi_dims_mask
    
    # spline degrees
    p = options.p
    # spline types (clamped vs. periodic)
    spl_kind = options.spl_kind
    # boundary conditions (Homogeneous Dirichlet or None)
    dirichlet_bc = options.dirichlet_bc
    # Number of quadrature points per histopolation cell
    nq_pr = options.nq_pr
    # Number of quadrature points per grid cell for L^2
    nquads = options.nquads
    # C^k smoothness at eta_1=0 for polar domains
    polar_ck = options.polar_ck
    # local commuting projectors
    local_projectors = options.local_projectors

    derham = Derham(
        Nel,
        p,
        spl_kind,
        dirichlet_bc=dirichlet_bc,
        nquads=nquads,
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
        print(f"GL quad pts (L2):".ljust(25), nquads)
        print(f"GL quad pts (hist):".ljust(25), nq_pr)
        print(
            "MPI proc. per dir.:".ljust(25),
            derham.domain_decomposition.nprocs,
        )
        print("use polar splines:".ljust(25), derham.polar_ck == 1)
        print("domain on process 0:".ljust(25), derham.domain_array[0])

    return derham


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
