import glob
import importlib.util
import logging
import os
import shutil
import sys
from types import ModuleType

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy.geometry.base import Domain
from struphy.io.options import DerhamOptions
from struphy.topology.grids import TensorProductGrid

logger = logging.getLogger("struphy")


def import_parameters_py(params_path: str, name: str = "parameters") -> ModuleType:
    """Import a .py parameter file under the given module name and return it.

    The parameter file at ``params_path`` is loaded as a module using the
    provided ``name``, which is also used as the key in ``sys.modules``.
    By default, the module is registered under the name ``"parameters"``.
    """
    assert ".py" in params_path
    spec = importlib.util.spec_from_file_location(name, params_path)
    params_in = importlib.util.module_from_spec(spec)
    sys.modules[name] = params_in
    spec.loader.exec_module(params_in)
    return params_in


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
        logger.info(f"{d =}")
        logger.info(f"{out =}")
        logger.info(f"{d_default =}")
        logger.info(f"{d_opts =}")
        logger.info(f"{keys =}")
        logger.info(f"{depth =}")
        logger.info(f"{pop_again =}")

    if verbose:
        logger.info(f"{d =}")
        logger.info(f"{out =}")
        logger.info(f"{d_default =}")
        logger.info(f"{d_opts =}")
        logger.info(f"{keys =}")
        logger.info(f"{depth =}")
        logger.info(f"{pop_again =}")

    count = 0
    for key, val in d.items():
        count += 1

        if verbose:
            logger.info(f"\n{keys =} | {key =}, {type(val) =}, {count =}\n")

        if isinstance(val, list):
            # create default parameter dict "out"

            if verbose:
                logger.info(f"{val =}")

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
                logger.info(f"{out =}")

            if verbose:
                logger.info(f"{out =}")

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
