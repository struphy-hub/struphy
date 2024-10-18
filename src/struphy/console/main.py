#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
import importlib.metadata
import os
import pickle
import sys
from argparse import HelpFormatter, RawTextHelpFormatter, _SubParsersAction

import argcomplete
import yaml


def prepare_messages(libpath, i_path, o_path, b_path):
    """Prepare version and path messages."""

    __version__ = importlib.metadata.version("struphy")
    # version message
    version_message = f"Struphy {__version__}\n"
    version_message += "Copyright 2019-2024 (c) Struphy dev team | Max Planck Institute for Plasma Physics\n"
    version_message += "MIT license\n"

    # path message
    path_message = f"Struphy installation path: {libpath}\n"
    path_message += f"current input:             {i_path}\n"
    path_message += f"current output:            {o_path}\n"
    path_message += f"current batch scripts:     {b_path}"

    return version_message, path_message


def load_model_info(libpath):
    """Load available models and their messages."""
    try:
        with open(os.path.join(libpath, "models", "models_list"), "rb") as fp:
            list_models = pickle.load(fp)
        with open(os.path.join(libpath, "models", "models_message"), "rb") as fp:
            model_messages = pickle.load(fp)
    except Exception:
        list_models = []
        model_messages = ["", "", "", "", ""]
    
    # model_message, 
    fluid_message, kinetic_message, hybrid_message, toy_message = tuple(model_messages)
    
    model_messages_dict = {
        'model_message':'model_message',
        'fluid_message':fluid_message,
        'kinetic_message':kinetic_message,
        'hybrid_message':hybrid_message,
        'toy_message':toy_message,
    }
    
    return list_models, model_messages_dict


def create_argument_parser(
    version_message,
    path_message,
    list_models,
    model_messages_dict,
    params_files,
    out_folders,
    batch_files,
):
    """Create the main argument parser with all subparsers."""
    epilog_message = (
        'Type "struphy COMMAND --help" for more information on a command.\n\n'
        "For more help on how to use Struphy, see https://struphy.pages.mpcdf.de/struphy/index.html"
    )
    parser = argparse.ArgumentParser(
        prog="struphy",
        formatter_class=CustomFormatter,
        description="Struphy: STRUcture-Preserving HYbrid codes for plasma physics.",
        epilog=epilog_message,
    )

    # Add basic options
    add_basic_options(parser, version_message, path_message)

    # Create subparsers
    subparsers = parser.add_subparsers(
        title="available commands", metavar="COMMAND", dest="command"
    )
    print(model_messages_dict.keys())
    # Add sub-commands
    add_compile_subparser(subparsers)
    add_run_subparser(subparsers, list_models, model_messages_dict['model_message'], params_files, batch_files)
    add_units_subparser(subparsers, list_models, model_messages_dict['model_message'], params_files)
    add_params_subparser(subparsers, list_models, model_messages_dict['model_message'])
    add_profile_subparser(subparsers)
    add_pproc_subparser(subparsers, out_folders)
    add_test_subparser(subparsers, list_models)

    return parser


def add_basic_options(parser, version_message, path_message):
    """Add basic command-line options to the parser."""
    parser.add_argument("-v", "--version", action="version", version=version_message)
    parser.add_argument(
        "-p",
        "--path",
        action="version",
        version=path_message,
        help="default installations and i/o paths",
    )
    parser.add_argument(
        "-s", "--short-help", action="store_true", help="display short help"
    )
    parser.add_argument(
        "--fluid", action="store_true", help="display available fluid models"
    )
    parser.add_argument(
        "--kinetic", action="store_true", help="display available kinetic models"
    )
    parser.add_argument(
        "--hybrid", action="store_true", help="display available hybrid models"
    )
    parser.add_argument(
        "--toy", action="store_true", help="display available toy models"
    )
    parser.add_argument(
        "--refresh-models",
        action="store_true",
        help="refresh list of available model names",
    )
    parser.add_argument(
        "--set-i", type=str, metavar="PATH", help="set new default Input folder"
    )
    parser.add_argument(
        "--set-o", type=str, metavar="PATH", help="set new default Output folder"
    )
    parser.add_argument(
        "--set-b",
        type=str,
        metavar="PATH",
        help="set new default Batch folder",
    )
    parser.add_argument(
        "--set-iob",
        type=str,
        metavar="PATH",
        help="set new default folder for io/inp/, io/out and io/batch",
    )


def add_compile_subparser(subparsers):
    """Add the 'compile' sub-command parser."""
    parser_compile = subparsers.add_parser(
        "compile",
        help="compile computational kernels, install psydac (on first call only)",
        description="Compile Struphy kernels using pyccel, https://github.com/pyccel/pyccel.",
    )
    parser_compile.add_argument(
        "--language",
        type=str,
        metavar="LANGUAGE",
        help='either "c" (default) or "fortran"',
        default="c",
    )
    parser_compile.add_argument(
        "--compiler",
        type=str,
        metavar="COMPILER",
        help="compiler choice",
        default="GNU",
    )
    parser_compile.add_argument(
        "--omp-pic", action="store_true", help="compile PIC kernels with OpenMP"
    )
    parser_compile.add_argument(
        "--omp-feec", action="store_true", help="compile FEEC kernels with OpenMP"
    )
    parser_compile.add_argument(
        "-d", "--delete", action="store_true", help="remove .f90/.c and .so files"
    )
    parser_compile.add_argument(
        "-s",
        "--status",
        action="store_true",
        help="print current Struphy compilation status",
    )
    parser_compile.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="call pyccel with --verbose compiler option",
    )
    parser_compile.add_argument(
        "--dependencies",
        action="store_true",
        help="print Struphy kernels to be compiled",
    )
    parser_compile.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="say yes to prompt when changing the language",
    )


def add_run_subparser(
    subparsers, list_models, model_message, params_files, batch_files
):
    """Add the 'run' sub-command parser."""
    parser_run = subparsers.add_parser(
        "run",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=30
        ),
        help="run a Struphy model",
        description="Run a Struphy model.",
        epilog="For more info on Struphy models, visit https://struphy.pages.mpcdf.de/struphy/sections/models.html",
    )
    parser_run.add_argument(
        "model", type=str, choices=list_models, metavar="MODEL", help=model_message
    )
    parser_run.add_argument(
        "-i",
        "--inp",
        type=str,
        choices=params_files,
        metavar="FILE",
        help="parameter file in current I/O path",
    )
    parser_run.add_argument(
        "--input-abs", type=str, metavar="FILE", help="parameter file, absolute path"
    )
    parser_run.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="DIR",
        help="output directory relative to current I/O path",
        default="sim_1",
    )
    parser_run.add_argument(
        "--output-abs", type=str, metavar="DIR", help="output directory, absolute path"
    )
    parser_run.add_argument(
        "-b",
        "--batch",
        type=str,
        choices=batch_files,
        metavar="FILE",
        help="batch script in current I/O path",
    )
    parser_run.add_argument(
        "--batch-abs", type=str, metavar="FILE", help="batch script, absolute path"
    )
    parser_run.add_argument(
        "--runtime",
        type=int,
        metavar="N",
        help="maximum wall-clock time in minutes",
        default=300,
    )
    parser_run.add_argument(
        "-s",
        "--save-step",
        type=int,
        metavar="N",
        help="save data every N time steps",
        default=1,
    )
    parser_run.add_argument(
        "-r",
        "--restart",
        action="store_true",
        help="restart the simulation in the specified output folder",
    )
    parser_run.add_argument(
        "--mpi", type=int, metavar="N", help="number of MPI processes", default=1
    )
    parser_run.add_argument(
        "--debug", action="store_true", help="launch a Cobra debug run"
    )
    parser_run.add_argument("--cprofile", action="store_true", help="run with Cprofile")
    parser_run.add_argument("--likwid", action="store_true", help="run with Likwid")
    parser_run.add_argument(
        "-li",
        "--likwid-inp",
        type=str,
        metavar="FILE",
        help="likwid parameter file in current I/O path",
    )
    parser_run.add_argument(
        "--likwid-input-abs",
        type=str,
        metavar="FILE",
        help="likwid parameter file, absolute path",
    )
    parser_run.add_argument(
        "-lr",
        "--likwid-repetitions",
        type=int,
        help="number of repetitions of the same simulation",
        default=1,
    )


def add_units_subparser(subparsers, list_models, model_message, params_files):
    """Add the 'units' sub-command parser."""
    parser_units = subparsers.add_parser(
        "units",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=30
        ),
        help="show physical units of a Struphy model",
        description="Show physical units of a Struphy model.",
        epilog="For more info on Struphy models, visit https://struphy.pages.mpcdf.de/struphy/sections/models.html",
    )
    parser_units.add_argument(
        "model", type=str, choices=list_models, metavar="MODEL", help=model_message
    )
    parser_units.add_argument(
        "-i",
        "--input",
        type=str,
        choices=params_files,
        metavar="FILE",
        help="parameter file relative to current I/O path",
    )
    parser_units.add_argument(
        "--input-abs", type=str, metavar="FILE", help="parameter file, absolute path"
    )


def add_params_subparser(subparsers, list_models, model_message):
    """Add the 'params' sub-command parser."""
    parser_params = subparsers.add_parser(
        "params",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=30
        ),
        help="create default parameter file for a model, or show model's options",
        description="Creates a default parameter file for a specific model, or shows a model's options.",
    )
    parser_params.add_argument(
        "model", type=str, choices=list_models, metavar="MODEL", help=model_message
    )
    parser_params.add_argument(
        "-f",
        "--file",
        type=str,
        metavar="FILE",
        help="name of the parameter file to be created",
    )
    parser_params.add_argument(
        "-o", "--options", action="store_true", help="show model options"
    )
    parser_params.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Say yes on prompt to overwrite .yml FILE",
    )


def add_profile_subparser(subparsers):
    """Add the 'profile' sub-command parser."""
    parser_profile = subparsers.add_parser(
        "profile",
        help="profile finished Struphy runs",
        description="Compare profiling data of finished Struphy runs.",
    )
    parser_profile.add_argument(
        "dirs", type=str, nargs="+", metavar="DIR", help="simulation output folders"
    )
    parser_profile.add_argument(
        "--replace",
        action="store_true",
        help="replace module names with class names for better info",
    )
    parser_profile.add_argument(
        "--all",
        action="store_true",
        help="display the 50 most expensive function calls",
    )
    parser_profile.add_argument(
        "--n-lines",
        type=int,
        metavar="N",
        help="number of lines to display in profiling analysis",
        default=6,
    )
    parser_profile.add_argument(
        "--print-callers",
        type=str,
        metavar="STR",
        help="print callers for functions matching STR",
        default=None,
    )
    parser_profile.add_argument(
        "--savefig", type=str, metavar="NAME", help="save the profile figure under NAME"
    )


def add_pproc_subparser(subparsers, out_folders):
    """Add the 'pproc' sub-command parser."""
    parser_pproc = subparsers.add_parser(
        "pproc",
        help="post process data of a finished Struphy run",
        description="Post-process data of a finished Struphy run to prepare for diagnostics.",
    )
    parser_pproc.add_argument(
        "-d",
        "--dirr",
        type=str,
        choices=out_folders,
        metavar="DIR",
        help="simulation output folder",
        default="sim_1",
    )
    parser_pproc.add_argument(
        "--dir-abs",
        type=str,
        metavar="DIR",
        help="simulation output folder, absolute path",
    )
    parser_pproc.add_argument(
        "-s",
        "--step",
        type=int,
        metavar="N",
        help="do post-processing every N-th time step",
        default=1,
    )
    parser_pproc.add_argument(
        "--celldivide",
        type=int,
        metavar="N",
        help="divide each grid cell by N for field evaluation",
        default=1,
    )
    parser_pproc.add_argument(
        "--physical",
        action="store_true",
        help="evaluate physical components in addition to logical components",
    )


def add_test_subparser(subparsers, list_models):
    """Add the 'test' sub-command parser."""
    parser_test = subparsers.add_parser(
        "test",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=30
        ),
        help="run Struphy tests",
        description="Run available unit tests or test Struphy models or tutorials.",
    )
    test_groups = list_models + [
        "models",
        "unit",
        "tutorials",
        "timings",
        "fluid",
        "kinetic",
        "hybrid",
        "toy",
    ]
    parser_test.add_argument(
        "group",
        type=str,
        choices=test_groups,
        metavar="GROUP",
        help="test group or model name",
    )
    parser_test.add_argument(
        "--mpi",
        type=int,
        metavar="N",
        help="number of MPI processes used in tests",
        default=2,
    )
    parser_test.add_argument(
        "-f",
        "--fast",
        action="store_true",
        help="test model(s) just in slab geometry (Cuboid)",
    )
    parser_test.add_argument(
        "--with-desc", action="store_true", help="include DESC equilibrium in tests"
    )
    parser_test.add_argument(
        "-v", "--verbose", action="store_true", help="print timings to screen"
    )
    parser_test.add_argument(
        "--monitor", action="store_true", help="use pytest-monitor in tests"
    )
    parser_test.add_argument(
        "-n", type=int, help="specific tutorial simulation to run", default=None
    )
    parser_test.add_argument(
        "-T",
        "--Tend",
        type=float,
        help="simulation end time in units of the model",
        default=None,
    )


def handle_basic_options(parser, args, state, libpath):
    """Handle basic options like --short-help, --set-i, etc."""
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit()

    if args.short_help:
        print_short_help(parser)
        sys.exit()

    if args.fluid or args.kinetic or args.hybrid or args.toy:
        display_model_subset(args)
        sys.exit()

    if args.set_i or args.set_o or args.set_b or args.set_iob:
        set_paths(args, state, libpath)
        sys.exit()

    if args.refresh_models:
        refresh_models(libpath)
        sys.exit()


def print_short_help(parser):
    """Print a short help message."""
    lines = parser.format_help().splitlines()
    start = next(i for i, line in enumerate(lines) if "Struphy" in line)
    end = next(i for i, line in enumerate(lines) if "available commands:" in line)
    print("\n".join(lines[start:end]))
    print("\n".join(lines[end:]))


def display_model_subset(args):
    """Display a subset of models based on the provided option."""
    if args.fluid:
        print_model_message("fluid")
    if args.kinetic:
        print_model_message("kinetic")
    if args.hybrid:
        print_model_message("hybrid")
    if args.toy:
        print_model_message("toy")
    print(
        "For more info on Struphy models, visit https://struphy.pages.mpcdf.de/struphy/sections/models.html"
    )


def print_model_message(model_type):
    """Print the message for a specific model type."""
    messages = {
        "fluid": "Fluid models:\n-------------\n",
        "kinetic": "Kinetic models:\n---------------\n",
        "hybrid": "Hybrid models:\n--------------\n",
        "toy": "Toy models:\n-----------\n",
    }
    print(messages.get(model_type, ""))


def set_paths(args, state, libpath):
    """Set input, output, or batch paths based on the arguments."""
    if args.set_i:
        i_path = resolve_path(args.set_i, libpath, 'io/inp')
        state['i_path'] = i_path
        print(f'New input path has been set to {i_path}')
    if args.set_o:
        o_path = resolve_path(args.set_o, libpath, 'io/out')
        state['o_path'] = o_path
        print(f'New output path has been set to {o_path}')
    if args.set_b:
        b_path = resolve_path(args.set_b, libpath, 'io/batch')
        state['b_path'] = b_path
        print(f'New batch path has been set to {b_path}')
    if args.set_iob:
        path = resolve_base_path(args.set_iob, libpath)
        state['i_path'] = os.path.join(path, 'io/inp')
        state['o_path'] = os.path.join(path, 'io/out')
        state['b_path'] = os.path.join(path, 'io/batch')
        print(f'New I/O paths have been set under {path}')
    save_state(state, libpath)


def resolve_path(arg_path, libpath, default_subpath):
    """Resolve the path based on the argument."""
    if arg_path == ".":
        path = os.getcwd()
    elif arg_path == "d":
        path = os.path.join(libpath, default_subpath)
    else:
        path = os.path.abspath(arg_path)
        os.makedirs(path, exist_ok=True)
    return path


def resolve_base_path(arg_path, libpath):
    """Resolve the base path for --set-iob."""
    if arg_path == ".":
        path = os.getcwd()
    elif arg_path == "d":
        path = libpath
    else:
        path = os.path.abspath(arg_path)
    return path


def refresh_models(libpath):
    """Refresh the list of available models."""
    print('Collecting available models ...')
    import inspect
    from struphy.models import fluid, kinetic, hybrid, toy

    model_info = {
        'fluid': (fluid, []),
        'kinetic': (kinetic, []),
        'hybrid': (hybrid, []),
        'toy': (toy, [])
    }

    model_message = 'run one of the following models:\n'
    list_models = []
    for model_type, (module, models_list) in model_info.items():
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name not in {'StruphyModel', 'Propagator'}:
                models_list.append(name)
                list_models.append(name)
    print(models_list)
    exit()
    # Save models
    with open(os.path.join(libpath, 'models', 'models_list'), "wb") as fp:
        pickle.dump(list_models, fp)

    
   
    model_message += '\n' + fluid_message
    model_message += '\n' + kinetic_message
    model_message += '\n' + hybrid_message
    model_message += '\n' + toy_message


    # Prepare and save messages
    model_messages = []
    for model_type, (_, models_list) in model_info.items():
        message = f'{model_type.capitalize()} models:\n' + '-' * (len(model_type) + 7) + '\n'
        message += '\n'.join(f'"{name}"' for name in models_list)
        model_messages.append(message)
        print('message',message)

    # Save model messages
    with open(os.path.join(libpath, 'models', 'models_message'), "wb") as fp:
        pickle.dump(model_messages, fp)

    print('Done.')


def execute_subcommand(args):
    """Execute the appropriate sub-command function based on the arguments."""
    subcommand_func = globals().get(f"struphy_{args.command}")
    if subcommand_func:
        # Prepare kwargs by removing irrelevant entries
        kwargs = vars(args).copy()
        for key in [
            "command",
            "short_help",
            "fluid",
            "kinetic",
            "hybrid",
            "toy",
            "set_i",
            "set_o",
            "set_b",
            "set_iob",
            "refresh_models",
        ]:
            kwargs.pop(key, None)
        subcommand_func(**kwargs)
    else:
        print("Unknown command.")
        sys.exit(1)


def get_params_files(i_path):
    """Get parameter files from the input path."""
    if os.path.exists(i_path) and os.path.isdir(i_path):
        params_files = recursive_get_files(i_path)
    else:
        print("Path to input files missing! Set it with `struphy --set-i PATH`")
        params_files = []
    return params_files


def get_output_folders(o_path):
    """Get output folders from the output path."""
    out_folders = []
    if os.path.exists(o_path) and os.path.isdir(o_path):
        all_folders = os.listdir(o_path)
        for name in all_folders:
            if "." not in name:
                out_folders.append(name)
    else:
        print("Path to outputs directory missing! Set it with `struphy --set-o PATH`")
    return out_folders


def get_batch_files(b_path):
    """Get batch files from the batch path."""
    if os.path.exists(b_path) and os.path.isdir(b_path):
        batch_files = recursive_get_files(b_path, contains=(".sh"))
    else:
        print("Path to batch files missing! Set it with `struphy --set-b PATH`")
        batch_files = []
    return batch_files


def struphy():
    """
    Struphy main executable. Performs argument parsing and sub-command call.
    """

    # Get struphy path
    import struphy
    import struphy.utils.utils as utils
    from struphy.console.compile import struphy_compile
    from struphy.console.params import struphy_params
    from struphy.console.pproc import struphy_pproc
    from struphy.console.profile import struphy_profile
    from struphy.console.run import struphy_run
    from struphy.console.test import struphy_test
    from struphy.console.units import struphy_units
    from struphy.utils.utils import get_paths  # Import the utility functions
    from struphy.utils.utils import read_state, save_state

    libpath = struphy.__path__[0]

    # Load state and paths
    state = read_state()
    i_path, o_path, b_path = get_paths(state)

    # Save state (in case defaults were used)
    save_state(state)

    # Prepare messages
    version_message, path_message = prepare_messages(libpath, i_path, o_path, b_path)

    # Collect available models and messages
    list_models, model_messages_dict = load_model_info(libpath)
    print('list_models',list_models)
    
    # Get parameter files, output folders, and batch files
    params_files = get_params_files(i_path)
    out_folders = get_output_folders(o_path)
    batch_files = get_batch_files(b_path)

    # Create argument parser
    parser = create_argument_parser(
        version_message,
        path_message,
        list_models,
        model_messages_dict,
        params_files,
        out_folders,
        batch_files,
    )

    # Parse arguments
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # Handle basic options and exit if necessary
    handle_basic_options(parser, args, state, libpath)

    # Load and execute the appropriate sub-command function
    execute_subcommand(args)


class NoSubparsersMetavarFormatter(HelpFormatter):
    """
    Removes redundant COMMANDS printing in help message of argument parser.
    """

    def _format_action(self, action):
        result = super()._format_action(action)
        if isinstance(action, _SubParsersAction):
            # fix indentation on first line
            return "%*s%s" % (self._current_indent, "", result.lstrip())
        return result

    def _format_action_invocation(self, action):
        if isinstance(action, _SubParsersAction):
            # remove metavar and help line
            return ""
        return super()._format_action_invocation(action)

    def _iter_indented_subactions(self, action):
        if isinstance(action, _SubParsersAction):
            try:
                get_subactions = action._get_subactions
            except AttributeError:
                pass
            else:
                # remove indentation
                yield from get_subactions()
        else:
            yield from super()._iter_indented_subactions(action)


class CustomFormatter(NoSubparsersMetavarFormatter, RawTextHelpFormatter):
    """
    Removes redundant COMMANDS printing in help message of argument parser and enables line breaks.
    """

    pass


def recursive_get_files(path, contains=(".yml", ".yaml"), out=[], prefix=[]):
    all_names = os.listdir(path)
    n_folders = 0
    # count folders in path
    for name in all_names:
        if os.path.isdir(os.path.join(path, name)):
            n_folders += 1
    # add specified files to out or descend
    for name in all_names:
        if any([cont in name for cont in contains]):
            if len(prefix) == 0:
                out += [name]
            else:
                out += [os.path.join(prefix[-1], name)]
        elif os.path.isdir(os.path.join(path, name)):
            if len(prefix) == 0:
                prefix = [name]
            else:
                prefix += [os.path.join(prefix[-1], name)]
            recursive_get_files(
                os.path.join(path, name), contains=contains, out=out, prefix=prefix
            )
    if n_folders == 0 and len(prefix) != 0:
        prefix.pop()

    return out
