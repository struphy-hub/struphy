#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
import glob
import importlib
import importlib.metadata
import os
import pickle
import site
import subprocess
import sys
from argparse import HelpFormatter, RawTextHelpFormatter, _SubParsersAction

import argcomplete
import yaml

# struphy path
import struphy
from struphy.utils import utils

libpath = struphy.__path__[0]
__version__ = importlib.metadata.version("struphy")

# version message
version_message = f"Struphy {__version__}\n"
version_message += "Copyright 2019-2025 (c) Struphy dev team | Max Planck Institute for Plasma Physics\n"
version_message += "MIT license\n"


def struphy():
    """Struphy main executable. Performs argument parsing and sub-command call."""

    # create argument parser
    epilog_message = 'Type "struphy COMMAND --help" for more information on a command.\n\n'
    epilog_message += "For more help on how to use Struphy, see https://struphy-hub.github.io/struphy/"

    parser = argparse.ArgumentParser(
        prog="struphy",
        formatter_class=CustomFormatter,
        description="Struphy: STRUcture-Preserving HYbrid code for plasma physics.",
        epilog=epilog_message,
    )

    # Read struphy state file
    state = utils.read_state()

    # Update and save state file
    utils.update_state(state=state)
    utils.save_state(state=state)

    # Load the models and messages
    model_message = "All models are listed on https://struphy-hub.github.io/struphy/sections/models.html"
    list_models = []
    ml_path = os.path.join(libpath, "models", "models_list")
    if not os.path.isfile(ml_path):
        utils.refresh_models()

    with open(ml_path, "rb") as fp:
        list_models = pickle.load(fp)
    with open(os.path.join(libpath, "models", "models_message"), "rb") as fp:
        model_message, fluid_message, kinetic_message, hybrid_message, toy_message = pickle.load(
            fp,
        )

    # 0. basic options
    add_parser_basic_options(parser)

    # create sub-commands and save name of sub-command into variable "command"
    subparsers = parser.add_subparsers(
        title="available commands",
        metavar="COMMAND",
        dest="command",
    )

    # 1. "compile" sub-command
    add_parser_compile(subparsers)

    # 2. "params" sub-command
    add_parser_params(subparsers, list_models, model_message)

    # 3. "profile" sub-command
    add_parser_profile(subparsers)

    # 4. "likwid_profile" sub-command
    add_parser_likwid_profile(subparsers)

    # 5. "test" sub-command
    add_parser_test(subparsers, list_models)

    # 6. "format" and "lint" sub-commands
    add_parser_format(subparsers)

    # parse argument
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # Set args.config if user ran struphy format / struphy lint
    set_args_format_config(args, parser)

    # if no arguments are passed, print help and exit
    if all(v is None or not v for v in vars(args).values()):
        parser.print_help()
        sys.exit(0)

    # display short help and exit
    if args.short_help:
        print_short_help(parser)
        sys.exit(0)

    # display subset of models
    model_flags = [
        (args.fluid, fluid_message),
        (args.kinetic, kinetic_message),
        (args.hybrid, hybrid_message),
        (args.toy, toy_message),
    ]

    for flag, message in model_flags:
        if flag:
            print(message)
            print("For more info on Struphy models, visit https://struphy.pages.mpcdf.de/struphy/sections/models.html")
            sys.exit(0)

    if args.refresh_models:
        utils.refresh_models()

    # load sub-command function
    command_map = {
        "compile": ("struphy.console.compile", "struphy_compile"),
        "lint": ("struphy.console.format", "struphy_lint"),
        "format": ("struphy.console.format", "struphy_format"),
        "likwid_profile": ("struphy.console.likwid", "struphy_likwid_profile"),
        "params": ("struphy.console.params", "struphy_params"),
        "profile": ("struphy.console.profile", "struphy_profile"),
        "test": ("struphy.console.test", "struphy_test"),
    }

    # import struphy.console.MODULE.FUNC_NAME as func
    if args.command in command_map:
        module_path, func_name = command_map[args.command]
        func = getattr(importlib.import_module(module_path), func_name)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    # transform parser Namespace object to dictionary and remove "command" key
    kwargs = vars(args)
    for key in [
        "command",
        "short_help",
        "fluid",
        "kinetic",
        "hybrid",
        "toy",
        "refresh_models",
        # These options are stored in kwargs.config
        "input_type",
        "path",
        "linters",
        "iterations",
        "output_format",
    ]:
        kwargs.pop(key, None)

    # start sub-command function with all parameters of that function
    # for k, v in kwargs.items():
    #     print(k, v)
    func(**kwargs)


def add_parser_basic_options(parser):
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=version_message,
    )
    parser.add_argument(
        "-s",
        "--short-help",
        action="store_true",
        help="display short help",
    )
    parser.add_argument(
        "--fluid",
        action="store_true",
        help="display available fluid models",
    )
    parser.add_argument(
        "--kinetic",
        action="store_true",
        help="display available kinetic models",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="display available hybrid models",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="display available toy models",
    )
    parser.add_argument(
        "--refresh-models",
        help="refresh list of available model names",
        action="store_true",
    )


def add_parser_compile(
    subparsers,
):
    parser_compile = subparsers.add_parser(
        "compile",
        help="compile computational kernels (including psydac)",
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
        help='either "GNU" (default), "intel", "PGI", "nvidia" or "LLVM"',
        default="GNU",
    )

    parser_compile.add_argument(
        "--compiler-config",
        type=str,
        metavar="COMPILER_CONFIG",
        help="Path to a JSON compiler file.",
        default=None,
    )

    parser_compile.add_argument(
        "--omp-pic",
        help="compile PIC kernels with OpenMP",
        action="store_true",
    )

    parser_compile.add_argument(
        "--omp-feec",
        help="compile FEEC kernels with OpenMP",
        action="store_true",
    )

    parser_compile.add_argument(
        "-d",
        "--delete",
        help="remove .f90/.c and .so files (for running pure Python code)",
        action="store_true",
    )

    parser_compile.add_argument(
        "-s",
        "--status",
        help="print current Struphy compilation status on screen",
        action="store_true",
    )

    parser_compile.add_argument(
        "-v",
        "--verbose",
        help="call pyccel with --verbose compiler option",
        action="store_true",
    )

    parser_compile.add_argument(
        "--dependencies",
        help="print Struphy kernels to be compiled (.py) and their dependencies (.so) on screen",
        action="store_true",
    )

    parser_compile.add_argument(
        "--time-execution",
        help="Prints the time spent in each section of the pyccelization.",
        action="store_true",
    )

    parser_compile.add_argument(
        "-y",
        "--yes",
        help="say yes to prompt when changing the language",
        action="store_true",
    )


def add_parser_params(subparsers, list_models, model_message):
    parser_params = subparsers.add_parser(
        "params",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog,
            max_help_position=35,
        ),
        help="create default parameter file for a model, or show model's options",
        description="Create default parameter file (.py) for a specific model.",
    )

    parser_params.add_argument(
        "model_name",
        type=str,
        choices=list_models,
        metavar="MODEL",
        help=model_message,
    )

    parser_params.add_argument(
        "--check-file",
        type=str,
        metavar="FILE",
        help="check if the parameters in the .yml file are valid",
    )

    parser_params.add_argument(
        "-y",
        "--yes",
        help="Say yes on prompt to overwrite PATH",
        action="store_true",
    )


def add_parser_profile(subparsers):
    parser_profile = subparsers.add_parser(
        "profile",
        help="profile finished runs",
        description="Compare profiling data of finished Struphy runs. For each function in a predefined filter, displays: ncalls, tottime, percall and cumtime.",
    )

    parser_profile.add_argument(
        "dirs",
        type=str,
        nargs="+",
        metavar="DIR",
        help="simulation ouput folders",
    )

    parser_profile.add_argument(
        "--replace",
        help="replace module names with class names for better info",
        action="store_true",
    )

    parser_profile.add_argument(
        "--all",
        help="display the 50 most expensive function calls, without applying the predefined filter",
        action="store_true",
    )

    parser_profile.add_argument(
        "--n-lines",
        type=int,
        metavar="N",
        help="plot the N most time consuming calls in profiling analysis (default=6)",
        default=6,
    )

    parser_profile.add_argument(
        "--print-callers",
        type=str,
        metavar="STR",
        help="string STR that identifies functions for which to print callers (default=None)",
        default=None,
    )

    parser_profile.add_argument(
        "--savefig",
        type=str,
        metavar="NAME",
        help="save (and dont display) the profile figure under NAME, relative to current output path.",
    )


def add_parser_likwid_profile(subparsers):
    try:
        import pylikwid

        add_likwid_parser = True
    except (ModuleNotFoundError, ImportError):
        add_likwid_parser = False

    if add_likwid_parser:
        parser_likwid_profile = subparsers.add_parser(
            "likwid_profile",
            help="Profile finished runs with likwid",
            description="Compare profiling data of finished Struphy runs. Run the plot files script with a given directory.",
        )

        parser_likwid_profile.add_argument(
            "--dir",
            type=str,
            nargs="+",
            required=True,
            help="Paths to the data directories (space-separated, supports wildcards)",
        )
        parser_likwid_profile.add_argument(
            "--title",
            type=str,
            default="Testing",
            help="Name of the project",
        )
        parser_likwid_profile.add_argument(
            "--output",
            type=str,
            default=".",
            help="Output directory",
        )
        parser_likwid_profile.add_argument(
            "--groups",
            type=str,
            default=["*"],
            nargs="+",
            required=False,
            help="Likwid groups to include (space-separated, supports wildcards). Default: ['*'].",
        )
        parser_likwid_profile.add_argument(
            "--skip",
            type=str,
            default=[],
            nargs="+",
            required=False,
            help="Likwid groups to skip (space-separated, supports wildcards). Default: [].",
        )
        parser_likwid_profile.add_argument(
            "--plots",
            type=str,
            default=[
                "pinning",
                "speedup",
                "barplots",
                "loadbalance",
                "roofline",
            ],
            nargs="+",
            required=False,
            help="Types of plots to plot (space-separated). Default: [pinning, speedup. barplots, loadbalance, roofline]",
        )


def add_parser_test(subparsers, list_models):
    try:
        import pytest_mpi

        add_test_parser = True
    except ModuleNotFoundError:
        add_test_parser = False

    if add_test_parser:
        parser_test = subparsers.add_parser(
            "test",
            formatter_class=lambda prog: argparse.RawTextHelpFormatter(
                prog,
                max_help_position=30,
            ),
            help="run tests",
            description="Run available unit tests or test Struphy models.",
        )

        parser_test.add_argument(
            "group",
            type=str,
            choices=list_models
            + ["models"]
            + ["unit"]
            + ["fluid"]
            + ["kinetic"]
            + ["hybrid"]
            + ["toy"]
            + ["verification"],
            metavar="GROUP",
            help='can be either:\na) a model name \
                                    \nb) "models" for testing of all models (or "fluid", "kinetic", "hybrid", "toy" for testing just a sub-group) \
                                    \nc) "verification" for running all verification tests \
                                    \nd) "unit" for performing unit tests',
        )

        parser_test.add_argument(
            "--mpi",
            type=int,
            metavar="N",
            help="set number of MPI processes used in tests (default=1))",
            default=1,
        )

        parser_test.add_argument(
            "--with-desc",
            help="include DESC equilibrium in tests (mem consuming)",
            action="store_true",
        )

        parser_test.add_argument(
            "-v",
            "--vrbose",
            help="print output of testing on screen",
            action="store_true",
        )

        parser_test.add_argument(
            "--nclones",
            type=int,
            metavar="N",
            help="number of domain clones (default=1)",
            default=1,
        )

        parser_test.add_argument(
            "--show-plots",
            help="show plots of tests",
            action="store_true",
        )


def add_parser_format(subparsers):
    try:
        import autopep8
        import isort
        import ruff

        add_lintformat_parser = True
    except ModuleNotFoundError:
        add_lintformat_parser = False

    if is_installed_editable("struphy") and add_lintformat_parser:
        parser_format = subparsers.add_parser(
            "format",
            help="format source files",
            description="Format source files based on the given input.",
        )

        parser_lint = subparsers.add_parser(
            "lint",
            help="lint and analyze source files",
            description="Check code statistics and formatting compliance.",
        )

        # Common argument for both 'format' and 'lint'
        for subparser in [parser_format, parser_lint]:
            subparser.add_argument(
                "input_type",
                type=str,
                choices=["all", "staged", "branch", "__init__.py"],
                nargs="?",  # optional
                help="specify the files to process",
            )
            subparser.add_argument(
                "--path",
                type=str,
                # default=libpath,
                help="the path to the directory or file",
            )

            subparser.add_argument(
                "--verbose",
                action="store_true",
                help="use verbose output",
            )

        parser_format.add_argument(
            "--linters",
            type=str,
            nargs="+",
            default=["ruff"],
            choices=["add-trailing-comma", "isort", "autopep8", "ruff"],
            help="list of linters to use",
        )
        parser_format.add_argument(
            "--iterations",
            type=int,
            default=5,
            help="maximum number of times to run each formatter",
        )
        # Avoid interfering with --yes flags in other subparsers
        # by adding an argument group
        format_group = parser_format.add_argument_group("format options")
        format_group.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="say yes to prompt when asked if all files should be formatted",
        )

        parser_lint.add_argument(
            "--linters",
            type=str,
            nargs="+",
            default=["ruff", "omp_flags"],
            choices=["add-trailing-comma", "isort", "autopep8", "flake8", "pylint", "ruff", "omp_flags"],
            help="list of linters to use",
        )

        parser_lint.add_argument(
            "--output-format",
            type=str,
            default="table",
            choices=["table", "plain", "report"],
            help="specify the format of the output: 'table' for tabular output, 'plain' for regular output, or 'report' for saving a html report",
        )


def set_args_format_config(args, parser):
    if args.command == "format" or args.command == "lint":
        if not args.input_type and not args.path:
            parser.error("Use with either 'all', 'staged', 'branch', or '--path PATH'")
        args.config = {
            "input_type": args.input_type,
            "path": args.path,
            "linters": args.linters,
        }

    if args.command == "format":
        args.config["iterations"] = args.iterations
    if args.command == "lint":
        args.config["output_format"] = args.output_format


def print_short_help(parser):
    lines = parser.format_help().splitlines()
    bool_1 = [i for i, x in enumerate(lines) if "Struphy" in x]
    bool_2 = [i for i, x in enumerate(lines) if "available commands:" in x]
    print(lines[bool_1[0]])
    print(lines[bool_1[0] + 1])
    for li in lines[bool_2[0] :]:
        print(li)


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


def recursive_get_files(path, contains=(".yml", ".yaml"), out=None, prefix=None):
    if out is None:
        out = []

    if prefix is None:
        prefix = []

    # only search 1 level deep (e.g. a/test.yml will be found, but not a/b/test.yml)
    level_depth = 1

    all_names = os.listdir(path)
    n_folders = 0
    # count folders in path
    for name in all_names:
        if os.path.isdir(os.path.join(path, name)):
            n_folders += 1
    # add specified files to out or descend
    n_searched_folders = 0
    for name in all_names:
        if any([cont in name for cont in contains]):
            if len(prefix) == 0:
                out += [name]
            else:
                out += [os.path.join(prefix[-1], name)]
        elif os.path.isdir(os.path.join(path, name)) and len(prefix) < level_depth:
            n_searched_folders += 1
            if len(prefix) == 0:
                prefix = [name]
            else:
                prefix += [os.path.join(prefix[-1], name)]
            recursive_get_files(
                os.path.join(path, name),
                contains=contains,
                out=out,
                prefix=prefix,
            )

    if (n_folders == 0 or len(prefix) == level_depth or n_searched_folders == n_folders) and len(prefix) != 0:
        prefix.pop()

    return out


def is_installed_editable(package_name):
    """
    Check if a package is installed in editable mode by inspecting
    First: `pip show {package_name}`.
    Second: Check for `__editable__` file in site-packages

    Parameters
    ----------
    package_name : str
        Name of the package.
    """
    try:
        pip_show_output = subprocess.check_output(["pip", "show", package_name], text=True)

        if "Editable project location" in pip_show_output:
            # print(f"{package_name} is installed in editable mode.")
            return True

    except subprocess.CalledProcessError as e:
        print("Error while checking pip show:", e)
        return False

    for path in site.getsitepackages():
        editable_file = os.path.join(path, f"__editable__.{package_name.replace('-', '_')}-*.pth")
        if any(os.path.exists(f) for f in glob.glob(editable_file)):
            # print(f"{package_name} is installed in editable mode.")
            # print(f"{editable_file} found in site-packages")
            return True

    # print(f"{package_name} is not installed in editable mode.")
    return False
