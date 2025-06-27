#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import glob
import importlib.metadata
import os
import site
import subprocess
import sys
from argparse import HelpFormatter, RawTextHelpFormatter, _SubParsersAction

import argcomplete

__version__ = importlib.metadata.version("struphy")


def struphy():
    """Struphy main executable. Performs argument parsing and sub-command call."""

    import argparse
    import pickle

    import yaml

    # struphy path
    import struphy
    import struphy.utils.utils as utils

    libpath = struphy.__path__[0]

    # create argument parser
    epilog_message = 'Type "struphy COMMAND --help" for more information on a command.\n\n'
    epilog_message += "For more help on how to use Struphy, see https://struphy.pages.mpcdf.de/struphy/index.html"

    parser = argparse.ArgumentParser(
        prog="struphy",
        formatter_class=CustomFormatter,
        description="Struphy: STRUcture-Preserving HYbrid codes for plasma physics.",
        epilog=epilog_message,
    )

    # version message
    version_message = f"Struphy {__version__}\n"
    version_message += "Copyright 2019-2025 (c) Struphy dev team | Max Planck Institute for Plasma Physics\n"
    version_message += "MIT license\n"

    # Read struphy state file
    state = utils.read_state()

    if "i_path" in state:
        i_path = state["i_path"]
    else:
        i_path = os.path.join(libpath, "io/inp")
        state["i_path"] = i_path

    if "o_path" in state:
        o_path = state["o_path"]
    else:
        o_path = os.path.join(libpath, "io/out")
        state["o_path"] = o_path

    if "b_path" in state:
        b_path = state["b_path"]
    else:
        b_path = os.path.join(libpath, "io/batch")
        state["b_path"] = b_path

    utils.save_state(state)

    # path message
    path_message = f"Struphy installation path: {libpath}\n"
    path_message += f"current input:             {i_path}\n"
    path_message += f"current output:            {o_path}\n"
    path_message += f"current batch scripts:     {b_path}"

    # check parameter file in current input path:
    if os.path.exists(i_path) and os.path.isdir(i_path):
        params_files = recursive_get_files(i_path)
    else:
        print("Path to input files missing! Set it with `struphy --set-i PATH`")
        params_files = []

    # check output folders in current output path:
    out_folders = []
    if os.path.exists(o_path) and os.path.isdir(o_path):
        all_folders = os.listdir(o_path)
        for name in all_folders:
            if "." not in name:
                out_folders += [name]
    else:
        print("Path to outputs directory missing! Set it with `struphy --set-o PATH`")

    # check batch scripts in current batch path:
    if os.path.exists(b_path) and os.path.isdir(b_path):
        batch_files = recursive_get_files(
            b_path,
            contains=(".sh"),
            out=[],
            prefix=[],
        )
    else:
        print("Path to batch files missing! Set it with `struphy --set-b PATH`")
        batch_files = []

    try:
        with open(os.path.join(libpath, "models", "models_list"), "rb") as fp:
            list_models = pickle.load(fp)
        with open(os.path.join(libpath, "models", "models_message"), "rb") as fp:
            model_message, fluid_message, kinetic_message, hybrid_message, toy_message = pickle.load(
                fp,
            )
    except:
        list_models = []
        model_message = ""

    # 0. basic options
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=version_message,
    )
    parser.add_argument(
        "-p",
        "--path",
        action="version",
        version=path_message,
        help="default installations and i/o paths",
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
    parser.add_argument(
        "--set-i",
        type=str,
        metavar="PATH",
        help='make PATH the new default Input folder ("." to use cwd, "d" to use default <install-path>/io/inp/)',
    )
    parser.add_argument(
        "--set-o",
        type=str,
        metavar="PATH",
        help='make PATH the new default Output folder ("." to use cwd, "d" to use default <install-path>/io/out/)',
    )
    parser.add_argument(
        "--set-b",
        type=str,
        metavar="PATH",
        help='make PATH the new default Batch folder ("." to use cwd, "d" to use default <install-path>/io/batch/)',
    )
    parser.add_argument(
        "--set-iob",
        type=str,
        metavar="PATH",
        help='make PATH the new default folder for io/inp/, io/out and io/batch ("." to use cwd, "d" to use default <install-path>)',
    )

    # create sub-commands and save name of sub-command into variable "command"
    subparsers = parser.add_subparsers(
        title="available commands",
        metavar="COMMAND",
        dest="command",
    )

    # 1. "compile" sub-command
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

    # 2. "run" sub-command
    parser_run = subparsers.add_parser(
        "run",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog,
            max_help_position=30,
        ),
        help="run a Struphy model",
        description="Run a Struphy model.",
        epilog="For more info on Struphy models, visit https://struphy.pages.mpcdf.de/struphy/sections/models.html",
    )

    # parser_run.add_argument(
    #     "model",
    #     type=str,
    #     default=None,
    #     choices=list_models,
    #     metavar="MODEL",
    #     help=model_message,
    # )
    parser_run.add_argument(
        "model",
        type=str,
        nargs="?",  # makes it optional
        default=None,  # fallback if nothing is passed
        choices=list_models,
        metavar="MODEL",
        help=model_message + f" (default: None)",
    )

    parser_run.add_argument(
        "-i",
        "--inp",
        type=str,
        choices=params_files,
        metavar="FILE",
        help="parameter file (.yml) in current I/O path",
    )

    parser_run.add_argument(
        "--input-abs",
        type=str,
        metavar="FILE",
        help="parameter file (.yml), absolute path",
    )

    parser_run.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="DIR",
        help="output directory relative to current I/O path (default=sim_1)",
        default="sim_1",
    )

    parser_run.add_argument(
        "--output-abs",
        type=str,
        metavar="DIR",
        help="output directory, absolute path",
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
        "--batch-abs",
        type=str,
        metavar="FILE",
        help="batch script, absolute path",
    )

    parser_run.add_argument(
        "--runtime",
        type=int,
        metavar="N",
        help="maximum wall-clock time of program in minutes (default=300)",
        default=300,
    )

    parser_run.add_argument(
        "-s",
        "--save-step",
        type=int,
        metavar="N",
        help='how often to save data in hdf5 file, i.e. every "save-step" time step (default=1, which is every time step)',
        default=1,
    )

    parser_run.add_argument(
        "--sort-step",
        type=int,
        metavar="N",
        help="sort markers in memory every N time steps (default=0, which means markers are sorted only at the start of simulation)",
        default=0,
    )

    parser_run.add_argument(
        "-r",
        "--restart",
        help="restart the simulation in the output folder specified under -o",
        action="store_true",
    )

    parser_run.add_argument(
        "--mpi",
        type=int,
        metavar="N",
        help='use "mpirun -n N" to launch a parallel Struphy run (default=1)',
        default=1,
    )

    parser_run.add_argument(
        "--nclones",
        type=int,
        metavar="N",
        help="number of domain clones (default=1)",
        default=1,
    )

    parser_run.add_argument(
        "--debug",
        help="launch a Cobra debug run, see https://docs.mpcdf.mpg.de/doc/computing/cobra-user-guide.html#interactive-debug-runs",
        action="store_true",
    )

    parser_run.add_argument(
        "--cprofile",
        help="run with Cprofile",
        action="store_true",
    )

    parser_run.add_argument(
        "-v",
        "--verbose",
        help="print info of struphy/main.py on screen",
        action="store_true",
    )

    # 3. "units" sub-command
    parser_units = subparsers.add_parser(
        "units",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog,
            max_help_position=30,
        ),
        help="show physical units of a Struphy model",
        description="Show physical units of a Struphy model.",
        epilog="For more info on Struphy models, visit https://struphy.pages.mpcdf.de/struphy/sections/models.html",
    )

    parser_units.add_argument(
        "model",
        type=str,
        choices=list_models,
        metavar="MODEL",
        help=model_message,
    )

    parser_units.add_argument(
        "-i",
        "--input",
        type=str,
        choices=params_files,
        metavar="FILE",
        help="parameter file (.yml) relative to current I/O path. If absent, default parameters are used.",
    )

    parser_units.add_argument(
        "--input-abs",
        type=str,
        metavar="FILE",
        help="parameter file (.yml), absolute path",
    )

    # 4. "params" sub-command
    parser_params = subparsers.add_parser(
        "params",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog,
            max_help_position=30,
        ),
        help="create default parameter file for a model, or show model's options",
        description="Creates a default parameter file for a specific model, or shows a model's options.",
    )

    parser_params.add_argument(
        "model",
        type=str,
        choices=list_models,
        metavar="MODEL",
        help=model_message,
    )

    parser_params.add_argument(
        "-f",
        "--file",
        type=str,
        metavar="FILE",
        help="name of the parameter file (.yml) to be created in the current I/O path (default=params_<model>.yml)",
    )

    parser_params.add_argument(
        "-o",
        "--options",
        help="show model options",
        action="store_true",
    )

    parser_params.add_argument(
        "-y",
        "--yes",
        help="Say yes on prompt to overwrite .yml FILE",
        action="store_true",
    )

    parser_performance = parser_run.add_argument_group(
        "Performance profiling options",
        "Arguments related to performance measurement. Note that hardware metrics requires a likwid installation.",
    )

    try:
        import pylikwid

        add_likwid_parser = True
    except (ModuleNotFoundError, ImportError):
        add_likwid_parser = False

    if add_likwid_parser:
        # Add Likwid-related arguments to the likwid group
        parser_performance.add_argument(
            "--likwid",
            help="run with Likwid",
            action="store_true",
        )

        parser_performance.add_argument(
            "-g",
            "--group",
            default="MEM_DP",
            type=str,
            help="likwid measurement group",
        )
        parser_performance.add_argument(
            "--nperdomain",
            default=None,  # Example: S:36 means 36 cores/socket
            type=str,
            help="Set the number of processes per node by giving an affinity domain and count",
        )

        parser_performance.add_argument(
            "--stats",
            help="Print Likwid statistics",
            action="store_true",
        )

        parser_performance.add_argument(
            "--marker",
            help="Activate Likwid marker API",
            action="store_true",
        )

        parser_performance.add_argument(
            "--hpcmd_suspend",
            help="Suspend the HPCMD daemon",
            action="store_true",
        )

        parser_performance.add_argument(
            "-lr",
            "--likwid-repetitions",
            type=int,
            help="Number of repetitions of the same simulation",
            default=1,
        )

    parser_performance.add_argument(
        "--time-trace",
        help="Measure time traces for each call of the regions measured with ProfileManager",
        action="store_true",
    )

    parser_performance.add_argument(
        "--sample-duration",
        help="Duration of samples when measuring time traces with ProfileManager",
        default=1.0,
    )

    parser_performance.add_argument(
        "--sample-interval",
        help="Time between samples when measuring time traces with ProfileManager",
        default=1.0,
    )

    # 5. "profile" sub-command
    parser_profile = subparsers.add_parser(
        "profile",
        help="profile finished Struphy runs",
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

    if add_likwid_parser:
        parser_likwid_profile = subparsers.add_parser(
            "likwid_profile",
            help="Profile finished Struphy runs with likwid",
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

    # 6. "pproc" sub-command
    parser_pproc = subparsers.add_parser(
        "pproc",
        help="post process data of a finished Struphy run",
        description="Post-process data of a finished Struphy run to prepare for diagnostics.",
    )

    parser_pproc.add_argument(
        "dirs",
        type=str,
        nargs="*",
        choices=out_folders,
        metavar="DIR",
        default=["sim_1"],
        help=("Simulation output folders to post-process (relative to current I/O path) (default: [sim_1])."),
    )

    parser_pproc.add_argument(
        "--dir-abs",
        type=str,
        metavar="DIR",
        help="simulation output folder to post-process, absolute path",
    )

    parser_pproc.add_argument(
        "-s",
        "--step",
        type=int,
        metavar="N",
        help="do post-processing every N-th time step (default=1).",
        default=1,
    )

    parser_pproc.add_argument(
        "--celldivide",
        type=int,
        metavar="N",
        help="divide each grid cell by N for field evaluation (default=1)",
        default=1,
    )

    parser_pproc.add_argument(
        "--physical",
        help="in addition to logical components, evaluates push-forwarded physical (xyz) components",
        action="store_true",
    )

    parser_pproc.add_argument(
        "--guiding-center",
        help="compute guiding-center coordinates (only from Particles6D)",
        action="store_true",
    )

    parser_pproc.add_argument(
        "--classify",
        help="classify guiding-center trajectories (passing, trapped or lost)",
        action="store_true",
    )

    parser_pproc.add_argument(
        "--no-vtk",
        help="whether vtk files creation should be skipped",
        action="store_true",
    )

    parser_pproc.add_argument(
        "--time-trace",
        help="whether to plot the time traces",
        action="store_true",
    )

    # 7. "test" sub-command
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
            help="run Struphy tests",
            description="Run available unit tests or test Struphy models.",
        )

        parser_test.add_argument(
            "group",
            type=str,
            choices=list_models + ["models"] + ["unit"] + ["fluid"] + ["kinetic"] + ["hybrid"] + ["toy"],
            metavar="GROUP",
            help='can be either:\na) a model name \
                                    \nb) "models" for testing of all models (or "fluid", "kinetic", "hybrid", "toy" for testing just a sub-group) \
                                    \nc) "unit" for performing unit tests',
        )

        parser_test.add_argument(
            "--mpi",
            type=int,
            metavar="N",
            help="set number of MPI processes used in tests (default=2))",
            default=2,
        )

        parser_test.add_argument(
            "-f",
            "--fast",
            help="test model(s) just in slab geometry (Cuboid)",
            action="store_true",
        )

        parser_test.add_argument(
            "--with-desc",
            help="include DESC equilibrium in tests (mem consuming)",
            action="store_true",
        )

        parser_test.add_argument(
            "-T",
            "--Tend",
            type=float,
            help="if GROUP=a), simulation end time in units of the model (default=0.015 with dt=0.005), data is only saved at TEND if set",
            default=None,
        )

        parser_test.add_argument(
            "-v",
            "--vrbose",
            help="print output of testing on screen",
            action="store_true",
        )

        parser_test.add_argument(
            "--verification",
            help="perform verification runs specified in io/inp/verification/",
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

    # 8/9 format and lint sub-command
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
                choices=["all", "staged", "branch"],
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

    # parse argument
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
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

    # if no arguments are passed, print help and exit
    print_help = True
    for key, val in args.__dict__.items():
        if val is None or not val:
            continue
        else:
            print_help = False

    if print_help:
        parser.print_help()
        sys.exit(0)

    # display short help and exit
    if args.short_help:
        lines = parser.format_help().splitlines()
        bool_1 = [i for i, x in enumerate(lines) if "Struphy" in x]
        bool_2 = [i for i, x in enumerate(lines) if "available commands:" in x]
        print(lines[bool_1[0]])
        print(lines[bool_1[0] + 1])
        for li in lines[bool_2[0] :]:
            print(li)
        sys.exit(0)

    # display subset of models
    if args.fluid:
        print(fluid_message)
        print("For more info on Struphy models, visit https://struphy.pages.mpcdf.de/struphy/sections/models.html")
        sys.exit(0)

    if args.kinetic:
        print(kinetic_message)
        print("For more info on Struphy models, visit https://struphy.pages.mpcdf.de/struphy/sections/models.html")
        sys.exit(0)

    if args.hybrid:
        print(hybrid_message)
        print("For more info on Struphy models, visit https://struphy.pages.mpcdf.de/struphy/sections/models.html")
        sys.exit(0)

    if args.toy:
        print(toy_message)
        print("For more info on Struphy models, visit https://struphy.pages.mpcdf.de/struphy/sections/models.html")
        sys.exit(0)

    # set default in path
    if args.set_i:
        if args.set_i == ".":
            i_path = os.getcwd()
        elif args.set_i == "d":
            i_path = os.path.join(libpath, "io/inp")
        else:
            i_path = args.set_i
            try:
                os.makedirs(i_path)
            except:
                pass

        i_path = os.path.abspath(i_path)

        state["i_path"] = i_path
        utils.save_state(state)

        print(f"New input path has been set to {state['i_path']}")
        sys.exit(0)

    # set default out path
    if args.set_o:
        if args.set_o == ".":
            o_path = os.getcwd()
        elif args.set_o == "d":
            o_path = os.path.join(libpath, "io/out")
        else:
            o_path = args.set_o
            try:
                os.makedirs(o_path)
            except:
                pass

        o_path = os.path.abspath(o_path)

        state["o_path"] = o_path
        utils.save_state(state)

        print(f"New output path has been set to {state['o_path']}")
        sys.exit(0)

    # set default out path
    if args.set_b:
        if args.set_b == ".":
            b_path = os.getcwd()
        elif args.set_b == "d":
            b_path = os.path.join(libpath, "io/batch")
        else:
            b_path = args.set_b
            try:
                os.makedirs(b_path)
            except:
                pass

        b_path = os.path.abspath(b_path)

        state["b_path"] = b_path
        utils.save_state(state)

        print(f"New batch path has been set to {state['b_path']}")
        sys.exit(0)

    # set paths for inp, out and batch (with io/inp etc. prefices)
    if args.set_iob:
        if args.set_iob == ".":
            path = os.getcwd()
        elif args.set_iob == "d":
            path = libpath
        else:
            path = args.set_iob

        i_path = os.path.join(path, "io/inp")
        o_path = os.path.join(path, "io/out")
        b_path = os.path.join(path, "io/batch")

        subprocess.run(["struphy", "--set-i", i_path])
        subprocess.run(["struphy", "--set-o", o_path])
        subprocess.run(["struphy", "--set-b", b_path])

        sys.exit(0)

    if args.refresh_models:
        print("Collecting available models ...")

        import inspect
        import pickle

        from struphy.models import fluid, hybrid, kinetic, toy

        list_fluid = []
        fluid_string = ""
        for name, obj in inspect.getmembers(fluid):
            if inspect.isclass(obj):
                if name not in {"StruphyModel", "Propagator"}:
                    list_fluid += [name]
                    fluid_string += '"' + name + '"\n'

        list_kinetic = []
        kinetic_string = ""
        for name, obj in inspect.getmembers(kinetic):
            if inspect.isclass(obj):
                if name not in {"StruphyModel", "Propagator"}:
                    list_kinetic += [name]
                    kinetic_string += '"' + name + '"\n'

        list_hybrid = []
        hybrid_string = ""
        for name, obj in inspect.getmembers(hybrid):
            if inspect.isclass(obj):
                if name not in {"StruphyModel", "Propagator"}:
                    list_hybrid += [name]
                    hybrid_string += '"' + name + '"\n'

        list_toy = []
        toy_string = ""
        for name, obj in inspect.getmembers(toy):
            if inspect.isclass(obj):
                if name not in {"StruphyModel", "Propagator"}:
                    list_toy += [name]
                    toy_string += '"' + name + '"\n'

        list_models = list_fluid + list_kinetic + list_hybrid + list_toy

        with open(os.path.join(libpath, "models", "models_list"), "wb") as fp:
            pickle.dump(list_models, fp)

        # fluid message
        fluid_message = "Fluid models:\n"
        fluid_message += "-------------\n"
        fluid_message += fluid_string

        # kinetic message
        kinetic_message = "Kinetic models:\n"
        kinetic_message += "---------------\n"
        kinetic_message += kinetic_string

        # hybrid message
        hybrid_message = "Hybrid models:\n"
        hybrid_message += "--------------\n"
        hybrid_message += hybrid_string

        # toy message
        toy_message = "Toy models:\n"
        toy_message += "-----------\n"
        toy_message += toy_string

        # model message
        model_message = "run one of the following models:\n"
        model_message += "\n" + fluid_message
        model_message += "\n" + kinetic_message
        model_message += "\n" + hybrid_message
        model_message += "\n" + toy_message

        with open(os.path.join(libpath, "models", "models_message"), "wb") as fp:
            pickle.dump(
                [
                    model_message,
                    fluid_message,
                    kinetic_message,
                    hybrid_message,
                    toy_message,
                ],
                fp,
            )

        print("Done.")
        sys.exit(0)

    # load sub-command function
    if args.command == "compile":
        from struphy.console.compile import struphy_compile as func
    elif args.command == "lint":
        from struphy.console.format import struphy_lint as func
    elif args.command == "format":
        from struphy.console.format import struphy_format as func
    elif args.command == "likwid_profile":
        from struphy.console.likwid import struphy_likwid_profile as func
    elif args.command == "params":
        from struphy.console.params import struphy_params as func
    elif args.command == "pproc":
        from struphy.console.pproc import struphy_pproc as func
    elif args.command == "profile":
        from struphy.console.profile import struphy_profile as func
    elif args.command == "run":
        from struphy.console.run import struphy_run as func
    elif args.command == "test":
        from struphy.console.test import struphy_test as func
    elif args.command == "units":
        from struphy.console.units import struphy_units as func

    # transform parser Namespace object to dictionary and remove "command" key
    kwargs = vars(args)
    kwargs.pop("command")
    kwargs.pop("short_help")
    kwargs.pop("fluid")
    kwargs.pop("kinetic")
    kwargs.pop("hybrid")
    kwargs.pop("toy")
    kwargs.pop("set_i")
    kwargs.pop("set_o")
    kwargs.pop("set_b")
    kwargs.pop("set_iob")
    kwargs.pop("refresh_models")

    # These options are stored in kwargs.config
    for key in ["input_type", "path", "linters", "iterations", "output_format"]:
        kwargs.pop(key, None)

    # start sub-command function with all parameters of that function
    # for k, v in kwargs.items():
    #     print(k, v)
    func(**kwargs)


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
