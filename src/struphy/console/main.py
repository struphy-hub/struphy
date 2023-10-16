#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from argparse import HelpFormatter, _SubParsersAction, RawTextHelpFormatter
import argcomplete
import importlib.metadata

__version__ = importlib.metadata.version("struphy")


def struphy():
    '''
    Struphy main executable. Performs argument parsing and sub-command call.
    '''

    import os
    import inspect
    import argparse
    from struphy.models import fluid, kinetic, hybrid, toy
    from struphy.console.compile import struphy_compile
    from struphy.console.run import struphy_run
    from struphy.console.units import struphy_units
    from struphy.console.params import struphy_params
    from struphy.console.profile import struphy_profile
    from struphy.console.pproc import struphy_pproc
    from struphy.console.tutorials import struphy_tutorials
    from struphy.console.test import struphy_test

    # create argument parser
    epilog_message = 'Run "struphy COMMAND --help" for more information on a command.\n\n'
    epilog_message += 'For more help on how to use Struphy, see https://struphy.pages.mpcdf.de/struphy/index.html'

    parser = argparse.ArgumentParser(prog='struphy',
                                     formatter_class=CustomFormatter,
                                     description='Struphy: STRUcture-Preserving HYbrid codes for plasma physics.',
                                     epilog=epilog_message)

    # version message
    version_message = f'Struphy {__version__}\n'
    version_message += 'Copyright 2019-2023 (c) Struphy dev team | Max Planck Institute for Plasma Physics\n'
    version_message += 'MIT license\n'

    # path message
    import struphy
    libpath = struphy.__path__[0]

    try:
        with open(os.path.join(libpath, 'i_path.txt'), 'r') as f:
            i_path = f.readlines()[0]
    except FileNotFoundError:
        # setting default input path
        with open(os.path.join(libpath, 'i_path.txt'), 'w') as f:
            i_path = os.path.join(libpath, 'io/inp')
            f.write(i_path)

    try:
        with open(os.path.join(libpath, 'o_path.txt'), 'r') as f:
            o_path = f.readlines()[0]
    except FileNotFoundError:
        # setting default output path
        with open(os.path.join(libpath, 'o_path.txt'), 'w') as f:
            o_path = os.path.join(libpath, 'io/out')
            f.write(o_path)

    try:
        with open(os.path.join(libpath, 'b_path.txt'), 'r') as f:
            b_path = f.readlines()[0]
    except FileNotFoundError:
        # setting default output path
        with open(os.path.join(libpath, 'b_path.txt'), 'w') as f:
            b_path = os.path.join(libpath, 'io/batch')
            f.write(b_path)

    path_message = f'Struphy installation path: {libpath}\n'
    path_message += f'current input:             {i_path}\n'
    path_message += f'current output:            {o_path}\n'
    path_message += f'current batch scripts:     {b_path}'
    
    # check parameter file in current input path:
    all_files = os.listdir(i_path)
    params_files = []
    for name in all_files:
        if '.yml' in name or '.yaml' in name:
            params_files += [name]
            
    # check output folders in current output path:
    all_folders = os.listdir(o_path)
    out_folders = []
    for name in all_folders:
        if '.' not in name:
            out_folders += [name]
            
    # check batch scripts in current batch path:
    all_files = os.listdir(b_path)
    batch_files = []
    for name in all_files:
        if '.sh' in name:
            batch_files += [name]

    parser.add_argument('-v', '--version', action='version',
                        version=version_message)
    parser.add_argument('-p', '--path', action='version',
                        version=path_message, help='default installations and i/o paths')
    parser.add_argument('--set-i',
                        type=str,
                        metavar='PATH',
                        help='make PATH the new default Input folder ("." to use cwd, "d" to use default <install-path>/io/inp/)',)
    parser.add_argument('--set-o',
                        type=str,
                        metavar='PATH',
                        help='make PATH the new default Output folder ("." to use cwd, "d" to use default <install-path>/io/out/)',)
    parser.add_argument('--set-b',
                        type=str,
                        metavar='PATH',
                        help='make PATH the new default Batch folder ("." to use cwd, "d" to use default <install-path>/io/batch/)',)
    parser.add_argument('--set-iob',
                        type=str,
                        metavar='PATH',
                        help='make PATH the new default folder for io/inp/, io/out and io/batch ("." to use cwd, "d" to use default <install-path>)',)

    # create sub-commands and save name of sub-command into variable "command"
    subparsers = parser.add_subparsers(title='available commands',
                                       metavar='COMMAND',
                                       dest='command')

    # 1. "compile" sub-command
    parser_compile = subparsers.add_parser('compile',
                                           help='compile computational kernels, install psydac (on first call only)',
                                           description='Compile Struphy kernels using pyccel, https://github.com/pyccel/pyccel.')

    parser_compile.add_argument('--no-openmp',
                                help='compile without OpenMP',
                                action='store_true')

    parser_compile.add_argument('-d', '--delete',
                                help='remove .f90 and .so files (for running pure Python code)',
                                action='store_true')

    parser_compile.add_argument('-v', '--verbose',
                                help='call pyccel with --verbose compiler option',
                                action='store_true')

    # 2. "run" sub-command
    parser_run = subparsers.add_parser('run',
                                       formatter_class=lambda prog: argparse.RawTextHelpFormatter(
                                           prog, max_help_position=30),
                                       help='run a Struphy model',
                                       description='Run a Struphy model.',
                                       epilog='For more info on Struphy models, visit https://struphy.pages.mpcdf.de/struphy/sections/models.html')

    list_fluid = []
    fluid_string = ''
    for name, obj in inspect.getmembers(fluid):
        if inspect.isclass(obj):
            if name not in {'StruphyModel', }:
                list_fluid += [name]
                fluid_string += '"' + name + '"\n'

    list_kinetic = []
    kinetic_string = ''
    for name, obj in inspect.getmembers(kinetic):
        if inspect.isclass(obj):
            if name not in {'StruphyModel', }:
                list_kinetic += [name]
                kinetic_string += '"' + name + '"\n'

    list_hybrid = []
    hybrid_string = ''
    for name, obj in inspect.getmembers(hybrid):
        if inspect.isclass(obj):
            if name not in {'StruphyModel', }:
                list_hybrid += [name]
                hybrid_string += '"' + name + '"\n'

    list_toy = []
    toy_string = ''
    for name, obj in inspect.getmembers(toy):
        if inspect.isclass(obj):
            if name not in {'StruphyModel', }:
                list_toy += [name]
                toy_string += '"' + name + '"\n'

    list_models = list_fluid + list_kinetic + list_hybrid + list_toy

    # model message
    model_message = 'Available models:\n'
    model_message += '\nFluid models:\n'
    model_message += '-------------\n'
    model_message += fluid_string
    model_message += '\nKinetic models:\n'
    model_message += '---------------\n'
    model_message += kinetic_string
    model_message += '\nHybrid models:\n'
    model_message += '--------------\n'
    model_message += hybrid_string
    model_message += '\nToy models:\n'
    model_message += '-----------\n'
    model_message += toy_string
    # version_message += 'MIT license\n'

    parser_run.add_argument('model',
                            type=str,
                            choices=list_models,
                            metavar='MODEL',
                            help=model_message,)

    parser_run.add_argument('-i', '--inp',
                            type=str,
                            choices=params_files,
                            metavar='FILE',
                            help='parameter file (.yml) in current I/O path',)

    parser_run.add_argument('--input-abs',
                            type=str,
                            metavar='FILE',
                            help='parameter file (.yml), absolute path',)

    parser_run.add_argument('-o', '--output',
                            type=str,
                            metavar='DIR',
                            help='output directory relative to current I/O path (default=sim_1)',
                            default='sim_1',)

    parser_run.add_argument('--output-abs',
                            type=str,
                            metavar='DIR',
                            help='output directory, absolute path',)

    parser_run.add_argument('-b', '--batch',
                            type=str,
                            choices=batch_files,
                            metavar='FILE',
                            help='batch script in current I/O path', )

    parser_run.add_argument('--batch-abs',
                            type=str,
                            metavar='FILE',
                            help='batch script, absolute path',)

    parser_run.add_argument('--runtime',
                            type=int,
                            metavar='N',
                            help='maximum wall-clock time of program in minutes (default=300)',
                            default=300,)

    parser_run.add_argument('-s', '--save-step',
                            type=int,
                            metavar='N',
                            help='how often to save data in hdf5 file, i.e. every "save-step" time step (default=1, which is every time step)',
                            default=1,)

    parser_run.add_argument('-r', '--restart',
                            help='restart the simulation in the output folder specified under -o',
                            action='store_true',)

    parser_run.add_argument('--mpi',
                            type=int,
                            metavar='N',
                            help='use "mpirun -n N" to launch a parallel Struphy run (default=1)',
                            default=1,)

    parser_run.add_argument('--debug',
                            help='launch a Cobra debug run, see https://docs.mpcdf.mpg.de/doc/computing/cobra-user-guide.html#interactive-debug-runs',
                            action='store_true',)

    # 3. "units" sub-command
    parser_units = subparsers.add_parser(
        'units',
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=30),
        help='show physical units of a Struphy model',
        description='Show physical units of a Struphy model.',
        epilog='For more info on Struphy models, visit https://struphy.pages.mpcdf.de/struphy/sections/models.html')

    parser_units.add_argument('model',
                              type=str,
                              choices=list_models,
                              metavar='MODEL',
                              help=model_message,)

    parser_units.add_argument('-i', '--input',
                              type=str,
                              choices=params_files,
                              metavar='FILE',
                              help='parameter file (.yml) relative to current I/O path. If absent, default parameters are used.',)

    parser_units.add_argument('--input-abs',
                              type=str,
                              metavar='FILE',
                              help='parameter file (.yml), absolute path',)

    # 4. "params" sub-command
    parser_params = subparsers.add_parser(
        'params',
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=30),
        help='create default parameter file for a model',
        description='Creates a default parameter file for a specific model.')

    parser_params.add_argument('model',
                               type=str,
                               choices=list_models,
                               metavar='MODEL',
                               help=model_message,)

    parser_params.add_argument('-f', '--file',
                               type=str,
                               metavar='FILE',
                               help='name of the parameter file (.yml) to be created in the current I/O path (default=params_<model>.yml)',)

    parser_params.add_argument('-o', '--options',
                               help='show model options',
                               action='store_true')
    
    parser_params.add_argument('-y', '--yes',
                               help='Say yes on prompt to overwrite .yml FILE',
                               action='store_true')

    # 5. "profile" sub-command
    parser_profile = subparsers.add_parser(
        'profile',
        help='profile finished Struphy runs',
        description='Compare profiling data of finished Struphy runs. For each function in a predefined filter, displays: ncalls, tottime, percall and cumtime.')

    parser_profile.add_argument('dirs',
                                type=str,
                                nargs='+',
                                metavar='DIR',
                                help='simulation ouput folders')

    parser_profile.add_argument('--replace',
                                help='replace module names with class names for better info',
                                action='store_true',)

    parser_profile.add_argument('--all',
                                help='display the 50 most expensive function calls, without applying the predefined filter',
                                action='store_true',)

    parser_profile.add_argument('--n-lines',
                                type=int,
                                metavar='N',
                                help='plot the N most time consuming calls in profiling analysis (default=6)',
                                default=6,)

    parser_profile.add_argument('--print-callers',
                                type=str,
                                metavar='STR',
                                help='string STR that identifies functions for which to print callers (default=None)',
                                default=None)

    parser_profile.add_argument('--savefig',
                                type=str,
                                metavar='NAME',
                                help='save (and dont display) the profile figure under NAME, relative to current output path.',)

    # 6. "pproc" sub-command
    parser_pproc = subparsers.add_parser(
        'pproc',
        help='post process data of a finished Struphy run',
        description='Post-process data of a finished Struphy run to prepare for diagnostics.')

    parser_pproc.add_argument('-d', '--dirr',
                              type=str,
                              choices=out_folders,
                              metavar='DIR',
                              help='simulation output folder to post-process relative to current I/O path (default=sim_1)',
                              default='sim_1',)

    parser_pproc.add_argument('--dir-abs',
                              type=str,
                              metavar='DIR',
                              help='simulation output folder to post-process, absolute path',)

    parser_pproc.add_argument('-s', '--step',
                              type=int,
                              metavar='N',
                              help='do post-processing every N-th time step (default=1).',
                              default=1)

    parser_pproc.add_argument('--celldivide',
                              type=int,
                              metavar='N',
                              help='divide each grid cell by N for field evaluation (default=1)',
                              default=1)

    # 7. "tutorials" sub-command
    parser_tutorials = subparsers.add_parser(
        'tutorials',
        help='run Struphy simulation(s) for notebook tutorials',
        description='Run Struphy simulation(s) necessary to run notebook tutorials.',
        epilog='Find the notebook tutorials at https://struphy.pages.mpcdf.de/struphy/sections/tutorials.html')

    parser_tutorials.add_argument('-n',
                                  type=int,
                                  help='specific tutorial simulation to run (int, optional)',
                                  default=None)

    # 8. "test" sub-command
    parser_test = subparsers.add_parser('test',
                                        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
                                            prog, max_help_position=30),
                                        help='run Struphy tests',
                                        description='Run available unit tests; or test Struphy models (with post processing) for all available options and for three different mappings (Cuboid, HollowTorus and Tokamak).')

    parser_test.add_argument('group',
                             type=str,
                             choices=list_models + ['codes'] + ['unit'],
                             metavar='GROUP',
                             help='can be a model name, "codes" (tests all models on 2 mpi processes) or "unit" (perform unit tests)',)

    parser_test.add_argument('--mpi',
                             type=int,
                             metavar='N',
                             help='run parallel unit tests (with N number of processes) instead of serial ones',
                             default=0)

    parser_test.add_argument('--fast',
                             help='test model(s) just in slab geometry (Cuboid)',
                             action='store_true')

    # parse argument
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # if no arguments are passed, print help and exit
    print_help = True
    for key, val in args.__dict__.items():
        if val is not None:
            print_help = False

    if print_help:
        parser.print_help()
        exit()

    # set default in path
    if args.set_i:
        if args.set_i == '.':
            i_path = os.getcwd()
        elif args.set_i == 'd':
            i_path = os.path.join(libpath, 'io/inp')
        else:
            i_path = args.set_i
            try:
                os.makedirs(i_path)
            except:
                pass

        i_path = os.path.abspath(i_path)

        with open(os.path.join(libpath, 'i_path.txt'), 'w') as f:
            f.write(i_path)

        print(f'New default Input path has been set:')
        import subprocess
        subprocess.run(['struphy', '-p'])

        exit()

    # set default out path
    if args.set_o:
        if args.set_o == '.':
            o_path = os.getcwd()
        elif args.set_o == 'd':
            o_path = os.path.join(libpath, 'io/out')
        else:
            o_path = args.set_o
            try:
                os.makedirs(o_path)
            except:
                pass

        o_path = os.path.abspath(o_path)

        with open(os.path.join(libpath, 'o_path.txt'), 'w') as f:
            f.write(o_path)

        print(f'New default Out path has been set:')
        import subprocess
        subprocess.run(['struphy', '-p'])

        exit()

    # set default out path
    if args.set_b:
        if args.set_b == '.':
            b_path = os.getcwd()
        elif args.set_b == 'd':
            b_path = os.path.join(libpath, 'io/batch')
        else:
            b_path = args.set_b
            try:
                os.makedirs(b_path)
            except:
                pass

        b_path = os.path.abspath(b_path)

        with open(os.path.join(libpath, 'b_path.txt'), 'w') as f:
            f.write(b_path)

        print(f'New default Batch path has been set:')
        import subprocess
        subprocess.run(['struphy', '-p'])

        exit()

    # set paths for inp, out and batch (with io/inp etc. prefices)
    if args.set_iob:
        if args.set_iob == '.':
            path = os.getcwd()
        elif args.set_iob == 'd':
            path = libpath
        else:
            path = args.set_iob

        i_path = os.path.join(path, 'io/inp')
        o_path = os.path.join(path, 'io/out')
        b_path = os.path.join(path, 'io/batch')

        import subprocess
        subprocess.run(['struphy', '--set-i', i_path])
        subprocess.run(['struphy', '--set-o', o_path])
        subprocess.run(['struphy', '--set-b', b_path])

        exit()

    # load sub-command function (see functions below)
    func = locals()['struphy_' + args.command]

    # transform parser Namespace object to dictionary and remove "command" key
    kwargs = vars(args)
    kwargs.pop('command')
    kwargs.pop('set_i')
    kwargs.pop('set_o')
    kwargs.pop('set_b')
    kwargs.pop('set_iob')

    # start sub-command function with all parameters of that function
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
