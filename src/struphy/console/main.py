#!/usr/bin/env python3
from argparse import HelpFormatter, _SubParsersAction, RawTextHelpFormatter

__version__ = "1.9.8"

def struphy():
    '''
    Struphy main executable. Performs argument parsing and sub-command call.
    '''

    import os, inspect
    import argparse
    from struphy.models import models
    from struphy.console.compile import struphy_compile
    from struphy.console.run import struphy_run
    from struphy.console.units import struphy_units
    from struphy.console.profile import struphy_profile
    from struphy.console.pproc import struphy_pproc
    from struphy.console.example import struphy_example
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
    version_message += 'Copyright 2022 (c) struphy dev team | CONTRIBUTING.md | Max Planck Institute for Plasma Physics\n'
    version_message += 'MIT license\n'

    # path message
    import struphy
    libpath = struphy.__path__[0]
    path_message = f'Struphy installation path: {libpath}\n'
    path_message += f'default input:             {libpath}/io/inp\n'
    path_message += f'default input:             {libpath}/io/out\n'
    path_message += f'template batch scripts:    {libpath}/io/batch'

    parser.add_argument('-v', '--version', action='version',
                        version=version_message)
    parser.add_argument('-p', '--path', action='version',
                        version=path_message, help='default installations and i/o paths')

    # create sub-commands and save name of sub-command into variable "command"
    subparsers = parser.add_subparsers(title='available commands',
                                       metavar='COMMAND',
                                       dest='command')

    # 1. "compile" sub-command
    parser_compile = subparsers.add_parser('compile',
                                           help='compile computational kernels, install psydac',
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
                                       formatter_class=lambda prog: argparse.HelpFormatter(
                                           prog, max_help_position=30),
                                       help='run a Struphy model',
                                       description='Run a Struphy model.',
                                       epilog='For more info on Struphy models, visit https://struphy.pages.mpcdf.de/struphy/sections/models.html')

    list_models = []
    for name, obj in inspect.getmembers(models):
        if inspect.isclass(obj):
            if name not in {'StruphyModel', }:
                list_models += [name]

    models_string = ''
    for mod in list_models[:-1]:
        models_string += '"' + mod + '", '

    models_string += 'or "' + list_models[-1] + '"'

    parser_run.add_argument('model',
                            type=str,
                            choices=list_models,
                            metavar='model',
                            help='which model to run (must be one of ' + models_string + ')',)

    parser_run.add_argument('-i', '--input',
                            type=str,
                            metavar='FILE',
                            help='parameter file (.yml) relative to <install_path>/struphy/io/inp/ (default=parameters.yml)',
                            default='parameters.yml',)

    parser_run.add_argument('--input-abs',
                            type=str,
                            metavar='FILE',
                            help='parameter file (.yml), absolute path',)

    parser_run.add_argument('-o', '--output',
                            type=str,
                            metavar='DIR',
                            help='output directory relative to <install_path>/struphy/io/out/ (default=sim_1)',
                            default='sim_1',)

    parser_run.add_argument('--output-abs',
                            type=str,
                            metavar='DIR',
                            help='output directory, absolute path',)

    parser_run.add_argument('-b', '--batch',
                            type=str,
                            metavar='FILE',
                            help='batch script relative to <install_path>/struphy/io/batch/ ', )

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
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=30),
        help='show physical units of a Struphy model',
        description='Show physical units of a Struphy model.',
        epilog='For more info on Struphy models, visit https://struphy.pages.mpcdf.de/struphy/sections/models.html')

    parser_units.add_argument('model',
                              type=str,
                              choices=list_models,
                              metavar='model',
                              help='which model to look at (must be one of ' + models_string + ')',)

    parser_units.add_argument('-i', '--input',
                              type=str,
                              metavar='FILE',
                              help='parameter file (.yml) relative to <install_path>/struphy/io/inp/ (default=parameters.yml)',
                              default='parameters.yml',)

    parser_units.add_argument('--input-abs',
                              type=str,
                              metavar='FILE',
                              help='parameter file (.yml), absolute path',)

    # 4. "profile" sub-command
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

    # 5. "pproc" sub-command
    parser_pproc = subparsers.add_parser(
        'pproc',
        help='post process data of a finished Struphy run',
        description='Post-process data of a finished Struphy run to prepare for diagnostics.')

    parser_pproc.add_argument('-d', '--dirr',
                              type=str,
                              metavar='DIR',
                              help='simulation output folder to post-process relative to <install_path>/struphy/io/out/ (default=sim_1)',
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

    # 6. "example" sub-command
    parser_example = subparsers.add_parser(
        'example',
        help='run a Struphy example',
        description='Run a complete Struphy example including prost-processing and plots.')
                
    files = os.listdir(os.path.join(libpath, 'examples'))

    list_examples = []
    for file in files:
        if file[-3:] == '.py' and file[0] != '_':
            list_examples += [file[:-3]]
            
    list_examples.sort()
                
    examples_string = ''
    for ex in list_examples[:-1]:
        examples_string += '"' + ex + '", '

    examples_string += 'or "' + list_examples[-1] + '"'

    parser_example.add_argument('case',
                                type=str,
                                metavar='case',
                                help='which example to run (must be one of ' + examples_string + ')')

    parser_example.add_argument('--mpi',
                                type=int,
                                metavar='N',
                                help='use "mpirun -n N" to launch the example in parallel (default=1)',
                                default=1)
    
    parser_example.add_argument('-d', '--diagnostics',
                                help='run diagnostics only, if output folder of example already exists',
                                action='store_true')

    # 7. "test" sub-command
    parser_test = subparsers.add_parser('test',
                                        help='run Struphy units tests',
                                        description='Run available tests. If no options are given, all units tests (serial and parallel) are run (2 processes for parallel tests).')

    parser_test.add_argument('--serial',
                             help='run serial unit tests only',
                             action='store_true')

    parser_test.add_argument('--mpi',
                             type=int,
                             metavar='N',
                             help='run parallel units tests only (with N number of processes)',
                             default=0)

    parser_test.add_argument('--codes',
                             help='run code tests',
                             action='store_true')

    # parse argument
    args = parser.parse_args()

    # if no arguments are passed, print help and exit
    if args.command is None:
        parser.print_help()
        exit()

    # handle argument dependencies in "sub-command"
    if args.command == 'test':

        # set default case "struphy test"
        if args.serial == False and args.mpi == 0:
            args.serial = True
            args.mpi = 2

        # if codes is given, don't run unit tests
        if args.codes:
            args.serial = False
            args.mpi = 0

    # load sub-command function (see functions below)
    func = locals()['struphy_' + args.command]

    # transform parser Namespace object to dictionary and remove "command" key
    kwargs = vars(args)
    kwargs.pop('command')

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