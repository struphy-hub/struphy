#!/usr/bin/env python3
from argparse import HelpFormatter, _SubParsersAction, RawTextHelpFormatter, ArgumentDefaultsHelpFormatter

__version__ = "1.9.7"

__all__ = ['struphy', 'main',
           'NoSubparsersMetavarFormatter', 'CustomFormatter']


def struphy():
    '''
    Struphy main executable. Performs argument parsing and sub-command call.
    '''

    import inspect
    import argparse
    from struphy.models import models
    from struphy.console import example
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
                                           help='compile computational kernels',
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
        help='post process data of finished Struphy runs',
        description='Post-process data of finished Struphy runs to prepare for diagnostics.')

    parser_pproc.add_argument('dirs',
                              type=str,
                              nargs='+',
                              metavar='DIR',
                              help='simulation ouput folders')

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

    list_examples = []
    for name, obj in inspect.getmembers(example):
        if inspect.isfunction(obj):
            if name not in {'struphy_example', }:
                list_examples += [name]

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


def main(model_name, parameters, path_out, restart=False, max_sim_time=300):
    """
    Run a Struphy model.

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
    """

    from struphy.models import models
    from struphy.post_processing.output_handling import DataContainer

    from psydac.linalg.stencil import StencilVector

    import os
    import time
    import yaml
    import sys

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # synchronize MPI processes and set start time of simulation
    comm.Barrier()
    start_simulation = time.time()

    # get command-line arguments if present
    if len(sys.argv) == 6:
        model_name = sys.argv[1]
        parameters = sys.argv[2]
        path_out = sys.argv[3]
        restart = sys.argv[4] == 'True'
        max_sim_time = float(sys.argv[5])

    # call pre-processing (preparation of output folder, information printing etc.)
    params = pre_processing(model_name, parameters,
                            path_out, restart, max_sim_time, rank, size)

    # instantiate STRUPHY model (will only allocate model objects and associated memory)
    model_class = getattr(models, model_name)
    model = model_class(params, comm)

    # data object for saving (will either create new hdf5 files if restart=False or open existing files if restart=True)
    data = DataContainer(path_out, comm=comm)

    # start of a new simulation (set initial conditions according to parameter file)
    if not restart:
        time_steps_done = 0
        model.initialize_from_params()

        # initial diagnostic data (will be saved in hdf5 file)
        model.update_scalar_quantities(0.)
        model.update_markers_to_be_saved()
        model.update_distr_function()

    # restart of an existing simulation (load restart data from hdf5 files, no diagnostics needed at this stage)
    else:
        time_steps_done = data.file['scalar/time'].size - 1
        model.initialize_from_restart(data.file)

    # save scalar quantities in group 'scalar/'
    for key, val in model.scalar_quantities.items():
        key_scalar = 'scalar/' + key
        data.add_data({key_scalar: val})

    # store grid_info only for runs with 512 ranks or smaller
    if size <= 512:
        data.file['scalar'].attrs['grid_info'] = model.derham.domain_array
    else:
        data.file['scalar'].attrs['grid_info'] = model.derham.domain_array[0]

    # save electromagentic fields/potentials data in group 'feec/'
    for key, val in model.em_fields.items():
        if 'params' not in key:
            key_field = 'feec/' + key
            key_field_restart = 'restart/' + key

            # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
            val['obj'].extract_coeffs(update_ghost_regions=False)

            # save numpy array to be updated each time step.
            if isinstance(val['obj'].vector_stencil, StencilVector):
                data.add_data({key_field: val['obj'].vector_stencil._data})
                data.add_data(
                    {key_field_restart: val['obj'].vector_stencil._data})
            else:
                for n in range(3):
                    key_component = key_field + '/' + str(n + 1)
                    key_component_restart = key_field_restart + \
                        '/' + str(n + 1)
                    data.add_data(
                        {key_component: val['obj'].vector_stencil[n]._data})
                    data.add_data(
                        {key_component_restart: val['obj'].vector_stencil[n]._data})

            # save field meta data
            data.file[key_field].attrs['space_id'] = val['obj'].space_id
            data.file[key_field].attrs['starts'] = val['obj'].starts
            data.file[key_field].attrs['ends'] = val['obj'].ends
            data.file[key_field].attrs['pads'] = val['obj'].pads

    # save fluid data in group 'feec/'
    for species, val in model.fluid.items():

        species_path = 'feec/' + species + '_'
        species_path_restart = 'restart/' + species + '_'

        for variable, subval in val.items():
            if 'params' not in variable:
                key_field = species_path + variable
                key_field_restart = species_path_restart + variable

                # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
                subval['obj'].extract_coeffs(update_ghost_regions=False)

                # save numpy array to be updated each time step.
                if isinstance(subval['obj'].vector_stencil, StencilVector):
                    data.add_data(
                        {key_field: subval['obj'].vector_stencil._data})
                    data.add_data(
                        {key_field_restart: subval['obj'].vector_stencil._data})
                else:
                    for n in range(3):
                        key_component = key_field + '/' + str(n + 1)
                        key_component_restart = key_field_restart + \
                            '/' + str(n + 1)
                        data.add_data(
                            {key_component: subval['obj'].vector_stencil[n]._data})
                        data.add_data(
                            {key_component_restart: subval['obj'].vector_stencil[n]._data})

                # save field meta data
                data.file[key_field].attrs['space_id'] = subval['obj'].space_id
                data.file[key_field].attrs['starts'] = subval['obj'].starts
                data.file[key_field].attrs['ends'] = subval['obj'].ends
                data.file[key_field].attrs['pads'] = subval['obj'].pads

    # save kinetic data in group 'kinetic/'
    for key, val in model.kinetic.items():
        key_spec = 'kinetic/' + key
        key_spec_restart = 'restart/' + key

        data.add_data({key_spec_restart: val['obj']._markers})

        for key1, val1 in val['kinetic_data'].items():
            key_dat = key_spec + '/' + key1

            if isinstance(val1, dict):
                for key2, val2 in val1.items():
                    key_f = key_dat + '/' + key2
                    data.add_data({key_f: val2})

                    dims = (len(key2) - 2)//3 + 1
                    for dim in range(dims):
                        data.file[key_f].attrs['bin_centers' + '_' + str(dim + 1)] = val['bin_edges'][key2][dim][:-1] + (
                            val['bin_edges'][key2][dim][1] - val['bin_edges'][key2][dim][0])/2

            else:
                data.add_data({key_dat: val1})

    # keys to be saved at each time step and only at end (restart)
    save_keys_each = []
    save_keys_end = []

    for key in data.dset_dict:
        if len(key) <= 7:
            save_keys_each.append(key)
        else:
            if key[:7] == 'restart':
                save_keys_end.append(key)
            else:
                save_keys_each.append(key)

    if rank == 0:
        print('\nInitial time series saved.')
        model.print_scalar_quantities()

    # start time integration
    dt = params['time']['dt']
    Tend = params['time']['Tend']
    split_algo = params['time']['split_algo']

    if rank == 0:
        print(
            f'\nStart time integration with {split_algo} splitting algorithm')
        print()

    # time loop
    while True:

        # synchronize MPI processes and check if simulation end is reached
        comm.Barrier()

        break_cond_1 = time_steps_done*dt >= Tend
        break_cond_2 = (time.time() - start_simulation) / \
            60 > max_sim_time

        # stop time loop?
        if break_cond_1 or break_cond_2:
            # save restart data
            data.save_data(keys=save_keys_end)
            # close output file and time loop
            data.file.close()
            # om.export_space_info() TODO: Psydac Derham functionaltiy not yet implemented.
            end_simulation = time.time()
            if rank == 0:
                print('wall-clock time of simulation [sec]: ',
                      end_simulation - start_simulation)
                print()
            break

        # call time integrator for time stepping
        model.integrate(dt, split_algo)
        time_steps_done += 1

        # update time series
        model.update_scalar_quantities(dt*time_steps_done)
        model.update_markers_to_be_saved()
        model.update_distr_function()

        # extract FEM coefficients
        for key, val in model.em_fields.items():
            if 'params' not in key:
                # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
                val['obj'].extract_coeffs(update_ghost_regions=False)

        for species, val in model.fluid.items():
            for variable, subval in val.items():
                if 'params' not in variable:
                    # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
                    subval['obj'].extract_coeffs(update_ghost_regions=False)

        # save data (everything but restart data)
        data.save_data(keys=save_keys_each)

        # print number of finished time steps and current energies
        if rank == 0 and time_steps_done % 1 == 0:
            total_steps = str(int(round(Tend/dt)))
            str_len = len(total_steps)
            step = str(time_steps_done).zfill(str_len)
            message = 'time steps finished : ' + step + '/' + total_steps
            print(message, end='\n')
            model.print_scalar_quantities()
            print()

#########################
##### Helper functions ##
#########################


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

        # clean output folder if it already exists
        else:

            # remove eval_fields folder
            folder = os.path.join(path_out, 'eval_fields')
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print('Removed folder ' + folder)

            # remove vtk folder
            folder = os.path.join(path_out, 'vtk')
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print('Removed folder ' + folder)

            # remove kinetic_data folder
            folder = os.path.join(path_out, 'kinetic_data')
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print('Removed folder ' + folder)

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

            # remove hdf5 files (if NOT a restart)
            if not restart:
                files = glob.glob(os.path.join(path_out, '*.hdf5'))
                for n, file in enumerate(files):
                    os.remove(file)
                    if n < 10:  # print only forty statements in case of many processes
                        print('Removed file ' + file)

    # save "parameters" dictionary as .yml file
    if isinstance(parameters, dict):
        parameters_path = os.path.join(path_out, 'parameters.yml')

        # write parameters to file
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
        # copy parameter file (if it does not already exist in output folder)
        if not os.path.exists(os.path.join(path_out, 'parameters.yml')):
            shutil.copy2(parameters_path, os.path.join(
                path_out, 'parameters.yml'))

    if mpi_rank == 0:

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

        # write meta data
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
            f.write('# processes: '.ljust(30) + str(mpi_size) + '\n')

    return params


if __name__ == '__main__':

    import os
    import struphy

    libpath = struphy.__path__[0]
    print(libpath)

    # default model
    model = 'Maxwell'

    # default parameter file
    input_file = os.path.join(libpath, 'io/inp/parameters.yml')

    # default output
    output_folder = os.path.join(libpath, 'io/out/sim_1')

    # run model
    main(model, input_file, output_folder, restart=False, max_sim_time=15)
