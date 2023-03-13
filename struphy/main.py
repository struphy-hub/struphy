#!/usr/bin/env python3

import argparse
import inspect
import time
import yaml
import datetime
import sysconfig
import sys
from mpi4py import MPI

from psydac.linalg.stencil import StencilVector
from struphy.post_processing.output_handling import DataContainer
from struphy.models import models


def main_test(run='Maxwell', compile=None):
    '''
    STRUPHY main execution file.
    '''

    # get arguments
    parser = argparse.ArgumentParser(
        prog = 'struphy',
        description = 'STRUcture-Preserving HYbrid codes',
        epilog = 'See https://gitlab.mpcdf.mpg.de/struphy/struphy for more information.',
        )
    
    list_models = []
    for name, obj in inspect.getmembers(models):
        if inspect.isclass(obj):
            if name not in {'StruphyModel', 'Propagator', 'PolarVector'}:
                list_models += [name]

    parser.add_argument('--run',
                        type=str,
                        choices=list_models,
                        metavar='<name>',
                        help='Which model to run: <name> in ' + str(list_models))
    
    parser.add_argument('--compile',
                        type=str,
                        choices=['all', 'no-openmp', 'delete', 'verbose'],
                        metavar='<opt>',
                        help='Compile kernels with or without openmp, with verbose screen output, or delete all .so files: <opt> in ' + str(['all', 'no-openmp', 'delete', 'verbose'])
                        )
    
    args = parser.parse_args()
    print(args.run)
    print(args.compile)

    if args.run is not None:
        run = args.run

    if args.compile is not None:
        compile = args.compile

    print(run)
    print(compile)


def main():
    '''
    Run a STRUPHY model.
    '''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # synchronize MPI processes and set start time of simulation
    comm.Barrier()
    start_simulation = time.time()

    # get arguments
    model_name = sys.argv[1]
    file_in = sys.argv[2]
    exit_flag = True

    if len(sys.argv) > 3:
        exit_flag = False
        path_out = sys.argv[3]
        file_meta = sys.argv[5]
        restart = sys.argv[6]
        max_sim_time = float(sys.argv[7])

    # convert restart string "false" or "true" to boolean
    if restart == 'true':
        restart = True
    else:
        restart = False
    
    # load simulation parameters
    with open(file_in) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
        
    # load STRUPHY model
    model_class = getattr(models, model_name)

    if exit_flag:
        model_class.print_units(params['model_units'])
        exit()
    # write meta data
    if rank == 0:
        with open(file_meta, 'w') as f:
            f.write('\ndate of simulation: '.ljust(20) +
                    str(datetime.datetime.now()) + '\n')
            f.write('platform: '.ljust(20) + sysconfig.get_platform() + '\n')
            f.write('python version: '.ljust(20) +
                    sysconfig.get_python_version() + '\n')
            f.write('model_name: '.ljust(20) + model_name + '\n')
            f.write('# processes: '.ljust(20) + str(comm.Get_size()) + '\n')

        print(
            f'\nMPI communicator initialized with {comm.Get_size()} process(es).\n')
        print('Starting model ' + model_name + '...\n')

    # load simulation parameters
    with open(file_in) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # instantiate STRUPHY model (will only allocate model objects and associated memory)
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
    if comm.Get_size() <= 512:
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
                data.add_data({key_field_restart: val['obj'].vector_stencil._data})
            else:
                for n in range(3):
                    key_component = key_field + '/' + str(n + 1)
                    key_component_restart = key_field_restart + '/' + str(n + 1)
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
                        key_component_restart = key_field_restart + '/' + str(n + 1)
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
        print(f'\nRank: {rank} | Initial time series saved.')
        model.print_scalar_quantities()

    # define stepping scheme
    dt = params['time']['dt']
    split_algo = params['time']['split_algo']

    def integrate_in_time():

        # First order in time
        if split_algo == 'LieTrotter':

            for propagator in model.propagators:
                propagator(dt)

        # Second order in time
        elif split_algo == 'Strang':

            assert len(model.propagators) > 1

            for propagator in model.propagators:
                propagator(dt/2.)

            for propagator in model.propagators[::-1]:
                propagator(dt/2.)

        else:
            raise NotImplementedError(
                f'Splitting scheme {split_algo} not available.')

    # start time integration
    if rank == 0:
        print('\nStart time integration: ' + split_algo)
        print()

    # time loop
    while True:

        # synchronize MPI processes and check if simulation end is reached
        comm.Barrier()

        break_cond_1 = time_steps_done * \
            params['time']['dt'] >= params['time']['Tend']
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
        integrate_in_time()
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
            total_steps = str(
                int(round(params['time']['Tend'] / params['time']['dt'])))
            str_len = len(total_steps)
            step = str(time_steps_done).zfill(str_len)
            message = 'time steps finished : ' + step + '/' + total_steps
            print('\r', message, end='\n')
            model.print_scalar_quantities()
            print()


if __name__ == '__main__':
    #run()
    main()