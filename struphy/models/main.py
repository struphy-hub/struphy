def main(model_name, parameters, path_out, restart=False, runtime=300):
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

    restart : bool, optional
        Whether to restart a run (default=False).

    runtime : int, optional
        Maximum run time of simulation in minutes. Will finish the time integration once this limit is reached (default=300).
    """

    from struphy.models import models
    from struphy.models.utilities import pre_processing
    from struphy.post_processing.output_handling import DataContainer

    from psydac.linalg.stencil import StencilVector

    import time, yaml

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # synchronize MPI processes to set same start time of simulation for all processes
    comm.Barrier()
    start_simulation = time.time()

    # call pre-processing (preparation of parameters, output folder and printing information to screen)
    params = pre_processing(model_name, 
                            parameters,
                            path_out, 
                            restart, 
                            runtime, 
                            rank, 
                            size)

    # instantiate STRUPHY model (will only allocate model objects and associated memory)
    model = getattr(models, model_name)(params, comm)

    # data object for saving (will either create new hdf5 files if restart=False or open existing files if restart=True)
    data = DataContainer(path_out, comm=comm)

    # start a new simulation (set initial conditions according to parameter file)
    if not restart:
        time_steps_done = 0
        model.initialize_from_params() 

    # restart of an existing simulation (load restart data from hdf5 files, no diagnostics needed at this stage)
    else:
        time_steps_done = data.file['scalar/time'].size - 1
        model.initialize_from_restart(data.file)

    # print plasma params
    if rank == 0:
        model.print_plasma_params()

    # initial diagnostic data (will be saved in hdf5 file)
    model.update_scalar_quantities(0.)
    model.update_markers_to_be_saved()
    model.update_distr_function()
    
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
        run_time_now = (time.time() - start_simulation)/60

        # stop time loop?
        break_cond_1 = time_steps_done*dt >= Tend
        break_cond_2 = run_time_now > runtime
        
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


if __name__ == '__main__':

    import argparse
    import os
    import struphy

    libpath = struphy.__path__[0]
    
    parser = argparse.ArgumentParser(description='Run an Struphy model.')
    
    # model
    parser.add_argument('model',
                        type=str,
                        metavar='model',
                        help='the name of the model to run (default=Maxwell)')
    
    # input (absolute path)
    parser.add_argument('-i', '--input',
                        type=str,
                        metavar='FILE',
                        help='absolute path of parameter file (.yml) (default=<struphy_path>/io/inp/parameters.yml)',
                        default=os.path.join(libpath, 'io/inp/parameters.yml'))
    
    # output (absolute path)
    parser.add_argument('-o', '--output',
                        type=str,
                        metavar='DIR',
                        help='absolute path of output folder (default=<struphy_path>/io/out/sim_1)',
                        default=os.path.join(libpath, 'io/out/sim_1'))
    
    # restart
    parser.add_argument('-r', '--restart',
                         help='restart the simulation in the output folder specified under -o',
                         action='store_true')
    
    # runtime
    parser.add_argument('--runtime',
                        type=int,
                        metavar='N',
                        help='maximum wall-clock time of program in minutes (default=300)',
                        default=300)

    args = parser.parse_args()

    # solve the model
    main(args.model, 
         args.input, 
         args.output, 
         args.restart, 
         args.runtime)
