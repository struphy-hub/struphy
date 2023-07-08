def main(model_name, parameters, path_out, restart=False, runtime=300, save_step=1):
    """
    Run a Struphy model.

    Parameters
    ----------
    model_name : str
        The name of the model to run. Type "struphy run --help" in your terminal to see a list of available models.

    parameters : dict | str
        The simulation parameters. Can either be a dictionary OR a string (path of .yml parameter file)

    path_out : str
        The output directory. Will create a folder if it does not exist OR cleans the folder for new runs.

    restart : bool, optional
        Whether to restart a run (default=False).

    runtime : int, optional
        Maximum run time of simulation in minutes. Will finish the time integration once this limit is reached (default=300).

    save_step : int, optional
        When to save data output: every time step (save_step=1), every second time step (save_step=2), etc (default=1).
    """

    from struphy.models import fluid, kinetic, hybrid, toy
    from struphy.models.setup import pre_processing
    from struphy.models.output_handling import DataContainer

    import numpy as np
    import time

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
    objs = [fluid, kinetic, hybrid, toy]
    for obj in objs:
        try:
            model_class = getattr(obj, model_name)
        except AttributeError: 
            pass

    model = model_class(params, comm)

    # data object for saving (will either create new hdf5 files if restart==False or open existing files if restart==True)
    data = DataContainer(path_out, comm=comm)

    # time quantities (current time value and current time index)
    time_state = {}
    time_state['value'] = np.zeros(1, dtype=float)
    time_state['index'] = np.zeros(1, dtype=int)

    # save time quantities in group 'time/'
    for key, val in time_state.items():
        key_time = 'time/' + key
        key_time_restart = 'restart/time/' + key
        data.add_data({key_time: val})
        data.add_data({key_time_restart: val})

    # start a new simulation (set initial conditions according to parameter file)
    time_params = params['time']

    if rank == 0:
        print('\nINITIAL CONDITIONS:')
        
    if not restart:
        model.initialize_from_params()
        total_steps = str(
            int(round(time_params['Tend']/time_params['dt'])))

    # restart of an existing simulation (overwrite time quantities and load restart data from hdf5 files)
    else:
        time_state['value'][0] = data.file['restart/time/value'][-1]
        time_state['index'][0] = data.file['restart/time/index'][-1]

        total_steps = str(
            int(round((time_params['Tend'] - time_state['value'][0])/time_params['dt'])))

        model.initialize_from_restart(data)

    # list of model methods for diagnostics
    model_updates = []
    for method in dir(model):
        if 'update' in method:
            model_updates.append(getattr(model, method))

    # initial diagnostic data (will be saved in hdf5 file)
    for method in model_updates:
        method()

    # prepare hdf5 file structure
    save_keys_all, save_keys_end = model.initialize_data_output(data, size)

    if rank == 0:
        print('\nINITIAL SCALAR QUANTITIES:')
        model.print_scalar_quantities()

    # ======================== main time loop ======================
    if rank == 0:
        split_algo = time_params['split_algo']
        print(
            f'\nSTART TIME STEPPING WITH "{split_algo}" SPLITTING:')

    # time loop
    while True:

        # synchronize MPI processes and check if simulation end is reached
        comm.Barrier()
        run_time_now = (time.time() - start_simulation)/60

        # stop time loop?
        break_cond_1 = time_state['value'][0] >= time_params['Tend']
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

        # integrate the model for a time step dt
        model.integrate(time_params['dt'], time_params['split_algo'])

        # update time and index (round time to 10 decimals for a clean time grid!)
        time_state['value'][0] = round(
            time_state['value'][0] + time_params['dt'], 10)
        time_state['index'][0] += 1

        # update diagnostics data and save data
        if time_state['index'][0] % save_step == 0:

            # call diagnostics updates
            for method in model_updates:
                method()

            # extract FEM coefficients
            for key, val in model.em_fields.items():
                if 'params' not in key:
                    # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
                    val['obj'].extract_coeffs(update_ghost_regions=False)

            for _, val in model.fluid.items():
                for variable, subval in val.items():
                    if 'params' not in variable:
                        # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
                        subval['obj'].extract_coeffs(
                            update_ghost_regions=False)

            # save data (everything but restart data)
            data.save_data(keys=save_keys_all)

            # print current time and scalar quantities to screen
            if rank == 0:
                step = str(time_state['index'][0]).zfill(len(total_steps))

                message = 'time: {0:12.8f}/{1:12.8f}'.format(time_state['value'][0], time_params['Tend'])
                message += ' | ' + 'time step: ' + step + '/' + str(total_steps) 

                print(message, end='\n')
                model.print_scalar_quantities()
                print()
    # ===================================================================

    with open(path_out + '/meta.txt', 'a') as f:
        f.write('wall-clock time [min]:'.ljust(30) +
                str((end_simulation - start_simulation)/60.) + '\n')


if __name__ == '__main__':

    import argparse
    import os
    import struphy

    libpath = struphy.__path__[0]
    
    with open(os.path.join(libpath, 'io_path.txt')) as f:
        io_path = f.readlines()[0]

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
                        default=os.path.join(io_path, 'io/inp/parameters.yml'))

    # output (absolute path)
    parser.add_argument('-o', '--output',
                        type=str,
                        metavar='DIR',
                        help='absolute path of output folder (default=<struphy_path>/io/out/sim_1)',
                        default=os.path.join(io_path, 'io/out/sim_1'))

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

    # runtime
    parser.add_argument('-s', '--save-step',
                        type=int,
                        metavar='N',
                        help='how often to skip data saving (default=1, which means data is saved every time step)',
                        default=1)

    args = parser.parse_args()

    # solve the model
    main(args.model,
         args.input,
         args.output,
         args.restart,
         args.runtime,
         args.save_step)
