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
    from struphy.io.setup import pre_processing
    from struphy.io.output_handling import DataContainer

    import numpy as np
    import time

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # synchronize MPI processes to set same start time of simulation for all processes
    comm.Barrier()
    start_simulation = time.time()

    # loading of simulation parameters, creating output folder and printing information to screen
    params = pre_processing(model_name,
                            parameters,
                            path_out,
                            restart,
                            runtime,
                            rank,
                            size)

    # instantiate Struphy model (will allocate model objects and associated memory)
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

    # add time quantities to data object for saving
    for key, val in time_state.items():
        key_time = 'time/' + key
        key_time_restart = 'restart/time/' + key
        data.add_data({key_time: val})
        data.add_data({key_time_restart: val})

    time_params = params['time']

    # set initial conditions for all variables
    if not restart:
        model.initialize_from_params()

        total_steps = str(
            int(round(time_params['Tend']/time_params['dt'])))

    else:
        model.initialize_from_restart(data)

        time_state['value'][0] = data.file['restart/time/value'][-1]
        time_state['index'][0] = data.file['restart/time/index'][-1]

        total_steps = str(
            int(round((time_params['Tend'] - time_state['value'][0])/time_params['dt'])))

    # compute initial scalars and kinetic data
    model.update_scalar_quantities()
    model.update_markers_to_be_saved()
    model.update_distr_function()

    # add all variables to be saved to data object
    save_keys_all, save_keys_end = model.initialize_data_output(data, size)

    # ======================== main time loop ======================
    if rank == 0:
        print('\nINITIAL SCALAR QUANTITIES:')
        model.print_scalar_quantities()

        split_algo = time_params['split_algo']
        print(
            f'\nSTART TIME STEPPING WITH "{split_algo}" SPLITTING:')

    # time loop
    run_time_now = 0.0
    while True:

        comm.Barrier()

        # stop time loop?
        break_cond_1 = time_state['value'][0] >= time_params['Tend']
        break_cond_2 = run_time_now > runtime

        if break_cond_1 or break_cond_2:
            # save restart data (other data already saved below)
            data.save_data(keys=save_keys_end)
            data.file.close()
            end_simulation = time.time()
            if rank == 0:
                print('wall-clock time of simulation [sec]: ',
                      end_simulation - start_simulation)
                print()
            break

        # perform one time step dt
        t0 = time.time()
        model.integrate(time_params['dt'], time_params['split_algo'])
        t1 = time.time()

        # update time and index (round time to 10 decimals for a clean time grid!)
        time_state['value'][0] = round(
            time_state['value'][0] + time_params['dt'], 10)
        time_state['index'][0] += 1
        
        run_time_now = (time.time() - start_simulation)/60

        # update diagnostics data and save data
        if time_state['index'][0] % save_step == 0:

            # compute scalars and kinetic data
            model.update_scalar_quantities()
            model.update_markers_to_be_saved()
            model.update_distr_function()

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

                message = 'time step: ' + step + '/' + str(total_steps)
                message += ' | ' + 'time: {0:10.5f}/{1:10.5f}'.format(
                    time_state['value'][0], time_params['Tend'])
                message += ' | ' + \
                    'wall clock [s]: {0:8.4f} | last step duration [s]: {1:8.4f}'.format(
                        run_time_now*60, t1 - t0)

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
    import yaml

    libpath = struphy.__path__[0]

    with open(os.path.join(libpath, 'state.yml')) as f:
        state = yaml.load(f, Loader=yaml.FullLoader)

    o_path = state['o_path']

    parser = argparse.ArgumentParser(description='Run an Struphy model.')

    # model
    parser.add_argument('model',
                        type=str,
                        metavar='model',
                        help='the name of the model to run')

    # input (absolute path)
    parser.add_argument('-i', '--input',
                        type=str,
                        metavar='FILE',
                        help='absolute path of parameter file (.yml)',)

    # output (absolute path)
    parser.add_argument('-o', '--output',
                        type=str,
                        metavar='DIR',
                        help='absolute path of output folder (default=<out_path>/sim_1)',
                        default=os.path.join(o_path, 'sim_1'))

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
