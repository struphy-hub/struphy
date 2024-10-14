def main(
    model_name: str,
    parameters: dict | str,
    path_out: str,
    *,
    restart: bool = False,
    runtime: int = 300,
    save_step: int = 1,
    supress_out: bool = False,
):
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

    supress_out : bool
        Whether to supress screen output during time integration.
    """

    from struphy.models.base import StruphyModel
    from struphy.feec.psydac_derham import Derham
    from struphy.models import fluid, kinetic, hybrid, toy
    from struphy.io.setup import pre_processing, setup_domain_cloning
    from struphy.profiling.profiling import ProfileRegion
    from struphy.io.output_handling import DataContainer
    from pyevtk.hl import gridToVTK

    import copy
    import numpy as np
    import time
    import os

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # synchronize MPI processes to set same start time of simulation for all processes
    comm.Barrier()
    start_simulation = time.time()

    # loading of simulation parameters, creating output folder and printing information to screen
    params = pre_processing(
        model_name, parameters, path_out, restart, runtime, save_step, rank, size
    )

    # Setup domain cloning communicators
    # MPI.COMM_WORLD     : comm
    # within a clone:    : sub_comm
    # between the clones : inter_comm
    # A copy of the params is used since the parker params are updated.
    params, inter_comm, sub_comm = setup_domain_cloning(
        comm, copy.deepcopy(params), params['grid']['Nclones'])

    # instantiate Struphy model (will allocate model objects and associated memory)
    objs = [fluid, kinetic, hybrid, toy]
    for obj in objs:
        try:
            model_class = getattr(obj, model_name)
        except AttributeError:
            pass

    with ProfileRegion('model_class_setup'):
        model = model_class(params=params, comm=sub_comm,
                            inter_comm=inter_comm)

    assert isinstance(model, StruphyModel)

    # store geometry vtk
    if rank == 0:
        grids_log = [
            np.linspace(1e-6, 1.0, 32),
            np.linspace(0.0, 1.0, 32),
            np.linspace(0.0, 1.0, 32),
        ]

        tmp = model.domain(*grids_log)
        grids_phy = [tmp[0], tmp[1], tmp[2]]

        pointData = {}
        det_df = model.domain.jacobian_det(*grids_log)
        pointData['det_df'] = det_df

        if model.mhd_equil is not None:
            absB0 = model.mhd_equil.absB0(*grids_log)
            p0 = model.mhd_equil.p0(*grids_log)
            pointData['absB0'] = absB0
            pointData['p0'] = p0
        elif model.braginskii_equil is not None:
            absB0 = model.braginskii_equil.absB0(*grids_log)
            p0 = model.braginskii_equil.p0(*grids_log)
            pointData['absB0'] = absB0
            pointData['p0'] = p0

        gridToVTK(os.path.join(path_out, 'geometry'),
                  *grids_phy, pointData=pointData)

    # data object for saving (will either create new hdf5 files if restart==False or open existing files if restart==True)
    # use MPI.COMM_WORLD as communicator when storing the outputs
    data = DataContainer(path_out, comm=comm)

    # time quantities (current time value, value in seconds and index)
    time_state = {}
    time_state['value'] = np.zeros(1, dtype=float)
    time_state['value_sec'] = np.zeros(1, dtype=float)
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

        total_steps = str(int(round(time_params['Tend'] / time_params['dt'])))

    else:
        model.initialize_from_restart(data)

        time_state['value'][0] = data.file['restart/time/value'][-1]
        time_state['value_sec'][0] = data.file['restart/time/value_sec'][-1]
        time_state['index'][0] = data.file['restart/time/index'][-1]

        total_steps = str(
            int(
                round(
                    (time_params['Tend'] - time_state['value']
                     [0]) / time_params['dt']
                )
            )
        )

    # compute initial scalars and kinetic data, pass time state to all propagators
    model.update_scalar_quantities()
    model.update_markers_to_be_saved()
    model.update_distr_functions()
    model.add_time_state(time_state['value'])

    # add all variables to be saved to data object
    save_keys_all, save_keys_end = model.initialize_data_output(data, size)

    # ======================== main time loop ======================
    model.update_scalar_quantities()
    if rank == 0:
        print('\nINITIAL SCALAR QUANTITIES:')
        model.print_scalar_quantities()

        split_algo = time_params['split_algo']
        print(f"\nSTART TIME STEPPING WITH '{split_algo}' SPLITTING:")

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
                print(
                    'wall-clock time of simulation [sec]: ',
                    end_simulation - start_simulation,
                )
                print()
            break

        # perform one time step dt
        t0 = time.time()
        with ProfileRegion('model.integrate'):
            model.integrate(time_params['dt'], time_params['split_algo'])
        t1 = time.time()

        # update time and index (round time to 10 decimals for a clean time grid!)
        time_state['value'][0] = round(
            time_state['value'][0] + time_params['dt'], 10)
        time_state['value_sec'][0] = round(
            time_state['value_sec'][0] +
            time_params['dt'] * model.units['t'], 10
        )
        time_state['index'][0] += 1

        run_time_now = (time.time() - start_simulation) / 60

        # update diagnostics data and save data
        if time_state['index'][0] % save_step == 0:

            # compute scalars and kinetic data
            model.update_scalar_quantities()
            model.update_markers_to_be_saved()
            model.update_distr_functions()

            # extract FEM coefficients
            for key, val in model.em_fields.items():
                if 'params' not in key:
                    field = val['obj']
                    assert isinstance(field, Derham.Field)
                    # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
                    field.extract_coeffs(update_ghost_regions=False)

            for _, val in model.fluid.items():
                for variable, subval in val.items():
                    if 'params' not in variable:
                        field = subval['obj']
                        assert isinstance(field, Derham.Field)
                        # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
                        field.extract_coeffs(update_ghost_regions=False)

            for key, val in model.diagnostics.items():
                if 'params' not in key:
                    field = val['obj']
                    assert isinstance(field, Derham.Field)
                    # in-place extraction of FEM coefficients from field.vector --> field.vector_stencil!
                    field.extract_coeffs(update_ghost_regions=False)

            # save data (everything but restart data)
            data.save_data(keys=save_keys_all)

            # print current time and scalar quantities to screen
            if rank == 0 and not supress_out:
                step = str(time_state['index'][0]).zfill(len(total_steps))

                message = 'time step: ' + step + '/' + str(total_steps)
                message += ' | ' + 'time: {0:10.5f}/{1:10.5f}'.format(
                    time_state['value'][0], time_params['Tend']
                )
                message += ' | ' + 'phys. time [s]: {0:12.10f}/{1:12.10f}'.format(
                    time_state['value_sec'][0], time_params['Tend'] *
                    model.units['t']
                )
                message += (
                    ' | '
                    + 'wall clock [s]: {0:8.4f} | last step duration [s]: {1:8.4f}'.format(
                        run_time_now * 60, t1 - t0
                    )
                )

                print(message, end='\n')
                model.print_scalar_quantities()
                print()
    # ===================================================================

    with open(path_out + '/meta.txt', 'a') as f:
        # f.write('wall-clock time [min]:'.ljust(30) + str((end_simulation - start_simulation)/60.) + '\n')
        f.write(
            f"{rank} {inter_comm.Get_rank()} {sub_comm.Get_rank()} {'wall-clock time[min]: '.ljust(30)}{(end_simulation - start_simulation) / 60}\n"
        )
    comm.Barrier()
    if rank == 0:
        print('struphy run finished')

    sub_comm.Free()
    inter_comm.Free()


if __name__ == '__main__':

    import argparse
    import os
    import struphy
    import yaml
    from struphy.profiling.profiling import (
        ProfileRegion,
        set_likwid,
        pylikwid_markerinit,
        pylikwid_markerclose,
    )

    libpath = struphy.__path__[0]

    with open(os.path.join(libpath, 'state.yml')) as f:
        state = yaml.load(f, Loader=yaml.FullLoader)

    o_path = state['o_path']

    parser = argparse.ArgumentParser(description='Run an Struphy model.')

    # model
    parser.add_argument(
        'model', type=str, metavar='model', help='the name of the model to run'
    )

    # input (absolute path)
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        metavar='FILE',
        help='absolute path of parameter file (.yml)',
    )

    # output (absolute path)
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        metavar='DIR',
        help='absolute path of output folder (default=<out_path>/sim_1)',
        default=os.path.join(o_path, 'sim_1'),
    )

    # restart
    parser.add_argument(
        '-r',
        '--restart',
        help='restart the simulation in the output folder specified under -o',
        action='store_true',
    )

    # runtime
    parser.add_argument(
        '--runtime',
        type=int,
        metavar='N',
        help='maximum wall-clock time of program in minutes (default=300)',
        default=300,
    )

    # save step
    parser.add_argument(
        '-s',
        '--save-step',
        type=int,
        metavar='N',
        help='how often to skip data saving (default=1, which means data is saved every time step)',
        default=1,
    )

    # supress screen output
    parser.add_argument(
        '--supress-out',
        help='supress screen output during time integration',
        action='store_true',
    )

    # likwid
    parser.add_argument(
        '--likwid',
        help='run with Likwid',
        action='store_true',
    )

    args = parser.parse_args()

    # Enable profiling if likwid == True
    set_likwid(args.likwid)
    pylikwid_markerinit()
    with ProfileRegion('main'):
        # solve the model
        main(
            args.model,
            args.input,
            args.output,
            restart=args.restart,
            runtime=args.runtime,
            save_step=args.save_step,
            supress_out=args.supress_out,
        )
    pylikwid_markerclose()
