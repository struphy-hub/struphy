""" An executable for quick access to the diagnostic tools in diagn_tools.py """

#!/usr/bin/env python3
import numpy as np
import argparse
import os
import h5py
import yaml

import struphy
import struphy.utils.utils as utils
from struphy.diagnostics.diagn_tools import plot_scalars, plot_distr_fun, phase_space_video


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('actions', nargs='+', type=str, default=[None],
                        help='''which actions to perform:\
                            \n - plot_scalars : plots the scalar quantities that were saved during the simulation\
                            \n - plot_distr   : plots the distribution function and delta-f (if available)\
                            \n                  set points for slicing with options below (default is middle of the space)\
                            \n - phase_space  : make a video of the distribution function (minus the background) in a 2D slice of phase space
                        ''')
    parser.add_argument('-f', nargs=1, type=str, default=['sim_1'],
                        help='name of the folder for the simulation data (in io/out)')
    parser.add_argument('-scalars', nargs='+', action='append', default=[],
                        help='(for plot_scalars) which quantities to plot')
    parser.add_argument('--log', action='store_true',
                        help='(for plot_scalars) if logarithmic y-axis should be used')
    parser.add_argument('--show', action='store_true',
                        help='(for plot_scalars) if the plot should be shown')
    parser.add_argument('--nosave', action='store_true',
                        help='(for plot_scalars) if the plot should not be displayed')
    parser.add_argument('--fit', action='store_true',
                        help='(for plot_scalars) if a fit should be done (using maxima)')
    parser.add_argument('--minfit', action='store_true',
                        help='(for plot_scalars) if a fit should be done using minima')
    parser.add_argument('-degree', nargs=1, type=int, default=[1],
                        help='(for plot_scalars --fit) the degree of the fit curve')
    parser.add_argument('-extrema', nargs=1, type=int, default=[4],
                        help='(for plot_scalars --fit) how many extrema should be used for the fit')
    parser.add_argument('-startextr', nargs=1, type=int, default=[0],
                        help='(for plot_scalars --fit) which extremum should be used first for the fit (0 = first)')
    parser.add_argument('-order', nargs=1, type=int, default=[4],
                        help='(for plot_scalars --fit) how many neighbouring points should be used for determining the extrema')
    parser.add_argument('-t', nargs=1, type=float, default=[0.],
                        help='(for plot_distr) at which time to plot the distribution function')
    parser.add_argument('-e1', nargs=1, type=float, default=[0.5],
                        help='(for plot_distr) at which position in eta1 direction to plot')
    parser.add_argument('-e2', nargs=1, type=float, default=[0.5],
                        help='(for plot_distr) at which position in eta2 direction to plot')
    parser.add_argument('-e3', nargs=1, type=float, default=[0.5],
                        help='(for plot_distr) at which position in eta3 direction to plot')
    parser.add_argument('-v1', nargs=1, type=float, default=[None],
                        help='(for plot_distr) at which point in v1 direction to plot')
    parser.add_argument('-v2', nargs=1, type=float, default=[None],
                        help='(for plot_distr) at which point in v2 direction to plot')
    parser.add_argument('-v3', nargs=1, type=float, default=[None],
                        help='(for plot_distr) at which point in v3 direction to plot')

    # Parse the arguments
    args = parser.parse_args()
    actions = args.actions
    foldername = args.f[0]
    time = args.t[0]
    do_log = args.log
    show = args.show
    nosave = args.nosave
    if len(args.scalars) != 0:
        scalars_plot = args.scalars[0]
    else:
        scalars_plot = args.scalars

    # Arguments for fitting
    do_fit = args.fit
    fit_minima = args.minfit
    if fit_minima and do_fit:
        do_fit = False
    no_extrema = args.extrema[0]
    order = args.order[0]
    degree = args.degree[0]
    start_extremum = args.startextr[0]

    # Read struphy state file
    state = utils.read_state()

    o_path = state['o_path']

    path = os.path.join(o_path, foldername)

    grid_slices = {
        'e1': args.e1[0], 'e2': args.e2[0], 'e3': args.e3[0],
        'v1': args.v1[0], 'v2': args.v2[0], 'v3': args.v3[0]
    }

    # Get fields
    file = h5py.File(os.path.join(path, 'data/', 'data_proc0.hdf5'), 'r')
    saved_scalars = file['scalar']
    saved_time = file['time']['value'][:]

    # read in parameters
    with open(path + '/parameters.yml') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # Get model name
    with open(path + '/meta.txt', 'r') as file:
        for line in file.readlines():
            if line[0:10] == 'model_name':
                model_name = line.split(':')[1].strip()

    if 'plot_scalars' in actions:
        plot_scalars(
            time=saved_time, scalar_quantities=saved_scalars, scalars_plot=scalars_plot,
            do_log=do_log, do_fit=do_fit, fit_minima=fit_minima, order=order,
            no_extrema=no_extrema, degree=degree, show_plot=show, start_extremum=start_extremum,
            save_plot=not nosave, savedir=path
        )

    if 'plot_distr' or 'phase_space' in actions:
        for species in params['kinetic'].keys():
            # Get model class
            from struphy.models import fluid, kinetic, hybrid, toy
            objs = [fluid, kinetic, hybrid, toy]
            for obj in objs:
                try:
                    model_class = getattr(obj, model_name)
                except AttributeError:
                    pass

            # get particles class name
            species_dict = model_class.species()
            particles_class_name = species_dict['kinetic'][species]

            # Get default background of particles class
            from struphy.pic import particles
            default_bckgr_type = getattr(
                particles, particles_class_name).default_bckgr_params()['type']

            # Get default background parameters
            from struphy.kinetic_background import maxwellians

            keys_given = []

            bckgr_fun = None
            if "background" in params['kinetic'][species].keys():
                bckgr_type = params['kinetic'][species]["background"]["type"]

                for fi in bckgr_type:
                    if fi[-2] == '_':
                        fi_type = fi[:-2]
                    else:
                        fi_type = fi

                    bckgr_params = params['kinetic'][species]["background"][fi]
                    keys_given += bckgr_params.keys()

                    if bckgr_fun is None:
                        bckgr_fun = getattr(maxwellians, fi_type)(
                            maxw_params=bckgr_params
                        )
                    else:
                        bckgr_fun = bckgr_fun + getattr(maxwellians, fi_type)(
                            maxw_params=bckgr_params
                        )
            else:
                bckgr_fun = getattr(maxwellians, default_bckgr_type)()

            # Get values of background shifts in velocity space
            positions = [np.array([grid_slices['e' + str(k)]])
                         for k in range(1, 4)]
            u = bckgr_fun.u(*positions)
            eval_params = {'u' + str(k+1): u[k][0] for k in range(3)}

            # Set velocity point of evaluation to velocity shift if not given by input
            for k in range(1, 4):
                if grid_slices['v' + str(k)] is None:
                    key = 'u' + str(k)
                    if key in keys_given:
                        grid_slices['v' + str(k)] = eval_params[key]

            # Plot the distribution function
            if 'plot_distr' in actions:
                # Get index of where to plot in time
                time_idx = np.argmin(np.abs(time - saved_time))

                plot_distr_fun(
                    path=os.path.join(
                        path, 'post_processing', 'kinetic_data', species
                    ),
                    time_idx=time_idx,
                    grid_slices=grid_slices,
                    save_plot=True, savepath=path,
                )

            # Create a video of the phase space
            if 'phase_space' in actions:
                for slice_name in os.listdir(os.path.join(
                    path, 'post_processing', 'kinetic_data', species, 'distribution_function'
                )):
                    phase_space_video(
                        t_grid=saved_time, grid_slices=grid_slices,
                        slice_name=slice_name,
                        marker_type=params['kinetic'][species]['markers']['type'],
                        species=species,
                        path=path,
                        model_name=model_name,
                        background_params=params['kinetic'][species]['background']
                    )

    file.close()


if __name__ == '__main__':
    main()
