""" An executable for quick access to the diagnostic tools in diagn_tools.py """

#!/usr/bin/env python3
import numpy as np
import argparse
import os
import h5py
import yaml

import struphy
from struphy.diagnostics.diagn_tools import plot_scalars, plot_distr_fun


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('actions', nargs='+', type=str, default=[None],
                        help='''which actions to perform:\
                            \n - plot_scalars : plots the scalar quantities that were saved during the simulation\
                            \n - plot_distr   : plots the distribution function and delta-f (if available)\
                            \n                  set points for slicing with options below (default is middle of the space)''')
    parser.add_argument('-f', nargs=1, type=str, default=['sim_1'],
                        help='in which folder the simulation data has been stored')
    parser.add_argument('-scalars', nargs='+', action='append', default=[],
                        help='(for plot_scalars) which quantities to plot')
    parser.add_argument('--log', action='store_true',
                        help='(for plot_scalars) if logarithmic y-axis should be used')
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

    args = parser.parse_args()
    actions = args.actions
    foldername = args.f[0]
    time = args.t[0]
    do_log = args.log
    if len(args.scalars) != 0:
        scalars_plot = args.scalars[0]
    else:
        scalars_plot = args.scalars

    libpath = struphy.__path__[0]
    with open(os.path.join(libpath, 'state.yml')) as f:
        state = yaml.load(f, Loader=yaml.FullLoader)

    o_path = state['o_path']

    path = os.path.join(o_path, foldername)

    grid_slices = {'e1': args.e1[0], 'e2': args.e2[0], 'e3': args.e3[0],
                   'v1': args.v1[0], 'v2': args.v2[0], 'v3': args.v3[0]}

    # Get fields
    file = h5py.File(os.path.join(path, 'data/', 'data_proc0.hdf5'), 'r')
    saved_scalars = file['scalar']
    saved_time = file['time']['value'][:]

    # read in parameters
    with open(path + '/parameters.yml') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    if 'plot_scalars' in actions:
        plot_scalars(saved_time,
                     saved_scalars,
                     scalars_plot=scalars_plot,
                     do_log=do_log,
                     save_plot=True,
                     savedir=path)

    if 'plot_distr' in actions:
        for species in params['kinetic'].keys():
            # Set velocity point of evaluation to v_shift of background params if not given by input
            if params['kinetic'][species]['markers']['type'] == 'full_f':
                for k in range(1, 4):
                    if grid_slices['v' + str(k)] is None:
                        bckgr_type = params['kinetic'][species]['init']['type']
                        bckgr_param = params['kinetic'][species]['init'][bckgr_type]['u' + str(
                            k)]
                        if isinstance(bckgr_param, dict):
                            grid_slices['v' + str(k)] = \
                                bckgr_param['u0' + str(k)]
                        else:
                            grid_slices['v' + str(k)] = bckgr_param
            elif params['kinetic'][species]['markers']['type'] == 'delta_f' \
                    or params['kinetic'][species]['markers']['type'] == 'control_variate':
                for k in range(1, 4):
                    if grid_slices['v' + str(k)] is None:
                        bckgr_type = params['kinetic'][species]['background']['type']
                        bckgr_param = params['kinetic'][species]['background'][bckgr_type]['u' + str(
                            k)]
                        if isinstance(bckgr_param, dict):
                            grid_slices['v' + str(k)] = \
                                bckgr_param['u0' + str(k)]
                        else:
                            grid_slices['v' + str(k)] = bckgr_param

            # Get index of where to plot in time
            time_idx = np.argmin(np.abs(time - saved_time))

            plot_distr_fun(path=os.path.join(path, 'post_processing', 'kinetic_data', species),
                           time_idx=time_idx,
                           grid_slices=grid_slices,
                           save_plot=True, savepath=path)

    file.close()


if __name__ == '__main__':
    main()
