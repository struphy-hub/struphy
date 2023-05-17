#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftfreq, fftn
import matplotlib.colors as colors
import argparse
import pickle
import os
import h5py
import yaml

import struphy
from struphy.dispersion_relations import analytic


def main():
    """
    TODO
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('actions', nargs='+', type=str, default=[None],
                        help='''which actions to perform:\
                            \n - fourier_1d   : performs Fourier analysis of the fields and plots the results\
                            \n - plot_scalars : plots the scalar quantities that were saved during the simulation\
                            \n - plot_distr   : plots the distribution function and delta-f (if available)\
                            \n                  set points for slicing with options below (default is middle of the space)''')
    parser.add_argument('-f', nargs=1, type=str, default=['sim_1'],
                        help='in which folder the simulation data has been stored')
    parser.add_argument('-scalars', nargs='+', action='append', default=[[]],
                        help='(for plot_scalars) which quantities to plot')
    parser.add_argument('--log', action='store_true',
                        help='(for plot_scalars) if logarithmic y-axis should be used')
    parser.add_argument('-t', nargs=1, type=float, default=[0.],
                        help='(for plot_distr) at which time to plot the distribution function')
    parser.add_argument('-e1', nargs=1, type=float, default=[None],
                        help='(for plot_distr) at which position in eta1 direction to plot')
    parser.add_argument('-e2', nargs=1, type=float, default=[None],
                        help='(for plot_distr) at which position in eta2 direction to plot')
    parser.add_argument('-e3', nargs=1, type=float, default=[None],
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
    scalars_plot = args.scalars[0]
    path = os.path.join(os.path.dirname(struphy.__file__),
                        'io/out', foldername)

    grid_slices = {'e': {'e1': args.e1[0], 'e2': args.e2[0], 'e3': args.e3[0]},
                   'v': {'v1': args.v1[0], 'v2': args.v2[0], 'v3': args.v3[0]}}

    # code name
    with open(path + '/meta.txt', 'r') as f:
        lines = f.readlines()

    code = lines[-2].split()[-1]

    # Get fields
    file = h5py.File(path + '/data_proc0.hdf5', 'r')
    field_names = list(file['feec'].keys())
    saved_scalars = file['scalar']
    saved_time = file['time']['value'][:]

    # read in parameters
    with open(path + '/parameters.yml') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    if 'fourier_1d' in actions:
        assert os.path.exists(os.path.join(path, 'post_processing', 'fields_data')), \
            'For Fourier analysis needs fields in the model.'

        point_data_log = []
        # load data dicts for e_field
        for k in range(len(field_names)):
            with open(os.path.join(path, 'post_processing', 'fields_data', field_names[k], '_log.bin'), 'rb') as handle:
                point_data_log += [pickle.load(handle)]

        point_data_phys = []
        for k in range(len(field_names)):
            with open(os.path.join(path, 'post_processing', 'fields_data', field_names[k], '_phy.bin'), 'rb') as handle:
                point_data_phys += [pickle.load(handle)]

        # load grids
        with open(os.path.join(path, 'post_processing', 'fields_data', field_names[k], '_log.bin'), 'rb') as handle:
            grids = pickle.load(handle)

        with open(os.path.join(path, 'post_processing', 'fields_data', field_names[k], '_phy.bin'), 'rb') as handle:
            grids_mapped = pickle.load(handle)

        if code == 'LinearMHD':
            equil_type = params['mhd_equilibrium']['name']

            if equil_type == 'HomogenSlab':
                B0x = params['mhd_equilibrium']['HomogenSlab']['B0x']
                B0y = params['mhd_equilibrium']['HomogenSlab']['B0y']
                B0z = params['mhd_equilibrium']['HomogenSlab']['B0z']

                p0 = (2*params['mhd_equilibrium']['HomogenSlab']
                      ['beta']/100)/(B0x**2 + B0y**2 + B0z**2)
                n0 = params['mhd_equilibrium']['HomogenSlab']['n0']

                gamma = 5/3

            else:
                raise NotImplementedError(
                    f'Dispersion relations for MHD equilibrium of type {equil_type} has not been implemented yet!')

            disp_params = {'B0x': B0x, 'B0y': B0y, 'B0z': B0z,
                           'p0': p0, 'n0': n0, 'gamma': gamma}

            # fft in (t, z) of first component of u_field on physical grid
            fourier_1d(point_data_log[3], field_names[3], code, grids,
                       grids_mapped=grids_mapped, component=0, slice_at=[0, 0, None],
                       do_plot=True, disp_name='Mhd1D', disp_params=disp_params,
                       save_plot=True, save_name=os.path.join(path, code + '_' + field_names[3]))

            # fft in (t, z) of pressure on physical grid
            fourier_1d(point_data_log[2], field_names[2], code, grids,
                       grids_mapped=grids_mapped, component=0, slice_at=[0, 0, None],
                       do_plot=True, disp_name='Mhd1D', disp_params=disp_params,
                       save_plot=True, save_name=os.path.join(path, code + '_' + field_names[2]))

        elif code == 'Maxwell':
            # fft in (t, z) of first component of e_field on physical grid
            fourier_1d(point_data_log[1], field_names[1], code, grids,
                       grids_mapped=grids_mapped, component=0, slice_at=[0, 0, None],
                       do_plot=True, disp_name='Maxwell1D',
                       save_plot=True, save_name=os.path.join(path, code + '_' + field_names[1]))

        else:
            raise NotImplementedError(
                f'1D Fourier analysis is not yet implemented for the model {code}')

    if 'plot_scalars' in actions:
        plot_scalars(saved_time,
                     saved_scalars,
                     scalars_plot=scalars_plot,
                     do_log=do_log,
                     save_plot=True,
                     savename=os.path.join(path, code))

    if 'plot_distr' in actions:
        for species in params['kinetic'].keys():
            time_idx = np.argmin(np.abs(time - saved_time))
            plot_distr_fun(path=os.path.join(path, 'post_processing', 'kinetic_data', species),
                           time_idx=time_idx,
                           grid_slices=grid_slices,
                           save_plot=True, savepath=path)

    file.close()


def fourier_1d(values, name, code, grids,
               grids_mapped=None, component=0, slice_at=[None, 0, 0],
               do_plot=False, disp_name=None, disp_params={},
               save_plot=False, save_name=None, file_format='png'):
    """
    Perform fft in space-time, (t, x) -> (omega, k), where x can be a logical or physical coordinate.
    Returns values if plot=False.

    Parameters
    ----------
        values : dict
            Dictionary holding values of a B-spline FemField on the grid as 3d np.arrays:
            values[n] contains the values at time step n, where n = 0:Nt-1:step with 0<step.

        name : str
            Name of the FemField. 

        code : str
            From which code the data has been obtained.

        grids : 3-list
            1d logical grids in each eta-direction with Nel[i]*npts_per_cell[i] + 1 entries in each direction. 

        grids_mapped : 3-list
            Mapped grids obtained by domain(). If None, the fft is performed on the logical grids.

        component : int
            Which component of a FemField to consider; is 0 for 0-and 3-forms, is in {0, 1, 2} for 1- and 2-forms.

        slice_at : 3-list
            At which indices i, j the 1d slice data (t, eta)_(i, j) should be obtained. 
            One entry must be "None"; this is the direction of the fft. 
            Default: [None, 0, 0] performs the eta1-fft at (eta2[0], eta3[0]). 

        do_plot : boolean
            Plot result if True, otherwise return things.

        disp_name : str
            The name of the dispersion relation class in struphy.dispersion_relations.analytic to be used for analytic comparison.

        disp_params : dict
            Parameters needed for analytical dispersion relation, see struphy.dispersion_relations.analytic.

        save_plot : boolean
            Save figure if True. Then a path has to be given.

        save_name : str
            Name under which the plot of the result should be saved.

        file_format : str
            Type of file which the plot of the result should be saved.

    Returns
    -------
        omega : np.array
            1d array of angular frequency.

        kvec : np.array
            1d array of wave vector.

        dispersion : np.array
            2d array of shape (omega.size, kvce.size) holding the fft.
    """

    print(f'code: {code}')

    keys = list(values.keys())

    # check uniform grid in time
    dt = keys[1] - keys[0]
    print(f'time step: {dt}')
    assert np.all([np.abs(y - x - dt) < 1e-12 for x,
                  y in zip(keys[:-1], keys[1:])])

    # create 4d np.array with shape (time, eta1, eta2, eta3)
    dim_t = len(keys)
    dim_eta = values[keys[0]][component].shape

    temp = np.zeros((dim_t, *dim_eta))

    for n, (time, snapshot) in enumerate(values.items()):
        temp[n, :, :, :] = snapshot[component]

    # Extract 2d data (t, eta) for fft
    if slice_at[0] == None:

        data = temp[:, :, slice_at[1], slice_at[2]]
        grid = grids[0]
        if grids_mapped is not None:
            grid = grids_mapped[0][:, slice_at[1], slice_at[2]]

    elif slice_at[1] == None:

        data = temp[:, slice_at[0], :, slice_at[2]]
        grid = grids[1]
        if grids_mapped is not None:
            grid = grids_mapped[1][slice_at[0], :, slice_at[2]]

    elif slice_at[2] == None:

        data = temp[:, slice_at[0], slice_at[1], :]
        grid = grids[2].flatten()
        if grids_mapped is not None:
            grid = grids_mapped[2][slice_at[0], slice_at[1], :]

    else:
        AssertionError('One entry of slice_at must be "None".')

    # extract uniform grid in space
    Nt = data.shape[0]
    Nx = grid.size
    dx = grid[1] - grid[0]
    print(f'space step: {dx}')
    assert np.allclose(grid[1:] - grid[:-1], dx*np.ones_like(grid[:-1]))

    dispersion = (2./Nt)*(2./Nx)*np.abs(fftn(data))[:Nt//2, :Nx//2]
    kvec = 2*np.pi*fftfreq(Nx, dx)[:Nx//2]
    omega = 2*np.pi*fftfreq(Nt, dt)[:Nt//2]

    if do_plot:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        colormap = 'plasma'
        K, W = np.meshgrid(kvec, omega)
        lvls = np.logspace(-15, -1, 27)
        disp_plot = ax.contourf(K, W, dispersion**2 / (dispersion**2).max(),
                                cmap=colormap, norm=colors.LogNorm(), levels=lvls)
        plt.colorbar(ticks=[1e-12, 1e-9, 1e-6, 1e-3],
                     mappable=disp_plot, format='%.0e')
        title = name + ' component ' + \
            str(component + 1) + ' from code: ' + code
        ax.set_title(title)
        ax.set_xlabel('$k$ [a.u.]')
        ax.set_ylabel('$\omega$ [a.u.]')

        # analytic solution:
        disp_class = getattr(analytic, disp_name)
        disp = disp_class(**disp_params)

        kpara = kvec

        branches = disp(kpara)
        set_min = 0.
        set_max = 0.
        for key, branch in branches.items():
            vals = np.real(branch)
            ax.plot(kvec, vals, '--', label=key)
            tmp = np.min(vals)
            if tmp < set_min:
                set_min = tmp
            tmp = np.max(vals)
            if tmp > set_max:
                set_max = tmp

        ax.legend()
        ax.set_xlim(0, kvec[-1])
        ax.set_ylim(set_min*1.1, set_max*1.1)

        if save_plot:
            assert save_name is not None, 'When wanting to save the plot a path has to be given!'
            plt.savefig(save_name + '.' + file_format)
        else:
            plt.show()

    else:
        return kvec, omega, dispersion


def plot_scalars(time, scalar_quantities, scalars_plot=[], do_log=False, save_plot=False, savename=None, file_format='png'):
    """
    Plot the scalar quantities and the relative error in the total energy for a simulation.

    Parameters
    ----------
    scalar_quantities : dict
        HDF5 dictionary dataset containing the scalar quantities that were saved during the simulation

    scalars_plot : list
        list of names of scalars that should be plotted. If empty then all are plotted

    do_log : boolean
        Do a logarithmic plot in the y-axis if True.

    save_plot : boolean
        Save figure if True. Then a path has to be given.

    savename : str
        Name under which the plot of the result should be saved.

    file_format : str
        Type of file which the plot of the result should be saved.
    """

    if 'en_tot' in scalar_quantities.keys():
        en_tot = scalar_quantities['en_tot'][:]

        plt.figure('en_tot')
        if do_log:
            plt.semilogy(time, en_tot)
        else:
            plt.plot(time, en_tot)

        if save_plot:
            assert savename is not None, 'When wanting to save the plot a path has to be given!'
            plt.savefig(savename + '_en_tot' + '.' + file_format)
        else:
            plt.show()

        plt.figure('en_tot_rel_err')
        plt.plot(time[1:], np.divide(
            np.abs(en_tot[1:] - en_tot[0]), en_tot[0]))

        if save_plot:
            assert savename is not None, 'When wanting to save the plot a path has to be given!'
            plt.savefig(savename + '_en_tot_rel_err' + '.' + file_format)
        else:
            plt.show()

    plt.figure('scalars')
    if len(scalars_plot) == 0:
        for key, quantity in scalar_quantities.items():
            if key not in ['time', 'en_tot']:
                if do_log:
                    plt.semilogy(time, quantity[:], label=key)
                else:
                    plt.plot(time, quantity[:], label=key)
    else:
        for key in scalars_plot:
            if do_log:
                plt.semilogy(time, scalar_quantities[key][:], label=key)
            else:
                plt.plot(time, scalar_quantities[key][:], label=key)

    plt.legend()
    plt.xlabel('time')

    if save_plot:
        assert savename is not None, 'When wanting to save the plot a path has to be given!'
        plt.savefig(savename + '_scalars' + '.' + file_format)
    else:
        plt.show()


def plot_distr_fun(path, time_idx, grid_slices, save_plot=False, savepath=None, file_format='png'):
    """
    Plot the binned distribution function at given slices of the phase space.

    Parameters
    ----------
    path : str
        Path to the kinetic data of the species.

    time : float
        at which point in time to plot

    grid_slices : dict
        dictionary with keys e and v that hold dictionaries with directions and values
        that indicate which slices of the data should be plotted

    save_plot : boolean
        Save figure if True. Then a path has to be given.

    savepath : str
        Path under which the plot of the result should be saved.

    file_format : str
        Type of file which the plot of the result should be saved.
    """

    species = str(path.split('/')[-1])
    path = os.path.join(path, 'distribution_function')

    # make empty dictionaries
    f = {'e': None, 'v': None}
    delta_f = {'e': None, 'v': None}
    grids = {'e': [], 'v': []}

    # Loop over files and load distribution function data
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)

        # load full distribution functions and compute inds_to_names and names_to_inds
        if filename[:3] == 'f_e':
            f['e'] = np.load(filepath)
            for comp in filename.split('_')[1:]:
                comp = comp.split('.')[0]
                grids['e'] += [np.load(os.path.join(path, 'grid_' + comp + '.npy'))]
        elif filename[:3] == 'f_v':
            f['v'] = np.load(filepath)
            for comp in filename.split('_')[1:]:
                comp = comp.split('.')[0]
                grids['v'] += [np.load(os.path.join(path, 'grid_' + comp + '.npy'))]

        # load delta f
        elif filename[:9] == 'delta_f_e':
            delta_f['e'] = np.load(filepath)
        elif filename[:9] == 'delta_f_v':
            delta_f['v'] = np.load(filepath)

    for typ in ['e', 'v']:
        for k in range(f[typ].ndim - 1):

            f_slicing = [0] * f[typ].ndim
            f_slicing[k + 1] = slice(None)
            # plot delta_f
            if delta_f[typ] is not None:
                plt.figure('delta_f')
                plt.plot(grids[typ][k], delta_f[typ][tuple(f_slicing)].squeeze())
                plt.xlabel(typ + str(k + 1))
                plt.ylabel(r'$\delta f$')
                print(f'Created plot for delta_f in {typ + str(k + 1)}')

                if save_plot:
                    assert savepath is not None, 'When wanting to save the plot a path has to be given!'
                    savename = os.path.join(savepath, species + '_delta_f_'
                                            + typ + str(k + 1) + '.' + file_format)
                    plt.savefig(savename)
                else:
                    plt.show()
                plt.close()

            # plot full f
            if f[typ] is not None:
                plt.figure('f')
                plt.plot(grids[typ][k], f[typ][tuple(f_slicing)].squeeze())
                plt.xlabel(typ + str(k + 1))
                plt.ylabel(r'$f$')
                print(f'Created plot for f in {typ + str(k + 1)}')

                if save_plot:
                    assert savepath is not None, 'When wanting to save the plot a path has to be given!'
                    savename = os.path.join(savepath, species + '_f_'
                                            + typ + str(k + 1) + '.' + file_format)
                    plt.savefig(savename)
                else:
                    plt.show()
                plt.close()


if __name__ == '__main__':
    main()
