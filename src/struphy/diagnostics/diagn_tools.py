""" A collection of tools used for diagnostics """

#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftfreq, fftn
import matplotlib.colors as colors
import os

from struphy.dispersion_relations import analytic


def power_spectrum_2d(values, name, code, grids,
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


def plot_scalars(time, scalar_quantities, scalars_plot=[], do_log=False, save_plot=False, savedir=None, file_format='png'):
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

    savedir : str
        Name of the folder in which the plot of the result should be saved.

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
            assert savedir is not None, 'When wanting to save the plot a path has to be given!'
            plt.savefig(os.path.join(savedir, 'en_tot' + '.' + file_format))
        else:
            plt.show()

        plt.figure('en_tot_rel_err')
        plt.plot(time[1:], np.divide(
            np.abs(en_tot[1:] - en_tot[0]), en_tot[0]))

        if save_plot:
            assert savedir is not None, 'When wanting to save the plot a path has to be given!'
            plt.savefig(os.path.join(savedir, 'en_tot_rel_err' + '.' + file_format))
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
        assert savedir is not None, 'When wanting to save the plot a path has to be given!'
        plt.savefig(os.path.join(savedir, 'scalars' + '.' + file_format))
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

    # Loop over folders and plot for each of them
    for folder in os.listdir(path):

        grids = []
        f = None
        delta_f = None

        subpath = os.path.join(path, folder)

        # Loop over the files in this subdirectory
        for filename in os.listdir(subpath):
            filepath = os.path.join(subpath, filename)

            # load full distribution functions
            if filename == 'f_binned.npy':
                f = np.load(filepath)

            # load delta f
            elif filename == 'delta_f_binned.npy':
                delta_f = np.load(filepath)

        assert f is not None, "No distribution function file found!"

        # Load grid
        directions = folder.split('_')
        for direction in directions:
            grids += [np.load(os.path.join(subpath,
                              'grid_' + direction + '.npy'))]

        # Get indices of where to plot in other directions
        grid_idxs = {}
        for k in range(f.ndim - 1):
            grid_idxs[directions[k]] = np.argmin(
                np.abs(grids[k] - grid_slices[directions[k]]))

        for k in range(f.ndim - 1):
            # Prepare slicing
            f_slicing = [0] * f.ndim
            # time index
            f_slicing[0] = time_idx
            # direction in which to plot
            f_slicing[k + 1] = slice(None)
            # directions in which f is evaluated at a point
            for j in range(1, f.ndim):
                if j == k + 1:
                    continue
                f_slicing[j] = grid_idxs[directions[k]]

            # plot delta_f
            if delta_f is not None:
                plt.figure('delta_f')
                plt.plot(grids[k], delta_f[tuple(f_slicing)].squeeze())
                plt.xlabel(directions[k])
                plt.ylabel(r'$\delta f$')
                print(f'Created plot for delta_f in {directions[k]}')

                if save_plot:
                    assert savepath is not None, 'When wanting to save the plot a path has to be given!'
                    savename = os.path.join(savepath, species + '_delta_f_'
                                            + directions[k] + '.' + file_format)
                    plt.savefig(savename)
                else:
                    plt.show()
                plt.close()

            # plot full f
            if f is not None:
                plt.figure('f')
                plt.plot(grids[k], f[tuple(f_slicing)].squeeze())
                plt.xlabel(directions[k])
                plt.ylabel(r'$f$')
                print(f'Created plot for f in {directions[k]}')

                if save_plot:
                    assert savepath is not None, 'When wanting to save the plot a path has to be given!'
                    savename = os.path.join(savepath, species + '_f_'
                                            + directions[k] + '.' + file_format)
                    plt.savefig(savename)
                else:
                    plt.show()
                plt.close()

        del grids
        del f
        del delta_f
