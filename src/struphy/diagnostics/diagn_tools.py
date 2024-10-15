#!/usr/bin/env python3
import os
import subprocess
import numpy as np
from scipy.fft import fftfreq, fftn
from scipy.signal import argrelextrema
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from struphy.dispersion_relations import analytic


def power_spectrum_2d(values, name, code, grids,
                      grids_mapped=None, component=0, slice_at=[None, 0, 0],
                      do_plot=False, disp_name=None, disp_params={},
                      save_plot=False, save_name=None, file_format='png'):
    """ Perform fft in space-time, (t, x) -> (omega, k), where x can be a logical or physical coordinate.
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


def plot_scalars(time, scalar_quantities, scalars_plot=[], do_log=False,
                 do_fit=False, fit_minima=False, order=4, no_extrema=4, start_extremum=0, degree=1,
                 show_plot=False, save_plot=False, savedir=None, file_format='png'):
    """ Plot the scalar quantities and the relative error in the total energy for a simulation.

    Parameters
    ----------
    scalar_quantities : dict
        HDF5 dictionary dataset containing the scalar quantities that were saved during the simulation

    scalars_plot : list
        list of names of scalars that should be plotted. If empty then all are plotted

    do_log : boolean
        Do a logarithmic plot in the y-axis if True.

    do_fit : boolean
        Do a fit to maxima if True.

    fit_minima : boolean
        Do a fit to minima if True. Will set do_fit to False if True.

    order : int
        How many neighbouring points should be used for finding extrema.

    no_extrema : int
        How many extrema should be used for the fit.

    start_extremum : int
        Which extremum should be used first for the fit.

    show_plot : boolean
        Display the figure if True.

    save_plot : boolean
        Save the figure if True. Then a path has to be given.

    savedir : str
        Name of the folder in which the plot of the result should be saved.

    file_format : str
        Type of file which the plot of the result should be saved.
    """

    # Only have one of the two as True
    if fit_minima and do_fit:
        do_fit = False

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
            plt.savefig(os.path.join(
                savedir, 'en_tot_rel_err' + '.' + file_format))
        if show_plot:
            plt.show()

    # Dict with label as key and time series as value
    plot_quantities = {}
    if len(scalars_plot) == 0:
        for key, quantity in scalar_quantities.items():
            if key not in ['time', 'en_tot']:
                plot_quantities[key] = quantity[:]
    else:
        for key in scalars_plot:
            plot_quantities[key] = scalar_quantities[key][:]

    # Make the figure
    plt.figure('scalars')
    for key, plot_quantity in plot_quantities.items():
        # Get the indices of the extrema
        if do_fit:
            inds_exs = argrelextrema(plot_quantity, np.greater, order=order)
        elif fit_minima:
            inds_exs = argrelextrema(plot_quantity, np.less, order=order)
        else:
            inds_exs = None

        if inds_exs is not None:
            # Get x-values and y-values of data to fit to
            quantity_extrema = plot_quantity[inds_exs][start_extremum:start_extremum+no_extrema]
            times_extrema = time[inds_exs][start_extremum:start_extremum+no_extrema]

            # for plotting take a bit more time at start and end
            if len(inds_exs[0]) >= 2:
                time_start_idx = np.max(
                    [0, 2*inds_exs[0][start_extremum] - inds_exs[0][start_extremum+1]])
                time_end_idx = np.min(
                    [len(time) - 1, 2*inds_exs[0][start_extremum+no_extrema-1] - inds_exs[0][start_extremum+no_extrema-2]])
                time_cut = time[time_start_idx:time_end_idx]
            else:
                time_cut = time

        if do_log:
            # plot quantity, extrema, and fit
            plt.semilogy(time, plot_quantity[:], '.', label=key, markersize=2)

            if inds_exs is not None:
                # do the fitting
                coeffs = np.polyfit(times_extrema, np.log(
                    quantity_extrema), deg=degree)
                plt.plot(times_extrema, quantity_extrema,
                         'r*', label='local extrema')
                plt.plot(
                    time_cut,
                    np.exp(coeffs[0] * time_cut + coeffs[1]),
                    label=r"$a * \exp(m x)$ with" +
                    f"\na={np.round(np.exp(coeffs[1]), 3)} m={np.round(coeffs[0], 3)}"
                )
        else:
            plt.plot(time, quantity[:], '.', label=key, markersize=2)

            if inds_exs is not None:
                # do the fitting
                coeffs = np.polyfit(
                    times_extrema, quantity_extrema, deg=degree)

                # plot quantity, extrema, and fit
                plt.plot(times_extrema, quantity_extrema,
                         'r*', label='local extrema')
                plt.plot(
                    time_cut,
                    np.exp(coeffs[0] * time_cut + coeffs[1]),
                    label=r"$a x + b$ with" +
                    f"\na={np.round(coeffs[1], 3)} b={np.round(coeffs[0], 3)}"
                )

    plt.legend()
    plt.xlabel('time')

    if save_plot:
        assert savedir is not None, 'When wanting to save the plot a path has to be given!'
        plt.savefig(os.path.join(savedir, 'scalars' + '.' + file_format))
    if show_plot:
        plt.show()


def plot_distr_fun(path, time_idx, grid_slices,
                   save_plot=False, savepath=None, file_format='png'):
    """ Plot the binned distribution function at given slices of the phase space.

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


def phase_space_video(t_grid, grid_slices, slice_name, marker_type, species, path, model_name, background_params=None):
    """ Create a video of all 2D slices of the distribution function over time.

    Parameters
    ----------
    t_grid : np.ndarray
        1D-array containing all the times

    grid_slices : dict
        holds the names of the directions as keys and the values at where the function should
        be evaluated as values

    slice_name : str
        The name of the slicing, e.g. e2_v1_v2

    marker_type : str
        one of full_f, control_variate, delta_f

    species : str
        the name of the species

    path : str
        the path to the data of which the videos should be created

    model_name : str
        name of the model that was run

    background_params : dict [optional]
        parameters of the maxwellian background type if a full_f method was used
    """
    # Make sure that the slice that was saved during the simulation is at least 2D
    if '_' not in slice_name:
        return

    data_path = os.path.join(
        path, 'post_processing', 'kinetic_data', species, 'distribution_function', slice_name
    )

    # Check how many slicings have been given and make slices_2d for all
    # combinations of spatial and velocity dimensions
    slices_2d = []
    directions = slice_name.split('_')
    for direc1 in directions:
        if direc1[0] == 'e':
            for direc2 in directions:
                if direc2[0] == 'v':
                    slices_2d += [direc1 + '_' + direc2]
    print(
        f"Found {len(slices_2d)} 2D slicing(s) for {species}, proceeding to generate images")

    # Load all the grids
    grids = []
    for direction in directions:
        grids += [
            np.load(
                os.path.join(data_path, 'grid_' + direction + '.npy')
            )
        ]

    # Create folder for images of video
    vid_folder = os.path.join(path, 'videos')
    if not os.path.exists(vid_folder):
        os.mkdir(vid_folder)

    # If simulation was for full-f subtract the background function
    if marker_type == 'full_f':
        assert background_params is not None

        # Load background
        from struphy.kinetic_background import maxwellians
        background_type = background_params['type']
        if background_type in background_params.keys():
            background_function = getattr(
                maxwellians,
                background_params['type']
            )(background_params[background_type])
        else:
            background_function = getattr(
                maxwellians,
                background_params['type']
            )()

        bckgr_grids = []
        k = 0
        for direc in ['e1', 'e2', 'e3', 'v1', 'v2', 'v3']:
            if direc in directions:
                bckgr_grids += [grids[k]]
                k += 1
            else:
                bckgr_grids += [np.array(grid_slices[direc])]

        bckgr_mesh = np.meshgrid(*bckgr_grids, indexing='ij')

        background_data = background_function(*bckgr_mesh)

        df_data = np.load(
            os.path.join(
                data_path,
                'f_binned.npy'
            )
        ) - background_data[None, :, :, :, :, :, :].squeeze()
    elif marker_type in ['control_variate', 'delta_f']:
        df_data = np.load(
            os.path.join(
                data_path,
                'delta_f_binned.npy'
            )
        )
    else:
        raise NotImplementedError(
            f"Making a video for marker type {marker_type} is not implemented!")

    assert df_data is not None

    # Make plot series for each 2D slice
    for slc in slices_2d:
        # Get indices of where to plot in other directions
        grid_idxs = {}
        for k in range(df_data.ndim - 1):
            grid_idxs[directions[k]] = np.argmin(
                np.abs(grids[k] - grid_slices[directions[k]]))

        eta_grid = np.load(
            os.path.join(
                data_path,
                'grid_' + slc[:2] + '.npy'
            )
        )
        v_grid = np.load(
            os.path.join(
                data_path,
                'grid_' + slc[-2:] + '.npy'
            )
        )

        # Prepare slicing
        f_slicing = [0] * df_data.ndim
        for k in range(df_data.ndim):
            # directions in which f is evaluated at a point
            if directions[k - 1] in slc:
                f_slicing[k] = slice(None)
            else:
                f_slicing[k] = grid_idxs[directions[k - 1]]

        # Create folder for saving the images series
        imgs_folder = os.path.join(vid_folder, slc)
        if not os.path.exists(imgs_folder):
            os.mkdir(imgs_folder)

        phase_space_plots(
            t_grid=t_grid,
            eta_grid=eta_grid,
            v_grid=v_grid,
            df_binned=df_data[tuple(f_slicing)].squeeze(),
            save_path=imgs_folder,
            model_name=model_name,
            eta_label=slc[:2],
            v_label=slc[-2:]
        )
        print("Phase space plots have been successfully created!")

        try:
            import cv2
        except:
            yn = input(
                "It seems like cv2 is not installed. Would you like to install it now (Y/n)?")

            if yn in ('', 'Y', 'y', 'yes', 'Yes'):
                subprocess.run(
                    ["python3", "-m", "pip", "install", "opencv-python"])
            else:
                return

        images = [
            img for img in sorted(
                os.listdir(imgs_folder)
            ) if img.endswith(".png")
        ]
        frame = cv2.imread(os.path.join(imgs_folder, images[0]))
        height, width, _ = frame.shape

        fps = 15
        video = cv2.VideoWriter(
            os.path.join(
                vid_folder,
                'video_' + slc + '.avi',
            ),
            0, fps, (width, height)
        )

        print("Creating video now")
        for image in tqdm(images):
            video.write(cv2.imread(os.path.join(imgs_folder, image)))

        cv2.destroyAllWindows()
        video.release()


def phase_space_plots(t_grid, eta_grid, v_grid, df_binned, save_path, model_name, eta_label=None, v_label=None):
    """ Create a time series of phase space plots for given delta-f data

    Parameters
    ----------
    t_grid : np.ndarray
        1D-array containing all the times

    eta_grid : np.ndarray
        1D-array containing all values in spatial direction

    v_grid : np.ndarray
        1D-array containing all values in velocity direction

    df_binned : np.ndarray
        3D-array containing all values of the distribution function

    save_path : str
        the path to where the images should be stored

    model_name : str
        name of the model that was run

    eta_label : str
        name of the spatial direction

    v_label : str
        name of the velocity direction
    """
    assert t_grid.ndim == eta_grid.ndim == v_grid.ndim == 1, f"Input arrays must be 1D!"
    assert df_binned.shape[0] == t_grid.size, f"{df_binned.shape =}, {t_grid.shape =}"
    assert df_binned.shape[1] == eta_grid.size, f"{df_binned.shape =}, {eta_grid.shape =}"
    assert df_binned.shape[2] == v_grid.size, f"{df_binned.shape =}, {v_grid.shape =}"

    ee1, vv1 = np.meshgrid(eta_grid, v_grid, indexing='ij')

    nt = len(t_grid)
    log_nt = int(np.log10(nt)) + 1
    len_dt = len(str(t_grid[1]).split('.')[1])

    cmap = 'Oranges'
    vmin = np.min(df_binned)
    vmax = np.max(df_binned)

    plt.figure(figsize=(9, 6))
    for n in tqdm(range(nt)):
        t = f'%.{len_dt}f' % t_grid[n]
        plt.pcolor(ee1, vv1, df_binned[n], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(f'$t=${t}, from Struphy model "{model_name}"')
        if eta_label is not None:
            plt.xlabel(fr"$\eta_{eta_label[-1]}$")
        if v_label is not None:
            plt.ylabel(fr"$v_{v_label[-1]}$")
        plt.savefig(
            os.path.join(
                save_path,
                'step_{0:0{1}d}.png'.format(n, log_nt)
            ),
            bbox_inches="tight"
        )
        plt.clf()
