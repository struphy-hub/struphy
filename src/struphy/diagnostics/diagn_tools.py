#!/usr/bin/env python3
import os
import shutil
import subprocess

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftfreq, fftn
from scipy.signal import argrelextrema
from tqdm import tqdm

from struphy.dispersion_relations import analytic


def power_spectrum_2d(
    values: dict,
    name: str,
    grids: tuple,
    grids_mapped: tuple = None,
    component: int = 0,
    slice_at: tuple = (None, 0, 0),
    do_plot: bool = False,
    disp_name: str = None,
    disp_params: dict = {},
    fit_branches: int = 0,
    noise_level: float = 0.1,
    extr_order: int = 10,
    fit_degree: tuple = (1,),
    save_plot: bool = False,
    save_name: str = None,
    file_format: str = "png",
):
    """Perform fft in space-time, (t, x) -> (omega, k), where x can be a logical or physical coordinate.
    Returns values if plot=False.

    Parameters
    ----------
    values : dict
        Dictionary holding values of a B-spline FemField on the grid as 3d np.arrays:
        values[n] contains the values at time step n, where n = 0:Nt-1:step with 0<step.

    name : str
        Name of the FemField.

    grids : 3-tuple
        1d logical grids in each eta-direction with Nel[i]*npts_per_cell[i] + 1 entries in each direction.

    grids_mapped : 3-tuple
        Mapped grids obtained by domain(). If None, the fft is performed on the logical grids.

    component : int
        Which component of a FemField to consider; is 0 for 0-and 3-forms, is in {0, 1, 2} for 1- and 2-forms.

    slice_at : 3-tuple
        At which indices i, j the 1d slice data (t, eta)_(i, j) should be obtained.
        One entry must be "None"; this is the direction of the fft.
        Default: [None, 0, 0] performs the eta1-fft at (eta2[0], eta3[0]).

    do_plot : boolean
        Plot result if True, otherwise return things.

    disp_name : str
        The name of the dispersion relation class in struphy.dispersion_relations.analytic to be used for analytic comparison.

    disp_params : dict
        Parameters needed for analytical dispersion relation, see struphy.dispersion_relations.analytic.

    fit_branches: int
        How many branches to fit in the dispersion relation.
        Default=0 means no fits are made.
        
    noise_level: float
        Sets the threshold above which local maxima in the power spectrum are taken into account.
        Computed as threshold = max(spectrum) * noise_level. 
        
    extr_oder: int
        Order given to argrelextrema.
        
    fit_degree: tuple[int]
        Degree of fitting polynomial for each branch (fit_branches) of power spectrum.

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
        2d array of shape (omega.size, kvec.size) holding the fft.
        
    coeffs : list[list]
        List of fitting coefficients (lenght is fit_branches).
    """

    keys = list(values.keys())

    # check uniform grid in time
    dt = keys[1] - keys[0]
    assert np.all([np.abs(y - x - dt) < 1e-12 for x, y in zip(keys[:-1], keys[1:])])

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
    assert np.allclose(grid[1:] - grid[:-1], dx * np.ones_like(grid[:-1]))

    dispersion = (2.0 / Nt) * (2.0 / Nx) * np.abs(fftn(data))[: Nt // 2, : Nx // 2]
    kvec = 2 * np.pi * fftfreq(Nx, dx)[: Nx // 2]
    omega = 2 * np.pi * fftfreq(Nt, dt)[: Nt // 2]

    coeffs = None
    if fit_branches > 0:
        assert len(fit_degree) == fit_branches
        # determine maxima for each k
        k_start = kvec.size // 8 # take only first half of k-vector
        k_end = kvec.size // 2 # take only first half of k-vector
        k_fit = []
        omega_fit = {}
        for n in range(fit_branches):
            omega_fit[n] = []
        for k, f_of_omega in zip(kvec[k_start:k_end], dispersion[:, k_start:k_end].T):
            threshold = np.max(f_of_omega) * noise_level
            extrms = argrelextrema(f_of_omega, np.greater, order=extr_order)[0]
            above_noise = np.nonzero(f_of_omega > threshold)[0]
            intersec = list(set(extrms) & set(above_noise))
            # intersec = list(set(extrms))
            if not intersec:
                continue
            intersec.sort()
            # print(f"{intersec = }")
            # print(f"{[omega[intersec[n]] for n in range(fit_branches)]}")
            assert len(intersec) == fit_branches, f"Number of found branches {len(intersec)} is not {fit_branches = }! \
                Try to lower 'noise_level' or increase 'extr_order'."
            k_fit += [k]
            for n in range(fit_branches):
                omega_fit[n] += [omega[intersec[n]]]
        
        # fit
        coeffs = []
        for m, om in omega_fit.items():
            coeffs += [np.polyfit(k_fit, om, deg=fit_degree[n])]    
        print(f"\nFitted {coeffs = }")

    if do_plot:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        colormap = "plasma"
        K, W = np.meshgrid(kvec, omega)
        lvls = np.logspace(-15, -1, 27)
        disp_plot = ax.contourf(
            K,
            W,
            dispersion**2 / (dispersion**2).max(),
            cmap=colormap,
            norm=colors.LogNorm(),
            levels=lvls,
        )
        plt.colorbar(
            ticks=[1e-12, 1e-9, 1e-6, 1e-3],
            mappable=disp_plot,
            format="%.0e",
        )
        title = name + ", component " + str(component + 1)
        ax.set_title(title)
        ax.set_xlabel("$k$ [a.u.]")
        ax.set_ylabel(r"$\omega$ [a.u.]")
        
        if fit_branches > 0:
            for n, cs in enumerate(coeffs):
                def fun(k):
                    out = k*0.0
                    for i, c in enumerate(np.flip(cs)):
                        out += c * k**i
                    return out
                ax.plot(kvec, fun(kvec), "r:", label=f"fit_{n + 1}")

        # analytic solution:
        disp_class = getattr(analytic, disp_name)
        disp = disp_class(**disp_params)

        kpara = kvec

        branches = disp(kpara)
        set_min = 0.0
        set_max = 0.0
        for key, branch in branches.items():
            vals = np.real(branch)
            ax.plot(kvec, vals, "--", label=key)
            tmp = np.min(vals)
            if tmp < set_min:
                set_min = tmp
            tmp = np.max(vals)
            if tmp > set_max:
                set_max = tmp

        ax.legend()
        ax.set_xlim(0, kvec[-1])
        ax.set_ylim(set_min * 1.1, set_max * 1.1)

        if save_plot:
            assert save_name is not None, "When wanting to save the plot a path has to be given!"
            plt.savefig(save_name + "." + file_format)
        else:
            plt.show()

    return omega, kvec, dispersion, coeffs


def plot_scalars(
    time,
    scalar_quantities,
    scalars_plot=None,
    do_log=False,
    do_fit=False,
    fit_minima=False,
    order=4,
    no_extrema=4,
    start_extremum=0,
    degree=1,
    show_plot=False,
    save_plot=False,
    savedir=None,
    file_format="png",
):
    """Plot the scalar quantities and the relative error in the total energy for a simulation.

    Parameters
    ----------
    scalar_quantities : dict
        HDF5 dictionary dataset containing the scalar quantities that were saved during the simulation

    scalars_plot : list | tuple
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

    if "en_tot" in scalar_quantities.keys():
        en_tot = scalar_quantities["en_tot"][:]

        plt.figure("en_tot")
        if do_log:
            plt.semilogy(time, en_tot)
        else:
            plt.plot(time, en_tot)

        if save_plot:
            assert savedir is not None, "When wanting to save the plot a path has to be given!"
            plt.savefig(os.path.join(savedir, "en_tot" + "." + file_format))
        else:
            plt.show()

        plt.figure("en_tot_rel_err")
        plt.plot(
            time[1:],
            np.divide(
                np.abs(en_tot[1:] - en_tot[0]),
                en_tot[0],
            ),
        )

        if save_plot:
            assert savedir is not None, "When wanting to save the plot a path has to be given!"
            plt.savefig(
                os.path.join(
                    savedir,
                    "en_tot_rel_err" + "." + file_format,
                ),
            )
        if show_plot:
            plt.show()

    # Dict with label as key and time series as value
    plot_quantities = {}
    if scalars_plot is None:
        for key, quantity in scalar_quantities.items():
            if key not in ["time", "en_tot"]:
                plot_quantities[key] = quantity[:]
    else:
        for key in scalars_plot:
            plot_quantities[key] = scalar_quantities[key][:]

    # Make the figure
    plt.figure("scalars")
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
            quantity_extrema = plot_quantity[inds_exs][start_extremum : start_extremum + no_extrema]
            times_extrema = time[inds_exs][start_extremum : start_extremum + no_extrema]

            # for plotting take a bit more time at start and end
            if len(inds_exs[0]) >= 2:
                time_start_idx = np.max(
                    [0, 2 * inds_exs[0][start_extremum] - inds_exs[0][start_extremum + 1]],
                )
                time_end_idx = np.min(
                    [
                        len(time) - 1,
                        2 * inds_exs[0][start_extremum + no_extrema - 1] - inds_exs[0][start_extremum + no_extrema - 2],
                    ],
                )
                time_cut = time[time_start_idx:time_end_idx]
            else:
                time_cut = time

        if do_log:
            # plot quantity, extrema, and fit
            plt.semilogy(time, plot_quantity[:], ".", label=key, markersize=2)

            if inds_exs is not None:
                # do the fitting
                coeffs = np.polyfit(
                    times_extrema,
                    np.log(
                        quantity_extrema,
                    ),
                    deg=degree,
                )
                plt.plot(
                    times_extrema,
                    quantity_extrema,
                    "r*",
                    label="local extrema",
                )
                plt.plot(
                    time_cut,
                    np.exp(coeffs[0] * time_cut + coeffs[1]),
                    label=r"$a * \exp(m x)$ with" + f"\na={np.round(np.exp(coeffs[1]), 3)} m={np.round(coeffs[0], 3)}",
                )
        else:
            plt.plot(time, plot_quantity[:], ".", label=key, markersize=2)

            if inds_exs is not None:
                # do the fitting
                coeffs = np.polyfit(
                    times_extrema,
                    quantity_extrema,
                    deg=degree,
                )

                # plot quantity, extrema, and fit
                plt.plot(
                    times_extrema,
                    quantity_extrema,
                    "r*",
                    label="local extrema",
                )
                plt.plot(
                    time_cut,
                    np.exp(coeffs[0] * time_cut + coeffs[1]),
                    label=r"$a x + b$ with" + f"\na={np.round(coeffs[1], 3)} b={np.round(coeffs[0], 3)}",
                )

    plt.legend()
    plt.xlabel("time")

    if save_plot:
        assert savedir is not None, "When wanting to save the plot a path has to be given!"
        plt.savefig(os.path.join(savedir, "scalars" + "." + file_format))
    if show_plot:
        plt.show()


def plot_distr_fun(
    path,
    time_idx,
    grid_slices,
    save_plot=False,
    savepath=None,
    file_format="png",
):
    """Plot the binned distribution function at given slices of the phase space.

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

    species = str(path.split("/")[-1])
    path = os.path.join(path, "distribution_function")

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
            if filename == "f_binned.npy":
                f = np.load(filepath)

            # load delta f
            elif filename == "delta_f_binned.npy":
                delta_f = np.load(filepath)

        assert f is not None, "No distribution function file found!"

        # Load grid
        directions = folder.split("_")
        for direction in directions:
            grids += [
                np.load(
                    os.path.join(
                        subpath,
                        "grid_" + direction + ".npy",
                    ),
                ),
            ]

        # Get indices of where to plot in other directions
        grid_idxs = {}
        for k in range(f.ndim - 1):
            grid_idxs[directions[k]] = np.argmin(
                np.abs(grids[k] - grid_slices[directions[k]]),
            )

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
                plt.figure("delta_f")
                plt.plot(grids[k], delta_f[tuple(f_slicing)].squeeze())
                plt.xlabel(directions[k])
                plt.ylabel(r"$\delta f$")
                plt.title(f"time step n={time_idx}")
                print(f"Created plot for delta_f in {directions[k]}")

                if save_plot:
                    assert savepath is not None, "When wanting to save the plot a path has to be given!"
                    savename = os.path.join(
                        savepath,
                        species + "_delta_f_" + directions[k] + "." + file_format,
                    )
                    plt.savefig(savename)
                else:
                    plt.show()
                plt.close()

            # plot full f
            if f is not None:
                plt.figure("f")
                plt.plot(grids[k], f[tuple(f_slicing)].squeeze())
                plt.xlabel(directions[k])
                plt.ylabel(r"$f$")
                plt.title(f"time step n={time_idx}")
                print(f"Created plot for f in {directions[k]}")

                if save_plot:
                    assert savepath is not None, "When wanting to save the plot a path has to be given!"
                    savename = os.path.join(
                        savepath,
                        species + "_f_" + directions[k] + "." + file_format,
                    )
                    plt.savefig(savename)
                else:
                    plt.show()
                plt.close()

        del grids
        del f
        del delta_f


def plots_videos_2d(
    t_grid,
    grid_slices,
    slice_name,
    plot_full_f,
    species,
    path,
    model_name,
    output: str = "overview",
    background_params=None,
    n_times=6,
    show_plot=False,
    save_plot=True,
    polar_params={},
):
    """TODO"""
    choices = ["overview", "video"]
    assert output in choices, f"Can only do one of {choices=} but got {output=}"

    # Make sure that the slice that was saved during the simulation is at least 2D
    if "_" not in slice_name:
        return

    if polar_params == {}:
        do_polar = False
    else:
        do_polar = polar_params["do_polar"]

    data_path = os.path.join(
        path,
        "post_processing",
        "kinetic_data",
        species,
        "distribution_function",
        slice_name,
    )

    # Create a folder for the diagnostics
    diagn_path = os.path.join(path, "diagnostics")
    if (output == "overview" and save_plot) or output == "video":
        if not os.path.exists(diagn_path):
            os.mkdir(diagn_path)

    slices_2d, grids, directions, df_data = get_slices_grids_directions_and_df_data(
        plot_full_f=plot_full_f,
        background_params=background_params,
        grid_slices=grid_slices,
        data_path=data_path,
        slice_name=slice_name,
    )

    # Make plot series for each 2D slice
    for slc in slices_2d:
        # Assign some nicer names
        label_1 = slc[:2]
        label_2 = slc[-2:]

        # Only needed for "video" option
        images_path = None
        if output == "video":
            # Create folder for saving the images series
            images_path = os.path.join(
                diagn_path,
                "video_frames_" + slc,
            )
            if os.path.exists(images_path):
                shutil.rmtree(images_path)

            os.mkdir(images_path)

        # Get indices of where to plot in other directions
        grid_idxs = {}
        for k in range(df_data.ndim - 1):
            direc = directions[k]
            grid_idxs[direc] = np.argmin(
                np.abs(grids[direc] - grid_slices[direc]),
            )

        grid_1 = np.load(
            os.path.join(
                data_path,
                "grid_" + label_1 + ".npy",
            ),
        )
        grid_2 = np.load(
            os.path.join(
                data_path,
                "grid_" + label_2 + ".npy",
            ),
        )

        # Prepare slicing
        f_slicing = [0] * df_data.ndim
        for k in range(df_data.ndim):
            # directions in which f is evaluated at a point
            if directions[k - 1] in slc:
                f_slicing[k] = slice(None)
            else:
                f_slicing[k] = grid_idxs[directions[k - 1]]

        df_binned = df_data[tuple(f_slicing)].squeeze()

        assert t_grid.ndim == grid_1.ndim == grid_2.ndim == 1, f"Input arrays must be 1D!"
        assert df_binned.shape[0] == t_grid.size, f"{df_binned.shape =}, {t_grid.shape =}"
        assert df_binned.shape[1] == grid_1.size, f"{df_binned.shape =}, {grid_1.shape =}"
        assert df_binned.shape[2] == grid_2.size, f"{df_binned.shape =}, {grid_2.shape =}"

        # Scale the coordinates to cartesian sizes for plot to be more obvious
        if do_polar:
            for sl, var in zip([label_1, label_2], [grid_1, grid_2]):
                if sl in polar_params.values():
                    if polar_params["radial_coord"] == sl:
                        var *= polar_params["r_max"] - polar_params["r_min"]
                        var += polar_params["r_min"]
                    elif polar_params["angular_coord"] == sl:
                        var *= 2 * np.pi

        grid_1_mesh, grid_2_mesh = np.meshgrid(grid_1, grid_2, indexing="ij")

        if output == "video":
            plots_2d_video(
                t_grid=t_grid,
                grid_1_mesh=grid_1_mesh,
                grid_2_mesh=grid_2_mesh,
                df_binned=df_binned,
                model_name=model_name,
                label_1=label_1,
                label_2=label_2,
                do_polar=do_polar,
                images_path=images_path,
            )

            video_2d(
                slc=slc,
                diagn_path=diagn_path,
                images_path=images_path,
            )

        elif output == "overview":
            plots_2d_overview(
                t_grid=t_grid,
                grid_1_mesh=grid_1_mesh,
                grid_2_mesh=grid_2_mesh,
                slc=slc,
                df_binned=df_binned,
                save_path=diagn_path,
                model_name=model_name,
                label_1=label_1,
                label_2=label_2,
                do_polar=do_polar,
                n_times=n_times,
                show_plot=show_plot,
                save_plot=save_plot,
            )

        else:
            raise NotImplementedError(f"{output=} is not implemented!")


def video_2d(slc, diagn_path, images_path):
    """Create a video of all 2D slices of the distribution function over time.

    Parameters
    ----------
    t_grid : np.ndarray
        1D-array containing all the times

    grid_slices : dict
        holds the names of the directions as keys and the values at where the function should
        be evaluated as values

    slice_name : str
        The name of the slicing, e.g. e2_v1_v2

    plot_full_f : bool
        whether to plot full-f instead of delta-f data

    species : str
        the name of the species

    path : str
        the path to the data of which the videos should be created

    model_name : str
        name of the model that was run

    background_params : dict [optional]
        parameters of the maxwellian background type if a full_f method was used
    """

    try:
        import cv2
    except:
        yn = input(
            "It seems like cv2 is not installed. Would you like to install it now (Y/n)?",
        )

        if yn in ("", "Y", "y", "yes", "Yes"):
            subprocess.run(
                ["python3", "-m", "pip", "install", "opencv-python"],
            )
        else:
            return

    images = [
        img
        for img in sorted(
            os.listdir(images_path),
        )
        if img.endswith(".png")
    ]
    frame = cv2.imread(os.path.join(images_path, images[0]))
    height, width, _ = frame.shape

    fps = 15
    video = cv2.VideoWriter(
        os.path.join(
            diagn_path,
            "video_" + slc + ".avi",
        ),
        0,
        fps,
        (width, height),
    )

    print("Creating video now")
    for image in tqdm(images):
        video.write(cv2.imread(os.path.join(images_path, image)))

    cv2.destroyAllWindows()
    video.release()


def plots_2d_video(
    t_grid,
    grid_1_mesh,
    grid_2_mesh,
    df_binned,
    model_name,
    label_1=None,
    label_2=None,
    do_polar=False,
    images_path=None,
):
    # Best color scheme
    cmap = "seismic"

    vmin = []
    vmax = []

    # Get parameters for time and labelling for it
    nt = len(t_grid)
    log_nt = int(np.log10(nt)) + 1
    len_dt = len(str(t_grid[1]).split(".")[1])

    # Get the correct scale for the plots
    vmin += [np.min(df_binned[:]) / 3]
    vmax += [np.max(df_binned[:]) / 3]
    vmin = np.min(vmin)
    vmax = np.max(vmax)
    vscale = np.max(np.abs([vmin, vmax]))

    # Set up the figure and axis once
    if do_polar:
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection="polar"))
        im = ax.pcolormesh(grid_2_mesh, grid_1_mesh, df_binned[0], cmap=cmap, vmin=-vscale, vmax=vscale)
    else:
        fig, ax = plt.subplots(figsize=(9, 9))
        im = ax.pcolormesh(grid_1_mesh, grid_2_mesh, df_binned[0], cmap=cmap, vmin=-vscale, vmax=vscale)

    # Create the colorbar once
    fig.colorbar(im, ax=ax)

    for k in tqdm(range(nt)):
        obj = plt
        t = f"%.{len_dt}f" % t_grid[k]

        # Set the title including the time
        fig.suptitle(rf"Struphy model '{model_name}', $t=${t}")

        # Update the plot data. pcolormesh returns a QuadMesh; update its array.
        # Note: set_array expects a 1D array, so we flatten the data.
        im.set_array(df_binned[k].ravel())

        # Force a re-draw of the canvas
        fig.canvas.draw_idle()

        # Only add axis labels for non-polar plots since it confuses
        if not do_polar:
            if label_1[0] == "e":
                obj.xlabel(rf"$\eta_{label_1[-1]}$")
            else:
                obj.xlabel(rf"$v_{label_1[-1]}$")
            if label_2[0] == "e":
                obj.ylabel(rf"$\eta_{label_2[-1]}$")
            else:
                obj.ylabel(rf"$v_{label_2[-1]}$")

        # Save the current frame
        plt.savefig(
            os.path.join(
                images_path,
                "step_{0:0{1}d}.png".format(k, log_nt),
            ),
            bbox_inches="tight",
            dpi=150,
        )

    # Clear the figure
    plt.clf()

    plt.close("all")


def plots_2d_overview(
    t_grid,
    grid_1_mesh,
    grid_2_mesh,
    slc,
    df_binned,
    save_path,
    model_name,
    label_1=None,
    label_2=None,
    do_polar=False,
    n_times=1,
    show_plot=False,
    save_plot=True,
):
    # Best color scheme
    cmap = "seismic"

    times = []
    for k in range(n_times):
        times += [int((len(t_grid) - 1) * k / n_times)]

    # Get parameters for time and labelling for it
    len_dt = len(str(t_grid[1]).split(".")[1])

    # Assign some values and change them below
    vmin = []
    vmax = []
    n_rows = 1
    n_cols = 1
    fig_size = (1, 1)
    fig_height = 1

    # Make nice layout for subplots
    if n_times in [1, 2, 3]:
        n_cols = n_times
        n_rows = 1
        fig_height = 4.5
    elif n_times == 4:
        n_cols = 2
        n_rows = 2
        fig_height = 8.5
    else:
        n_cols = 3
        n_rows = int(np.ceil(n_times / n_cols))
        fig_height = 4 * n_rows

    fig_size = (4 * n_cols, fig_height)

    # Get the correct scale for the plots
    for time in times:
        vmin += [np.min(df_binned[time]) / 3]
        vmax += [np.max(df_binned[time]) / 3]
    vmin = np.min(vmin)
    vmax = np.max(vmax)
    vscale = np.max(np.abs([vmin, vmax]))

    # Plot options for polar plots
    subplot_kw = dict(projection="polar") if do_polar else None

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, subplot_kw=subplot_kw)

    # So we an use .flatten() even for just 1 plot
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # fig.tight_layout(h_pad=5.0, w_pad=5.0)
    # fig.tight_layout(pad=5.0)
    plt.subplots_adjust(
        left=0.05,
        bottom=0.1,
        right=0.85,
        top=0.9,
        wspace=0.3,
        hspace=0.35,
    )

    # Set the suptitle
    fig.suptitle(f"Struphy model '{model_name}'")

    for k in np.arange(n_times):
        obj = axes.flatten()[k]
        n = times[k]
        t = f"%.{len_dt}f" % t_grid[n]

        obj.title.set_text(rf"$t=${t}")

        # Plot the data
        if not do_polar:
            im = obj.pcolor(grid_1_mesh, grid_2_mesh, df_binned[n], cmap=cmap, vmin=-vscale, vmax=vscale)
        else:
            im = obj.pcolor(grid_2_mesh, grid_1_mesh, df_binned[n], cmap=cmap, vmin=-vscale, vmax=vscale)

        # Only add axis labels for non-polar plots since it confuses
        if not do_polar:
            if label_1[0] == "e":
                obj.set_xlabel(rf"$\eta_{label_1[-1]}$")
            else:
                obj.set_xlabel(rf"$v_{label_1[-1]}$")
            if label_2[0] == "e":
                obj.set_ylabel(rf"$\eta_{label_2[-1]}$")
            else:
                obj.set_ylabel(rf"$v_{label_2[-1]}$")

    # Add global colorbar
    cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax)

    if save_plot:
        plt.savefig(
            os.path.join(
                save_path,
                "overview_" + slc + ".png",
            ),
            dpi=150,
        )

    if show_plot:
        plt.show()

    # Clear the figure
    plt.clf()

    plt.close("all")


def get_slices_grids_directions_and_df_data(plot_full_f, grid_slices, data_path, slice_name, background_params=None):
    """Prepare the lists of slices, grids, and directions from the given data and extract the delta-f data.

    Parameters
    ----------
    plot_full_f : bool
        whether to plot full-f instead of delta-f data

    grid_slices : dict
        holds the names of the directions as keys and the values at where the function should
        be evaluated as values

    data_path : str
        the path to the data which should be prepared

    slice_name : str
        The name of the slicing, e.g. e2_v1_v2

    background_params : dict [optional]
        parameters of the maxwellian background type if a full_f method was used

    Returns
    -------
    slices_2d : list[string]
        A list of all the slicings

    grids : list[np.ndarray]
        A list of all grids according to the slices

    directions : list[string]
        A list of the directions that appear in all slices

    df_data : np.ndarray
        The data of delta-f (in case of full-f: distribution function minus background)
    """

    directions = slice_name.split("_")

    # Load all the grids
    grids = {}
    for direction in directions:
        grids[direction] = np.load(
            os.path.join(data_path, "grid_" + direction + ".npy"),
        )

    # If simulation was for full-f subtract the background function
    if plot_full_f:
        _name = "f_binned.npy"
    else:
        _name = "delta_f_binned.npy"
    _data = np.load(os.path.join(data_path, _name))

    # Check how many slicings have been given and make slices_2d for all
    # combinations of spatial and velocity dimensions
    slices_2d = []
    for direc1 in directions:
        for direc2 in directions:
            if (direc1 != direc2) and (direc2 + "_" + direc1) not in slices_2d:
                slices_2d += [direc1 + "_" + direc2]

    return slices_2d, grids, directions, _data
