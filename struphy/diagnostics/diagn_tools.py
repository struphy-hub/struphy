#!/usr/bin/env python3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from scipy.fft import fftfreq, fftn
import matplotlib.colors as colors

from struphy.dispersion_relations import analytic

import pickle
import os, shutil


def fourier_1d(values, name, code, grids, masks, grids_mapped=None, component=0, slice_at=[None, 0, 0], plot=False, disp_name=None, disp_params={}):
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
            1d logical grids in each eta-direction with Nel[i] * npts_per_cell[i] entries. 
            All break points other than 0. and 1. appear twice; double entries can be eliminated by using masks. 
            
        masks : 3-list
            Each entry is a boolean list of same size as the corresponding grids entry. 
            It is False where a double counted break point appears, and True otherwise.
            Hence grids[i][masks[i]] gives an equally spaced 1d logical grid.

        grids_mapped : 3-list
            Mapped grids obtained by domain.evaluate(). If None, the fft is performed on the logical grids.

        component : int
            Which component of a FemField to consider; is 0 for 0-and 3-forms, is in {0, 1, 2} for 1- and 2-forms.

        slice_at : 3-list
            At which indices i, j the 1d slice data (t, eta)_(i, j) should be obtained. 
            One entry must be "None"; this is the direction of the fft. 
            Default: [None, 0, 0] performs the eta1-fft at (eta2[0], eta3[0]). 

        plot : boolean
            Plot result if True, otherwise return things.

        disp_name : str
            The name of the dispersion relation class in struphy.dispersion_relations.analytic to be used for analytic comparison.

        disp_params : dict
            Parameters needed for analytical dispersion relation, see struphy.dispersion_relations.analytic.

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

        data_brk = temp[:, :, slice_at[1], slice_at[2]]
        grid_brk = grids[0].flatten()
        if grids_mapped is not None:
            grid_brk = grids_mapped[0][:, slice_at[1], slice_at[2]]
        mask = masks[0]

    elif slice_at[1] == None:

        data_brk = temp[:, slice_at[0], :, slice_at[2]]
        grid_brk = grids[1].flatten()
        if grids_mapped is not None:
            grid_brk = grids_mapped[1][slice_at[0], :, slice_at[2]]
        mask = masks[1]

    elif slice_at[2] == None:

        data_brk = temp[:, slice_at[0], slice_at[1], :]
        grid_brk = grids[2].flatten()
        if grids_mapped is not None:
            grid_brk = grids_mapped[2][slice_at[0], slice_at[1], :]
        mask = masks[2]

    else:
        AssertionError('One entry of slice_at must be "None".')

    # eliminate double appearance of break points in grid
    grid = grid_brk[mask]

    # eliminate double output of data at break points
    Nt = data_brk.shape[0]
    Nx = grid.size
    data = np.empty((Nt, Nx), dtype=float)
    data[:, :] = data_brk[:, mask]

    # extract uniform grid in space
    dx = grid[1] - grid[0]
    print(f'space step: {dx}')
    assert np.allclose(grid[1:] - grid[:-1], dx*np.ones_like(grid[:-1]))

    dispersion = (2./Nt)*(2./Nx)*np.abs(fftn(data))[:Nt//2, :Nx//2]
    kvec = 2*np.pi*fftfreq(Nx, dx)[:Nx//2]
    omega = 2*np.pi*fftfreq(Nt, dt)[:Nt//2]

    if plot:

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
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
        kperp = None

        branches = disp(kpara, kperp=kperp)
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

        plt.show()

    else:
        return kvec, omega, dispersion


def show_poloidal(path, name):
    '''Show poloidal planes at eta3=0 and eta3=1, for testing. Also plot grid.
    This is for comparison with paraview.'''

    with open(path + '/eval_fields/' + name + '_logical.bin', 'rb') as handle:
        point_data_logical = pickle.load(handle)

    with open(path + '/eval_fields/' + name + '_physical.bin', 'rb') as handle:
        point_data_physical = pickle.load(handle)

    with open(path + '/eval_fields/grids.bin', 'rb') as handle:
        grids = pickle.load(handle)

    with open(path + '/eval_fields/grids_mapped.bin', 'rb') as handle:
        grids_mapped = pickle.load(handle)

    fig0 = plt.figure()
    ax0 = fig0.add_subplot(projection='3d')

    ax0.plot_wireframe(grids_mapped[0][:, :, 0], grids_mapped[1][:, :, 0], np.zeros_like(
        grids_mapped[1][:, :, 0]))
    ax0.plot_wireframe(grids_mapped[0][:, -1, :], np.ones_like(
        grids_mapped[1][:, -1, :]), grids_mapped[2][:, -1, :])
    ax0.plot_wireframe(np.zeros_like(grids_mapped[1][-1, :, :]), grids_mapped[1][-1, :, :], grids_mapped[2][-1, :, :])
    plt.show()

    # directory for png files
    png_path = path + '/eval_fields/' + name 
    try:
        os.mkdir(png_path + '/png/')
    except:
        shutil.rmtree(png_path + '/png/')
        os.mkdir(png_path + '/png/')

    for n, (t, val) in enumerate(point_data_physical.items()):

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        im0 = axs[0].pcolor(grids_mapped[0][:, :, 0], grids_mapped[1][:, :, 0], val[0][:, :, 0], vmin=-1e-3, vmax=1e-3)
        im1 = axs[1].pcolor(grids_mapped[0][:, :, -1], grids_mapped[1][:, :, -1], val[0][:, :, -1], vmin=-1e-3, vmax=1e-3)
        axs[0].axis([0., 1., 0., 1.])
        axs[1].axis([0., 1., 0., 1.])
        plt.colorbar(im0, ax=axs[0])
        plt.colorbar(im1, ax=axs[1])
        axs[0].set_title(f'poloidal plane at eta3 = 0.0, n={n}')
        axs[1].set_title(f'poloidal plane at eta3 = 1.0, n={n}')
        plt.savefig(png_path + '/png/step_{0:04d}'.format(n))
        plt.close(fig)
        print(f'{n + 1} png files saved.')


if __name__ == '__main__':
    path = '/home/spossann/git_repos/struphy/struphy/io/out/sim_1'
    show_poloidal(path, 'e_field')
