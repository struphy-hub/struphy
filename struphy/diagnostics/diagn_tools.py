#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from scipy.fft import fftfreq ,fftn
import matplotlib.colors as colors
import yaml
import h5py

from struphy.dispersion_relations import analytic

from struphy.diagnostics.post_processing import create_femfields, eval_femfields

from psydac.fem.vector import ProductFemSpace

import sysconfig
import matplotlib.pyplot as plt
import numpy as np


def fourier_1d(values, code, grids, grids_phys=None, component=0, slice_at=[None, 0, 0], plot=False):
    """
    Perform fft in space-time, (t, x) -> (omega, k), where x can be a logical or physical coordinate.
    Returns values if plot=False.
    
    Parameters
    ----------
        values : dict
            Dictionary holding values of a B-spline FemField on the grid as 3d np.arrays:
            values[n] contains the values at time step n, where n = 0:Nt-1:step with 0<step. 
            
        code : str
            From which code the data has been obtained.

        grids : 3-list
            Logical grids corresponding to values. Each entry is 1d grid in one
            eta-direction in the format (Nel[i], npts_per_cell[i]). Grids are equally spaced.

        grids_phys : 3-list
            Physical grid corresponding to values. Each entry is a 3d np.arrays
            which is the mapping component F_i evaluated at meshgrid(*grids).
            If None, the fft is performed on the logical grids.

        component : int
            Which component of a FemField to consider; is 0 for 0-and 3-forms, is in {0, 1, 2} for 1- and 2-forms.

        slice_at : 3-list
            At which indices i, j the 1d slice data (t, eta)_(i, j) should be obtained. 
            One entry must be "None"; this is the direction of the fft. 
            Default: [None, 0, 0] performs the eta1-fft at (eta2[0], eta3[0]). 

        plot : boolean
            Plot result if True, otherwise return things.

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
        grid = grids[0].flatten()
        if grids_phys is not None:
            grid = grids_phys[0][:, slice_at[1], slice_at[2]]

    elif slice_at[1] == None:

        data = temp[:, slice_at[0], :, slice_at[2]]
        grid = grids[1].flatten()
        if grids_phys is not None:
            grid = grids_phys[1][slice_at[0], :, slice_at[2]]

    elif slice_at[2] == None:

        data = temp[:, slice_at[0], slice_at[1], :]
        grid = grids[2].flatten()
        if grids_phys is not None:
            grid = grids_phys[2][slice_at[0], slice_at[1], :]

    else:
        AssertionError('One entry of slice_at must be "None".')

    Nt, Nx = data.shape
    # check uniform grid in space
    dx = grid[1] - grid[0]
    print(f'space step: {dx}')
    assert np.allclose(grid[1:] - grid[:-1], dx*np.ones_like(grid[:-1]))

    dispersion      = (2./Nt)*(2./Nx)*np.abs(fftn(data))[:Nt//2, :Nx//2] 
    kvec            = 2*np.pi*fftfreq(Nx, dx)[:Nx//2]
    omega           = 2*np.pi*fftfreq(Nt, dt)[:Nt//2] 

    if plot:
        
        fig, ax     = plt.subplots(1, 1, figsize=(10,10))
        colormap    = 'plasma'
        K, W        = np.meshgrid(kvec, omega)
        ax.contourf(K, W, dispersion**2/ (dispersion**2).max(), cmap=colormap, norm=colors.LogNorm())
        title = 'code: ' + code 
        ax.set_title(title) 
        ax.set_xlabel('$k$ [a.u.]')
        ax.set_ylabel('$\omega$ [a.u.]')

        # analytic solution:
        if code == 'maxwell_psydac' or code == 'maxwell':
            disp = analytic.Maxwell1D()
            kpara = kvec
            kperp = None
        else:
            raise NotImplementedError('Analytic dispersion relation not implemented.')

        branches = disp.spectrum(kpara, kperp=kperp)
        set_min = 0.
        set_max = 0.
        for key, branch in branches.items():
            vals = np.real(branch)
            ax.plot(kvec, vals, '--', label=key)
            tmp = np.min(vals)
            if tmp < set_min: set_min = tmp 
            tmp = np.max(vals)
            if tmp > set_max: set_max = tmp

        ax.legend()
        ax.set_xlim(0, kvec[-1])
        ax.set_ylim(set_min*1.1, set_max*1.1)

        plt.show()

    else:
        return kvec, omega, dispersion


