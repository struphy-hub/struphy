#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from scipy.fft import fftfreq ,fftn
import matplotlib.colors as colors
import yaml
import h5py

from struphy.models.dispersion_relations import analytic


def fourier_1d(values, code, grids, grids_phys=None, component=0, slice_at=[None, 0, 0], plot=False):
    """
    Perform fft in space-time, (t, x) -> (omega, k), where x can be a logical or physical coordinate.
    
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


def load_values_from_path(path_out, params_name='parameters.yml', data_name='eval_data.hdf5'):
    """
    Load evaluated data from output folder
    
    path : string 
        path to output folder
        
    params_name : string
        name of params data

    data_name : string
        name of evaluatid data

    Returns
    -------
    params : yml file
        parameter
    
    data : hdf5 file
        data
    """

    params_file = path_out + params_name
    data_file   = path_out + data_name

    with open(params_file) as file:
        params      = yaml.load(file, Loader=yaml.FullLoader)
        data        = h5py.File(data_file, 'r')

    return params, data

def get_dispersion(params, data, quantity):
    """
    Perform 1d fft for evaluated data in x, y or z direction (physical space)
    
    params : yml file
        parameter of the simulation

    data : hdf5 file
        evaluated data file 

    quantity : string 
        key value for the data file

    Returns
    -------
    k : ndarray
        wave wector
    
    w : ndarray
        angular frequency

    dispersion : ndarray
    """
         
    xId, yId, zId   = [0,0,0]
    eval_values     = data[quantity][0]

    Nt, Nx, Ny, Nz  = eval_values.shape
    dt              = params['time']['dt']
    domain_type     = params['geometry']['type']
    dx              = (params['geometry']['params_' + domain_type]['e1'] - params['geometry']['params_' + domain_type]['b1']) / Nx
    dy              = (params['geometry']['params_' + domain_type]['e2'] - params['geometry']['params_' + domain_type]['b2']) / Ny
    dz              = (params['geometry']['params_' + domain_type]['e3'] - params['geometry']['params_' + domain_type]['b3']) / Nz
    direction       =  params['initialization']['params_noise']['direction']

    if direction == 'x':
        dispersion      = (2./Nt)*(2./Nx)*np.abs(fftn(eval_values[:, :, yId, zId]))[:Nt//2, :Nx//2] 
        w               = 2*np.pi*fftfreq(Nt, dt)[:Nt//2] 
        k               = 2*np.pi*fftfreq(Nx, dx)[:Nx//2]

    elif direction == 'y': 
        dispersion    = (2./Nt)*(2./Ny)*np.abs(fftn(eval_values[:, xId, :, zId]))[:Nt//2, :Ny//2] 
        w               = 2*np.pi*fftfreq(Nt, dt)[:Nt//2] 
        k               = 2*np.pi*fftfreq(Ny, dy)[:Ny//2]

    elif direction == 'z':  
        dispersion    = (2./Nt)*(2./Nz)*np.abs(fftn(eval_values[:, xId, yId , :]))[:Nt//2, :Nz//2] 
        w               = 2*np.pi*fftfreq(Nt, dt)[:Nt//2] 
        k               = 2*np.pi*fftfreq(Nz, dz)[:Nz//2]

    return k, w, dispersion

def descend_dict(mydict, file, indent):
    '''
    Iterate through dictionary until non-dict values are found and print values to file.
    '''

    for k, v in mydict.items():
        sp = indent
        if isinstance(v, dict):
            file.write(sp + k + ': ' + '\n')
            descend_dict(v, file, sp + '   ')
        else:
            file.write(sp + k + ': ' + str(v) + '\n')

def descend_obj(obj, dsets, sep='\t'):
    """
    Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
    """

    import h5py

    if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
        for key in obj.keys():
            #print(sep,'-',key,':',obj[key])
            descend_obj(obj[key], dsets, sep=sep+'\t')
    elif type(obj)==h5py._hl.dataset.Dataset:
        #for key in obj.attrs.keys():
            #print(sep+'\t','-',key,':',obj.attrs[key])
        # print('obj attributes:')
        # print(obj.name)
        # print(obj.shape)
        # print(obj.size)
        # print(obj.ndim)
        # print(obj[()])
        dsets.append(obj.name)

def get_data(param_file, path_out=None):
    '''Get parameters and save simulation data to dictionary.
    
    Parameters
    ----------
        param_file : str
            Name of parameter file (absolute path).

        path_out :
            Name of .hdf5 output file (absolute path), default=None.
    '''

    import yaml
    import h5py

    with open(param_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    #print("Diagnostics, parameters:")
    #print(params)

    data = dict()

    if path_out==None: 
        return params, data

    names = []
    with h5py.File(path_out, 'r') as obj:
        descend_obj(obj['/'], names) 
        for name in names:
            dset = obj[name]
            data[dset.name] = dset[()]

    return params, data

def plot_data(data, name=None):

    import matplotlib.pyplot as plt

    for k, v in data.items():
        #print(k)
        #print(v)
        if name==None:
            if v.ndim==1:
                plt.figure()
                plt.semilogy(v)
                plt.title(k)
        else:
            if name in k:
                plt.figure()
                plt.semilogy(v)
                plt.title(k)

    plt.show()


    
