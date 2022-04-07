#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from scipy.fft import fftfreq ,fftn
import matplotlib.colors as colors
import yaml
import h5py

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


    
