import h5py
import ctypes
import os

import numpy as np


class DataContainer:
    """
    Save data during simulation (each process locally).

    Parameters
    ----------
        path_out : str
            Path to hdf5 data files.

        data_dict : dict
            Name-object pairs to save during time stepping, e.g. {key : val}. key must be a string and val must be a np.array of fixed shape. Scalar values (floats) must therefore be passed as 1d arrays of size 1.

        data_name : str
            Name to hdf5 file (otional).

        comm : MPI communicator
    """

    def __init__(self, path_out, data_dict=None, data_name=None, comm=None):

        # set name of hdf5 file
        if comm is None:
            self._rank = None
            _affix = ''
        else:
            self._rank = comm.Get_rank()
            _affix = '_proc' + str(self._rank)

        if data_name is None:
            self._data_name = 'data' + _affix + '.hdf5'
        else:
            if data_name.find(".hdf5") == -1:
                self._data_name = data_name + '.hdf5'
            else:
                self._data_name = data_name

        # write mode, deletes existing file
        self._file = h5py.File(path_out + self._data_name, 'w')
        self._obj_ids = dict()
        
        # list of dataset keys
        self._dset_keys = []

        # create data sets corresponding to given data_dict and save initial values
        if data_dict is not None:
            for key, val in data_dict.items():
                
                assert isinstance(val, np.ndarray)

                # floats saved as 1d arrays of size 1
                if val.size == 1:
                    self._file.create_dataset(
                        key, (1,), maxshape=(None,), dtype=float, chunks=True)
                    self._file[key][0] = val[0]
                else:
                    self._file.create_dataset(
                        key, (1,) + val.shape, maxshape=(None,) + val.shape, dtype=float, chunks=True)
                    self._file[key][0] = val
                
                # replace object with its id
                self._obj_ids[key] = id(val)
                
                # add key to list of dataset keys
                self._dset_keys += [key]

    @property
    def file(self):
        """ The hdf5 file.
        """
        return self._file
    
    @property
    def obj_ids(self):
        """ The IDs of the objects to be saved.
        """
        return self._obj_ids
    
    @property
    def dset_keys(self):
        """ List of dataset keys.
        """
        return self._dset_keys

    
    def delete(self):
        """ Deletes hdf5 file.
        """
        
        os.remove(self._data_name)
    
    
    def add_data(self, data_dict):
        """
        Add data object to be saved during simulation.

        Parameters
        ----------
            data_dict : dict
            Name-object pairs to save during time stepping, e.g. {key : val}. key must be a string and val must be a np.array of fixed shape. Scalar values (floats) must therefore be passed as 1d arrays of size 1.
        """
        
        for key, val in data_dict.items():
                
            assert isinstance(val, np.ndarray)

            # floats saved as 1d arrays of size 1
            if val.size == 1:
                self._file.create_dataset(
                    key, (1,), maxshape=(None,), dtype=float, chunks=True)
                self._file[key][0] = val[0]
            else:
                self._file.create_dataset(
                    key, (1,) + val.shape, maxshape=(None,) + val.shape, dtype=float, chunks=True)
                self._file[key][0] = val

            # replace object with its id
            self._obj_ids[key] = id(val)
            
            # add key to list of dataset keys
            self._dset_keys += [key]

    
    def save_data(self, keys=None):
        """
        Save data objects to hdf5 file.

        Parameters
        ----------
            keys : list
                Keys to the data objects specified when using "add_data". Default saves all specified data objects.
        """

        if keys is None:
            
            for key in self._dset_keys:
                
                self._file[key].resize(self._file[key].shape[0] + 1, axis=0)
                self._file[key][-1] = ctypes.cast(self._obj_ids[key], ctypes.py_object).value

        else:

            for key in keys:

                self._file[key].resize(self._file[key].shape[0] + 1, axis=0)
                self._file[key][-1] = ctypes.cast(self._obj_ids[key], ctypes.py_object).value

    def info(self):
        """ Print info of data sets to screen.  
        """
        
        for key in self._dset_keys:
            print(f'\nData set name: {key}')
            print('Shape:', self._file[key].shape)
            print('Attributes:')
            for attr, val in self._file[key].attrs.items():
                print(attr, val)
        
