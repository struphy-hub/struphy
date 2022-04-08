import h5py
import ctypes


class Data_container:
    '''Save data during simulation (each process locally).

    Parameters
    ----------
        path_out : string
            path to hdf5 data files

        data_dict : dict
            Name-object pairs to save during time stepping, e.g. {'x': x_mat}. x_mat must be np.array with fixed shape.

        data_name : string
            name to hdf5 file (otional)

    '''

    def __init__(self, path_out, data_dict=None, data_name=None, comm=None):

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

        # Write mode, deletes existing file
        self._f = h5py.File(path_out + self._data_name, 'w')
        self._ids = dict()

        if data_dict is not None:
            for key, obj in data_dict.items():

                self._f.create_dataset(
                    key, (1,) + obj.shape, maxshape=(None,) + obj.shape, dtype=float, chunks=True)
                # replace object with its id
                self._ids[key] = id(obj)
                # Add object id as attribute
                #self._f[key].attrs['id'] = id(obj)
                print(f'Rank: {self._rank} | ' + key.ljust(20) +
                      'will be saved to ' + self._data_name)

    @property
    def file(self):
        '''The h5py file.'''
        return self._f

    def add_data(self, data_dict):
        '''Add data object to be saved during simulation.

        Parameters
        ----------
            data_dict : dict
                Name-object pairs to save during time stepping, e.g. {'x': x_mat}. x_mat must be np.array with fixed shape.
        '''

        for key, obj in data_dict.items():

            self._f.create_dataset(
                key, (1,) + obj.shape, maxshape=(None,) + obj.shape, dtype=float, chunks=True)
            # replace object with its id
            self._ids[key] = id(obj)
            # Add object id as attribute
            #self._f[key].attrs['id'] = id(obj)
            # save initial value
            self._f[key][-1] = obj
            print(f'Rank: {self._rank} | ' + key.ljust(20) +
                  'will be saved to ' + self._data_name)

    def save_data(self, keys=None):
        '''Save data objects to hdf5 file.

        Parameters
        ----------
            keys : list
                Keys to the data objects specified when using "add_data". Default saves all specified data objects.
        '''

        if keys == None:

            for key, obj in self._ids.items():

                #print(ctypes.cast(obj, ctypes.py_object).value) 
                self._f[key].resize(self._f[key].shape[0] + 1, axis = 0)
                self._f[key][-1] = ctypes.cast(obj, ctypes.py_object).value
                #print(key + ' saved to data.hdf5')


        else:

            for key in keys:

                self._f[key].resize(self._f[key].shape[0] + 1, axis=0)
                #self._f[key][-1] = ctypes.cast(self.ids[key], ctypes.py_object).value
                self._f[key][-1] = ctypes.cast(self._f[key].attrs['id'],
                                               ctypes.py_object).value
                #print(key + ' saved to data.hdf5')


class Data_container_psydac:
    '''Save data during simulation (each process locally).

    Parameters
    ----------
        path_out : string
            path to hdf5 data files

        data_dict : dict
            Name-object pairs to save during time stepping, e.g. {'x': x_mat}. x_mat must be np.array with fixed shape.

        data_name : string
            name to hdf5 file (otional)

    '''

    def __init__(self, path_out, data_dict=None, data_name=None, comm=None):

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

        # Write mode, deletes existing file
        self._f = h5py.File(path_out + self._data_name, 'w')
        #self._ids = dict()

        if data_dict is not None:
            for key, obj in data_dict.items():

                self._f.create_dataset(
                    key, (1,) + obj.shape, maxshape=(None,) + obj.shape, dtype=float, chunks=True)
                # replace object with its id
                #self._ids[key] = id(obj)
                # Add object id as attribute
                self._f[key].attrs['id'] = id(obj)
                print(f'Rank: {self._rank} | ' + key.ljust(20) +
                      'will be saved to ' + self._data_name)

    @property
    def f(self):
        '''The h5py file.'''
        return self._f

    def add_data(self, data_dict):
        '''Add data object to be saved during simulation.

        Parameters
        ----------
            data_dict : dict
                Name-object pairs to save during time stepping, e.g. {'x': x_mat}. x_mat must be np.array with fixed shape.
        '''

        for key, obj in data_dict.items():

            self._f.create_dataset(
                key, (1,) + obj.shape, maxshape=(None,) + obj.shape, dtype=float, chunks=True)
            # replace object with its id
            #self._ids[key] = id(obj)
            # Add object id as attribute
            self._f[key].attrs['id'] = id(obj)
            # save initial value
            self._f[key][-1] = obj
            print(f'Rank: {self._rank} | ' + key.ljust(20) +
                  'will be saved to ' + self._data_name)

    def save_data(self, keys=None):
        '''Save data objects to hdf5 file.

        Parameters
        ----------
            keys : list
                Keys to the data objects specified when using "add_data". Default saves all specified data objects.
        '''

        if keys == None:

            for key, dset in self._f.items():

                #print(ctypes.cast(obj, ctypes.py_object).value)
                dset.resize(dset.shape[0] + 1, axis=0)
                dset[-1] = ctypes.cast(dset.attrs['id'],
                                       ctypes.py_object).value
                #print(key + ' saved to data.hdf5')

        else:

            for key in keys:

                self._f[key].resize(self._f[key].shape[0] + 1, axis=0)
                #self._f[key][-1] = ctypes.cast(self.ids[key], ctypes.py_object).value
                self._f[key][-1] = ctypes.cast(self._f[key].attrs['id'],
                                               ctypes.py_object).value
                #print(key + ' saved to data.hdf5')

    def info(self):
        '''Print info of data sets to screen.  
        '''
        for key, dset in self._f.items():
            print(f'\nData set name: {key}')
            print('Attributes:')
            for attr, val in dset.attrs.items():
                print(attr, val)

