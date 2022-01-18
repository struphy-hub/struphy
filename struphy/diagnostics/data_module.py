import h5py
import ctypes

class Data_container:
    '''Save data during simulation.
    
    Parameters
    ----------
        data_dict : dict
            Name-object pairs to save during time stepping, e.g. {'x': x_mat}. x_mat must be np.array with fixed shape.
    '''

    def __init__(self, path_out, data_dict=None):

        self.file = h5py.File(path_out + 'data.hdf5', 'a')
        self.ids  = dict()
        
        try:
            for key, obj in data_dict.items():

                self.file.create_dataset(key, (1,) + obj.shape, maxshape=(None,) + obj.shape, dtype=float, chunks=True)
                # replace object with its id
                self.ids[key] = id(obj)
                print(key.ljust(16) + 'will be saved to data.hdf5')
        except:
            pass


    def add_data(self, data_dict):
        '''Add data object to be saved during simulation.
    
        Parameters
        ----------
            data_dict : dict
                Name-object pairs to save during time stepping, e.g. {'x': x_mat}. x_mat must be np.array with fixed shape.
        '''

        for key, obj in data_dict.items():

            self.file.create_dataset(key, (1,) + obj.shape, maxshape=(None,) + obj.shape, dtype=float, chunks=True)
            # replace object with its id
            self.ids[key] = id(obj)
            # save initial value
            self.file[key][-1] = obj
            print(key.ljust(16) + 'will be saved to data.hdf5')


    def save_data(self, keys=None):
        '''Save data objects to hdf5 file.
        
        Parameters
        ----------
            keys : list
                Keys to the data objects specified when using "add_data". Default saves all specified data objects.
        '''

        if keys==None:

            for key, obj in self.ids.items():

                #print(ctypes.cast(obj, ctypes.py_object).value) 
                self.file[key].resize(self.file[key].shape[0] + 1, axis = 0)
                self.file[key][-1] = ctypes.cast(obj, ctypes.py_object).value
                #print(key + ' saved to data.hdf5')

        else:

            for key in keys:

                self.file[key].resize(self.file[key].shape[0] + 1, axis = 0)
                self.file[key][-1] = ctypes.cast(self.ids[key], ctypes.py_object).value
                #print(key + ' saved to data.hdf5')


    def print_data_to_screen(self, keys, time_points=None):
        '''Print saved dataset(s) to screen.
        
        Parameters
        ----------
            keys : list
                Keys to the datasets specified when using "add_data".  
            time_points : list
                Time indices (integer) for which to print data, default is all times.  
        '''

        if time_points==None:
            for key in keys:
                print(key + ' at all times:')
                print(self.file[key][:])

        else:
            for time in time_points:
                for key in keys:
                    print(key + ' at step ' + str(time) + ':', self.file[key][time])


