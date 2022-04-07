import h5py
import ctypes

class Data_container:
    '''Save data during simulation.
    
    Parameters
    ----------
        path_out : string
            path to hdf5 data files

        data_dict : dict
            Name-object pairs to save during time stepping, e.g. {'x': x_mat}. x_mat must be np.array with fixed shape.
    
        data_name : string
            name to hdf5 file (otional)

    '''
    
    def __init__(self, path_out, data_dict=None, data_name=None):
    
        if data_name is None:
            self.data_name = 'data.hdf5'
        else:
            if data_name.find(".hdf5") ==-1:
                self.data_name = data_name + '.hdf5'
            else: 
                self.data_name = data_name

        self.file = h5py.File(path_out + self.data_name, 'w') # restart needs to ba added
        self.ids  = dict()
        
        try:
            for key, obj in data_dict.items():

                self.file.create_dataset(key, (1,) + obj.shape, maxshape=(None,) + obj.shape, dtype=float, chunks=True)
                # replace object with its id
                self.ids[key] = id(obj)
                print(key.ljust(20) + 'will be saved to ' + self.data_name)
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
            print(key.ljust(20) + 'will be saved to ' + self.data_name)


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


