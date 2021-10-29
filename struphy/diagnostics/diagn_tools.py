#!/usr/bin/env python3

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



def get_data(path_in, path_out=None):

    import yaml
    import h5py

    with open(path_in + 'parameters.yml') as f:
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


    
