import sys, os, shutil, h5py
import pickle
import numpy as np

import struphy.post_processing.post_processing_tools as pproc

cell_divide = int(sys.argv[1])

assert cell_divide > 0

for path in sys.argv[2:]:

    print('')
    
    # check for fields and kinetic data in hdf5 file that need post processing
    file = h5py.File(path + '/data_proc0.hdf5', 'r')
    
    if 'feec' in file.keys():
        exist_fields = True
    else:
        exist_fields = False
        
    kinetic_species = []
    if 'kinetic' in file.keys():
        exist_kinetic = [[], []]
        
        for name in file['kinetic'].keys():
            kinetic_species += [name]
            
            # check for saved markers
            if 'markers' in file['kinetic'][name]:
                exist_kinetic[0] += [True]
            else:
                exist_kinetic[0] += [False]
                
            # check for saved distribution function
            if 'f' in file['kinetic'][name]:
                exist_kinetic[1] += [True]
            else:
                exist_kinetic[1] += [False]
    else:
        exist_kinetic = False
            
    file.close()
    
    if exist_fields:
    
        fields, space_ids, code = pproc.create_femfields(path)
        point_data_logic, point_data_phys, grids, grids_mapped = pproc.eval_femfields(path, fields, space_ids, cell_divide=cell_divide)

        # directory for evaluated field data 
        try:
            os.mkdir(path + 'eval_fields/')
        except:
            shutil.rmtree(path + 'eval_fields/')
            os.mkdir(path + 'eval_fields/')

        # save data dicts for each field
        for name, val in point_data_logic.items():

            with open(path + 'eval_fields/' + name + '_log.bin', 'wb') as handle:
                pickle.dump(val, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

            with open(path + 'eval_fields/' + name + '_phy.bin', 'wb') as handle:
                pickle.dump(point_data_phys[name], handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        # save grids
        with open(path + 'eval_fields/grids_log.bin', 'wb') as handle:
                pickle.dump(grids, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        with open(path + 'eval_fields/grids_phy.bin', 'wb') as handle:
                pickle.dump(grids_mapped, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
    
    if np.any(exist_kinetic):
        
        # directory for evaluated kinetic data
        try:
            os.mkdir(path + 'kinetic_data/')
        except:
            shutil.rmtree(path + 'kinetic_data/')
            os.mkdir(path + 'kinetic_data/')
            
    # kinetic post processing for each species
    for n, species in enumerate(kinetic_species):
        
        try:
            os.mkdir(path + 'kinetic_data/' + species + '/')
        except:
            shutil.rmtree(path + 'kinetic_data/' + species + '/')
            os.mkdir(path + 'kinetic_data/' + species + '/')
        
        # markers
        if exist_kinetic[0][n]:
            pproc.post_process_markers(path, species)
        
        # distribution function
        if exist_kinetic[1][n]:
            pproc.post_process_f(path, species) 
