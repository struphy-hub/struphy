import sys, os, shutil, h5py
import pickle
from struphy.post_processing.post_processing_tools import create_femfields, eval_femfields, post_process_markers

ppcell = int(sys.argv[1])

if ppcell == 1:
    ppcell = None

for path in sys.argv[2:]:

    print('')
    
    # check for fields and kinetic data in hdf5 file
    file = h5py.File(path + '/data_proc0.hdf5', 'r')
    
    if 'fields' in file.keys():
        exist_fields = True
    else:
        exist_fields = False
        
    if 'kinetic' in file.keys():
        exist_kinetic = True
    else:
        exist_kinetic = False
        
    file.close()
    
    
    if exist_fields:
    
        fields, space_ids, code = create_femfields(path)
        point_data_logic, point_data_phys, grids, grids_mapped, masks = eval_femfields(path, fields, space_ids, npts_per_cell=1)

        # directory for evaluated field data 
        try:
            os.mkdir(path + '/eval_fields/')
        except:
            shutil.rmtree(path + '/eval_fields/')
            os.mkdir(path + '/eval_fields/')

        # save data dicts for each field
        for name, val in point_data_logic.items():

            with open(path + '/eval_fields/' + name + '_log.bin', 'wb') as handle:
                pickle.dump(val, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

            with open(path + '/eval_fields/' + name + '_phy.bin', 'wb') as handle:
                pickle.dump(point_data_phys[name], handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        # save grids
        with open(path + '/eval_fields/grids_log.bin', 'wb') as handle:
                pickle.dump(grids, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        with open(path + '/eval_fields/grids_phy.bin', 'wb') as handle:
                pickle.dump(grids_mapped, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        with open(path + '/eval_fields/masks.bin', 'wb') as handle:
                pickle.dump(masks, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
    
    if exist_kinetic:
        post_process_markers(path)