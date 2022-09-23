import sys
import pickle
import yaml
from struphy.post_processing import Cprofile_analyser


if sys.argv[1] == 'true':
    list_of_funcs = None
else:
    list_of_funcs = ['assemble_', 'propagator', 'accumulate', '_fill', 'pusher', 'update_ghost_regions', 'schur', 'pcg', 'bicgstab', 'pbicgstab']
    print('\nKeyword search enabled:')
    print(list_of_funcs)

dicts = []
nproc = []
Nel = []
for path in sys.argv[2:]:

    print('')
    Cprofile_analyser.get_cprofile_data(path)
    
    with open(path + 'profile_dict.sav', 'rb') as f:
        dicts += [pickle.load(f)]

    with open(path + 'meta.txt', 'r') as f:
        lines = f.readlines()

    nproc += [int(lines[-1].split()[-1])]
   
    with open(path + 'parameters.yml', 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    Nel += [params['grid']['Nel']]

# loop over keys (should be same in each dict)
count = 0
for key in dicts[0].keys():

    count += 1

    if list_of_funcs == None:

        for dict, path, n, dim in zip(dicts, sys.argv[1:], nproc, Nel):
            print(f'# processes: {n:4d}, count: {count:2d}  ', key.ljust(60), dict[key]['cumtime'])
        
        if count == 60: break

    elif any(func in key for func in list_of_funcs) and 'dependencies_' not in key:
        
        print('')
        
        for dict, path, n, dim in zip(dicts, sys.argv[1:], nproc, Nel):
            print(f'# processes: {n:4d}, Nel: {dim}  ', key.ljust(60), dict[key]['cumtime'])
            
