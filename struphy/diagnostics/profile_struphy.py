import sys
import pickle
from struphy.diagnostics import Cprofile_analyser

list_of_funcs = ['assemble_', 'update', 'step_']

dicts = []
nproc = []
for path in sys.argv[1:]:

    print('')
    Cprofile_analyser.get_cprofile_data(path)
    
    with open(path + 'profile_dict.sav', 'rb') as f:
        dicts += [pickle.load(f)]

    with open(path + 'meta.txt', 'r') as f:
        lines = f.readlines()

    nproc += [int(lines[-1].split()[-1])]

# loop over keys (should be same in each dict)
for key in dicts[0].keys():

    if any(func in key for func in list_of_funcs) and 'dependencies_' not in key:
        print('')
        for dict, path, n in zip(dicts, sys.argv[1:], nproc):
            print(f'# processes: {n},   ', key.ljust(60), dict[key]['cumtime'])
