#!/usr/bin/env python3

'''
Main execution file.
'''

import sys
import sysconfig
import datetime
import yaml
import struphy.diagnostics.diagn_tools as tools

#print('Number of arguments:', len(sys.argv), 'arguments.')
#print('Argument List:', sys.argv)

code       = sys.argv[1]
file_in    = sys.argv[2]
path_out   = sys.argv[3] 
path_batch = sys.argv[4]
file_meta  = sys.argv[5] 
mode  = sys.argv[6] 

with open(file_in) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

print('\nParameters:')
print('Nel          :', params['grid']['Nel'])
print('geometry     :', params['geometry']['type'])
print('params map   :', params['geometry']['params_' + params['geometry']['type']])
print('mhd equil.   :', params['mhd_equilibrium']['type'])
print('dt           :', params['time']['dt'])
print('nuh          :', params['hot_equilibrium']['nuh'])
print('Np           :', params['markers']['Np'])
print('control var. :', params['markers']['control'])

with open(file_meta, 'w+') as f:
    f.write('date of simulation: '.ljust(20) + str(datetime.datetime.now()) + '\n')
    f.write('platform: '.ljust(20) + sysconfig.get_platform() + '\n')
    f.write('python version: '.ljust(20) + sysconfig.get_python_version() + '\n')
    f.write('code: '.ljust(20) + code + '\n')
    f.write('parameters from: '.ljust(20) + file_in + '\n')
    f.write('output data to: '.ljust(20) + path_out + '\n')
    f.write('batch script used: '.ljust(20) + path_batch + '\n\n')
    tools.descend_dict(params, f, '')
    # for k, v in params.items():
    #     #print(type(v), isinstance(v, dict))
    #     f.write(k + ': ' + str(v) + '\n')

# start simulation:    
if code=='cc_lin_mhd_6d':
    from struphy.models.codes import cc_lin_mhd_6d 
    cc_lin_mhd_6d.execute(file_in, path_out, mode=='True')


