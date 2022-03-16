#!/usr/bin/env python3

'''
Main execution file.
'''

import sys
import sysconfig
import datetime
import struphy.diagnostics.diagn_tools as tools

#print('Number of arguments:', len(sys.argv), 'arguments.')
#print('Argument List:', sys.argv)

code       = sys.argv[1]
file_in    = sys.argv[2]
path_out   = sys.argv[3] 
path_batch = sys.argv[4]
file_meta  = sys.argv[5] 
mode       = sys.argv[6]

with open(file_meta, 'w+') as f:
    f.write('date of simulation: '.ljust(20) + str(datetime.datetime.now()) + '\n')
    f.write('platform: '.ljust(20) + sysconfig.get_platform() + '\n')
    f.write('python version: '.ljust(20) + sysconfig.get_python_version() + '\n')
    f.write('code: '.ljust(20) + code + '\n')
    f.write('parameters from: '.ljust(20) + file_in + '\n')
    f.write('output data to: '.ljust(20) + path_out + '\n')
    f.write('batch script used: '.ljust(20) + path_batch + '\n\n')
    #tools.descend_dict(params, f, '')
    # for k, v in params.items():
    #     #print(type(v), isinstance(v, dict))
    #     f.write(k + ': ' + str(v) + '\n')

# start simulation:   
if code=='lin_mhd':
    from struphy.models.codes import lin_mhd
    lin_mhd.execute(file_in, path_out, mode=='a') 

elif code=='lin_mhd_MF':
    from struphy.models.codes import lin_mhd_MF
    lin_mhd_MF.execute(file_in, path_out, mode=='a') 

elif code=='lin_mhd_psydac':
    from struphy.models.codes import lin_mhd_psydac
    lin_mhd_psydac.execute(file_in, path_out, mode=='a') 

elif code=='cc_lin_mhd_6d':
    from struphy.models.codes import cc_lin_mhd_6d 
    cc_lin_mhd_6d.execute(file_in, path_out, mode=='a')

elif code=='cc_lin_mhd_6d_MF':
    from struphy.models.codes import cc_lin_mhd_6d_MF 
    cc_lin_mhd_6d_MF.execute(file_in, path_out, mode=='a')

elif code=='pc_lin_mhd_6d_MF_full':
    from struphy.models.codes import pc_lin_mhd_6d_MF_full 
    pc_lin_mhd_6d_MF_full.execute(file_in, path_out, mode=='a')

elif code=='pc_lin_mhd_6d_MF_perp':
    from struphy.models.codes import pc_lin_mhd_6d_MF_perp 
    pc_lin_mhd_6d_MF_perp.execute(file_in, path_out, mode=='a')
    
elif code=='kinetic_extended':
    from struphy.models.codes import kinetic_extended
    kinetic_extended.execute(file_in, path_out, mode=='a')

elif code=='maxwell':
    from struphy.models.codes import maxwell 
    maxwell.execute(file_in, path_out, mode=='a')

else:
    raise NotImplementedError('Model not implemented.')


