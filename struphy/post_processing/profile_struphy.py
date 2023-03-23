import sys
import pickle
import yaml
import numpy as np
from matplotlib import pyplot as plt

from struphy.post_processing.cprofile_analyser import get_cprofile_data, replace_keys


def main():
    """
    TODO
    """
    print(sys.argv)

    # check --all option
    if sys.argv[1] == 'true':
        list_of_funcs = None
    else:
        list_of_funcs = ['assemble_',
                         'propagator',
                         'accumulate',
                         '_fill',
                         'pusher',
                         'update_ghost_regions',
                         'solver',
                         'class ',
                         'stencil',
                         'block',
                         'integrate_in_time']
        print('\nKeyword search enabled with keywords\n')
        print(list_of_funcs)

    # replace propagator keys or not
    do_replace_keys = sys.argv[2] == 'true'

    # plot n_lines most time consuming calls in profiling analysis
    n_lines = int(sys.argv[3])

    # load data
    dicts_pre = []
    nproc = []
    Nel = []
    for path in sys.argv[4:]:

        print('')
        get_cprofile_data(path)

        with open(path + 'profile_dict.sav', 'rb') as f:
            dicts_pre += [pickle.load(f)]

        with open(path + 'meta.txt', 'r') as f:
            lines = f.readlines()

        nproc += [int(lines[-1].split()[-1])]

        with open(path + 'parameters.yml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        Nel += [params['grid']['Nel']]

    # Nicer key names for output:
    dicts = []
    for d in dicts_pre:

        tmp = {}
        for key, val in d.items():
            #tmp[key] = float(val['cumtime'])
            tmp[key] = val

        if do_replace_keys:
            tmp2 = replace_keys(tmp)
        else:
            tmp2 = tmp

        dicts += [tmp2]

    # loop over keys (should be same in each dict)
    d_saved = {}
    print('#processes \tpos  \t\tfunction\t\t\t\t\t\t\t\t\t  ncalls\t totime\t\tpercall\t       cumtime')
    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    for position, key in enumerate(dicts[0].keys()):

        if list_of_funcs == None:

            for dict, path, n, dim in zip(dicts, sys.argv[4:], nproc, Nel):

                string = f'{n:4d}\t\t{position:2d}\t' + str(key.ljust(90))
                for value in dict[key].values():
                    string += str(value).ljust(15)
                    # if len(str(value)) < 7:
                    #     string += '\t\t'
                    # else:
                    #     string += '\t'
                print(string)
            print('')

            if position == 50:

                exit()

        elif any(func in key for func in list_of_funcs) and 'dependencies_' not in key and '_dot' not in key:

            d_saved[key] = {'mpi_size': [], 'Nel': [], 'time': []}

            for dict, path, n, dim in zip(dicts, sys.argv[4:], nproc, Nel):

                string = f'{n:4d}\t\t{position:2d}\t' + str(key.ljust(90))
                for value in dict[key].values():
                    string += str(value).ljust(15)
                    #string += '\t\t'
                print(string)

                d_saved[key]['mpi_size'] += [n]
                d_saved[key]['Nel'] += [dim]
                d_saved[key]['time'] += [dict[key]]
            print('')

            if position >= 200:
                exit()

    # save profiling date in each sim path
    for path in sys.argv[4:]:
        with open(path + 'comparison_dict.sav', 'w+b') as f:
            pickle.dump(d_saved, f)

    # plot results
    fig = plt.figure(figsize=(10, 10))
    for n, (key, val) in enumerate(d_saved.items()):
        if n < n_lines and '__init__' not in key and 'mass' not in key and 'set_backend' not in key:
            #print(key, val)

            # strong scaling plot
            if all([Nel == val['Nel'][0] for Nel in val['Nel']]):
                plt.loglog(val['mpi_size'], val['time'], label=key)
                plt.xlabel('mpi_size')
                plt.ylabel('time [s]')
                plt.title('Strong scaling for Nel=' +
                          str(val['Nel'][0]) + ' cells')
                plt.legend(loc='lower left')
                plt.loglog(val['mpi_size'], val['time'][0]/2 **
                           np.arange(len(val['time'])), 'k--', alpha=0.3)
            # weak scaling plot
            else:
                plt.plot(val['mpi_size'], val['time'], label=key)
                plt.xlabel('mpi_size')
                plt.ylabel('time [s]')
                plt.title('Weak scaling for cells/mpi_size=' +
                          str(np.prod(val['Nel'][0])/val['mpi_size'][0]) + '=const.')
                plt.legend(loc='upper left')
                #plt.loglog(val['mpi_size'], val['time'][0]*np.ones_like(val['time']), 'k--', alpha=0.3)
                plt.xscale('log')

    plt.show()


if __name__ == '__main__':
    main()
