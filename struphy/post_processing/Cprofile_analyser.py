def get_cprofile_data(path, n_stats=None):
    '''Prepare Cprofile data and save to "profile_dict.sav".
    
    Parameters
    ----------
        path : str
            Path to file "profile_tmp" (usually in output folder).

        n_stats : int
            Number of stats to show and save (cumtime ordered).
    '''

    import pstats
    from pstats import SortKey
    import time
    import pickle

    p = pstats.Stats(path + 'profile_tmp')
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(0)
    #p.strip_dirs().sort_stats(SortKey.TIME).print_stats(n_stats)
    #p.print_title()

    stdout = open(path + "profile_out.txt", "w+")
    p = pstats.Stats(path + 'profile_tmp', stream=stdout)
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(n_stats)
    #p.strip_dirs().sort_stats(SortKey.TIME).print_stats(n_stats)
    stdout.close()

    data_cprofile = dict()
    with open(path + 'profile_out.txt') as f:
        lines = f.readlines()
        #print(len(lines))
        search = False
        for n, line in enumerate(lines):
            if search:
                li = line.split()
                #print(li)
                #print(len(name_li), len(li))
                if len(li)==0: 
                    search = False            
                    continue
                #print(name_li[0], li[0])
                #print(name_li[1], li[1])
                #print(name_li[2], li[2])
                #print(name_li[3], li[3])
                #print(name_li[4], li[4])
                data_cprofile[li[-1]] = {name_li[0]: li[0],
                                         name_li[1]: li[1],
                                         name_li[2]: li[2],
                                         name_li[3]: li[3],
                                         name_li[4]: li[4],}
                #time.sleep(1)
                
            if 'filename:lineno' in line:
                #print(n, repr(line))
                name_li = line.split()
                #print(name_li)
                search = True

    with open(path + 'profile_dict.sav', 'w+b') as f:       
        pickle.dump(data_cprofile, f)



def compare_cprofile_data(path, list_of_funcs=None):
    '''Print Cprofile data from "profile_dict.sav" to screen (see get_cprofile_data).
    
    Parameters
    ----------
        path : str
            Path to file "profile_dict.sav" (usually in output folder).

        list_of_funcs : list
            Strings to watch for in "function name" of Cprofile data, allows to look at data of specific functions. 
            If "None", the 50 functions with the longest cumtime are listed.
    '''

    import pickle

    with open(path + 'profile_dict.sav', 'rb') as f:
        data_cprofile = pickle.load(f)

    if list_of_funcs==None:
        print('-'*76)
        print('function name'.ljust(60), 'cumulative time')
        print('-'*76)
    else:
        print('-'*76)
        print('function name, keywords: {}'.format(list_of_funcs).ljust(60), 'cumulative time')
        print('-'*76)

    counter = 0
    for k, v in data_cprofile.items():

        counter += 1
        if list_of_funcs==None:
            print(k.ljust(60), v['cumtime'])
            if counter>49:
                break
        elif any(func in k for func in list_of_funcs) and 'dependencies_' not in k:
            print(k.ljust(60), v['cumtime'])


