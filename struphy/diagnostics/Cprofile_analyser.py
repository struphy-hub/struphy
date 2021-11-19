def get_cprofile_data(path):

    import pstats
    from pstats import SortKey
    import time
    import pickle

    p = pstats.Stats(path + 'profile_tmp')
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats('substep', 20)
    p.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)

    stdout = open(path + "profile_out.txt", "w+")
    p = pstats.Stats(path + 'profile_tmp', stream=stdout)
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats('substep', 20)
    p.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)
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



def compare_cprofile_data(list_of_paths):

    import pickle
    import struphy.diagnostics.diagn_tools as tools

    print()
    print('Np'.rjust(8), 'substep_1', 'substep_2', 'substep_3', 'substep_4', 'substep_5', 'substep_6')
    for path in list_of_paths:
        with open(path + 'profile_dict.sav', 'rb') as f:
            data_cprofile = pickle.load(f)
        params, data = tools.get_data(path)

        for k, v in data_cprofile.items():
            if 'substep_1' in k:
                ct1 = float(v['cumtime'])
            elif 'substep_2' in k:
                ct2 = float(v['cumtime'])
            elif 'substep_3' in k:
                ct3 = float(v['cumtime'])
            elif 'substep_4' in k:
                ct4 = float(v['cumtime'])
            elif 'substep_5' in k:
                ct5 = float(v['cumtime'])
            elif 'substep_6' in k:
                ct6 = float(v['cumtime'])

        print('{0:8d} {1:9.3f} {2:9.3f} {3:9.3f} {4:9.3f} {5:9.3f} {6:9.3f}'.format(params['Np'], ct1, ct2, ct3, ct4, ct5, ct6))
        #print(data_cprofile)
        #print()

        

if __name__ == '__main__':
    get_cprofile_data('my_struphy_sims/sim_1/')
    # compare_cprofile_data(['my_struphy_sims/sim_1/', 'my_struphy_sims/sim_2/', 
                        # 'my_struphy_sims/sim_3/', 'my_struphy_sims/sim_4/',])
