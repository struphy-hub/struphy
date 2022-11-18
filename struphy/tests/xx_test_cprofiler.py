def test_Cprofiler():
    '''Test code profiler.'''

    import sysconfig
    from struphy.diagnostics import cprofile_analyser 
    
    try:
        path = sysconfig.get_path("platlib") + '/struphy/io/out/sim_2/'
        print()
        cprofile_analyser.get_cprofile_data(path)
        cprofile_analyser.compare_cprofile_data(path)
        cprofile_analyser.compare_cprofile_data(path, ['step_', '_dot'])
    except:
        pass

    try:
        path = sysconfig.get_path("platlib") + '/struphy/io/out/sim_3/'
        print()
        cprofile_analyser.get_cprofile_data(path)
        #cprofile_analyser.compare_cprofile_data(path)
        cprofile_analyser.compare_cprofile_data(path, ['step_', '_dot'])
    except:
        pass
    
    try:
        path = sysconfig.get_path("platlib") + '/struphy/io/out/sim_4/'
        print()
        cprofile_analyser.get_cprofile_data(path)
        #cprofile_analyser.compare_cprofile_data(path)
        cprofile_analyser.compare_cprofile_data(path, ['step_', '_dot'])
    except:
        pass
    
    try:
        path = sysconfig.get_path("platlib") + '/struphy/io/out/sim_5/'
        print()
        cprofile_analyser.get_cprofile_data(path)
        #cprofile_analyser.compare_cprofile_data(path)
        cprofile_analyser.compare_cprofile_data(path, ['step_', '_dot'])
    except:
        pass
    
    try:
        path = sysconfig.get_path("platlib") + '/struphy/io/out/sim_6/'
        print()
        cprofile_analyser.get_cprofile_data(path)
        #cprofile_analyser.compare_cprofile_data(path)
        cprofile_analyser.compare_cprofile_data(path, ['step_', '_dot'])
    except:
        pass
    

if __name__ == '__main__':
    test_Cprofiler()