def test_Cprofiler():
    '''Test code profiler.'''

    import sysconfig
    from struphy.diagnostics import Cprofile_analyser 

    path = sysconfig.get_path("platlib") + '/struphy/io/out/sim_2/'
    print()
    Cprofile_analyser.get_cprofile_data(path)
    Cprofile_analyser.compare_cprofile_data(path)
    Cprofile_analyser.compare_cprofile_data(path, ['step_', '_dot'])

    path = sysconfig.get_path("platlib") + '/struphy/io/out/sim_3/'
    print()
    Cprofile_analyser.get_cprofile_data(path)
    #Cprofile_analyser.compare_cprofile_data(path)
    Cprofile_analyser.compare_cprofile_data(path, ['step_', '_dot'])

    path = sysconfig.get_path("platlib") + '/struphy/io/out/sim_4/'
    print()
    Cprofile_analyser.get_cprofile_data(path)
    #Cprofile_analyser.compare_cprofile_data(path)
    Cprofile_analyser.compare_cprofile_data(path, ['step_', '_dot'])

    path = sysconfig.get_path("platlib") + '/struphy/io/out/sim_5/'
    print()
    Cprofile_analyser.get_cprofile_data(path)
    #Cprofile_analyser.compare_cprofile_data(path)
    Cprofile_analyser.compare_cprofile_data(path, ['step_', '_dot'])

    path = sysconfig.get_path("platlib") + '/struphy/io/out/sim_6/'
    print()
    Cprofile_analyser.get_cprofile_data(path)
    #Cprofile_analyser.compare_cprofile_data(path)
    Cprofile_analyser.compare_cprofile_data(path, ['step_', '_dot'])


if __name__ == '__main__':
    test_Cprofiler()