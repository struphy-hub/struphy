def struphy_test(serial=True, mpi=0, codes=False):
    """
    Run Struphy unit and/or code tests.
    
    Parameters
    ----------
    serial : bool
        Whether to run serial units tests.
        
    mpi : int
        If >0, parallel units tests are run with mpi number of processes.
        
    codes : bool
        Whether to run code tests.
    """
    
    import os
    import subprocess
    import struphy

    libpath = struphy.__path__[0]
    
    if serial:
        subprocess.run(['pytest',
                        'tests/tests_serial'], 
                        check=True, cwd=libpath)
        
    if mpi > 0:
        subprocess.run(['mpirun',
                        '-n',
                        str(mpi),
                        'pytest',
                        '--with-mpi',
                        'tests/tests_mpi'],
                        check=True, cwd=libpath)
        
    if codes:
        
        # test Maxwell
        subprocess.run(['struphy', 'run', 'Maxwell',
                        '-i', 'tests/params_maxwell_1.yml',
                        '-o', 'sim_test_1'], check=True)
        subprocess.run(['struphy', 'run', 'Maxwell',
                        '-i', 'tests/params_maxwell_2.yml',
                        '-o', 'sim_test_2'], check=True)
        subprocess.run(['struphy', 'run', 'Maxwell',
                        '-i', 'tests/params_maxwell_2.yml',
                        '-o', 'sim_test_3',
                        '--mpi', '2'], check=True)
        
        subprocess.run(['struphy', 'pproc',
                        '-d', 'sim_test_1'], check=True)
        subprocess.run(['struphy', 'pproc', 
                        '-d', 'sim_test_2'], check=True)
        subprocess.run(['struphy', 'pproc', 
                        '-d', 'sim_test_3'], check=True)
        
        # test LinearMHD
        subprocess.run(['struphy', 'run', 'LinearMHD',
                        '-i', 'tests/params_linearmhd.yml',
                        '-o', 'sim_test_4'], check=True)
        subprocess.run(['struphy', 'run', 'LinearMHD',
                        '-i', 'tests/params_linearmhd.yml',
                        '-o', 'sim_test_5',
                        '--mpi', '2'], check=True)
        
        subprocess.run(['struphy', 'pproc', 
                        '-d', 'sim_test_4', 
                        '-s', '2',
                        '--celldivide', '3'], check=True)
        subprocess.run(['struphy', 'pproc', 
                        '-d', 'sim_test_5',
                        '-s', '3',
                        '--celldivide', '2'], check=True)
        
        # test LinearMHDVlasovCC
        subprocess.run(['struphy', 'run', 'LinearMHDVlasovCC',
                        '-i', 'tests/params_hybridmhdvlasovcc.yml',
                        '-o', 'sim_test_6',
                        '--mpi', '2'], check=True)
        subprocess.run(['struphy', 'run', 'LinearMHDVlasovCC',
                        '-i', 'tests/params_hybridmhdvlasovcc_control.yml',
                        '-o', 'sim_test_7',
                        '--mpi', '2'], check=True)
        
        subprocess.run(['struphy', 'pproc', 
                        '-d', 'sim_test_6'], check=True)
        subprocess.run(['struphy', 'pproc',
                        '-d', 'sim_test_7'], check=True)
       
        # test LinearMHDVlasovPC
        subprocess.run(['struphy', 'run', 'LinearMHDVlasovPC',
                        '-i', 'tests/params_hybridmhdvlasovpc.yml',
                        '-o', 'sim_test_8',
                        '--mpi', '2'], check=True)
        
        # test LinearVlasovMaxwell
        subprocess.run(['struphy', 'run', 'LinearVlasovMaxwell',
                        '-i', 'tests/params_linvlasovmaxwell.yml',
                        '-o', 'sim_test_9',
                        '--mpi', '2'], check=True)
        
        subprocess.run(['struphy', 'pproc', 
                        '-d', 'sim_test_9'], check=True)
        