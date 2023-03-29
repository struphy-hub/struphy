def struphy_compile(no_openmp=False, delete=False, verbose=False):
    """
    Compile Struphy kernels.
    
    Parameters
    ----------
    no_openmp : bool, optional
        Whether to compile kernels with (no_openmp=False) or without (no_openmp=True).
        
    delete : bool, optional
        If True, deletes generated Fortran and .so files.
        
    verbose : bool, optional
        Call pyccel in verbose mode.
    """
    
    import subprocess
    import struphy

    libpath = struphy.__path__[0]
        
    if delete:
        
        # (change dir not to be in source path)
        print('\nDeleting .f90 and .so files ...')
        subprocess.run(['make', 
                        'clean', 
                        '-f',
                        'compile_struphy.mk',
                        ], check=True, cwd=libpath)
        print('Done.')
        
    else:
        
        # gvec_to_python (change dir not to be in source path)
        print('\nCompiling gvec_to_python kernels ...')
        subprocess.run(['compile_gvec_to_python'], check=True, cwd=libpath)
        print('Done.')
        
        # struphy and psydac (change dir not to be in source path)
        flag_omp_pic = '--openmp'
        flag_omp_mhd = ''
        if no_openmp:
            flag_omp_pic = ''
            
        flag_verb = ''
        if verbose:
            flag_verb = '--verbose'
            
        print('\nCompiling Struphy and Psydac kernels ...')
        subprocess.run(['make', 
                        '-f', 
                        'compile_struphy.mk',
                        'flags=' + flag_verb,
                        'flags_openmp_pic=' + flag_omp_pic,
                        'flags_openmp_mhd=' + flag_omp_mhd,
                        ], check=True, cwd=libpath)
        print('Done.')  