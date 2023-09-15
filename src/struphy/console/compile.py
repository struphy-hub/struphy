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
    import os
    import pyccel

    libpath = struphy.__path__[0]
    
    if any([s==' ' for s in libpath]):
        raise NameError(f'Stuphy installation path MUST NOT contain blank spaces. Please rename "{libpath}".')

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

        # struphy and psydac (change dir not to be in source path)
        flag_omp_pic = '--openmp'
        flag_omp_mhd = ''
        if no_openmp:
            flag_omp_pic = ''

        # pyccel flags
        flags = ''
        
        _li = pyccel.__version__.split('.')
        _num = int(_li[0])*100 + int(_li[1])*10 + int(_li[2])
        if _num >= 180:
            flags += '--conda-warnings off'
            
        if verbose:
            flags += ' --verbose'

        # install psydac from wheel if not there
        current_ver = '0.1.2'
        psydac_file = 'psydac-' + current_ver + '-py3-none-any.whl'
        
        try:
            import psydac
            import importlib.metadata

            your_ver = importlib.metadata.version("psydac")
            
            if current_ver != your_ver:
                print(f'You have psydac version {your_ver}, but version {current_ver} is available.\n')
                subprocess.run(['pip',
                                'uninstall',
                                '-y',
                                'psydac'])
                print('\nInstalling Psydac ...')
                subprocess.run(['pip',
                                'install',
                                os.path.join(
                                    libpath, psydac_file),
                                ], check=True)
                print('Done.')
                
        except:
            print('\nInstalling Psydac ...')
            subprocess.run(['pip',
                            'install',
                            os.path.join(
                                libpath, psydac_file),
                            ], check=True)
            print('Done.')

        # gvec_to_python (change dir not to be in source path)
        subprocess.run(['compile-gvec-tp'], check=True, cwd=libpath)

        print('\nCompiling Struphy and Psydac kernels ...')
        subprocess.run(['make',
                        '-f',
                        'compile_struphy.mk',
                        'flags=' + flags,
                        'flags_openmp_pic=' + flag_omp_pic,
                        'flags_openmp_mhd=' + flag_omp_mhd,
                        ], check=True, cwd=libpath)
        print('Done.')
