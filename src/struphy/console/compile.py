def struphy_compile(language, compiler, omp_pic, omp_feec, delete, verbose):
    """
    Compile Struphy kernels.

    Parameters
    ----------
    language : str
        Either "c" (default) or "fortran".
    
    compiler : str
        Either "GNU" (default), "intel", "PGI", "nvidia" or the path to a JSON compiler file.
        Only "GNU" is regularly tested at the moment.
    
    omp_pic : bool
        Whether to compile PIC kernels with OpenMP (default=False).

    omp_feec : bool
        WHether to compile FEEC kernels with OpenMP (default=False).

    delete : bool
        If True, deletes generated Fortran/C files and .so files (default=False).

    verbose : bool
        Call pyccel in verbose mode (default=False).
    """

    import subprocess
    import struphy
    import os
    import pyccel

    libpath = struphy.__path__[0]

    if any([s == ' ' for s in libpath]):
        raise NameError(
            f'Stuphy installation path MUST NOT contain blank spaces. Please rename "{libpath}".')

    if delete:

        # (change dir not to be in source path)
        print('\nDeleting .f90/.c and .so files ...')
        subprocess.run(['make',
                        'clean',
                        '-f',
                        'compile_struphy.mk',
                        ], check=True, cwd=libpath)
        print('Done.')

    else:

        # struphy and psydac (change dir not to be in source path)
        flag_omp_pic = ''
        flag_omp_feec = ''
        if omp_pic:
            flag_omp_pic = '--openmp'
        if omp_feec:
            flag_omp_feec = '--openmp'

        # pyccel flags
        flags = '--language=' +  language
        flags += ' --compiler=' + compiler

        _li = pyccel.__version__.split('.')
        _num = int(_li[0])*100 + int(_li[1])*10 + int(_li[2])
        if _num >= 180:
            flags += ' --conda-warnings=off'

        if verbose:
            flags += ' --verbose'

        # install psydac from wheel if not there
        current_ver = '0.1.3'
        psydac_file = 'psydac-' + current_ver + '-py3-none-any.whl'

        try:
            import psydac
            import importlib.metadata

            your_ver = importlib.metadata.version("psydac")

            if current_ver != your_ver:
                print(
                    f'You have psydac version {your_ver}, but version {current_ver} is available.\n')
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
        subprocess.run(['compile-gvec-tp',
                        '--language=' +  language,
                        '--compiler=' + compiler], check=True, cwd=libpath)

        print('\nCompiling Struphy and Psydac kernels ...')
        subprocess.run(['make',
                        '-f',
                        'compile_struphy.mk',
                        'flags=' + flags,
                        'flags_openmp_pic=' + flag_omp_pic,
                        'flags_openmp_mhd=' + flag_omp_feec,
                        ], check=True, cwd=libpath)
        print('Done.')
