def struphy_compile(language, compiler, omp_pic, omp_feec, delete, status, verbose, dependencies, yes):
    """
    Compile Struphy kernels. All files that contain "kernels" are detected automatically and saved to state.yml.

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

    status : bool
        If true, prints the current Struphy compilation status on screen.

    verbose : bool
        Call pyccel in verbose mode (default=False).

    dependencies : bool
        Whether to print Struphy kernels (to be compiled) and their dependencies on screen.
        
    yes : bool
        Whether to say yes to prompt when changing the language.
    """

    import subprocess
    import struphy
    import os
    import pyccel
    import yaml
    import sysconfig
    import struphy.dependencies as depmod

    libpath = struphy.__path__[0]

    so_suffix = sysconfig.get_config_var('EXT_SUFFIX')

    if any([s == ' ' for s in libpath]):
        raise NameError(
            f'Stuphy installation path MUST NOT contain blank spaces. Please rename "{libpath}".')

    # struphy state
    with open(os.path.join(libpath, 'state.yml')) as f:
        state = yaml.load(f, Loader=yaml.FullLoader)

    # collect kernels
    if 'kernels' not in state:
        state['kernels'] = []
        for subdir, dirs, files in os.walk(libpath):
            for file in files:
                if 'kernels' in file and '.py' in file and '_tmp.py' not in file and 'test' not in file and 'legacy' not in subdir and '__pycache__' not in subdir:
                    state['kernels'] += [os.path.join(subdir, file)]

        # set initial compiler infos to None
        state['last_used_language'] = None
        state['last_used_compiler'] = None
        state['last_used_omp_pic'] = None
        state['last_used_omp_feec'] = None

        with open(os.path.join(libpath, 'state.yml'), 'w') as f:
            yaml.dump(state, f)

    # source files
    sources = ' '.join(state['kernels'])

    # actions
    if delete:

        # (change dir not to be in source path)
        print('\nDeleting .f90/.c and .so files ...')
        subprocess.run(['make',
                        'clean',
                        '-f',
                        'compile_struphy.mk',
                        'sources=' + sources,
                        ], check=True, cwd=libpath)
        print('Done.')

        subprocess.run(['struphy',
                       'compile',
                        '--status',
                        ], check=True, cwd=libpath)

    elif status:

        # update status
        count_c = 0
        count_f90 = 0
        list_not_compiled = [s for s in state['kernels']]
        for subdir, dirs, files in os.walk(libpath):
            # print(f'{subdir = }')
            if subdir[-10:] == '__pyccel__' and '__epyccel__' not in subdir:
                dir_stem = '/'.join(subdir.split('/')[:-1])
                # print(f'{dir_stem = }')
                for file in files:
                    if file[-2:] == '.c' and 'wrapper' not in file and 'bind_c_' not in file:
                        stem = file[:-2]
                        is_c = True
                    elif file[-4:] == '.f90' and 'wrapper' not in file and 'bind_c_' not in file:
                        stem = file[:-4]
                        is_c = False
                    else:
                        continue

                    py_file = stem + '.py'
                    matches = [ker for ker in state['kernels']
                               if py_file in ker and dir_stem in ker]
                    # print(f'{matches = }')
                    for match in matches:
                        py_ker = match.split('/')[-1]
                        if py_ker == py_file:
                            matching = match
                    matching_so = matching.replace('.py', so_suffix)
                    # print(f'{matching_so = }')
                    if os.path.isfile(matching_so):
                        if is_c and state['last_used_language'] == 'c':
                            count_c += 1
                        elif not is_c and state['last_used_language'] == 'fortran':
                            count_f90 += 1
                        if matching in list_not_compiled:
                            list_not_compiled.remove(matching)

        n_kernels = len(state['kernels'])
        print('')
        print(f'{count_c} of {n_kernels} Struphy kernels are compiled with language C.')
        print(
            f'{count_f90} of {n_kernels} Struphy kernels are compiled with language Fortran.')
        print(f'{n_kernels - count_c - count_f90} of {n_kernels} Struphy kernels are not compiled (pure Python).')
        print(
            f'\ncompiler={state["last_used_compiler"]}\nflags_omp_pic={state["last_used_omp_pic"]}\nflags_omp_feec={state["last_used_omp_feec"]}')
        if len(list_not_compiled) > 0:
            print('\nPure Python kernels (not compiled) are:')
            for ker in list_not_compiled:
                print(ker)

        state['kernels_n'] = n_kernels
        state['compiled_in_c'] = count_c
        state['compiled_in_fortran'] = count_f90
        state['compiled_not_n'] = n_kernels - count_c - count_f90
        state['compiled_not'] = list_not_compiled

        with open(os.path.join(libpath, 'state.yml'), 'w') as f:
            yaml.dump(state, f)

    elif dependencies:
        print('\nAuto-detect dependencies ...')
        for ker in state['kernels']:
            deps = depmod.get_dependencies(ker.replace('.py', so_suffix))
            deps_li = deps.split(' ')
            print('-'*28)
            print(f'{ker = }')
            for dep in deps_li:
                print(f'{dep = }')

    else:

        # struphy and psydac (change dir not to be in source path)
        flag_omp_pic = ''
        flag_omp_feec = ''
        if omp_pic:
            flag_omp_pic = '--openmp'
        if omp_feec:
            flag_omp_feec = '--openmp'

        # pyccel flags
        flags = '--language=' + language
        flags += ' --compiler=' + compiler

        # state
        if state['last_used_language'] not in (language, None):
            if yes:
                yesno = 'Y'
            else:
                yesno = input(
                    f'Kernels compiled in language {state["last_used_language"]} exist, will be deleted, continue (Y/n)?')
                
            if yesno in ('', 'Y', 'y', 'yes'):
                subprocess.run(['struphy',
                                'compile',
                                '--delete',], check=True, cwd=libpath)
            else:
                return

        state['last_used_language'] = language
        state['last_used_compiler'] = compiler
        state['last_used_omp_pic'] = flag_omp_pic
        state['last_used_omp_feec'] = flag_omp_feec

        with open(os.path.join(libpath, 'state.yml'), 'w') as f:
            yaml.dump(state, f)

        _li = pyccel.__version__.split('.')
        _num = int(_li[0])*100 + int(_li[1])*10 + int(_li[2])
        if _num >= 180:
            flags += ' --conda-warnings=off'

        if verbose:
            flags += ' --verbose'

        # install psydac from wheel if not there
        current_ver = '0.1.7'
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

        # compilation
        subprocess.run(['compile-gvec-tp',
                        '--language=' + language,
                        '--compiler=' + compiler], check=True, cwd=libpath)

        print('\nCompiling Struphy and Psydac kernels ...')
        subprocess.run(['make',
                        '-f',
                        'compile_struphy.mk',
                        'sources=' + sources,
                        'flags=' + flags,
                        'flags_openmp_pic=' + flag_omp_pic,
                        'flags_openmp_mhd=' + flag_omp_feec,
                        ], check=True, cwd=libpath)
        print('Done.')

        subprocess.run(['struphy',
                       'compile',
                        '--status',
                        ], check=True, cwd=libpath)
