import os
import subprocess
import yaml
import importlib
import struphy
import pyccel


def struphy_test(group, mpi=2, fast=False, verbose=False, monitor=False, n=None, Tend=None):
    """
    Run Struphy unit and/or code tests.

    Parameters
    ----------
    mpi : int
        Number of MPI processes used in tests (must be >1, default=2).

    fast : bool
        Whether to test models just in slab geometry.

    verbose : bool
        Whether to print timings to screen.

    monitor : bool
        Whether to use pytest-monitor in tests.

    n : int
        Number of specific tutorial simulation to run. If None, all tutorial simulations are run.
        
    Tend : float
        If group is a model name, simulation end time in units of the model (default=0.015 with dt=0.005). Data is only saved at Tend if set.
    """

    assert mpi >= 2, 'Tests require at least 2 MPI processes.'

    libpath = struphy.__path__[0]

    pymon_abs = os.path.join(libpath, '_pymon/')
    if not os.path.exists(pymon_abs):
        os.mkdir(pymon_abs)

    with open(os.path.join(libpath, 'state.yml')) as f:
        state = yaml.load(f, Loader=yaml.FullLoader)

    if 'unit' in group:
        pymon_file = os.path.join(pymon_abs, '.pymon_unit')
        if os.path.isfile(pymon_file):
            os.remove(pymon_file)

        subprocess.run(['mpirun',
                        '-n',
                        str(mpi),
                        'pytest',
                        '--no-monitor'*(not monitor),
                        '-k',
                        'not _models and not _propagators and not _tutorial and not pproc',
                        '--with-mpi',
                        '--restrict-scope-to',
                        'module',
                        '--db',
                        '_pymon/.pymon_unit',
                        '--description',
                        f'language={state["last_used_language"]}, compiler={state["last_used_compiler"]}, omp_pic={state["last_used_omp_pic"]}, omp_feec={state["last_used_omp_feec"]}',
                        '--tag',
                        f'struphy={importlib.metadata.version("struphy")}',
                        '--tag',
                        f'pyccel={pyccel.__version__}',],
                       check=True, cwd=libpath)

    elif 'models' in group:
        pymon_file = os.path.join(pymon_abs, '.pymon_models')
        if os.path.isfile(pymon_file):
            os.remove(pymon_file)

        if not fast:
            subprocess.run(['mpirun',
                            '-n',
                            str(mpi),
                            'pytest',
                            '--no-monitor'*(not monitor),
                            '-k',
                            '_models',
                            '-s',
                            '--with-mpi',
                            '--restrict-scope-to',
                            'module',
                            '--db',
                            '_pymon/.pymon_models',
                            '--description',
                            f'language={state["last_used_language"]}, compiler={state["last_used_compiler"]}, omp_pic={state["last_used_omp_pic"]}, omp_feec={state["last_used_omp_feec"]}',
                            '--tag',
                            f'struphy={importlib.metadata.version("struphy")}',
                            '--tag',
                            f'pyccel={pyccel.__version__}',
                            '--tag',
                            f'model_opt_fast={fast}'],
                           check=True, cwd=libpath)
        else:
            subprocess.run(['mpirun',
                            '-n',
                            str(mpi),
                            'pytest',
                            '--no-monitor'*(not monitor),
                            '-k',
                            '_models',
                            '-s',
                            '--with-mpi',
                            '--restrict-scope-to',
                            'module',
                            '--db',
                            '_pymon/.pymon_models',
                            '--description',
                            f'language={state["last_used_language"]}, compiler={state["last_used_compiler"]}, omp_pic={state["last_used_omp_pic"]}, omp_feec={state["last_used_omp_feec"]}',
                            '--tag',
                            f'struphy={importlib.metadata.version("struphy")}',
                            '--tag',
                            f'pyccel={pyccel.__version__}',
                            '--tag',
                            f'model_opt_fast={fast}',
                            '--fast'],
                           check=True, cwd=libpath)

        subprocess.run(['pytest',
                        '-k',
                        'pproc',
                        '-s',
                        '--no-monitor',],
                       check=True, cwd=libpath)

    elif 'propagators' in group:
        pymon_file = os.path.join(pymon_abs, '.pymon_propagators')
        if os.path.isfile(pymon_file):
            os.remove(pymon_file)

        subprocess.run(['mpirun',
                        '-n',
                        str(mpi),
                        'pytest',
                        '--no-monitor'*(not monitor),
                        '-k',
                        '_propagators',
                        '--with-mpi',
                        '--restrict-scope-to',
                        'module',
                        '--db',
                        '_pymon/.pymon_propagators',
                        '--description',
                        f'language={state["last_used_language"]}, compiler={state["last_used_compiler"]}, omp_pic={state["last_used_omp_pic"]}, omp_feec={state["last_used_omp_feec"]}',
                        '--tag',
                        f'struphy={importlib.metadata.version("struphy")}',
                        '--tag',
                        f'pyccel={pyccel.__version__}',],
                       check=True, cwd=libpath)

    elif 'tutorials' in group:
        pymon_file = os.path.join(pymon_abs, '.pymon_tutorials')
        if os.path.isfile(pymon_file):
            os.remove(pymon_file)

        import pytest

        if n is None:
            test_only = ''
        else:
            test_only = str(n).zfill(2)

        subprocess.run(['mpirun',
                        '-n',
                        str(mpi),
                        'pytest',
                        '-k',
                        test_only,
                        '-s',
                        '--with-mpi',
                        '--restrict-scope-to',
                        'function',
                        '--db',
                        '_pymon/.pymon_tutorials',
                        '--description',
                        f'language={state["last_used_language"]}, compiler={state["last_used_compiler"]}, omp_pic={state["last_used_omp_pic"]}, omp_feec={state["last_used_omp_feec"]}',
                        '--tag',
                        f'struphy={importlib.metadata.version("struphy")}',
                        '--tag',
                        f'pyccel={pyccel.__version__}',
                        'tutorials/',
                        '--no-monitor'*(not monitor),],
                        check=True, cwd=libpath)

        # retcode = pytest.main(['-k',
        #                        test_only,
        #                        '-s',
        #                        '--restrict-scope-to',
        #                        'function' + ' --no-monitor'*(not monitor),
        #                        '--db',
        #                        os.path.join(
        #                            libpath, '_pymon/.pymon_tutorials'),
        #                        '--description',
        #                        f'language={state["last_used_language"]}, compiler={state["last_used_compiler"]}, omp_pic={state["last_used_omp_pic"]}, omp_feec={state["last_used_omp_feec"]}',
        #                        '--tag',
        #                        f'struphy={importlib.metadata.version("struphy")}',
        #                        '--tag',
        #                        f'pyccel={pyccel.__version__}',
        #                        os.path.join(libpath, 'tutorials')])

    elif 'timings' in group:

        for gr in ['unit', 'models', 'propagators', 'tutorials']:
            _file = os.path.join(pymon_abs, '.pymon_' + gr)
            if os.path.isfile(_file):
                pymon_html_json(gr, verbose=verbose)

    else:
        from struphy.models.tests import test_toy_models, test_fluid_models, test_kinetic_models, test_hybrid_models
        from struphy.models.tests.test_xxpproc import test_pproc_codes

        objs = [test_toy_models.test_toy, test_fluid_models.test_fluid,
                test_kinetic_models.test_kinetic, test_hybrid_models.test_hybrid]
        for obj in objs:
            try:
                obj(('Cuboid', 'HomogenSlab'), fast, model=group, Tend=Tend)
                if fast:
                    print(
                        f'Fast is enabled, mappings other than Cuboid are skipped ...')
                else:
                    obj(('HollowTorus', 'AdhocTorus'), fast, model=group, Tend=Tend)
                    obj(('Tokamak', 'EQDSKequilibrium'), fast, model=group, Tend=Tend)
            except AttributeError:
                pass

        test_pproc_codes(group)


def pymon_html_json(group, verbose=False):
    '''Use sqlite3 to create Struphy timings in .html and .json format from .pymon files.

    Output is written to _pymon/ in the struphy root directory.

    Parameters
    ----------
    group : str
        One of "unit", "models", "propagators", "tutorials" or "pproc".
    '''

    libpath = struphy.__path__[0]

    pymon_abs = os.path.join(libpath, '_pymon/')
    if not os.path.exists(pymon_abs):
        os.mkdir(pymon_abs)

    with open(os.path.join(libpath, 'state.yml')) as f:
        state = yaml.load(f, Loader=yaml.FullLoader)

    # identify the comiled language
    language = state['last_used_language']

    # create html files
    subprocess.run(['sqlite3',
                    '_pymon/.pymon_' + group,
                    '-cmd',
                    '.mode html',
                    '-cmd',
                    '.headers on',
                    '-cmd',
                    '.output _pymon/pymon_' + group + '_contexts_' + language + '.html',
                    'select CPU_COUNT, CPU_FREQUENCY_MHZ, CPU_TYPE, CPU_VENDOR, RAM_TOTAL_MB, MACHINE_TYPE, MACHINE_ARCH, SYSTEM_INFO, PYTHON_INFO from EXECUTION_CONTEXTS',
                    ], check=True, cwd=libpath)
    subprocess.run(['sqlite3',
                    '_pymon/.pymon_' + group,
                    '-cmd',
                    '.mode html',
                    '-cmd',
                    '.headers on',
                    '-cmd',
                    '.output _pymon/pymon_' + group + '_sessions_' + language + '.html',
                    'select RUN_DATE, RUN_DESCRIPTION from TEST_SESSIONS',
                    ], check=True, cwd=libpath)
    subprocess.run(['sqlite3',
                    '_pymon/.pymon_' + group,
                    '-cmd',
                    '.mode html',
                    '-cmd',
                    '.headers on',
                    '-cmd',
                    '.output _pymon/pymon_' + group + '_metrics_' + language + '.html',
                    'select ITEM_PATH, ITEM, TOTAL_TIME, MEM_USAGE from TEST_METRICS order by KIND DESC',
                    ], check=True, cwd=libpath)

    # add html TABLE keyword
    filename = os.path.join(pymon_abs, 'pymon_' +
                            group + '_contexts_' + language + '.html')
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write('<TABLE BORDER=1 WIDTH=1800>'.rstrip('\r\n') + '\n' + content)
    with open(filename, 'a+') as f:
        f.write('</TABLE>')

    filename = os.path.join(pymon_abs, 'pymon_' +
                            group + '_sessions_' + language + '.html')
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write('<TABLE BORDER=1 WIDTH=1200>'.rstrip('\r\n') + '\n' + content)
    with open(filename, 'a+') as f:
        f.write('</TABLE>')

    filename = os.path.join(pymon_abs, 'pymon_' +
                            group + '_metrics_' + language + '.html')
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write('<TABLE BORDER=1 WIDTH=1200>'.rstrip('\r\n') + '\n' + content)
    with open(filename, 'a+') as f:
        f.write('</TABLE>')

    # create json files
    subprocess.run(['sqlite3',
                    '_pymon/.pymon_' + group,
                    '.mode json',
                    '.output _pymon/pymon_' + group + '_contexts_' + language + '.json',
                    'select * from EXECUTION_CONTEXTS',
                    ], check=True, cwd=libpath)
    subprocess.run(['sqlite3',
                    '_pymon/.pymon_' + group,
                    '.mode json',
                    '.output _pymon/pymon_' + group + '_sessions_' + language + '.json',
                    'select * from TEST_SESSIONS',
                    ], check=True, cwd=libpath)
    subprocess.run(['sqlite3',
                    '_pymon/.pymon_' + group,
                    '.mode json',
                    '.output _pymon/pymon_' + group + '_metrics_' + language + '.json',
                    'select * from TEST_METRICS',
                    ], check=True, cwd=libpath)

    # print to screen
    if verbose:
        print('##########################' + '#' *
              len(group) + '##########################')
        print('######################### ' +
              group + ' #########################')
        print('##########################' + '#' *
              len(group) + '##########################')
        print('-------------------------- CONTEXTS: --------------------------')
        subprocess.run(['sqlite3',
                        '_pymon/.pymon_' + group,
                        '.mode line',
                        'select * from EXECUTION_CONTEXTS',
                        ], check=True, cwd=libpath)
        print('-------------------------- SESSIONS: --------------------------')
        subprocess.run(['sqlite3',
                        '_pymon/.pymon_' + group,
                        '.mode line',
                        'select * from TEST_SESSIONS',
                        ], check=True, cwd=libpath)
        print('-------------------------- METRICS: ---------------------------')
        subprocess.run(['sqlite3',
                        '_pymon/.pymon_' + group,
                        '.mode line',
                        'select * from TEST_METRICS',
                        ], check=True, cwd=libpath)
