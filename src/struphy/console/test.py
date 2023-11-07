def struphy_test(group, mpi=0, fast=False):
    """
    Run Struphy unit and/or code tests.

    Parameters
    ----------
    mpi : int
        If >0, parallel units tests are run with mpi number of processes.

    codes : bool
        Whether to run code tests.

    fast : bool
        Whether to test just in slab geometry.
    """

    import os
    import subprocess
    import struphy

    libpath = struphy.__path__[0]

    if 'unit' in group:
        if mpi == 0:
            subprocess.run(['pytest',
                            'tests/tests_serial'],
                           check=True, cwd=libpath)
        else:
            subprocess.run(['mpirun',
                            '-n',
                            str(mpi),
                            'pytest',
                            '--with-mpi',
                            'tests/tests_mpi'],
                           check=True, cwd=libpath)

    elif 'codes' in group:

        if not fast:
            subprocess.run(['mpirun',
                            '-n',
                            '2',
                            'pytest',
                            '--with-mpi',
                            'tests/test_codes'],
                           check=True, cwd=libpath)
        else:
            subprocess.run(['mpirun',
                            '-n',
                            '2',
                            'pytest',
                            '--with-mpi',
                            'tests/test_codes',
                            '--fast'],
                           check=True, cwd=libpath)

        subprocess.run(['pytest',
                        'tests/test_pproc'], check=True, cwd=libpath)

    else:
        from struphy.tests.test_codes import test_toy, test_fluid, test_kinetic, test_hybrid
        from struphy.tests.test_pproc.test_pproc import test_pproc_codes

        objs = [test_toy.test_toy, test_fluid.test_fluid,
                test_kinetic.test_kinetic, test_hybrid.test_hybrid]
        for obj in objs:
            try:
                obj(fast, ('Cuboid', 'HomogenSlab'), group)
                if not fast:
                    obj(fast, ('HollowTorus', 'AdhocTorus'), group)
                    obj(fast, ('Tokamak', 'EQDSKequilibrium'), group)
            except AttributeError:
                pass

        test_pproc_codes(group)
