def struphy_example(case, mpi=1):
    """
    Run a Struphy example.
    
    Parameters
    ----------
    case : str
        Name of the example case to run.
        
    mpi : int
        Number of MPI processes used to run the example.
    """
    
    # call example function from below with number of MPI procs
    globals()[case](mpi)

#---------------------
# examples:
# -------------------

def maxwell(mpi=1):
    """
    Run an example for the model "Maxwell".
    """
    
    import os
    import subprocess
    import struphy

    libpath = struphy.__path__[0]
    
    # run the model
    subprocess.run(['struphy', 
                    'run', 
                    'Maxwell',
                    '-i',
                    'examples/params_maxwell.yml',
                    '-o',
                    'sim_example_maxwell',
                    '--mpi',
                    str(mpi)], check=True)
    
    # perform post-processing
    subprocess.run(['struphy',
                    'pproc',
                    'sim_example_maxwell'], check=True)
    
    # run diagnostics
    subprocess.run(['python3',
                    'examples/example_diagnostics_1dfft_maxwell.py',
                    os.path.join(libpath, 'io/out/sim_example_maxwell')],
                    check=True, cwd=libpath)
    
    
def linearmhd(mpi=1):
    """
    Run an example for the model "LinearMHD".
    """
    
    import os
    import subprocess
    import struphy

    libpath = struphy.__path__[0]
    
    # run the model
    subprocess.run(['struphy', 
                    'run', 
                    'LinearMHD',
                    '-i',
                    'examples/params_linearmhd.yml',
                    '-o',
                    'sim_example_linearmhd',
                    '--mpi',
                    str(mpi)], check=True)
    
    # perform post-processing
    subprocess.run(['struphy',
                    'pproc',
                    'sim_example_linearmhd'], check=True)
    
    # run diagnostics
    subprocess.run(['python3',
                    'examples/example_diagnostics_1dfft_linearmhd.py',
                    os.path.join(libpath, 'io/out/sim_example_linearmhd')],
                    check=True, cwd=libpath)
    
    
def TAE_tokamak(mpi=1):
    """
    Run an example for the model "LinearMHD".
    """
    
    import os
    import subprocess
    import struphy

    libpath = struphy.__path__[0]
    
    # run MHD eigenvalue solver
    subprocess.run(['python3',
                    'eigenvalue_solvers/mhd_axisymmetric_main.py',
                    '-1',
                    '/io/inp/examples/params_TAE_tokamak.yml',
                    '/io/out/sim_example_TAE_tokamak',
                    'spec'], 
                    check=True, cwd=libpath)
    
    # run the model
    subprocess.run(['struphy', 
                    'run', 
                    'LinearMHD',
                    '-i',
                    '../out/sim_example_TAE_tokamak/parameters.yml',
                    '-o',
                    'sim_example_TAE_tokamak',
                    '--mpi',
                    str(mpi)], check=True)
    
    # perform post-processing
    subprocess.run(['struphy',
                    'pproc',
                    'sim_example_TAE_tokamak'], check=True)
    
    # run diagnostics
    subprocess.run(['python3',
                    'examples/example_TAE_tokamak.py',
                    os.path.join(libpath, 'io/out/sim_example_TAE_tokamak')], 
                    check=True, cwd=libpath)
    
    
    
def linearmhdvlasov_cc(mpi=1):
    """
    Run an example for the model "LinearMHDVlasovCC".
    """
    
    import os
    import subprocess
    import struphy

    libpath = struphy.__path__[0]
    
    # run the model
    subprocess.run(['struphy', 
                    'run', 
                    'LinearMHDVlasovCC',
                    '-i',
                    'examples/params_hybridmhdvlasovcc.yml',
                    '-o',
                    'sim_example_linearmhdvlasovcc',
                    '--mpi',
                    str(mpi)], check=True)
    
    # perform post-processing
    subprocess.run(['struphy',
                    'pproc',
                    'sim_example_linearmhdvlasovcc'], check=True)
    
    # run diagnostics
    subprocess.run(['python3',
                    'examples/example_diagnostics_hybridmhdvlasovcc.py',
                    os.path.join(libpath, 'io/out/sim_example_linearmhdvlasovcc')], 
                    check=True, cwd=libpath)
    

def linearmhdvlasov_pc(mpi=1):
    """
    Run an example for the model "LinearMHDVlasovPC".
    """
    
    import os
    import subprocess
    import struphy

    libpath = struphy.__path__[0]
    
    # run the model
    subprocess.run(['struphy', 
                    'run', 
                    'LinearMHDVlasovPC',
                    '-i',
                    'examples/params_hybridmhdvlasovpc.yml',
                    '-o',
                    'sim_example_linearmhdvlasovpc',
                    '--mpi',
                    str(mpi)], check=True)
    
    # perform post-processing
    subprocess.run(['struphy',
                    'pproc',
                    'sim_example_linearmhdvlasovpc'], check=True)
    
    # run diagnostics
    subprocess.run(['python3',
                    'examples/example_diagnostics_hybridmhdvlasovcc.py',
                    os.path.join(libpath, 'io/out/sim_example_linearmhdvlasovpc')],
                    check=True, cwd=libpath)
    
    
def orbits_tokamak(mpi=1):
    """
    Run an example for the model "Vlasov".
    """
    
    import os
    import subprocess
    import struphy

    libpath = struphy.__path__[0]
    
    # run the model
    subprocess.run(['struphy', 
                    'run', 
                    'Vlasov',
                    '-i',
                    'examples/params_orbits_tokamak.yml',
                    '-o',
                    'sim_example_orbits_tokamak',
                    '--mpi',
                    str(mpi)], check=True)
    
    # perform post-processing
    subprocess.run(['struphy',
                    'pproc',
                    'sim_example_orbits_tokamak'], check=True)
    
    # run diagnostics
    subprocess.run(['python3',
                    'examples/example_orbits_tokamak.py',
                    os.path.join(libpath, 'io/out/sim_example_orbits_tokamak') + '/'],
                    check=True, cwd=libpath)
    
def gc_orbits_tokamak(mpi=1):
    """
    Run an example for the model "DriftKinetic".
    """
    
    import os
    import subprocess
    import struphy

    libpath = struphy.__path__[0]
    
    # run the model
    subprocess.run(['struphy', 
                    'run', 
                    'DriftKinetic',
                    '-i',
                    'examples/params_gc_orbits_tokamak.yml',
                    '-o',
                    'sim_example_gc_orbits_tokamak',
                    '--mpi',
                    str(mpi)], check=True)
    
    # perform post-processing
    subprocess.run(['struphy',
                    'pproc',
                    'sim_example_gc_orbits_tokamak'], check=True)
    
    # run diagnostics
    subprocess.run(['python3',
                    'examples/example_orbits_tokamak.py',
                    os.path.join(libpath, 'io/out/sim_example_gc_orbits_tokamak') + '/'], 
                    check=True, cwd=libpath)
    
    