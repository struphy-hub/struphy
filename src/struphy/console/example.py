def struphy_example(case, diagnostics, mpi=1):
    """
    Run a Struphy example.
    
    Parameters
    ----------
    case : str
        Name of the example case to run.
        
    diagnostics : bool
        Wether to do diagnostics and plot only, if example has already been run before, and the output folder still exists.
        
    mpi : int
        Number of MPI processes used to run the example.
    """
    
    import subprocess
    import struphy

    libpath = struphy.__path__[0]
    
    command = ['python3',
               'examples/' + case + '.py',
               '--mpi',
               str(mpi)]
    
    if diagnostics:
        command += ['--diagnostics']
    
    subprocess.run(command, check=True, cwd=libpath)