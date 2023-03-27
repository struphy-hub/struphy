def struphy_pproc(dirs, celldivide=1):
    """
    Post process data from finished Struphy runs.
    
    Parameters
    ----------
    dirs : list
        Names of output folders (srings).
        
    celldivide : int, optional
        Number of grid point in each cell used to create vtk files (default=1).
    """
    import subprocess
    import os
    import struphy

    libpath = struphy.__path__[0]
    
    # loop over output folders and call post-processing .py file
    for d in dirs:
        subprocess.run(['python3',
                        'post_processing/pproc_struphy.py',
                        str(celldivide),
                        os.path.join(libpath, 'io/out', d) + '/'], 
                        cwd=libpath)