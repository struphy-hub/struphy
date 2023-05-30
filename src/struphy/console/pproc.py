def struphy_pproc(dirr, dir_abs=None, step=1, celldivide=1):
    """
    Post process data from finished Struphy runs.

    Parameters
    ----------
    dirr : str
        Path of simulation output folder relative to <struphy_path>/io/out.

    dir_abs : str
        Absolute path to the simulation output folder.

    step : int, optional
        Whether to do post-processing at every time step (step=1, default), every second time step (step=2), etc.

    celldivide : int, optional
        Number of grid point in each cell used to create vtk files (default=1).
    """
    import subprocess
    import os
    import struphy

    libpath = struphy.__path__[0]
    
    with open(os.path.join(libpath, 'io_path.txt')) as f:
        io_path = f.readlines()[0]

    # create absolute path
    if dir_abs is None:
        dir_abs = os.path.join(io_path, 'io/out', dirr)

    # loop over output folders and call post-processing .py file
    subprocess.run(['python3',
                    'post_processing/pproc_struphy.py',
                    dir_abs,
                    '-s',
                    str(step),
                    '--celldivide',
                    str(celldivide)],
                    cwd=libpath, 
                    check=True)
