Installation
============

Requirements
------------

:abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` has been tested on Debian ``linux-x86_64`` systems; it requires

* Python 3 
* pip3
* Fortran compiler (gcc/gfortran)
* openmpi

as well as the following Ubuntu packages (``apt-get install``):

* libblas-dev 
* liblapack-dev

Necessary Python packages will be automatically installed with ``pip install .`` (list of packages in ``setup.py``).

.. _mac:

Mac with M1 chip
----------------

Numba must be installed from source::

    git clone https://github.com/numba/llvmlite.git
    cd llvmlite; python setup.py install
    git clone git://github.com/numba/numba.git
    cd numba
    python setup.py build_ext --inplace 
    python setup.py install

Installation of ``h5py``::

    HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.13.0 
    pip install h5py


.. _clusters:

Computing clusters
------------------

Specifics for the HPC system ``cobra`` at IPP::

    module purge
    module load gcc openmpi anaconda/3/2020.02 mpi4py
    module list

Struphy must be **installed, ran and profiled** with the user flag ``--user`` (see below) because the module environment cannot be installed to.
Add the relevant paths (this is done in the provided batch scripts as well)::

    export PATH="${PATH}:$HOME/.local/bin"
    export PYTHONPATH="${PYTHONPATH}:$(python3 -m site --user-site)" 

In order to suppress fork warnings in the slurm output::

    OMPI_MCA_mpi_warn_on_fork=0
    export OMPI_MCA_mpi_warn_on_fork 

 
.. _user_install:

From PYPI
---------

Not yet available.


From source
-----------

Clone and checkout the ``devel`` branch::

    git clone -b devel git@gitlab.mpcdf.mpg.de:clapp/hylife.git struphy
    cd struphy

User specific install::

    pip install --user .

For developers (path search in cloned repo first)::

    pip install -e .

Virtual environment install (recommended if not on computing cluster)::

    python3 -m pip install --user virtualenv
    python3 -m venv <env_name>
    source <env_name>/bin/activate
    pip install .

Next, install the submodules ``gvec_to_python`` and ``psydac``::

    git submodule init
    git submodule update
    cd psydac
    git pull origin devel
    export CC="mpicc"
    export HDF5_MPI="ON"
    export HDF5_DIR=/path/to/hdf5/openmpi
    python3 -m pip install -r requirements.txt
    python3 -m pip install -r requirements_extra.txt --no-build-isolation
    pip install .
    pip install sympde==0.13.1
    cd ..
    cd gvec_to_python
    python3 -m pip install . -r requirements.txt
    pip install sympy==1.6.1 
    cd ..
    
Quick help::

    struphy


Source
------

The source code of :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` can be found at `https://gitlab.mpcdf.mpg.de/clapp/hylife <https://gitlab.mpcdf.mpg.de/clapp/hylife>`_. 
In case of access problems please contact `Stefan Possanner <spossann@ipp.mpg.de>`_.
