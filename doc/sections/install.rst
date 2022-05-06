Installation
============

Requirements
------------

- Linux (Ubuntu 20 or higher). If you are using a different OS, please run the `Multipass virtual machine <https://multipass.run/>`_.
- Access to the `Gitlab.mpcdf <https://gitlab.mpcdf.mpg.de/>`_ packages `Struphy <https://gitlab.mpcdf.mpg.de/clapp/hylife>`_ and `gvec_to_python <https://gitlab.mpcdf.mpg.de/spossann/gvec_to_python>`_.

Linux packages
--------------

::

    sudo apt-get update
    sudo apt-get install gfortran
    sudo apt-get install gcc
    sudo apt-get install libblas-dev liblapack-dev
    sudo apt-get install libopenmpi-dev openmpi-bin
    sudo apt-get install libomp-dev libomp5
    sudo apt-get install python3-pip
    sudo apt-get install python3-mpi4py


.. _user_install:

Struphy package install from PYPI
---------------------------------

Not yet available.


.. _source_install:

Struphy package install from source
-----------------------------------

Clone the `Struphy repository <https://gitlab.mpcdf.mpg.de/clapp/hylife>`_, checkout the ``devel`` branch and name the repo ``<name>``::

    git clone -b devel git@gitlab.mpcdf.mpg.de:clapp/hylife.git <name>

Install the submodules ``psydac`` and ``gvec_to_python``::

    cd <name>
    git submodule init
    git submodule update
    pip install h5py
    cd psydac
    git pull origin devel
    export CC="mpicc"
    export HDF5_MPI="ON"
    export HDF5_DIR=/path/to/hdf5/openmpi
    python3 -m pip install -r requirements.txt
    python3 -m pip install -r requirements_extra.txt --no-build-isolation
    pip install .
    cd psydac/core
    pyccel kernels.py --language fortran
    cd -
    cd ..
    cd gvec_to_python
    python3 -m pip install . -r requirements.txt
    pip install sympy==1.6.1 
    cd ..

Install ``struphy`` via::

    pip install <option> .

where ``<option>`` is either empty (for Python environment installation), ``--user`` (for installation in ``.local/lib``) or ``-e`` (or installation in development mode).

In case of a user installation (``<option>=--user``), it is safe to extend the ``PATH`` variable::

    export PATH=$PATH:~/.local/bin

Quick help::

    struphy

Struphy compilation::

    struphy compile
    pip install -U pyccel

Computing clusters
------------------

Specifics for the HPC system ``cobra`` at IPP::

    module purge
    module load gcc openmpi anaconda/3/2020.02 mpi4py
    module list

Continue with :ref:`source_install` from above.

Extend the ``PYTHONPATH``::

    export PYTHONPATH="${PYTHONPATH}:$(python3 -m site --user-site)" 

In order to suppress fork warnings in the slurm output::

    OMPI_MCA_mpi_warn_on_fork=0
    export OMPI_MCA_mpi_warn_on_fork 

 

