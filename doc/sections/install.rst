Installation
============

Requirements
------------

- Linux (Ubuntu 20 or higher). If you are using a different OS, please run the :ref:`multipass`.
- Access to the `Gitlab.mpcdf <https://gitlab.mpcdf.mpg.de/>`_ packages `Struphy <https://gitlab.mpcdf.mpg.de/clapp/hylife>`_ and `gvec_to_python <https://gitlab.mpcdf.mpg.de/spossann/gvec_to_python>`_.


.. _linux_packages:

Linux packages
--------------

::

    sudo apt update
    sudo apt install -y gfortran gcc libblas-dev liblapack-dev libopenmpi-dev openmpi-bin libomp-dev libomp5
    sudo apt install -y libhdf5-openmpi-dev
    sudo apt install -y python3-pip python3-mpi4py


.. _user_install:

Struphy package install from PYPI
---------------------------------

Not yet available.


.. _source_install:

Struphy package install from source
-----------------------------------

Before starting, it is safe to extend the ``PATH`` variable with your local path::

    export PATH=$PATH:~/.local/bin

Clone the `Struphy repository <https://gitlab.mpcdf.mpg.de/clapp/hylife>`_, checkout the ``devel`` branch and name the repo ``<name>``::

    git clone -b devel git@gitlab.mpcdf.mpg.de:clapp/hylife.git <name>

Install the submodules ``psydac``::

    cd <name>
    git submodule init
    git submodule update
    cd psydac
    git pull origin devel
    export CC="mpicc"
    export HDF5_MPI="ON"

Determine the ``HDF5_DIR`` via::

    dpkg -L libhdf5-openmpi-dev

The correct path is the one that ends with ``hdf5/openmpi``, for example ``/usr/lib/x86_64-linux-gnu/hdf5/openmpi`` on a standard Ubuntu system. Set the correct path as in::

    export HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi

Install psydac::

    python3 -m pip install -r requirements.txt
    python3 -m pip install -r requirements_extra.txt --no-build-isolation
    pip install .
    cd ..

Install the submodule ``gvec_to_python``::

    cd gvec_to_python
    python3 -m pip install . -r requirements.txt
    pip install sympy==1.6.1 
    cd ..

Find out where psydac is installed::

    pip show psydac

which yields something like::

    Name: psydac
    Version: 0.1
    Summary: Python package for BSplines/NURBS
    Home-page: http://www.ahmed.ratnani.org
    Author: Ahmed Ratnani, Jalal Lakhlili, Yaman Güçlü, Said Hadjout
    Author-email: ratnaniahmed@gmail.com
    License: LICENSE.txt
    Location: $HOME/git_repos/struphy/env/lib/python3.8/site-packages
    Requires: gelato, pyyaml, tblib, sympy, sympde, igakit, pytest, numpy, pyevtk, scipy, pyccel, packaging, matplotlib, numba, h5py, mpi4py
    Required-by:

The ``<path>`` under ``Location:`` is what we are looking for. Compile psydac kernels via::

    pyccel <path>/psydac/core/kernels.py --language fortran

Install ``struphy`` via::

    pip install <option> .

where ``<option>`` is either empty (for Python environment installation), ``--user`` (for installation in ``.local/lib``) or ``-e`` (or installation in development mode).

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


.. _multipass:

Multipass Virtual Machine
-------------------------

Download and documentation at `https://multipass.run <https://multipass.run/>`_.

After intallation, launch a VM with the name ``<VM-name>`` via::

    multipass launch --name <VM-name> --cpus 4

Quick info::

    multipass info --all

Enter the shell::

    multipass shell <VM-name>

Continue with the installation of :ref:`linux_packages`, then proceed to :ref:`source_install`.

Exit the VM with::

    exit

In order to have acces to the `Gitlab.mpcdf <https://gitlab.mpcdf.mpg.de/>`_ repositories, in case there is none, generate an ssh key::

    ssh-keygen

Then copy the key under ``.ssh/id_rsa.pub`` to your Gitlab profile.

You can mirror a folder ``<folder-name>`` to your host machine (for using a nice editor for instance). 
For this, create a new folder on your host (e.g. MacOS) and open a new terminal where you type::

    multipass mount /Path/to/Folder/on/Host/ <VM-name>:/home/ubuntu/<folder-name>/ 



 

