Installation
============

Requirements
------------

System
^^^^^^

- Linux (Ubuntu 20 or higher). If you are using a different OS, please run the :ref:`multipass`.
- Access to the `Gitlab.mpcdf <https://gitlab.mpcdf.mpg.de/>`_ packages `Struphy <https://gitlab.mpcdf.mpg.de/clapp/hylife>`_ and `gvec_to_python <https://gitlab.mpcdf.mpg.de/spossann/gvec_to_python>`_.


.. _linux_packages:

Linux packages
^^^^^^^^^^^^^^

::

    sudo apt update -y
    sudo apt install -y gfortran gcc libblas-dev liblapack-dev libopenmpi-dev openmpi-bin libomp-dev libomp5
    sudo apt install -y libhdf5-openmpi-dev
    sudo apt install -y python3-pip python3-mpi4py


.. _gvec_to_python:

gvec_to_python
^^^^^^^^^^^^^^

Get ``gvec_to_python`` wheel file::

    curl -O --header "PRIVATE-TOKEN: glpat-5QH11Kx-65GGiykzR5xo" "https://gitlab.mpcdf.mpg.de/api/v4/projects/5368/jobs/1679220/artifacts/dist/gvec_to_python-0.1.2-py3-none-any.whl"

Install from wheel::

    pip install gvec_to_python-0.1.2-py3-none-any.whl

.. _psydac:

psydac
^^^^^^

Get ``psydac`` source code (no wheel available yet)::

    git clone https://github.com/pyccel/psydac.git

Set variables for hdf5::

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


.. _user_install:

User install
------------

Before starting, it is safe to extend the ``PATH`` variable with your local path::

    export PATH=$PATH:~/.local/bin

Get current ``struphy`` wheel file::

    curl -O --header "PRIVATE-TOKEN: glpat-6HCAcowp8yKfBzDYWqA_" "https://gitlab.mpcdf.mpg.de/api/v4/projects/2599/jobs/1679170/artifacts/dist/struphy-1.9.0-py3-none-any.whl"

Install from wheel::

    pip install struphy-1.9.0-py3-none-any.whl


.. _source_install:

Developer install
-----------------

Before starting, it is safe to extend the ``PATH`` variable with your local path::

    export PATH=$PATH:~/.local/bin

Clone the `Struphy repository <https://gitlab.mpcdf.mpg.de/clapp/hylife>`_, checkout the ``devel`` branch, update submodules 
and name the repo ``<name>`` via::

    git clone -b devel --recurse-submodules git@gitlab.mpcdf.mpg.de:clapp/hylife.git <name>

Install the submodule in ``<name>/psydac`` according to :ref:`psydac` (without cloning the repo). Install the submodule in ``<name>/gvec_to_python``
according to::

    cd gvec_to_python
    python3 -m pip install . -r requirements.txt
    pip install sympy==1.6.1 
    cd ..

Install ``struphy`` via::

    pip install <option> .

where ``<option>`` is either empty (for Python environment installation), ``--user`` (for installation in ``.local/lib``) or ``-e`` (or installation in development mode).

Quick help::

    struphy

Struphy compilation::

    struphy compile

Computing clusters
------------------

Specifics for the HPC systems ``cobra`` and ``draco`` at IPP::

    module purge
    module load gcc openmpi anaconda/3/2020.02 mpi4py
    module list

Continue with :ref:`user_install` from above.

Extend the ``PYTHONPATH``::

    export PYTHONPATH="${PYTHONPATH}:$(python3 -m site --user-site)" 

In order to suppress fork warnings in the slurm output::

    OMPI_MCA_mpi_warn_on_fork=0
    export OMPI_MCA_mpi_warn_on_fork 


.. _multipass:

Multipass Virtual Machine
-------------------------

Download and documentation at `https://multipass.run <https://multipass.run/>`_.

After intallation, launch a VM via::

    multipass launch --name <VM-name> --cpus 4 --mem 4G --disk 16G

These are the recommended options, you can choose anything for ``<VM-name>``. The standard user is named ``ubuntu``.

Quick info::

    multipass info --all

Enter the shell::

    multipass shell <VM-name>

Continue with the installation of :ref:`linux_packages`, then proceed to :ref:`source_install`.

To shut down the VM::

    exit

and stop it from the host machine::

    multipass stop <VM-name>

In order to have access to the `Gitlab.mpcdf <https://gitlab.mpcdf.mpg.de/>`_ repositories, in case there is none, generate an ssh key::

    ssh-keygen

Then copy the key under ``.ssh/id_rsa.pub`` to your Gitlab profile.

You can mirror a folder ``<folder-name>`` to your host machine (for using a nice editor for instance).
``<folder-name>`` should be empty, as any content would be overwritten during ``mount``.
For this, create a new folder on your host (e.g. MacOS) and open a new terminal where you type::

    multipass mount /Path/to/Folder/on/Host/ <VM-name>:/home/ubuntu/<folder-name>/ 

(You should do this **before** you put anything in these folders.)

It is possible to access the GUI of your VM by installing ``ubuntu-desktop``::

    sudo apt-get install ubuntu-desktop xrdp -y

Then set your password via::

    sudo passwd ubuntu

You can also first create another user for that purpose by::

    sudo adduser USERNAME

and giving it superuser-rights::

    sudo usermod -aG sudo USERNAME

Then you can access the VM via a remote connection tool 
(e.g. Microsoft Remote Desktop on Windows and MacOS, and Remmina for Linux). 
For this you need the ip-address of your VM which you can find by running::

    multipass info

on the host machine. Then add a new PC with the ip-address as the PC name and login with your username and password. 
You should not update the VM from the GUI !


 

