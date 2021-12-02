Installation
============

Requirements
------------

:abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` has been tested on Debian linux-x86_64 systems; it requires

* Python3 
* pip3
* Fortran compiler (gcc/gfortran))

Ubuntu packages (``apt-get install``)::

    libblas-dev 
    liblapack-dev
    libopenmpi-dev
    openmpi-bin
    libomp-dev 
    libomp5
    tree

Python packages (automatically installed via ``pip3 install --user .``)::

    h5py
    matplotlib
    mpi4py
    numpy<1.21,>=1.17
    pyccel==0.10.1
    PyYAML
    scipy
    sympy
    vtk
    pandas
    pytest
    sphinx
    sphinxcontrib-napoleon
    sphinx-rtd-theme 
    m2r2
    docutils==0.15


.. _user_install:

User install
------------

Not yet available.


Developer install
-----------------

You can clone the full repository and checkout the ``devel`` branch::

    git clone -b devel git@gitlab.mpcdf.mpg.de:clapp/hylife.git

After that you can install :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` to the Python user install directory for your platform, typically ``~/.local/``::

    pip install --user .

However, for developers we recommend the creation of a virtual environment and installation therein::

    python3 -m pip install --user virtualenv
    python3 -m venv <env_name>
    source <env_name>/bin/activate
    pip3 install .

Compilation of kernels::

    struphy -c


Source
------

The source code of :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` can be found at `https://gitlab.mpcdf.mpg.de/clapp/hylife <https://gitlab.mpcdf.mpg.de/clapp/hylife>`_. 
In case of access problems please contact `Stefan Possanner <spossann@ipp.mpg.de>`_.