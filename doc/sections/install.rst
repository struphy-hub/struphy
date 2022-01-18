Installation
============

Requirements
------------

:abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` has been tested on Debian ``linux-x86_64`` systems; it requires

* Python 3 
* pip3
* Fortran compiler (gcc/gfortran)

as well as the following Ubuntu packages (``apt-get install``):

* libblas-dev 
* liblapack-dev
* libopenmpi-dev
* openmpi-bin
* libomp-dev 
* libomp5
* tree

Necessary Python packages will be automatically installed with ``pip install .`` (list of packages in ``setup.py``).


.. _user_install:

User install
------------

Not yet available.


Developer install
-----------------

Clone and checkout the ``devel`` branch::

    git clone -b devel git@gitlab.mpcdf.mpg.de:clapp/hylife.git

Install :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` in the default local directory of your platform::

    pip install --user .

For developers the creation of a virtual environment is recommended::

    python3 -m pip install --user virtualenv
    python3 -m venv <env_name>
    source <env_name>/bin/activate
    pip3 install .
    
Quick help::

    struphy -h

Compilation of kernels::

    struphy -c

Run the default code ``lin_mhd`` with default parameters::

    struphy

We recommend to run the code outside of the cloned repository, such that the installed and compiled version of Struphy is called.


Source
------

The source code of :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` can be found at `https://gitlab.mpcdf.mpg.de/clapp/hylife <https://gitlab.mpcdf.mpg.de/clapp/hylife>`_. 
In case of access problems please contact `Stefan Possanner <spossann@ipp.mpg.de>`_.