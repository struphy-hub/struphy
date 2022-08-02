Installation
============

Before starting, it is safe to extend the ``PATH`` variable with your local path::

    export PATH=$PATH:~/.local/bin


.. _user_install:

User install
------------

Install ``gvec_to_python`` from wheel file (see :ref:`python_packages`)::

    pip install gvec_to_python-0.1.2-py3-none-any.whl

Install ``psydac`` by following :ref:`install_psydac`. 

Get current ``struphy`` wheel file::

    curl -O --header "PRIVATE-TOKEN: glpat-6HCAcowp8yKfBzDYWqA_" "https://gitlab.mpcdf.mpg.de/api/v4/projects/2599/jobs/1679170/artifacts/dist/struphy-1.9.0-py3-none-any.whl"

Install from wheel::

    pip install struphy-1.9.0-py3-none-any.whl

Compile kernels::

    struphy compile


.. _source_install:

Developer install
-----------------

Clone the `Struphy repository <https://gitlab.mpcdf.mpg.de/clapp/hylife>`_, checkout the ``devel`` branch, update submodules 
and name the repo ``<name>`` via::

    git clone -b devel --recurse-submodules git@gitlab.mpcdf.mpg.de:clapp/hylife.git <name>

Install the submodule in ``<name>/psydac`` according to :ref:`install_psydac`. Install the submodule in ``<name>/gvec_to_python``
according to::

    cd gvec_to_python
    python3 -m pip install . -r requirements.txt
    pip install sympy==1.6.1 
    cd ..

Install ``struphy`` via::

    pip install <option> .

where ``<option>`` is either empty (for Python environment installation), ``--user`` (for installation in ``.local/lib``) or ``-e`` (or installation in development mode).

Compile kernels::

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


.. _install_psydac:

Psydac installation instructions
--------------------------------

In the psydac repository type::

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

    pyccel <path>/psydac/core/kernels.py
    pyccel <path>/psydac/core/bsplines_pyccel.py


 

