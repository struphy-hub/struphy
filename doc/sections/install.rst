Installation
============

Struphy can be installed in the following ways:

    1. Using Docker images, also suited :ref:`for developers <docker_devs>`
    2. From PyPI using ``pip install struphy``
    3. From source 

Option 1 is recommended as it works on all `docker-viable architectures <https://www.docker.com/>`_.
Options 2 and 3 are tested on ``x86_64`` with ``Ubuntu 20.04`` 
and on the `MPCDF HPC facilities <https://docs.mpcdf.mpg.de/doc/computing/index.html>`_.


.. _docker_install:

Docker
------

With this installation you will be able to run struphy in a `docker container <https://www.docker.com/resources/what-container/>`_, 
encapsulated from your host machine.
The container is launched from an `image <https://docs.docker.com/get-started/overview/#docker-objects>`_ 
which you can download and run immediately, irrespective of your architecture and OS.


.. _user_install:

User install
^^^^^^^^^^^^

To use struphy via docker, perform the following steps:

1. `Install Docker Desktop <https://docs.docker.com/desktop/>`_ and start it.

On Mac, it is recommended to read the `Mac OS permission requirements <https://docs.docker.com/desktop/mac/permission-requirements/>`_.
(REMARK: older versions of Mac OS may require `older docker desktop versions <https://docs.docker.com/desktop/release-notes/#docker-desktop-471>`_.)

On Windows, it is recommended to read the `Windows permission requirements <https://docs.docker.com/desktop/windows/permission-requirements/>`_

On Linux, if you do not want to preface the docker command with ``sudo``, you can 
`create a Unix group <https://docs.docker.com/engine/install/linux-postinstall/>`_ 
called ``docker`` and add your user to it.
If you are uncomfortable with running `sudo`, you can `run docker in "rootless" mode <https://docs.docker.com/engine/security/rootless/>`_.

2. Login to the MPCDF Gitlab registry using a predefined struphy user and token::

    docker login gitlab-registry.mpcdf.mpg.de -u struphy_group_api -p glpat-JW4kjd_YMvRinSzKxSxs

3. Run the latest release of struphy in a container (about 2 GB in size)::

    docker run -it gitlab-registry.mpcdf.mpg.de/struphy/struphy/release

The option ``-i`` stands for interactive while ``-t`` gives you a terminal. Test the container by typing ``struphy``,
which should display the struphy help. ``struphy compile`` will show that all kernels are already compiled. 
Type ``exit`` to exit and close the container.


Important docker commands
^^^^^^^^^^^^^^^^^^^^^^^^^

* ``docker images`` shows the images available on your computer.
* ``docker run -d -t --name <container_name> IMAGE`` runs the container in the background (detached).
* ``docker exec <container_name> COMMAND`` gives a bash command to a detached container.
* ``docker stop <container_name>`` stops the container.
* ``docker ps -l`` lists all containers (also exited/stopped).
* ``docker restart <container_name>`` restarts the container in detached mode.
* ``docker attach <container_name>`` opens a terminal to a detached container.

* Mirror default struphy output to ``~/<dir>`` on the host machine::
    
    docker run -it -v ~/<dir>:<install_path>/io/out gitlab-registry.mpcdf.mpg.de/struphy/struphy/release

.. _docker_devs:

Docker for devs
^^^^^^^^^^^^^^^

Docker is well-suited for developers on any kind of platform. 
After installing and launching docker desktop (1. from above) and logging in to the registry (2. from above),
the relevant docker image to run is ``gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu20``.
It initializes the recommended dev environment for struphy (``Ubuntu 20.04``) 
with all dependencies installed and compiled. 

In order to interact with ``gitlab.mpcdf`` you need to mirror your **private ssh key** into the container 
with the ``-v`` option. For a ``rsa`` key this is done with::

    docker run -it -v ~/.ssh/id_rsa:/root/.ssh/id_rsa gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu20

On OS other than Linux ``~/.ssh/id_rsa`` must be replaced with the path to the private rsa key.

You can now install struphy in developer mode::

    git clone git@gitlab.mpcdf.mpg.de:struphy/struphy.git
    cd struphy
    pip install -e .
    struphy

In order to develop inside the container, we recommend to use `Visual Studio Code <https://code.visualstudio.com/>`_.
Once installed, you can click on **Extensions** (red circle below) and install the ``Remote - Containers``
extension (green box). Now you will be able to edit container files in VScode by clicking on the green symbol
in the bottom-left corner (yellow circle). Choose ``Attach to a running container ...`` and select 
the container in which you want to edit. By doing ``File - Open Folder...`` you are able to
open any folder from the container.

In order to have Python highlighting we recommend to install the ``Python`` extension in VScode.

.. image:: ../pics/vscode_docker_red.png



.. _pypi_install:

PyPI
----

WARNING: this install is tested only on ``x86_64`` with ``Ubuntu 20.04`` 
and on the `MPCDF HPC facilities <https://docs.mpcdf.mpg.de/doc/computing/index.html>`_.

Struphy is not yet released on the `Python Packaging Index <https://pypi.org/>`_, coming soon! 

Preliminary Notes:

Required Linux packages (``.deb``)::

    sudo apt update -y
    sudo apt install -y gfortran gcc libblas-dev liblapack-dev libopenmpi-dev openmpi-bin libomp-dev libomp5
    sudo apt install -y libhdf5-openmpi-dev
    sudo apt install -y python3-pip python3-mpi4py

Two dependencies have to be installed "by hand":

1. Install ``gvec_to_python`` from wheel file::

    curl -O --header "PRIVATE-TOKEN: glpat-5QH11Kx-65GGiykzR5xo" "https://gitlab.mpcdf.mpg.de/api/v4/projects/5368/jobs/1679220/artifacts/dist/gvec_to_python-0.1.2-py3-none-any.whl"
    pip install gvec_to_python-0.1.2-py3-none-any.whl

2. Install ``psydac`` by following :ref:`install_psydac`. 

3. Install ``struphy``::

    pip install struphy

5. Compile kernels::

    struphy compile


.. _source_install:

Source
------

WARNING: this install is tested only on ``x86_64`` with ``Ubuntu 20.04`` 
and on the `MPCDF HPC facilities <https://docs.mpcdf.mpg.de/doc/computing/index.html>`_.

Required Linux packages (``.deb``)::

    sudo apt update -y
    sudo apt install -y gfortran gcc libblas-dev liblapack-dev libopenmpi-dev openmpi-bin libomp-dev libomp5
    sudo apt install -y libhdf5-openmpi-dev
    sudo apt install -y python3-pip python3-mpi4py

1. Clone the `struphy repository <https://gitlab.mpcdf.mpg.de/struphy/struphy>`_, update submodules 
and name the repo ``<name>`` via::

    git clone --recurse-submodules git@gitlab.mpcdf.mpg.de:struphy/struphy.git <name>
    cd <name>

2. Install the submodule in ``<name>/gvec_to_python`` according to::

    cd gvec_to_python
    python3 -m pip install . -r requirements.txt
    pip install sympy==1.6.1 
    cd ..

3. Install the submodule in ``<name>/psydac`` according to :ref:`install_psydac`.

4. Install ``struphy`` via::

    pip install <option> .

where ``<option>`` is either empty (for Python environment installation), ``--user`` (for installation in ``.local/lib``) or ``-e`` (or installation in development mode).

5. Compile kernels::

    struphy compile


MPCDF computing clusters
------------------------

Specifics for the HPC systems ``cobra`` and ``draco`` at `MPCDF HPC facilities <https://docs.mpcdf.mpg.de/doc/computing/index.html>`_::

    module purge
    module load gcc openmpi anaconda/3/2020.02 mpi4py
    module list

Continue with one of the three install methods from above.

Extend the ``PYTHONPATH``::

    export PYTHONPATH="${PYTHONPATH}:$(python3 -m site --user-site)" 

In order to suppress fork warnings in the slurm output::

    OMPI_MCA_mpi_warn_on_fork=0
    export OMPI_MCA_mpi_warn_on_fork 


.. _install_psydac:

Psydac installation instructions
--------------------------------

In the psydac repository type::

    git checkout struphy-branch
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


 

