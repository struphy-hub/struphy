Installation
============

Struphy can be installed in the following ways:

    1. Using :ref:`docker_install` images, also suited :ref:`for developers <docker_devs>`
    2. From :ref:`pypi_install` using ``pip install struphy``
    3. From :ref:`source_install` 

Option 1. is recommended as it works on all `docker-viable architectures <https://www.docker.com/>`_.
Options 2. and 3. are tested on ``x86_64`` with ``Ubuntu 20.04`` 
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

1. `Install docker desktop <https://docs.docker.com/desktop/>`_. Test your installation by typing ``docker info`` in the terminal.

(REMARK: older versions of Mac OS may require `older docker desktop versions <https://docs.docker.com/desktop/release-notes/#docker-desktop-471>`_.)

2. Login to the MPCDF Gitlab registry using a predefined struphy user and token::

    docker login gitlab-registry.mpcdf.mpg.de -u struphy_group_api -p glpat-JW4kjd_YMvRinSzKxSxs

3. Run the latest release of struphy in a container::

    docker run -i -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/struphy_release

The option ``-i`` stands for interactive while ``-t`` gives you a terminal. Test the container by typing ``struphy``,
which should display the struphy help. ``struphy compile`` will show that all kernels are already compiled. 
Type ``exit`` to exit and close the container.


Important docker commands
^^^^^^^^^^^^^^^^^^^^^^^^^

* ``docker run -d -t --name <container_name> IMAGE`` runs the container in the background (detached).
* ``docker ps`` lists running containers.
* ``docker exec <container_name> COMMAND`` gives a command to a detached container.
* ``docker stop <container_name>`` stops the container.
* ``docker images`` shows the images available on your computer.
* ``docker run -i -t -v ~/<dir>:<install_path>/io/out gitlab-registry.mpcdf.mpg.de/struphy/struphy/struphy_release`` mirrors struphy output to ``~/<dir>`` on the host machine.

.. _docker_devs:

Docker for devs
^^^^^^^^^^^^^^^

Docker is well-suited for developers on any kind of platform. 
After installing docker desktop (1.) and logging in to the registry (2.),
you can run the following container::

    docker run -i -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu20

This launches the recommended dev environment for struphy (``Ubuntu 20.04``) 
with all dependencies installed and compiled. Type ``pip list`` to view the installed Python packages.
Type ``exit`` to exit and close the container.

In order to interact with ``gitlab.mpcdf`` you need to mirror your **private ssh key** into the container.
On Linux this is done via::

    docker run -i -t -v ~/.ssh/id_rsa:/root/.ssh/id_rsa gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu20

On other OS ``~/.ssh/id_rsa`` must be replaced with the path to the private key.

You can now install struphy in developer mode::

    git clone git@gitlab.mpcdf.mpg.de:struphy/struphy.git
    cd struphy
    pip install -e .
    struphy

In order to develop inside the container, we recommend to use `Visual Studio Code <https://code.visualstudio.com/>`_.
Once installed, you can click on *Extensions* (fifth icon on the left sidebar) and search and install the ``Docker``
extension. Now you will be able to edit container files in VScode, as shown in the picture below.
Clicking on the new docker icon (red) on the left sidebar opens a box of running and exited containers (green).
You can click on a running container and go down the file tree to edit the file of choice.

.. image:: ../pics/vscode_docker_red.png



.. _pypi_install:

PyPI
----

Struphy is not yet released on the `Python Packaging Index <https://pypi.org/>`_, instead we describe here the similar
installation from the latest ``.whl`` file.

Two dependencies have to be installed "by hand":

1. Install ``gvec_to_python`` from wheel file::

    curl -O --header "PRIVATE-TOKEN: glpat-5QH11Kx-65GGiykzR5xo" "https://gitlab.mpcdf.mpg.de/api/v4/projects/5368/jobs/1679220/artifacts/dist/gvec_to_python-0.1.2-py3-none-any.whl"
    pip install gvec_to_python-0.1.2-py3-none-any.whl

2. Install ``psydac`` by following :ref:`install_psydac`. 

3. Download the latest ``struphy`` wheel file::

    curl -O --header "PRIVATE-TOKEN: glpat-6HCAcowp8yKfBzDYWqA_" "https://gitlab.mpcdf.mpg.de/api/v4/projects/2599/jobs/1778199/artifacts/dist/struphy-latest.whl"
    
4. Install ``struphy``::

    pip install struphy-1.9.1-py3-none-any.whl

where ``<option>`` is either empty (for Python environment installation), ``--user`` (for installation in ``.local/lib``) or ``-e`` (or installation in development mode).

5. Compile kernels::

    struphy compile


.. _source_install:

Source
------

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


 

