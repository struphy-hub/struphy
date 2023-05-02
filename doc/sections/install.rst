.. _install:

Installation
============

*Struphy* can be installed in the following ways:

1. From :ref:`PyPI <pypi_install>` :math:`\to` ``pip install struphy``
2. From :ref:`source <source_install>` 
3. Using :ref:`Docker images <docker_install>`, also suited :ref:`for developers <docker_devs>`


.. _require:

Requirements
------------

- Python 3 and pip3
- A Fortran compiler like gfortran, gcc
- Linear algebra packages BLAS and LAPACK
- An MPI library like open-mpi, mpich
- OpenMP
- Git version control system

Sample environment on **Debian-Ubuntu-Mint**::

    apt update
    apt install -y python3-pip
    apt install -y gcc
    apt install -y gfortran
    apt install -y libblas-dev liblapack-dev
    apt install -y libopenmpi-dev openmpi-bin
    apt install -y libomp-dev libomp5
    apt install -y git

Sample environment on **Fedora-CentOS-RHEL**::

    dnf check-update
    dnf install -y python3-pip
    dnf install -y gcc
    dnf install -y gfortran
    dnf install -y blas-devel lapack-devel
    dnf install -y openmpi openmpi-devel
    dnf install -y libgomp
    dnf install -y git
    dnf install -y environment-modules
    dnf install -y python3-mpi4py-openmpi
    dnf install -y python3-devel

Sample environment on **Mac OS**::

    brew update
    brew install -y python3-pip
    brew install -y gcc
    brew install -y openblas
    brew install -y lapack
    brew install -y open-mpi
    brew install -y libomp
    brew install -y git

On Windows systems we recommend the use of a virtual machine, for instance the :ref:`multipass`.


.. _pypi_install:

PyPI
----

On **Fedora-CentOS-RHEL** you must::

    module load mpi/openmpi-$(arch)

Install package::

    pip install struphy

Compile kernels::

    struphy compile


.. _source_install:

Source
------

Clone the `Struphy repository <https://gitlab.mpcdf.mpg.de/struphy/struphy>`_ and update submodules::

    git clone --recurse-submodules git@gitlab.mpcdf.mpg.de:struphy/struphy.git 
    cd struphy

Update pip and install package::

    pip install --upgrade pip
    pip install <option> .

where ``<option>`` is either empty (Python environment installation), ``--user`` (installation in ``.local/lib``) or ``-e`` (installation in development mode).

Compile kernels::

    struphy compile


.. _docker_install:

Docker
------

With this installation you will be able to run *struphy* in a `docker container <https://www.docker.com/resources/what-container/>`_, 
encapsulated from your host machine.
The container is launched from an `image <https://docs.docker.com/get-started/overview/#docker-objects>`_ 
which you can download and run immediately, irrespective of your architecture and OS.


.. _user_install:

User install
^^^^^^^^^^^^

To use *struphy* via docker, perform the following steps:

1. `Install Docker Desktop <https://docs.docker.com/desktop/>`_ and start it.

On Mac, it is recommended to read the `Mac OS permission requirements <https://docs.docker.com/desktop/mac/permission-requirements/>`_.
(REMARK: older versions of Mac OS may require `older docker desktop versions <https://docs.docker.com/desktop/release-notes/#docker-desktop-471>`_.)

On Windows, it is recommended to read the `Windows permission requirements <https://docs.docker.com/desktop/windows/permission-requirements/>`_

On Linux, if you do not want to preface the docker command with ``sudo``, you can 
`create a Unix group <https://docs.docker.com/engine/install/linux-postinstall/>`_ 
called ``docker`` and add your user to it.
If you are uncomfortable with running `sudo`, you can `run docker in "rootless" mode <https://docs.docker.com/engine/security/rootless/>`_.

2. Login to the MPCDF Gitlab registry using a predefined *struphy* user and token::

    docker login gitlab-registry.mpcdf.mpg.de -u struphy_group_api -p glpat-JW4kjd_YMvRinSzKxSxs

3. Pull one of the following environment images (< 1 GB in size)::

    docker pull gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu20:latest
    docker pull gitlab-registry.mpcdf.mpg.de/struphy/struphy/fedora36:latest

4. Run the container::

    docker run -it gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu20:latest

The option ``-i`` stands for interactive while ``-t`` gives you a terminal.

5. Continue with one of the installation methods above (PyPI or source).


Important docker commands
^^^^^^^^^^^^^^^^^^^^^^^^^

* ``docker images`` shows the images available on your computer.
* ``docker run -d -t --name <container_name> IMAGE`` runs the container in the background (detached).
* ``docker exec <container_name> COMMAND`` gives a bash command to a detached container.
* ``docker stop <container_name>`` stops the container.
* ``docker ps -l`` lists all containers (also exited/stopped).
* ``docker restart <container_name>`` restarts the container in detached mode.
* ``docker attach <container_name>`` opens a terminal to a detached container.

* Mirror default *struphy* output to ``~/<dir>`` on the host machine::
    
    docker run -it -v ~/<dir>:<install_path>/io/out gitlab-registry.mpcdf.mpg.de/struphy/struphy/release

.. _docker_devs:

Docker for devs
^^^^^^^^^^^^^^^

Docker is well-suited for developers on any kind of platform. 
In order to interact with ``gitlab.mpcdf`` you need to mirror your **private ssh key** into the container 
with the ``-v`` option. For a ``rsa`` key this is done with::

    docker run -it -v ~/.ssh/id_rsa:/root/.ssh/id_rsa gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu20:latest

On OS other than Linux ``~/.ssh/id_rsa`` must be replaced with the path to the private rsa key.

You can now install *struphy* from source (see above).

In order to develop inside the container, we recommend to use `Visual Studio Code <https://code.visualstudio.com/>`_.
Once installed, you can click on **Extensions** (red circle below) and install the ``Dev Containers``
extension (green box). Now you will be able to edit container files in VScode by clicking on the green symbol
in the bottom-left corner (yellow circle). Choose ``Attach to a running container ...`` and select 
the container in which you want to edit. By doing ``File - Open Folder...`` you are able to
open any folder from the container.

We recommend to install the following VScode extensions inside the container:

    - ``Python`` extension 
    - ``Python Extensions`` extension
    - ``Jupyter`` extension  

.. image:: ../pics/vscode_docker_red.png


MPCDF computing clusters
------------------------

Some specifics for the HPC systems ``cobra`` and ``draco`` at `MPCDF HPC facilities <https://docs.mpcdf.mpg.de/doc/computing/index.html>`_.

1. Load necessary modules::

    module purge
    module load gcc/9 openmpi anaconda/3/2021.11 mpi4py
    module list

2. Create a Python virtual environment::

    pip install -U virtualenv
    python3 -m venv <some_name>
    source <some_name>/bin/activate
    python3 -m pip install --upgrade pip

3. Continue with one of the install methods from above.

4. Test your installation with a debug run using up to 8 mpi processes (only on nodes ``cobra03-cobra06``)::

    struphy run Maxwell --mpi 8 --debug

5. When using slurm, include the following lines in your BATCH script::

    source <some_name>/bin/activate

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

Continue with the installation of :ref:`source_install`. To shut down the VM::

    exit

and stop it from the host machine::

    multipass stop <VM-name>

In order to have access to the `Struphy repository <https://gitlab.mpcdf.mpg.de/struphy/struphy>`_, 
generate an ssh key (if you do not already have one)::

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


 

