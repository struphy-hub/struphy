.. _install:

Installation
============

Struphy can be installed in the following ways:

1. From :ref:`PyPI <pypi_install>`, for running the latest release
2. From :ref:`source <source_install>`, for running or adding code 
3. Using :ref:`Docker images <docker_install>`, also suited :ref:`for developers <docker_devs>`


.. _require:

Requirements
------------

- Python >3.7 and <3.12 and pip3
- A Fortran compiler like gfortran, gcc
- Linear algebra packages BLAS and LAPACK
- An MPI library like open-mpi, mpich
- OpenMP
- Git version control system
- Pandoc (optional)
- **Struphy does not support** `Anaconda <https://www.anaconda.com/>`_, please use Python directly.

Sample environment on **Debian-Ubuntu-Mint**::

    sudo apt update
    sudo apt install -y python3-pip
    sudo apt install -y python3.10-venv
    sudo apt install -y gcc
    sudo apt install -y gfortran
    sudo apt install -y libblas-dev liblapack-dev
    sudo apt install -y libopenmpi-dev openmpi-bin
    sudo apt install -y libomp-dev libomp5
    sudo apt install -y git
    sudo apt install -y pandoc

Sample environment on **Fedora-CentOS-RHEL**::

    sudo dnf check-update
    sudo dnf install -y python3-pip
    sudo dnf install -y gcc
    sudo dnf install -y gfortran
    sudo dnf install -y blas-devel lapack-devel
    sudo dnf install -y openmpi openmpi-devel
    sudo dnf install -y libgomp
    sudo dnf install -y git
    sudo dnf install -y environment-modules
    sudo dnf install -y python3-mpi4py-openmpi
    sudo dnf install -y python3-devel
    sudo dnf install -y pandoc

Sample environment on **Mac OS**::

    brew update
    brew install python3
    brew install gcc
    brew install openblas
    brew install lapack
    brew install open-mpi
    brew install libomp
    brew install git
    brew install pandoc

In case you see problems with the `mpi4py` build on **Mac OS**, you can try to install the Xcode command line tools (160 MB)::

    xcode-select --install

On **Windows systems** we recommend the use of a virtual machine, for instance the :ref:`multipass`.


.. _virtualenv:

Virtual Python environment
--------------------------

In order for the Struphy installation not to interfere with the custom Python environment you have set up on your machine, 
we recommend the use of a `Virtual Python environment <https://pypi.org/project/virtualenv/>`_::

    pip install -U virtualenv

Create virtual environment::

    python3 -m venv <name>

Launch the virtual environment (from the location where created)::

    source <name>/bin/activate

Check the pre-installed packages and upgrade ``pip``::

    pip list
    pip install --upgrade pip

Continue with the Struphy installation. When finished, you can deactivate the virtual environment::

    deactivate

The environment is stored in the folder ``<name>`` and can be re-activated any time for working with Struphy.


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

Clone the `Struphy repository <https://gitlab.mpcdf.mpg.de/struphy/struphy>`_::

    git clone https://gitlab.mpcdf.mpg.de/struphy/struphy.git
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

With this installation you will be able to run Struphy in a `docker container <https://www.docker.com/resources/what-container/>`_, 
encapsulated from your host machine.
The container is launched from an `image <https://docs.docker.com/get-started/overview/#docker-objects>`_ 
which you can download and run immediately, irrespective of your architecture and OS.


.. _user_install:

User install
^^^^^^^^^^^^

To use Struphy via docker, perform the following steps:

1. `Install Docker Desktop <https://docs.docker.com/desktop/>`_ and start it.

On Mac, it is recommended to read the `Mac OS permission requirements <https://docs.docker.com/desktop/mac/permission-requirements/>`_.
(REMARK: older versions of Mac OS may require `older docker desktop versions <https://docs.docker.com/desktop/release-notes/#docker-desktop-471>`_.)

On Windows, it is recommended to read the `Windows permission requirements <https://docs.docker.com/desktop/windows/permission-requirements/>`_

On Linux, if you do not want to preface the docker command with ``sudo``, you can 
`create a Unix group <https://docs.docker.com/engine/install/linux-postinstall/>`_ 
called ``docker`` and add your user to it.
If you are uncomfortable with running `sudo`, you can `run docker in "rootless" mode <https://docs.docker.com/engine/security/rootless/>`_.

2. Login to the MPCDF Gitlab registry using a predefined Struphy user and token::

    docker login gitlab-registry.mpcdf.mpg.de -u docker_api -p glpat--z6kJtobeG-xM_LdL6k6

3. Pull one of the following environment images (< 1 GB in size)::

    docker pull gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu:latest
    docker pull gitlab-registry.mpcdf.mpg.de/struphy/struphy/fedora:latest

4. Run the container::

    docker run -it gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu:latest

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

* Mirror default Struphy output to ``~/<dir>`` on the host machine::
    
    docker run -it -v ~/<dir>:<install_path>/io/out gitlab-registry.mpcdf.mpg.de/struphy/struphy/release

.. _docker_devs:

Docker for devs
^^^^^^^^^^^^^^^

Docker is well-suited for developers on any kind of platform. 
In order to interact with ``gitlab.mpcdf`` you need to mirror your **private ssh key** into the container 
with the ``-v`` option. For a ``rsa`` key this is done with::

    docker run -it -v ~/.ssh/id_rsa:/root/.ssh/id_rsa gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu:latest

On OS other than Linux ``~/.ssh/id_rsa`` must be replaced with the path to the private rsa key.

You can now install Struphy from source (see above). An installation in **editable mode** (``pip install -e .``) can only be done
within a :ref:`virtualenv`.

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
    module load gcc/12 openmpi/4 anaconda/3/2023.03 git pandoc
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

After installation, create a new VM via::

    multipass launch <version> --name <VM-name> --cpus 4 --memory 4G --disk 16G

where ``<version>`` is the ubuntu version and can be  ``jammy`` (22), ``impish`` (21), ``focal`` (20), ... ,
and ``<VM-name>`` is the custom name of the VM. The default user in the VM is named ``ubuntu``.

Quick info::

    multipass info --all

You can enter the VM via::

    multipass shell <VM-name>

and exit it with::

    exit

To shut down the VM, run the following command in the host machine::

    multipass stop <VM-name>

In order to have access to the `Struphy repository <https://gitlab.mpcdf.mpg.de/struphy/struphy>`_, 
generate an ssh key (if you do not already have one)::

    ssh-keygen

Then copy the key under ``.ssh/id_rsa.pub`` to your Gitlab profile. Continue with the installation of :ref:`source_install`.

You can mirror a folder ``<folder-name>`` to your host machine (for using a nice editor for instance).
``<folder-name>`` should be empty, as any content would be overwritten during ``mount``.
For this, create a new folder on your host machine and open a new terminal where you type::

    multipass mount /Path/to/Folder/on/Host/ <VM-name>:/home/ubuntu/<folder-name>/ 

(You should do this **before** you put anything in these folders.)

**For Apple Silicon Users running a Multipass VM:**

Since the in-house SoC's by Apple are based on arm64 some packages are not available to them, most notably ``vtk``.
You can nevertheless install vtk on your VM via::

    sudo apt install python3-vtk9

This will give you all functionality, however it will not be recognized by ``pip``. You therefore have to install ``gvec_to_python`` from source,
commenting out ``vtk`` under ``install_requires``. Then do::

    python3 -m pip install .

You will further have to comment out ``vtk`` and ``gvec_to_python`` from the ``pyproject.toml`` file in the struphy repository. You then proceed with::

    python3 -m pip install <option> .

**Graphical Output from a Multipass VM**

Multipass runs the VM in command line only but it is possible to get graphical output by using **X11Forwarding**
and a ``ssh`` connection. (Note: the following has only been tested on MacOS.)

For this procedure you need to install a `X Window System <http://www.linfo.org/x.html>`_ client on your computer
(e.g. `XQuartz <https://www.xquartz.org/>`_ on MacOS or `Xming <http://www.straightrunning.com/XmingNotes/>`_ on Windows).

On your server (the Ubuntu VM) change the following lines in ``/etc/ssh/sshd_config``::

    X11Forwarding yes
    X11UseLocalHost yes

You can also manually set a port to be used there. Then restart the ssh service using ``service sshd restart``.

On your computer (the client) find the sshd config file (on MacOS it is in ``/etc/ssh/sshd-config``) and set the following values::

    X11Forwarding yes
    X11UseLocalHost no

You can also change the default port there to the one you used above. Then go to the ssh config file
(on MacOS it is in ``/etc/ssh/ssh_config``) and add the parameter::

    ForwardX11Timeout 596h

This will prevent your virtual machine from disconnecting to the X display after a couple of minutes.

You need to find the ip adress of your virtual machine using e.g. ``multipass list``; take note of it.

It is necessary to connect to the virtual machine using multipass's ssh keys; they are stored
in ``/var/root/Library/Application\ Support/multipassd/ssh-keys/id_rsa`` on MacOS and
in ``./System32/config/systemprofile/AppData/Roaming/multipassd/ssh-keys/id_rsa`` on Windows.

Then you can connect to your virtual machine using the appropriate path to the multipass ssh keys and the ip adress of your VM::
    
    sudo ssh -X -i path/to/id_rsa ubuntu@<ip-adress>

Running commands that prompt a window to open, should result in the opening of your X Window System client.

**Connecting to the VM with VS Code**

To connect to your vm via ``ssh`` in VS Code, install the "Remote SSH" extension. In the lower left corner of VS Code a green button will appear;
click it and select "Open SSH configuration file", then choose the standard one (``Users/<username>/.ssh/config``). It should be changed to look like this:

Host ``<alias>``

    HostName ``<ip-adress>``

    User ubuntu

where you can choose the alias (e.g. the name of your vm), and you have to find the ip adress of your vm which is shown
when you type ``multipass list`` on your host machine.

Click again the green buttons in the lower left corner of VS Code, choose "Connect to Host", and select the alias you just gave your machine.

Finally, you can activate syntax highlighting, etc. by installing the "Python" extension in VS Code in your VM: Open the extensions window in VS Code
and click "Install in SSH: ``<alias>``"
