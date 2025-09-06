.. _install:

Install
=======

.. _require:

Requirements
------------

- Python >=3.10 
- C or Fortran compiler like gcc, gfortran
- Linear algebra packages BLAS and LAPACK
- An MPI library like open-mpi, mpich
- OpenMP
- **Struphy is not currently supported** with `Anaconda <https://www.anaconda.com/>`_.

In order not to interfere with existing Python packages, 
it is highly recommended to install Struphy in a `Virtual Python environment <https://pypi.org/project/virtualenv/>`_::

    python3 -m pip install -U virtualenv

Create virtual environment::

    python3 -m venv <name>

Launch the virtual environment (from the location where created)::

    source <name>/bin/activate

Check the pre-installed packages and upgrade ``pip``::

    pip list
    pip install --upgrade pip

Continue with the Struphy installation; the quickest install is via :ref:`PyPI <pypi_install>`::
    
    pip install --no-cache-dir -U struphy
    struphy compile

When finished, you can deactivate the virtual environment::

    deactivate

The environment is stored in the folder ``<name>`` and can be re-activated any time for working with Struphy.


.. _install_modes:

Install modes
-------------

Struphy can be installed in the following ways:

1. From :ref:`PyPI <pypi_install>`, for running the latest release
2. From :ref:`source <source_install>`, for running or adding code 
3. Using :ref:`Docker images <docker_install>`, also suited :ref:`for developers <docker_devs>`

In case you encounter problems during install, check out :ref:`trouble_shoot`.


.. _sample_envs:

Sample environments
-------------------

Some Linux environments on which Struphy is continuously tested are:

.. tab-set::

    .. tab-item:: Ubuntu

        .. code-block::

            apt install -y software-properties-common
            add-apt-repository -y ppa:deadsnakes/ppa
            apt update -y
            apt install -y python3-pip 
            apt install -y python3-venv 
            apt install -y gfortran gcc 
            apt install -y liblapack-dev libopenmpi-dev 
            apt install -y libblas-dev openmpi-bin 
            apt install -y libomp-dev libomp5 
            apt install -y git
            apt install -y pandoc

    .. tab-item:: OpenSuse

        .. code-block::

            zypper refresh
            zypper install -y python311 python311-devel
            zypper install -y python311-pip python3-virtualenv
            zypper install -y gcc-fortran gcc 
            zypper install -y lapack-devel openmpi-devel 
            zypper install -y blas-devel openmpi 
            zypper install -y libgomp1 
            zypper install -y git 
            zypper install -y pandoc 
            zypper install -y vim 
            zypper install -y make

    .. tab-item:: AlmaLinux

        .. code-block::

            - yum install -y wget yum-utils make openssl-devel bzip2-devel libffi-devel zlib-devel
            - yum update -y 
            - yum clean all 
            - yum install -y gcc 
            - yum install -y gfortran  
            - yum install -y openmpi openmpi-devel  
            - yum install -y libgomp 
            - yum install -y git 
            - yum install -y environment-modules 
            - yum install -y sqlite-devel
            - wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz 
            - tar xzf Python-3.10.14.tgz 
            - cd Python-3.10.14 
            - ./configure --with-system-ffi --with-computed-gotos --enable-loadable-sqlite-extensions 
            - make -j ${nproc} 
            - make altinstall 
            - cd ..
            - export PATH="/usr/lib64/openmpi/bin:$PATH"
            - mv /usr/local/lib/libpython3.10.a libpython3.10.a.bak

    .. tab-item:: Fedora-CentOS-RHEL

        .. code-block::

            dnf install -y wget yum-utils make openssl-devel bzip2-devel libffi-devel zlib-devel
            dnf update -y  
            dnf install -y gcc
            dnf install -y gfortran  
            dnf install -y blas-devel lapack-devel  
            dnf install -y openmpi openmpi-devel 
            dnf install -y libgomp 
            dnf install -y git 
            dnf install -y environment-modules 
            dnf install -y python3-mpi4py-openmpi 
            dnf install -y pandoc
            dnf install -y sqlite-devel
            wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz 
            tar xzf Python-3.10.14.tgz 
            cd Python-3.10.14 
            ./configure --with-system-ffi --with-computed-gotos --enable-loadable-sqlite-extensions 
            make -j ${nproc} 
            make altinstall 
            cd .. 
            mv /usr/local/lib/libpython3.10.a libpython3.10.a.bak
            module load mpi/openmpi-$(arch)

    .. tab-item:: MacOS

        .. code-block::

            brew update
            brew install python3
            brew install gcc
            brew install openblas
            brew install lapack
            brew install open-mpi
            brew install libomp
            brew install git
            brew install pandoc

More details on the continuous test environments can be found in `.gitlab-ci.yml <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/.gitlab-ci.yml>`_
in the test stage.

On **Windows systems** we recommend the use of a virtual machine, for instance the :ref:`multipass`.


.. _trouble_shoot:

Trouble shooting
----------------

Install problems
^^^^^^^^^^^^^^^^

* Make sure that you can ``pip install -U mpi4py``.
* `mpi4py>=4.1.0` `provides binaries <https://github.com/mpi4py/mpi4py/releases/tag/4.1.0>`_` for common platforms. In case of "exotic" platforms you might try `pip install -U mpi4py --no-binary mpi4py`
* In many cases installing ``apt install openmpi-devel`` solves a problem with missing headers.
* On Mac OS, you can try to install the command line tools (160 MB) ``xcode-select --install``.
* Struphy is not supported with Conda; however, in case you insist you might try::

    conda install mpich
    conda install gxx_linux-64

Compilation problems
^^^^^^^^^^^^^^^^^^^^

* If compilation fails, ``struphy compile --delete`` can help to clean up the environment.
* 
  It can happen that during ``struphy compile`` you encounter::

    A module that was compiled using NumPy 1.x cannot be run in
    NumPy 2.2.1 as it may crash. To support both 1.x and 2.x
    versions of NumPy, modules must be compiled with NumPy 2.0.
    Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

  At the moment this error is resolved with::

    pip install numpy==1.26.4


.. _pypi_install:

PyPI
----

Install package::

    pip install --no-cache-dir -U struphy

For running (parallel) tests:

.. tab-set::

    .. tab-item:: bash

        .. code-block::

            pip install --no-cache-dir -U struphy

    .. tab-item:: zsh

        .. code-block::

            pip install --no-cache-dir -U struphy

Compile kernels in ``c`` (default)::

    struphy compile

You can compile in Fortran via::
    
    struphy compile --language=fortran

Other compile options can be accessed with::
    
    struphy compile -h


.. _source_install:

Source
------

Clone the `Struphy repository <https://gitlab.mpcdf.mpg.de/struphy/struphy>`_::

    git clone https://gitlab.mpcdf.mpg.de/struphy/struphy.git
    cd struphy

Update pip::

    pip install --upgrade pip

Install Struphy:

.. tab-set::

    .. tab-item:: Development mode

        .. code-block::

            pip install --no-cache-dir -e .

    .. tab-item:: Python environment

        .. code-block::

            pip install --no-cache-dir .

    .. tab-item:: .local/lib

        .. code-block::

            pip install --no-cache-dir --user .

Struphy features optional dependencies:

.. tab-set::

    .. tab-item:: bash

        * ``pip install .[phys]`` enables some physics packages, see `pyproject.toml <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/pyproject.toml?ref_type=heads>`_
        * ``pip install .[dev]`` enables the development environment (testing, linting, formatting)
        * ``pip install .[doc]`` enables :ref:`change_doc`

        These can also be combined, as for example in ``pip install .[phys,dev,doc]``.

    .. tab-item:: zsh

        * ``pip install ."[phys]"`` enables some physics packages, see `pyproject.toml <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/pyproject.toml?ref_type=heads>`_
        * ``pip install ."[dev]"`` enables the development environment (testing, linting, formatting)
        * ``pip install ."[doc]"`` enables :ref:`change_doc`

        These can also be combined, as for example in ``pip install ."[phys,dev,doc]"``.

Compile kernels in ``c`` (default)::

    struphy compile

You can compile in Fortran via::
    
    struphy compile --language=fortran

Other compile options can be accessed with::
    
    struphy compile -h


.. _args:

Argument completion
-------------------

Struphy provides console argument completion through the package `argcomplete <https://github.com/kislyuk/argcomplete>`_.
In order to enable it, make sure to have `bash <https://www.cyberciti.biz/faq/add-bash-auto-completion-in-ubuntu-linux/>`_ 
or `zsh <https://dev.to/zeromeroz/setting-up-zsh-and-oh-my-zhs-with-autocomplete-plugins-1nml>`_ tab comlpetion enabled. 
After Struphy installation type::

    activate-global-python-argcomplete

and follow the instructions. For activation you need to restart your shell, for instance with ``exec bash``.


.. _docker_install:

Docker
------

With this installation you will be able to run Struphy in a `docker container <https://www.docker.com/resources/what-container/>`_, 
encapsulated from your host machine.
The container is launched from an `image <https://docs.docker.com/get-started/overview/#docker-objects>`_ 
which you can download and run immediately, irrespective of your architecture and OS.

`Link to Struphy's container registry <https://gitlab.mpcdf.mpg.de/struphy/struphy/container_registry>`

Check out the `corresponding docker files <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/tree/devel/docker?ref_type=heads>`_.

.. _user_install:

User install
^^^^^^^^^^^^

To use Struphy via docker, perform the following steps:

1. `Install Docker Desktop <https://docs.docker.com/desktop/>`_ and start it. 

.. tab-set::

    .. tab-item:: Linux

        If you do not want to preface the docker command with ``sudo``, you can 
        `create a Unix group <https://docs.docker.com/engine/install/linux-postinstall/>`_ 
        called ``docker`` and add your user to it.
        If you are uncomfortable with running `sudo`, you can `run docker in "rootless" mode <https://docs.docker.com/engine/security/rootless/>`_.

    .. tab-item:: MacOS

        It is recommended to read the `Mac OS permission requirements <https://docs.docker.com/desktop/mac/permission-requirements/>`_.
        (REMARK: older versions of Mac OS may require `older docker desktop versions <https://docs.docker.com/desktop/release-notes/#docker-desktop-471>`_.)

    .. tab-item:: Windows

        It is recommended to read the `Windows permission requirements <https://docs.docker.com/desktop/windows/permission-requirements/>`_

2. Login to the MPCDF Gitlab registry using a predefined Struphy user and token::

    TOKEN=glpat-YzkatDxAYT1JZtyj9KjS; echo "$TOKEN" | docker login gitlab-registry.mpcdf.mpg.de -u struphy-hub-read-registry --password-stdin
    docker login gitlab-registry.mpcdf.mpg.de -u struphy-hub-read-registry -p glpat-YzkatDxAYT1JZtyj9KjS

3. Pull one of the availabale images listed above (< 1 GB in size), for instance::

    docker pull gitlab-registry.mpcdf.mpg.de/struphy/struphy/struphy_ubuntu_python_3_11

4. Run the container::

    docker run -it gitlab-registry.mpcdf.mpg.de/struphy/struphy/struphy_ubuntu_python_3_11

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
within a virtual environment.

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

Struphy is periodically tested on the `MPCDF HPC facilities <https://docs.mpcdf.mpg.de/doc/computing/index.html>`_.
Tests are performed with the `available MPCDF images <https://docs.mpcdf.mpg.de/doc/data/gitlab/gitlabrunners.html#docker-images-for-ci-with-mpcdf-environment-modules>`_.
The modules loaded in these tests can be found in Struphy's `.gitlab-ci.yml <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/.gitlab-ci.yml?ref_type=heads#L82>`_.

A common installation looks like this

1. Load necessary modules and create a virtual environment::

    module purge
    module load gcc/14 openmpi/5.0 python-waterboa/2024.06 git pandoc graphviz/8
    pip install -U virtualenv
    python3 -m venv <some_name>
    source <some_name>/bin/activate
    python3 -m pip install --upgrade pip

2. Install Struphy by not using the binaries of `mpi4py` (see install methods from above: :ref:`pypi_install` or :ref:`source_install`):

    pip install -U struphy --no-binary mpi4py

3. When using slurm, include the following lines in your BATCH script::

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

This will give you all functionality, however it will not be recognized by ``pip``. Then do::

    python3 -m pip install .

You will further have to comment out ``vtk`` from the ``pyproject.toml`` file in the struphy repository. You then proceed with::

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
