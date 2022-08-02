Requirements
============

System
------

- Linux (Ubuntu 20 or higher). If you are using a different OS, please run a virtual machine, for instance the :ref:`multipass`.
- Access to the `Gitlab.mpcdf <https://gitlab.mpcdf.mpg.de/>`_ packages `Struphy <https://gitlab.mpcdf.mpg.de/clapp/hylife>`_ and `gvec_to_python <https://gitlab.mpcdf.mpg.de/spossann/gvec_to_python>`_. Contact: `stefan.possanner@ipp.mpg.de <stefan.possanner@ipp.mpg.de>`_


.. _linux_packages:

Linux packages
--------------

::

    sudo apt update -y
    sudo apt install -y gfortran gcc libblas-dev liblapack-dev libopenmpi-dev openmpi-bin libomp-dev libomp5
    sudo apt install -y libhdf5-openmpi-dev
    sudo apt install -y python3-pip python3-mpi4py


.. _python_packages:

Unpublished Python packages
----------------------------

- `gvec_to_python <https://gitlab.mpcdf.mpg.de/spossann/gvec_to_python>`_
- `psydac <https://github.com/pyccel/psydac>`_

If you do a :ref:`source_install`, the source of these packages is included in the STRUPHY source via sub-packaging.

In case of a :ref:`user_install`, you need to get the source via::

    curl -O --header "PRIVATE-TOKEN: glpat-5QH11Kx-65GGiykzR5xo" "https://gitlab.mpcdf.mpg.de/api/v4/projects/5368/jobs/1679220/artifacts/dist/gvec_to_python-0.1.2-py3-none-any.whl"

and::

    git clone https://github.com/pyccel/psydac.git


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
