.. _quickstart:

Quickstart
==========

Local machine
-------------

:abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` is command-line based, i.e. there exists the binary ``.local/bin/struphy`` (or similar in a virtual environment). 

**Remark:** for testing, we recommend to run ``struphy`` outside of the folder containing the :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` repository. 
In this way you make sure to use the installed and compiled (!) version. 
For developing you can run inside the repo without installing after each change.

To get help type::

    struphy -h

The default simulation (few time steps of ``lin_mhd`` code) is started with::

    struphy

You can run a different code with the option ``-r``::

    struphy -r <code_name>

You now might want to change the input parameters. For this you first determine the location of the parameter file:: 

    struphy -p

You will see something like::

    Parameter files (.yml) are in     <some_path>/lib/python3.8/site-packages/struphy/io/inp/
    Output files (.hdf5, .txt) are in <some_path>/lib/python3.8/site-packages/struphy/io/out/
    Your batch script are in          <some_path>/lib/python3.8/site-packages/struphy/io/batch/

The default parameter file for the code ``<code_name>`` is::

    <some_path>/lib/python3.8/site-packages/struphy/io/inp/<code_name>/parameters.yml

You can open and edit this file to change input parameters, or copy this file (stay in the same folder) and give a new name::

    cd <some_path>/lib/python3.8/site-packages/struphy/io/inp/<code_name>/
    cp parameters.yml <file_name>.yml

You can run :abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` with the new parameter file::

    struphy -r <code_name> -i <file_name>

In case you don't want to overwrite the output of the previous simulation, you can also specify a new output directory::

    struphy -r <code_name> -i <file_name> -o <dir_name>

The parameter file ``<file_name>`` will be atomatically copied to the ouptut folder ``<dir_name>`` at runtime.


TOK cluster (IPP Garching)
--------------------------


COBRA (IPP Garching)
--------------------


DRACO (IPP Garching)
--------------------
