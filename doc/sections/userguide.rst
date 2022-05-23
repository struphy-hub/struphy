.. _userguide:

Userguide
=========


Compiling kernels
-----------------

:abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` uses pre-compiled FORTRAN kernels to achieve better performance, 
based on the library `pyccel <https://github.com/pyccel/pyccel>`_; launch compilation with::
        
        struphy compile <options> 

Options: 

* ``--no-openmp``: do not use the openmp flag during compilation.
* ``--delete``: remove all Fortran and shared object files (Python scripts will be called instead).


.. _running_codes:

Running codes
-------------

Run a Struphy code::

        struphy run <model> [<run options>]

Currently available models (``<model>``) in Struphy:

========================== =============================================================== ================================
Model                      Description                                                     
========================== =============================================================== ================================
``maxwell``                Maxwell equations in vacuum.                                    :ref:`equations <maxwell>`
========================== =============================================================== ================================

To add a new ``<model>``  please go to :ref:`developers` and follow section :ref:`add_model`. 
If no ``[<run options>]`` are specified, the input is taken from ``parameters.yml`` in the path listed when typing::

        struphy -p

which leads to an output like::

    Struphy installation path: $HOME/git_repos/struphy/struphy
    default input:             $HOME/git_repos/struphy/struphy/io/inp
    default output:            $HOME/git_repos/struphy/struphy/io/out
    template batch scripts:    $HOME/git_repos/struphy/struphy/io/batch

Output is written to the folder ``sim_1`` in the listed path, which is the **default**. 
Different input files or output folders **in the default paths** can be specified::

        struphy run <model> -i file_name -o folder_name

Absolute paths can be specified::

        struphy run <model> --input-abs abs_path_to_file --output-abs abs_path_for_output

`Slurm <https://slurm.schedmd.com/documentation.html>`_ jobs can be submitted via batch scripts. 
Some default batch scripts are provided in the default folder under ``\batch``, 
e.g. ``batch/cobra_0160proc.sh``. Those can be called via::

        struphy run <model> -b cobra_0160proc.sh

Again, an absolute path to a batch script can be specified::

        struphy run <model> --batch-abs abs_path_to_batch

Small parallel runs for testing can be called via::

        struphy run <model> --mpi <int>

where ``<int>`` denotes the number of processes. 

The above options can be combined (the order is not important).


.. _sim_params:

Simulation parameters
---------------------

The parameters for a run can be set in a single file usually called ``parameters.yml`` 
(can be any name specified with the run options ``-i`` or ``--input-abs``). 
A default is provided in ``struphy/io/inp/parameters.yml``. All models can run with the same default file.

When a Struphy code imports ``parameters.yml``, the parameters are given within a nested dictionary. 
The top level ``keys`` are ``geometry``, ``fields``, ``particles``, ``grid``, ``time`` and ``solvers``.


.. _profiling:

Code profiling
--------------

A finished run can be profiled (list individual run times of called sub-functions)::

        struphy profile <sim1>

Here, ``<sim1>`` is the absolute path to the output folder. Multiple runs **of the same model** can be compared::

        struphy profile <sim1> <sim2> ...

By default the profiler searches only for functions containing ``assemble_``, ``update`` or ``__call__`` in their name.
If you want to profile all sub-functions::

        struphy profile --all <sim1>        



