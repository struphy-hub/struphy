.. _running_codes:

The Struphy CLI
---------------

With Struphy it is simple to solve PDEs from the CLI (Command Line Interface).
A list of available model PDEs and run options can be accessed with::

    struphy run -h

The list of models is automatically updated whenever a developer adds a new model PDE.
All models are continuously auto-tested with many different input options. 

The basic command for solving a PDE is::

    struphy run [OPTIONS] MODEL 

If no ``[OPTIONS]`` are specified, a default parameter file ``params_MODEL.yml``
is created in the current input path (on prompt).
If we choose to do so, the file is created (or overwritten) and the code exits. 
To run the model with default parameters we can now type::

    struphy run MODEL

The current I/O paths can be obtained from::

    struphy -p

The I/O paths can be changed with the commands:: 

    struphy --set-i <path>
    struphy --set-o <path>

If ``<path>`` is ``.``, the current working directory is selected. 
The default I/O paths are ``<install_path>/io/inp`` and ``<install_path>/io/out``, respectively.
We can always revert back to the default I/O paths::

    struphy --set-i d
    struphy --set-o d

All parameters for a Struphy run are set in the parameter file ``params_MODEL.yml``.
We can edit/copy/rename this file to our liking and pass it via ``-i <new_name>.yml`` to the run command.
All models in Struphy share a set of "core" parameters which are discussed in :ref:`params_yml`.
Model specific parameters can be seen under the keyword ``options`` in the parameter file.
Each model species has its own ``options``, which are hard-coded as class methods.
We can inspect the model-specific options from the console::

    struphy params MODEL --options

Available options appear as lists in the options dictionary. 
The first list entry is the respective default option. 
The key structure of the options dictionary is exactly the same as in the parameter file.

By default, Struphy simulation data is written to ``sim_1/`` in the current output path. 
Different input files and/or output folders with respect the current I/O paths can be specified
with the ``-i`` and/or ``-o`` options, respectively::

    struphy run MODEL -i my_params.yml -o my_folder

Absolute paths (unrelated to the current I/O paths) can also be specified::

    struphy run MODEL --input-abs path/to/file.yml --output-abs path/to/folder

Small parallel runs (for debugging) can be called via::

    struphy run MODEL --mpi <int>

where ``<int>`` denotes the number of mpi processes. 
`Slurm <https://slurm.schedmd.com/documentation.html>`_ jobs can be submitted via batch scripts. 
The path to your batch scripts can be set via::

    struphy --set-b <path>

The logic of this command is the same as for ``--set-i`` above.
A model is run as a slurm job with the ``-b`` flag::

    struphy run MODEL -b my_batch_script.sh

Again, an absolute path to a batch script can be specified::

    struphy run MODEL --batch-abs path/to/batch.sh

Struphy ``[OPTIONS]`` can be combined (the order is not important).