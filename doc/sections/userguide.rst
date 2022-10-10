.. _userguide:

Userguide
=========

Basic struphy commands are explained in :ref:`quickstart`. Here, a more in-depth description
of struphy use is given. 

The main point of interaction for the user is the struphy parameter file; a default is provided in ``<install_path>/io/inp/parameters.yml``. 
This file is discussed in :ref:`params_yml`.


.. _running_codes:

Running struphy models
----------------------

The help for running struphy models can be accessed with::

    struphy run --help

The basic command is::

    struphy run MODEL [OPTIONS]

See :ref:`models` for valid ``MODEL`` names, which
are the listed class names, for example ``Maxwell``. 

If no ``[OPTIONS]`` are specified, the input is taken from ``<install_path>/io/inp/parameters.yml``,
where ``<install_path>`` is obtained from::

    struphy -p

By default, simulation data is written to ``<install_path>/io/out/sim_1/``. 
Different input files and/or output folders in ``<install_path>/io/`` can be specified
with the ``-i`` and/or ``-o`` flags, respectively::

    struphy run MODEL -i my_params.yml -o my_folder/

Absolute paths can also be specified::

    struphy run MODEL --input-abs path/to/file.yml --output-abs path/to/folder/

`Slurm <https://slurm.schedmd.com/documentation.html>`_ jobs can be submitted via batch scripts. 
Some default batch scripts are provided in ``<install_path>/io/batch``. 
A model is run as a slurm job with the ``-b`` flag::

        struphy run MODEL -b cobra_0160proc.sh

Again, an absolute path to a batch script can be specified::

        struphy run MODEL --batch-abs path/to/batch.sh

Small parallel runs for testing can be called via::

        struphy run MODEL --mpi <int>

where ``<int>`` denotes the number of processes. 

``[OPTIONS]`` can be combined (the order is not important).


.. _params_yml:

Setting simulation parameters
-----------------------------

The default parameter file ``<install_path>/io/inp/parameters.yml`` can be used for all available :ref:`models`. 
The user finds the simulation parameters categorized into the following top level keywords:

* :ref:`grid`
* :ref:`time`
* :ref:`geometry`
* :ref:`fields`
* :ref:`kinetic`
* :ref:`solvers`


.. _grid:

grid
^^^^

.. literalinclude:: ../../struphy/io/inp/parameters.yml
    :language: yaml
    :lineno-start: 1
    :lines: 1-7

.. _time:

time
^^^^

.. literalinclude:: ../../struphy/io/inp/parameters.yml
    :language: yaml
    :lineno-start: 10
    :lines: 10-14


.. _geometry:

geometry
^^^^^^^^

Available mappings :math:`F:(\eta_1, \eta_2, \eta_3) \mapsto (x, y, z)` are listed in :ref:`avail_mappings`.

.. literalinclude:: ../../struphy/io/inp/parameters.yml
    :language: yaml
    :lineno-start: 17
    :lines: 17-133


.. _fields:

fields
^^^^^^

There are three sub-keywords in ``fields``:

1. ``init``: defines the initial conditions of field variables. 

    Available initial perturbations to be added on :ref:`mhd_equil` are listed in :ref:`avail_inits`.

    The keyword ``coords`` must be set for the types ``ModesSin`` and ``ModesCos``.
    It specifies the coordinate system of the initial conditions.
    For instance, with ``type: ModesSin`` it is possible to initialize a sine wave as a functin of :math:`(x, y, z)`
    (before pullback) or as a function of :math:`(\eta_1, \eta_2, \eta_3)` (after pullback).

    The keyword ``comps`` must be set for the types ``ModesSin``, ``ModesCos`` and ``TorusModesSin``.
    It lets you define which components of a (vector-valued) p-form to perturb.
    Suppose you have two fields in your simulation, a scalar-valued 0-form and a vector-valued 1-form.
    In this case ``comps: [[False], [False, True, False]]`` would initialize only the second component
    of the 1-form, all other field pertubrations would be set to zero. 
    All components with ``True`` are initialized with the same ``type``.

    Types other than ``ModesSin``, ``ModesCos`` and ``TorusModesSin`` are model-specific.

2. ``mhd_u_space``: choose whether the momentum varibale in MHD models is represented in :math:`H(\textnormal{div})` or in :math:`(H^1)^3`.


3. ``mhd_equilibrium``: choose an MHD background (:math:`\mathbf{B}_0,\, p_0` and :math:`\rho_0` profiles). Available choices are listed in :ref:`mhd_equil`.


.. literalinclude:: ../../struphy/io/inp/parameters.yml
    :language: yaml
    :lineno-start: 136
    :lines: 136-208


.. _kinetic:

kinetic
^^^^^^^

Each kinetic species has its own sub-keyword which refers to the name of the species; 
in the example below there is just one species named ``hot_ions``, but in principle an arbitrary 
number of species can be addressed.

There are five sub-keywords for each kinetic species:

1. ``markers``: numerical marker parameters

2. ``attributes``: physical parameters of the species

3. ``background``: available background distribution functions are listed in :ref:`backgrounds`. The :ref:`vel_moments` of the backgrounds are defined through ``moms_spec``. An entry ``0`` means that the corresponding moment is constant in logical space. The corresponding value is then defined in ``moms_params``.

4. ``perturbations``: perturbations of velovity moments of the distribution function (see ``fields/init``).

5. ``save_data``: specify number of markers to be saved and under ``f`` possible binning of the distribution function (in case not all markers are saved).


.. literalinclude:: ../../struphy/io/inp/parameters.yml
    :language: yaml
    :lineno-start: 211
    :lines: 211-250


.. _solvers:

solvers
^^^^^^^

Define some parameters of the linear solvers used in a model. Each solver has its own sub-key ``solver_1``, ``solver_2``, etc.
In principle, there can be an arbitrary number of solvers in a model. Which solver refers to which propagator
has to be deduced from the model's source code.

Available solvers are listed in :ref:`avail_solvers`. Available preconditioners are listed in :ref:`preconditioner`.

.. literalinclude:: ../../struphy/io/inp/parameters.yml
    :language: yaml
    :lineno-start: 253
    :lines: 253-267


.. _pproc:

Post processing
---------------

Post processing means to prepare raw simulation data for diagnostics. The basic command for this is::

    struphy pproc sim_1 [<options> sim_2 ...]

Here, ``sim_1`` (and ``sim_2`` etc.) is relative to ``<install_path>/io/out/``. The command generates two kinds of data:

    1. ``vtk`` files for each time step in ``<install_path>/io/out/sim_1/vtk/`` 
    2. numpy arrays of evaluated fields and grids in ``<install_path>/io/out/sim_1/eval_fields/`` 


.. _diagnostics:

Diagnostics
-----------

Once the simulation data has been post processed, one can apply struphy's diagnostics tools:

.. automodule:: struphy.diagnostics.diagn_tools
    :members:
    :undoc-members:

An examples of the whole workflow is in ``struphy/examples/example_diagnostics_1dfft.py``.


.. _profiling:

Code profiling
--------------

Finished runs (with the smae ``MODEL``) can be profiled (=list individual run times of called sub-functions)::

        struphy profile sim_1 [sim_2 ...]

Here, ``sim_1`` is relative to ``<install_path>/io/out/``. By default the profiler searches only for functions including
one of the following strings:

.. literalinclude:: ../../struphy/post_processing/profile_struphy.py
    :language: python
    :lineno-start: 10
    :lines: 10-10

If you want to profile all sub-functions, inculde the ``--all`` flag::

        struphy profile --all sim_1 [sim_2 ...]       




