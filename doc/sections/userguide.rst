.. _userguide:

Userguide
=========

This section explains the use of existing models and physics features, and how to conduct
numerical experiments with STRUPHY.
The basic commands are explained in :ref:`quickstart`. Here, a more in-depth description
of the command line tools is given.

The main point of interaction for the **regular user** is the STRUPHY parameters file. 
A default of this file is provided in ``<install_path>/io/inp/parameters.yml``. 
This file is discussed in :ref:`params_yml`.


.. _running_codes:

Running STRUPHY models
----------------------

The basic command is::

        struphy run <model> [<run options>]

where ``[<run options>]`` is optional. Currently available models are listed in :ref:`models`. Valid choices for ``<model>``
are the listed class names, as in ``struphy.models.models.<model>``, for example ``Maxwell``.
To add a new ``<model>``  please go to :ref:`developers` and follow section :ref:`add_model`. 

If no ``[<run options>]`` are specified, the input is taken from ``<install_path>/io/inp/parameters.yml``,
where ``<install_path>`` is obtained from::

        struphy -p

By default, simulation data is written to ``<install_path>/io/out/sim_1/``. 
Different input files and/or output folders in ``<install_path>/io/`` can be specified when launching a run
with the ``-i`` and/or ``-o`` flags, respectively::

        struphy run <model> -i my_params.yml -o my_folder/

Absolute paths can also be specified::

        struphy run <model> --input-abs abs_path_to_file.yml --output-abs abs_path_to_folder/

`Slurm <https://slurm.schedmd.com/documentation.html>`_ jobs can be submitted via batch scripts. 
Some default batch scripts are provided in ``<install_path>/io/batch``, 
e.g. ``batch/cobra_0160proc.sh``. Those can be called with the ``-b`` flag::

        struphy run <model> -b cobra_0160proc.sh

Again, an absolute path to a batch script can be specified::

        struphy run <model> --batch-abs abs_path_to_batch.sh

Small parallel runs for testing can be called via::

        struphy run <model> --mpi <int>

where ``<int>`` denotes the number of processes. 

The ``[<run options>]`` can be combined (the order is not important).


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
    :lines: 1-8

.. _time:

time
^^^^

.. literalinclude:: ../../struphy/io/inp/parameters.yml
    :language: yaml
    :lineno-start: 11
    :lines: 11-15


.. _geometry:

geometry
^^^^^^^^

Available mappings :math:`F:(\eta_1, \eta_2, \eta_3) \mapsto (x, y, z)` are listed in :ref:`avail_mappings`.

.. literalinclude:: ../../struphy/io/inp/parameters.yml
    :language: yaml
    :lineno-start: 18
    :lines: 18-108


.. _fields:

fields
^^^^^^

Available initial perturbations to be added on :ref:`backgrounds` are listed in :ref:`avail_inits`.

The keyword ``coords`` specifies the coordinate system in which the initial conditions shall be prescribed.
For instance, with ``type: Modes_sin`` it is possible to initialize a sine wave as a functin of :math:`(x, y, z)`
(before pullback) or as a function of :math:`(\eta_1, \eta_2, \eta_3)` (after pullback).

The keyword ``comps`` lets you define which components of a (vector-valued) p-form to perturb.
Suppose you have two fields in your simulation, a scalar-valued 0-form and a vector-valued 1-form.
In this case ``comps: [[False], [False, True, False]]`` would initialize only the second component
of the 1-form, all other field pertubrations would be set to zero.

The keys ``mhd_`` are needed in case the field variables describe some set of MHD equations. With ``mhd_u_space``
you can choose whether the momentum varibale is represented in :math:`H(\textnormal{div})` or in :math:`(H^1)^3` space.
With ``mhd_equilibrium`` you choose the background; the available choices are listed in :ref:`mhd_equil`.

.. literalinclude:: ../../struphy/io/inp/parameters.yml
    :language: yaml
    :lineno-start: 111
    :lines: 111-174


.. _kinetic:

kinetic
^^^^^^^

Each kinetic species has its own dictionary under its key, which is for instance ``hot_ions`` or ``electrons``.
We show here one such dictionary.

You can choose between full-orbit species (``Particles6D``) and gyro-/driftkinetic species (``Particles5D``).

Available background distribution functions are listed in :ref:`backgrounds`. The :ref:`vel_moments` of the backgrounds are
defined through ``moms_spec``. 
An entry ``0`` means that the corresponding moment is constant in logical space.
The corresponding value is then defined in ``moms_params``.

Available perturbation functions (added to the background) are listed in :ref:`avail_inits`.
If you choose for instance ``Modes_sin``, the moments specified in ``moms`` will be perturbed
according to :math:`n = n_0 + \delta n`, where :math:`\delta n` will be a sinusoidal mode 
defined by the given parameters.


.. literalinclude:: ../../struphy/io/inp/parameters.yml
    :language: yaml
    :lineno-start: 184
    :lines: 184-219


.. _solvers:

solvers
^^^^^^^

Available solvers are listed in :ref:`avail_solvers`.

Available preconditioners are listed in :ref:`preconditioner`.

.. literalinclude:: ../../struphy/io/inp/parameters.yml
    :language: yaml
    :lineno-start: 222
    :lines: 222-236


.. _pproc:

Post processing
---------------

Post processing means to prepare raw simulation data for diagnostics. The basic command for this is::

    struphy pproc sim_1 [<options> sim_2 ...]

Here, ``sim_1`` (and ``sim_2`` etc.) is relative to ``<install_path>/io/out/``. The command generates two kinds of data:

    1. ``vtk`` files for each time step in ``<install_path>/io/out/sim_1/vtk/`` 
    2. numpy arrays of evaluated fields and grids in ``<install_path>/io/out/sim_1/eval_fields/`` 

The called STRUPHY routine is

.. automodule:: struphy.diagnostics.post_processing
    :members:
    :undoc-members:


.. _diagnostics:

Diagnostics
-----------

Once the simulation data has been post processed, one can apply STRUPHY's diagnostics tools:

.. automodule:: struphy.diagnostics.diagn_tools
    :members:
    :undoc-members:

An examples of the whole workflow is in ``struphy/examples/example_diagnostics_1dfft.py``.


.. _profiling:

Code profiling
--------------

Finished runs (with the smae ``<model>``) can be profiled (=list individual run times of called sub-functions)::

        struphy profile sim_1 [sim_2 ...]

Here, ``sim_1`` is relative to ``<install_path>/io/out/``. By default the profiler searches only for functions including
one of the following strings:

.. literalinclude:: ../../struphy/diagnostics/profile_struphy.py
    :language: python
    :lineno-start: 10
    :lines: 10-10

If you want to profile all sub-functions, inculde the ``--all`` flag::

        struphy profile --all sim_1 [sim_2 ...]       




