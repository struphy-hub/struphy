.. _toolkit:

Toolkit
=======

**This is a collection of modules/functions that help with the implementation of new models in Struphy.**


.. _model_base_class:

StruphyModel base class
-----------------------

Implemented models that inherit the base class are listed in :ref:`models`. 

.. autoclass:: struphy.models.base.StruphyModel
    :members:
    :undoc-members:


.. _prop_base_class:

Propagator base class
---------------------

Implemented propagators that inherit the base class are listed in :ref:`propagators`. 

.. autoclass:: struphy.propagators.base.Propagator
    :members:
    :undoc-members:
    :special-members:


.. _propagators:

Propagators
-----------

.. automodule:: struphy.propagators.propagators
    :members:
    :undoc-members:
    :exclude-members: push, variables


.. _derham:

Discrete Derham sequence (3d)
-----------------------------

Theoretical background can be found in the :ref:`appendix`.

.. autoclass:: struphy.psydac_api.psydac_derham.Derham
    :members:
    

.. _fields:

Finite element fields
---------------------

.. autoclass:: struphy.psydac_api.fields.Field
    :members: 


.. _particles:

Kinetic particles
-----------------

Full orbit
^^^^^^^^^^

.. autoclass:: struphy.pic.particles.Particles6D
    :members: 

Drift kinetic
^^^^^^^^^^^^^

.. autoclass:: struphy.pic.particles.Particles5D
    :members: 

.. _vel_moments:

Velocity moments
^^^^^^^^^^^^^^^^

.. automethod:: struphy.kinetic_background.moments_kernels.moments


.. _accumulators:

Particle accumulation functions 
-------------------------------

See :ref:`add_accum` for how to use these acumulation functions.

.. automodule:: struphy.pic.accum_kernels
    :members: 


.. _pushers:

Particle pushers 
----------------

.. automodule:: struphy.pic.pusher_kernels
    :members: 
    :undoc-members:


.. _linear_operators:

Linear operators
----------------

.. automodule:: struphy.psydac_api.linear_operators
    :members: 


.. _schur_solver:

Schur solver
------------

.. autoclass:: struphy.linear_algebra.schur_solver.SchurSolver
    :members: 


.. _preconditioner:

Preconditioner
--------------

.. automodule:: struphy.psydac_api.preconditioner
    :members: 
    :exclude-members: FFTSolver, is_circulant

.. _mhd_ops:

MHD operators
-------------

.. automodule:: struphy.psydac_api.mhd_ops_pure_psydac
    :members: 

.. _avail_solvers:

Iterative linear solvers
------------------------

.. automethod:: psydac.linalg.iterative_solvers.pcg

.. automodule:: struphy.linear_algebra.iterative_solvers
    :members: 
    :undoc-members:








