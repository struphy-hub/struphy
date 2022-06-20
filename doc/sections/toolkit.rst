.. _toolkit:

Toolkit
=======

**This is a collection of modules/functions that help with the implementation of new models in Struphy.**


.. _derham:

Discrete Derham sequence (3d)
-----------------------------

Theoretical background can be found in the :ref:`appendix`.

.. autoclass:: struphy.psydac_api.psydac_derham.Derham

**Properties:**

.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.Nel
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.breaks
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.p
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.spl_kind
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.nq_pr
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.quad_order
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.comm
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.der_as_mat
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.domain_array
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.neighbours
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.index_array_N
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.index_array_D
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.V0
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.V1
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.V2
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.V3
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.V0vec
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.grad
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.curl
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.div
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.P0
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.P1
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.P2
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.P3
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.P0vec
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.M0
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.M1
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.M2
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.M3
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.M0vec
.. autoproperty:: struphy.psydac_api.psydac_derham.Derham.F

**Methods:**

.. automethod:: struphy.psydac_api.psydac_derham.Derham.assemble_M0
.. automethod:: struphy.psydac_api.psydac_derham.Derham.assemble_M1
.. automethod:: struphy.psydac_api.psydac_derham.Derham.assemble_M2
.. automethod:: struphy.psydac_api.psydac_derham.Derham.assemble_M3
.. automethod:: struphy.psydac_api.psydac_derham.Derham.assemble_M0_nonsymb
.. automethod:: struphy.psydac_api.psydac_derham.Derham.assemble_M1_nonsymb
.. automethod:: struphy.psydac_api.psydac_derham.Derham.assemble_M2_nonsymb
.. automethod:: struphy.psydac_api.psydac_derham.Derham.assemble_M3_nonsymb
.. automethod:: struphy.psydac_api.psydac_derham.Derham.assemble_M0vec_nonsymb


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

.. autoclass:: struphy.linear_algebra.schur_solver.Schur_solver
    :members: 


.. _preconditioner:

Preconditioners
---------------

.. automodule:: struphy.psydac_api.preconditioner
    :members: 

.. _mhd_ops:

MHD operators
-------------

.. automodule:: struphy.psydac_api.mhd_ops_pure_psydac
    :members: 






