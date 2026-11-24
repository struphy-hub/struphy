FEEC
----

Derham complex
^^^^^^^^^^^^^^

.. automodule:: struphy.feec.psydac_derham
    :members:
    :special-members:
    :show-inheritance:
    :exclude-members: __init__

Grids
^^^^^

.. automodule:: struphy.topology.grids
    :members:
    :special-members:
    :show-inheritance:
    :exclude-members: __init__

Mass Operators (bilinear forms)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.feec.mass
    :members:
    :special-members:
    :show-inheritance:
    :exclude-members: __init__

Quadrature
^^^^^^^^^^

.. automodule:: struphy.feec.quadrature
    :members:
    :show-inheritance:
    :exclude-members: __init__


Projection of spline functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.feec.projectors
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. automodule:: struphy.feec.basis_projection_ops
    :members:
    :special-members:
    :show-inheritance:
    :exclude-members: __init__

Kronecker approximations of operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.feec.kronecker_block
    :members:
    :special-members:
    :show-inheritance:
    :exclude-members: __init__


Preconditioners
^^^^^^^^^^^^^^^

.. automodule:: struphy.feec.preconditioner
    :members:
    :show-inheritance:
    :exclude-members: __init__

Polar 2D x 1D Kronecker solvers used by the polar preconditioners (see :ref:`polar_splines`):

.. automodule:: struphy.polar.kron_2dx1d
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. _polar_gpu:

GPU backend of the polar operators
""""""""""""""""""""""""""""""""""

On the CuPy backend, extraction, derivatives, Kronecker operators and the polar solves keep their
coefficients on the device. Factorization and mass-preconditioner assembly run once on the host;
cached device factors apply subsequent solves. Identical factors are shared across polar spaces
and projector/preconditioner instances of the same Derham. Matrix contents and solver settings
are matched exactly; updated weights use new factors, and unused entries are released. Polar MPI collectives support CUDA-aware device
buffers or host staging through ``cunumpy.mpi.mpi_buffer``. The feectools halo exchanges require
CUDA-aware MPI on real GPUs. ``SplineFunction.__call__`` dispatches CUDA kernels for marker
and dense/sparse grid evaluation. The fixed local-grid helper ``eval_tp_fixed_loc`` remains
host-only; distributed field-evaluation MPI handling and coverage are separate follow-ups.

Geometric multigrid preconditioner:

.. automodule:: struphy.linear_algebra.multigrid.preconditioner
    :members:
    :show-inheritance:
    :exclude-members: __init__


Linear solvers
^^^^^^^^^^^^^^

.. automodule:: struphy.linear_algebra.schur_solver
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. automodule:: struphy.linear_algebra.solver
    :members:
    :show-inheritance:
    :exclude-members: __init__

Exact solver for operators that are invariant under shifts in eta_3 (e.g. axisymmetric equilibria), used as the
preconditioner ``"ToroidalFourierPreconditioner"`` of the shear Alfven step:

.. automodule:: struphy.linear_algebra.toroidal_fourier
    :members:
    :show-inheritance:
    :exclude-members: __init__
