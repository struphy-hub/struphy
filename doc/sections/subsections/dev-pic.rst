PIC
---

Particle base class
^^^^^^^^^^^^^^^^^^^

.. autoclass:: struphy.pic.base.Particles
    :members:
    :special-members:
    :show-inheritance:
    :exclude-members: __init__


Particle subclasses
^^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.pic.particles
    :members:
    :special-members:
    :show-inheritance:


Particel-to-grid accumulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.pic.accumulation.particles_to_grid
    :members:
    :special-members:
    :show-inheritance:


Accumulation kernels
^^^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.pic.accumulation.kernels


Pusher class
^^^^^^^^^^^^

Marker evaluation setup
~~~~~~~~~~~~~~~~~~~~~~~

``Propagator.init_kernels`` and ``Propagator.eval_kernels`` are tuples of
``KernelSetup`` instances. Each setup names the callable, its additional
arguments, its destination marker columns, and the evaluation weights ``alpha``.
Initialization kernels run once at the start of each push. Evaluation kernels
run before every stage/iteration, after sorting particles using their spatial
``alpha`` weights.

For example, register a vector evaluation with three explicit output columns::

    from struphy.pic.pushing.kernel_setup import KernelSetup
    from struphy.pic.pushing.kernels.unit_b_1form import unit_b_1form

    self.add_init_kernel(
        KernelSetup(
            kernel=unit_b_1form,
            args=(self.derham.args_derham, b1, b2, b3),
            output_indices=(first_free_idx, first_free_idx + 1, first_free_idx + 2),
        )
    )

Output indices are absolute marker columns in component order. Use ``None`` to
skip a component, for example ``(20, None, 24)``. Scalars use a one-element tuple;
tensors use nine destinations in row-major order. This replaces the previous
``column_nr`` and ``comps`` arguments. Kernels receive an integer array of output
indices, with ``-1`` representing a skipped component.

For an evaluation during iteration, pass a setup with the desired ``alpha`` to
``self.add_eval_kernel``. A scalar weight applies to all six phase-space
coordinates. A tuple supplies three to six weights; omitted velocity weights
are zero. Initialization setups must use ``alpha=0`` (the default).

.. autoclass:: struphy.pic.pushing.kernel_setup.KernelSetup
    :members:

.. autoclass:: struphy.pic.pushing.pusher.Pusher
    :members:
    :special-members:
    :show-inheritance:
    :exclude-members: __init__
