PIC
---

Particle base class
^^^^^^^^^^^^^^^^^^^

.. autoclass:: struphy.pic.base.Particles
    :members:
    :special-members:
    :show-inheritance:
    :exclude-members: __init__


Marker array layout
^^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.pic.marker_layout
    :members:


Particle subclasses
^^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.pic.particles
    :members:
    :special-members:
    :show-inheritance:


Particle-to-grid accumulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Three classes deposit marker data on the FE spaces:

* :class:`~struphy.pic.accumulation.particles_to_grid.Accumulator` assembles particle matrices (and optionally
  vectors), :math:`\sum_p \Lambda^\alpha_{\mu, ijk}(\boldsymbol \eta_p) F_{p, \mu\nu}
  \Lambda^\alpha_{\nu, mno}(\boldsymbol \eta_p)`; ``operators[0]`` is the assembled matrix as a linear operator;
* :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector` accumulates vectors only;
* :class:`~struphy.pic.accumulation.particles_to_grid.MarkerOperator` is the matrix-free counterpart of
  ``Accumulator.operators[0]``: a linear operator that applies the particle matrix to coefficients :math:`s` by
  evaluating their spline function :math:`S_h` (see :class:`~struphy.feec.psydac_derham.SplineFunction`) at the
  markers and depositing :math:`F_p\, S_h(\boldsymbol \eta_p)`, in each iteration of a linear solver. The fillings
  :math:`F_p` are stored in the auxiliary marker columns by the accumulation kernel of the propagator (``*_mf``);
  the generic kernels ``marker_operator_symm_dot``, ``marker_operator_asym_dot`` and ``marker_operator_rank1_dot``
  apply symmetric (6 entries), antisymmetric (3 entries) and rank-one (4 entries) fillings, and
  ``marker_operator_pressure_dot`` the 3x3 block matrix of the pressure coupling (``n_blocks=3``).
  Assembling costs the products of all pairs of basis functions per marker, :math:`\mathcal O((p+1)^6)`, once per
  step; the matrix-free application costs one spline evaluation and one deposition per marker,
  :math:`\mathcal O((p+1)^3)`, per solver iteration. It is faster for 3D grids with higher degrees, expensive
  fillings or few solver iterations. All propagators that assemble particle matrices (except the disabled
  ``FaradayExtended``) have the option ``matrix_free``:
  :class:`~struphy.propagators.current_coupling_5d_curlb.CurrentCoupling5DCurlb`,
  :class:`~struphy.propagators.current_coupling_5d_density.CurrentCoupling5DDensity`,
  :class:`~struphy.propagators.vlasov_ampere_coupling.VlasovAmpereCoupling`,
  :class:`~struphy.propagators.efield_weights_coupling.EfieldWeightsCoupling`,
  :class:`~struphy.propagators.current_coupling_6d_density.CurrentCoupling6DDensity`,
  :class:`~struphy.propagators.current_coupling_6d_current.CurrentCoupling6DCurrent` and
  :class:`~struphy.propagators.pressure_coupling_6d.PressureCoupling6D`. It is the default in the 5D current
  couplings and in the pressure coupling, whose assembly is the most expensive (nine blocks).

.. automodule:: struphy.pic.accumulation.particles_to_grid
    :members:
    :special-members:
    :show-inheritance:


Accumulation kernels
^^^^^^^^^^^^^^^^^^^^

.. automodule:: struphy.pic.accumulation.kernels

.. autofunction:: struphy.pic.accumulation.kernels.vlasov_maxwell.vlasov_maxwell_kernels.vlasov_maxwell


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


Pusher kernels
^^^^^^^^^^^^^^

.. autofunction:: struphy.pic.pushing.kernels.push_v_with_efield.push_v_with_efield_kernels.push_v_with_efield
