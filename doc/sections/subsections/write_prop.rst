.. _write_prop:

How to write a Propagator
-------------------------

A :class:`~struphy.propagators.base.Propagator` is where the magic
of your numerical method happens. It is where you implement the numerical
update rules for variables of your model.


.. _prop_basics:

Basics
======

Each Propagator must have at least the two magic methods ``__init__(self, ...)``
and ``__call__(self, dt)``. The former defines the variables to be updated and instantiates
all necessary objects and data structures used during the simulation.
The latter takes the time step ``dt`` as an argument and updates the variables 
by one time step.

The ``__init__(self, args, *, kwargs)`` constructor can take arguments ``args`` and keyword 
arguments ``kwargs``, separated by a star ``*``. The ``args`` are the variables to be updated, 
whereas the ``kwargs`` are parameters or options needed in the update step.
The latter usually come from the input parameter file, as described in :ref:`add_prop`.
As an example, let us look at :class:`struphy.propagators.propagators_coupling.VlasovAmpere`::

    def __init__(self,
        e: BlockVector,
        particles: Particles6D,
        *,
        c1: float = 1.,
        c2: float = 1.,
        solver=options(default=True)['solver'],
        ):

Here, a variable ``e`` of type :class:`~psydac.linalg.block.BlockVector` and a variable ``particles`` of type :class:`~struphy.pic.particles.Particles6D`
are updated by the Propagator.
Check available :ref:`data_structures` for possible types of variables.
The parameters of this Propagator are the two floats ``c1`` and ``c2``, as well as some ``solver``
parameters whose default is taken from the ``options`` attribute discussed below.


.. _prop_basics:

Accessing Struphy objects
=========================

Within a propagator, and within the ``__init__(self, ...)`` constructor in particular,
you have access to several features like spline spaces, geometry, mass matrices, Hodge operators
and projected MHD equilibria:

* ``self.derham`` returns the simulation instance of :class:`~struphy.feec.psydac_derham.Derham`
* ``self.domain`` returns the simulation instance of :class:`~struphy.geometry.base.Domain`
* ``self.mass_ops`` returns the simulation instance of :class:`~struphy.feec.mass.WeightedMassOperators`
* ``self.basis_ops`` returns the simulation instance of :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperators`
* ``self.projected_mhd_equil`` returns the simulation instance of :class:`~struphy.fields_background.projected_equils.ProjectedMHDequilibrium`

You can check the Struphy :ref:`api` for more details on how to use these objects.
As an example, here is a small snippet of :class:`struphy.propagators.propagators_fields.Maxwell`::

    # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
    _A = self.mass_ops.M1

    self._B = -1/2 * self.derham.curl.T @ self.mass_ops.M2
    self._C = 1/2 * self.derham.curl


.. _prop_particles:

Particle push and deposition
============================

Due to the large number :math:`N \gg 1` of particles in a PIC simulation, all particle
loops in Struphy are accelerated with `pyccel <https://github.com/pyccel/pyccel>`_,
and can be distributed with MPI.

In order to facilitate the integration of particle loops in Propagators,
Struphy provides the following "wrapper classes":

* :class:`~struphy.pic.pushing.pusher.Pusher` for solving particle ODEs (explicit, implicit, iterative)
* :class:`~struphy.pic.accumulation.particles_to_grid.Accumulator` for particle deposition into a matrix, and optionally a vector
* :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector` for particle deposition into a vector only

These wrapper classes take care of MPI communication, nonlinear iteration, multi-stage time stepping, among other
technical issues. All the developer has to provide is the particle loop **in the same way as for a single process, 
a single iteration, or a single stage of the chosen algorithm.**

The single-process/single-stage/single-iteration particle loop must be written into a so-called "kernel file".
The above wrapper classes then take the kernel as an input
(plus some other arguments discussed below) and take care of the mentioned abstractions. 
As an example, let us look at the Propagator :class:`struphy.propagators.propagators_markers.PushEta` 
for position update in logical space,

.. math::

    \frac{\textnormal d \boldsymbol \eta_p(t)}{\textnormal d t} = DF^{-1}(\boldsymbol \eta_p(t)) \,\mathbf v_p\,,

by means of an explicit Runge-Kutta scheme obtained from a :class:`~struphy.pic.pushing.pusher.ButcherTableau`::

    def __init__(self,
                 particles: Particles,
                 *,
                 algo: str = options(default=True)['algo'],
                 ):

        # base class constructor call
        super().__init__(particles)

        # get kernel
        kernel = pusher_kernels.push_eta_stage

        # define algorithm
        butcher = ButcherTableau(algo)
        args_kernel = (butcher.a,
                       butcher.b,
                       butcher.c)

        # instantiate Pusher
        self._pusher = Pusher(particles,
                              kernel,
                              args_kernel,
                              self.derham.args_derham,
                              self.domain.args_domain,
                              alpha_in_kernel=1.,
                              n_stages=butcher.n_stages,
                              mpi_sort='last')

    def __call__(self, dt):
        # push markers
        self._pusher(dt)

Some remarks to this code:

1. This Propagator updates one variable ``particles`` of type :class:`~struphy.pic.base.Particles`
2. The :class:`~struphy.propagators.base.Propagator` base class has to be instantiated with a call to ``super().__init__()`` 
3. The kernel (for a single RK stage particle loop) is ``push_eta_stage`` from the kernel file ``pusher_kernels.py``
4. The coefficients of the RK algorithm are retrieved from the :class:`~struphy.pic.pushing.pusher.ButcherTableau`; the ``algo`` is a keyword argument of the Propagator
5. The :class:`~struphy.pic.pushing.pusher.Pusher` is instantiated as ``self._pusher`` with its necessary arguments, in particular the ``particles`` object and the ``kernel``
6. The object ``self._pusher`` is callable and called in the ``__call__()`` method of the Propagator with the time step ``dt`` as the single argument.

Particle depositions have a similar structure. 
As an example, let us look at the Propagator :class:`struphy.propagators.propagators_coupling.VlasovAmpere` 
for the charge deposition in the Vlasov-Ampère system::

    def __init__(self,
                 e: BlockVector,
                 particles: Particles6D,
                 *,
                 c1: float = 1.,
                 c2: float = 1.,
                 solver=options(default=True)['solver']):

        super().__init__(e, particles)

        # get accumulation kernel
        accum_kernel = accum_kernels.vlasov_maxwell

        # Initialize Accumulator object
        self._accum = Accumulator(particles,
                                  'Hcurl',
                                  accum_kernel,
                                  self.mass_ops,
                                  self.domain.args_domain,
                                  add_vector=True,
                                  symmetry='symm')

        ...

    def __call__(self, dt):

        # accumulate
        self._accum()

        ...

Some remarks to this code:

1. This Propagator updates two variables: ``e`` of type :class:`~psydac.linalg.block.BlockVector` and  ``particles`` of type :class:`~struphy.pic.particles.Particles6D`
2. The :class:`~struphy.propagators.base.Propagator` base class has to be instantiated with a call to ``super().__init__()`` 
3. The kernel is ``vlasov_maxwell`` from the kernel file ``accum_kernels.py``
4. The :class:`~struphy.pic.accumulation.particles_to_grid.Accumulator` is instantiated as ``self._accum`` with its necessary arguments, in particular the ``particles`` object, the space ``'Hcurl'`` of the deposition and the ``accum_kernel``
5. The object ``self._accum`` is callable and called in the ``__call__()`` method of the Propagator to perform the accumulation.


.. _prop_kernels:

Particle kernels
================

A "kernel" is where the particle loops are written in Struphy.
The following **kernel files** are available:

* `pic/pushing/pusher_kernels.py <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/pic/pushing/pusher_kernels.py?ref_type=heads>`_ for general particle pushing
* `pic/pushing/pusher_kernels_gc.py <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/pic/pushing/pusher_kernels_gc.py?ref_type=heads>`_ for guiding-center pushing
* `pic/pushing/eval_kernels_gc.py <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/pic/pushing/eval_kernels_gc.py?ref_type=heads>`_ for particle evaluation of specific functions
* `pic/accumulation/accum_kernels.py <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/pic/accumulation/accum_kernels.py?ref_type=heads>`_ for general particle deposition 
* `pic/accumulation/accum_kernels_gc.py <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/pic/accumulation/accum_kernels_gc.py?ref_type=heads>`_ for particle deposition in guiding-center models

These kernel files are compiled when the ``struphy compile`` command is executed from the console.

In a kernel there are usually some of the following tasks to perform:

1. evaluate metric coefficients at the particle position :math:`\boldsymbol \eta_p`
2. evaluate FEEC spline fields at the particle position :math:`\boldsymbol \eta_p`
3. compute a sum over nearest-neighbor particles.

Within a kernel the metric coefficients of the map :math:`F:[0, 1]^3 \to \Omega` are available 
through the following module, imported at the top of the kernel files::

    import struphy.geometry.evaluation_kernels as evaluation_kernels

This `provides callables to all things mapping <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/geometry/evaluation_kernels.py?ref_type=heads>`_.
Linear algebra operations are available through the module::

    import struphy.linear_algebra.linalg_kernels as linalg_kernels

which provides `products, transpose, inverse, etc. <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/linear_algebra/linalg_kernels.py?ref_type=heads>`_

The evaluation of FEEC spline fields is managed through the following functions, 
which are imported at the top of the kernel files as well::

    get_spans
    eval_0form_spline_mpi
    eval_1form_spline_mpi
    eval_2form_spline_mpi
    eval_3form_spline_mpi
    eval_vectorfield_spline_mpi

Here is an example of a Struphy particle loop from the kernel :func:`struphy.pic.pushing.pusher_kernels.push_v_with_efield`,
for updating the Cartesian velocity

.. math::

        \frac{\text{d} \mathbf{v}_p}{\text{d} t} = \kappa \, DF^{-T} \mathbf{E}(\boldsymbol \eta_p)  \,,

using the following kernel::

    for ip in range(n_markers):

        # only do something if particle is a "true" particle (i.e. not a hole)
        if markers[ip, 0] == -1.:
            continue

        eta1 = markers[ip, 0]
        eta2 = markers[ip, 1]
        eta3 = markers[ip, 2]

        # evaluate Jacobian, result in dfm
        evaluation_kernels.df(eta1, eta2, eta3,
                              args_domain,
                              dfm)

        # metric coeffs
        det_df = linalg_kernels.det(dfm)
        linalg_kernels.matrix_inv_with_det(dfm, det_df, dfinv)
        linalg_kernels.transpose(dfinv, dfinvt)

        # spline evaluation
        span1, span2, span3 = get_spans(eta1, eta2, eta3, args_derham)

        # electric field: 1-form components
        eval_1form_spline_mpi(span1, span2, span3,
                              args_derham,
                              e1_1,
                              e1_2,
                              e1_3,
                              e_form)

        # electric field: Cartesian components
        linalg_kernels.matrix_vector(dfinvt, e_form, e_cart)

        # update velocities
        markers[ip, 3:6] += dt * const * e_cart

Some remarks to this code:

1. For each marker with index ``ip``, the Jacobian of the mapping ``df`` is evaluated at the logical marker position ``eta``
2. Other metric coefficients such as ``det_df``, ``dfinv`` and ``dfinvt`` are computed from ``df`` using ``linalg_kernels``
3. The spline evaluation happens in two steps:

   a. Compute the knot span indices (see below) in the three directions at ``eta`` using ``get_spans``
   b. Evaluate the 1-form by calling ``eval_1form_spline_mpi`` and store the result in ``e_form``

4. The marker velocities are updated in the last line in the ``markers`` array.

**Knot span index**: given a spline space of degree :math:`d`, at each particle position :math:`\boldsymbol \eta_p` there are 
:math:`d+1` non-zero B-spline basis functions. 
The :math:`d+1` indices of these basis functions are given by 
:math:`[s(\boldsymbol \eta_p)-d, \ldots, s(\boldsymbol \eta_p)]`, where :math:`s(\boldsymbol \eta_p) \in \mathbb N` 
is the so-called **knot span index** at position :math:`\boldsymbol \eta_p`. 

.. _prop_helper:

Helper classes
==============

* :class:`~struphy.pic.pushing.pusher.ButcherTableau` for choosing a Runge-Kutta method used in :class:`~struphy.pic.pushing.pusher.Pusher`.
* :class:`~struphy.linear_algebra.schur_solver.SchurSolver` for solving 2x2 block systems arising from mid-point rule
* :class:`~struphy.linear_algebra.schur_solver.SchurSolverFull` for solving 2x2 block systems with general rhs