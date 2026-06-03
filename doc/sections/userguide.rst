.. _userguide:

Userguide
=========

This guide is a practical, detailed companion to :ref:`quickstart`.
It focuses on common workflows in Struphy with copy-paste Python snippets
that you can adapt to your own cases.

For interactive notebooks and longer tutorial stories, see
`struphy-tutorials <https://github.com/struphy-hub/struphy-tutorials>`_.

.. highlight:: python

1. Overview
-----------

Struphy workflows are built around three concepts:

1. :class:`~struphy.Simulation`:
   The top-level runtime object. It owns global run setup (time stepping,
   geometry, grids, output folders, logging, MPI setup) and executes
   ``run()``.
2. :class:`~struphy.models.base.StruphyModel`:
   The PDE model definition. A model contains species, variables, and
   a collection of propagators used to evolve the unknowns.
3. :class:`~struphy.propagators.base.Propagator`:
   The elementary time-advancement operators. A model typically combines
   several propagators through a splitting scheme.

In practice, your script usually follows this pattern:

.. code-block:: python

    from struphy import Simulation
    from struphy.models import VlasovMaxwellOneSpecies

    model = VlasovMaxwellOneSpecies()
    sim = Simulation(model=model)
    sim.run()

For full API details, see :ref:`api_guide`.


2. Launching a simulation
-------------------------

The central entry point is :class:`~struphy.Simulation`.
For a minimal run with
:class:`~struphy.models.vlasov_maxwell_one_species.VlasovMaxwellOneSpecies`:

.. code-block:: python

    from struphy import Simulation
    from struphy.models import VlasovMaxwellOneSpecies

    model = VlasovMaxwellOneSpecies()

    sim = Simulation(
        model=model,
        params_path=__file__,
    )

    sim.run()

To print more runtime information, either set global logging level:

.. code-block:: python

    import logging
    from struphy import set_logging_level

    set_logging_level(logging.INFO)

or pass it directly when creating the simulation:

.. code-block:: python

    import logging
    from struphy import Simulation

    sim = Simulation(model=model, logging_level=logging.INFO)

Environment configuration is done with :class:`~struphy.EnvironmentOptions`.
This affects runtime behavior and output organization, not the physics:

.. code-block:: python

    from struphy import EnvironmentOptions, Simulation

    env = EnvironmentOptions(
        out_folders="./runs",
        sim_folder="vm1s_scan_A",
        restart=False,
        save_step=5,
        sort_step=20,
        max_runtime=180,
        profiling_activated=False,
    )

    sim = Simulation(model=model, env=env)

Tip: keep one ``sim_folder`` per parameter set to make post-processing and
comparisons reproducible.


3. Choosing a model
-------------------

A model defines which PDE system is solved and which species/variables exist.
Each concrete model is a subclass of
:class:`~struphy.models.base.StruphyModel`.

Example with optional model arguments:

.. code-block:: python

    from struphy.models import VlasovAmpereOneSpecies

    model = VlasovAmpereOneSpecies(
        alpha=1.0,
        epsilon=-1.0,
        with_B0=False,
    )

Example with explicit normalization units:

.. code-block:: python

    from struphy import BaseUnits
    from struphy.models import VlasovMaxwellOneSpecies

    units = BaseUnits(x=1.0, B=1.0, n=1.0)
    model = VlasovMaxwellOneSpecies(base_units=units)

Models are collections of species. Species hold the unknowns (variables)
of your PDE system. You can inspect them directly:

.. code-block:: python

    print(model.species.keys())
    print(model.field_species.keys())
    print(model.particle_species.keys())

See :ref:`species` in the API guide for
:class:`~struphy.models.species.FieldSpecies`,
:class:`~struphy.models.species.FluidSpecies`, and
:class:`~struphy.models.species.ParticleSpecies`.


4. Choosing the geometry
------------------------

Geometry is set through the ``domain=...`` argument of
:class:`~struphy.Simulation`.
The domain defines the map from logical coordinates :math:`\eta` to physical
coordinates :math:`x`.

Simple Cartesian box:

.. code-block:: python

    from struphy import domains

    domain = domains.Cuboid(
        l1=0.0,
        r1=2.0,
        l2=0.0,
        r2=1.0,
        l3=0.0,
        r3=1.0,
    )

Curvilinear example (hollow cylinder):

.. code-block:: python

    from struphy import domains

    domain = domains.HollowCylinder(
        a1=1.0,
        a2=10.0,
        Lz=10.0,
    )

Use the :ref:`avail_mappings` page to see all available domain classes and
their parameters.


5. Space and time grids
-----------------------

Two objects control temporal and spatial resolution:

1. :class:`~struphy.Time` for time stepping.
2. :class:`~struphy.grids.TensorProductGrid` for spatial mesh resolution.

Small "smoke test" run:

.. code-block:: python

    from struphy import Time, grids

    time_opts = Time(dt=0.01, Tend=0.05, split_algo="LieTrotter")
    grid = grids.TensorProductGrid(num_elements=(16, 16, 1))

Larger production-style setup:

.. code-block:: python

    from struphy import Time, grids

    time_opts = Time(dt=0.002, Tend=10.0, split_algo="Strang")
    grid = grids.TensorProductGrid(
        num_elements=(64, 128, 16),
        mpi_dims_mask=(True, True, False),
    )

In general:

1. decrease ``dt`` until your diagnostics are stable,
2. increase ``num_elements`` until spatial convergence is acceptable,
3. activate decomposition only in directions with enough elements per rank.


6. de Rham sequence
-------------------

Discrete FEEC spaces are configured with :class:`~struphy.DerhamOptions`.
Important arguments:

1. ``degree=(p1, p2, p3)``: spline degree in each logical direction.
2. ``bcs=(..., ..., ...)``: boundary conditions per direction.
   Use ``None`` for periodic directions.
3. ``nquads`` and ``nquads_proj``: quadrature rules.
4. ``polar_splines``: special smoothness treatment near polar singularities.
5. ``local_projectors``: local commuting projectors.

Periodic setup:

.. code-block:: python

    from struphy import DerhamOptions

    derham_opts = DerhamOptions(
        degree=(3, 3, 1),
        bcs=(None, None, None),
    )

Mixed periodic/non-periodic setup:

.. code-block:: python

    from struphy import DerhamOptions

    derham_opts = DerhamOptions(
        degree=(4, 3, 2),
        bcs=(
            ("dirichlet", "free"),
            None,
            ("free", "free"),
        ),
        nquads=(6, 6, 6),
        nquads_proj=(8, 8, 8),
        polar_splines=False,
        local_projectors=True,
    )

See :ref:`api_guide` for the class reference and :ref:`geomFE` for the FEEC
background and geometric interpretation.


7. MHD equilibrium
------------------

For MHD-type setups, choose an equilibrium object and pass it via
``equil=...`` into :class:`~struphy.Simulation`.

Homogeneous slab equilibrium:

.. code-block:: python

    from struphy import Simulation, equils

    equil = equils.HomogenSlab(B0z=1.0, beta=0.05, n0=1.0)
    sim = Simulation(model=model, domain=domain, grid=grid, equil=equil)

Sheared slab equilibrium:

.. code-block:: python

    from struphy import equils

    equil = equils.ShearedSlab(
        a=1.0,
        R0=3.0,
        B0=1.0,
        q0=1.05,
        q1=1.80,
        beta=0.1,
    )

You can also use axisymmetric/toroidal equilibria such as
``EQDSKequilibrium``, ``GVECequilibrium``, and ``DESCequilibrium`` when
appropriate for your domain and model.

For the complete list and options, see :ref:`equils_avail`.


8. Choosing particle parameters
-------------------------------

Particle species are configured through
:meth:`~struphy.models.species.ParticleSpecies.set_markers`.
This method collects all marker-related setup in one place:

1. marker loading,
2. weight handling,
3. boundary behavior,
4. sorting,
5. particle diagnostics output,
6. marker-array buffer size.

Detailed setup example:

.. code-block:: python

    from struphy import (
        BinningPlot,
        BoundaryParameters,
        KernelDensityPlot,
        LoadingParameters,
        SavingParameters,
        SortingParameters,
        WeightsParameters,
    )

    loading_params = LoadingParameters(
        Np=100000,
        loading="sobol_standard",
    )

    weights_params = WeightsParameters(
        control_variate=True,
    )

    boundary_params = BoundaryParameters()

    sorting_params = SortingParameters(
        do_sort=True,
        sorting_frequency=10,
        boxes_per_dim=(24, 24, 1),
    )

    phase_plot = BinningPlot(
        slice="e1_v1",
        n_bins=(128, 128),
        ranges=((0.0, 1.0), (-8.0, 8.0)),
    )

    kde_plot = KernelDensityPlot(
        pts_e1=128,
        pts_e2=128,
        pts_e3=1,
    )

    saving_params = SavingParameters(
        n_markers=0.01,
        binning_plots=(phase_plot,),
        kernel_density_plots=(kde_plot,),
    )

    model.kinetic_ions.set_markers(
        loading_params=loading_params,
        weights_params=weights_params,
        boundary_params=boundary_params,
        sorting_params=sorting_params,
        saving_params=saving_params,
        bufsize=0.4,
    )

Practical notes:

1. start with moderate ``Np`` and increase only after diagnostics look right,
2. ``do_sort=True`` is often beneficial for large runs,
3. ``bufsize`` trades memory for safer marker-array headroom.

See :ref:`api_guide` for all marker parameter classes.


9. Setting propagator options
-----------------------------

Each model provides a model-specific propagator collection under
``model.propagators``.
You configure each propagator through its own ``Options`` dataclass.

Default options:

.. code-block:: python

    model.propagators.maxwell.options = model.propagators.maxwell.Options()
    model.propagators.push_eta.options = model.propagators.push_eta.Options()

Options with explicit variable binding:

.. code-block:: python

    model.propagators.push_vxb.options = model.propagators.push_vxb.Options(
        b2_var=model.em_fields.b_field,
    )

Coupling/initial operators (if present in your model):

.. code-block:: python

    model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
    model.initial_poisson.options = model.initial_poisson.Options()

To discover available propagators for a model, inspect:

.. code-block:: python

    print(model.propagators)
    print(model.prop_list)

Always use options from the same propagator object, i.e.
``model.propagators.NAME.Options(...)``.


10. Setting initial conditions
------------------------------

Initial-condition setup depends on variable type. In the API guide, see
:class:`~struphy.models.variables.FEECVariable` and
:class:`~struphy.models.variables.PICVariable`.

FEEC variables (grid-based fields)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For FEEC variables, a common workflow is:

1. optional background via :class:`~struphy.FieldsBackground`,
2. optional perturbation via :class:`~struphy.initial.base.Perturbation`.

Example:

.. code-block:: python

    from struphy import FieldsBackground, perturbations

    model.em_fields.phi.add_background(
        FieldsBackground(type="LogicalConst", values=(0.0,), variable="phi0"),
    )

    model.em_fields.phi.add_perturbation(
        perturbations.ModesCos(ls=(1,), amps=(1e-3,)),
    )

PIC variables (particle distributions)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For particle variables, a background distribution is mandatory.
Without a particle background, allocation fails.

Example background and perturbed initial condition:

.. code-block:: python

    from struphy import maxwellians, perturbations

    # Mandatory background
    f0_a = maxwellians.Maxwellian3D(n=(0.5, None), u1=(2.0, None))
    f0_b = maxwellians.Maxwellian3D(n=(0.5, None), u1=(-2.0, None))
    background = f0_a + f0_b
    model.kinetic_ions.var.add_background(background)

    # Optional explicit initial condition: if omitted, initial condition = background
    pert = perturbations.ModesCos(ls=(1,), amps=(1e-3,))
    f1_a = maxwellians.Maxwellian3D(n=(0.5, pert), u1=(2.0, None))
    f1_b = maxwellians.Maxwellian3D(n=(0.5, pert), u1=(-2.0, None))
    init = f1_a + f1_b
    model.kinetic_ions.var.add_initial_condition(init)

Common pitfalls
^^^^^^^^^^^^^^^

1. Missing PIC background:
   ``model.kinetic_ions.var.add_background(...)`` must be called.
2. Inconsistent setup:
   choose ``domain``, ``grid``, and ``DerhamOptions`` jointly.
3. No output saved:
   set ``save_data=True`` on variables you want in diagnostics.

After initial conditions are set, launch the run:

.. code-block:: python

    sim.run()

   


