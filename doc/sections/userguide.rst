.. _userguide:

Userguide
=========

This section contains some basic commands to get started with Struphy. 
It is not a tutorial, but rather a collection of examples that show 
how to use the different features of Struphy. 
For more detailed tutorials, please refer to `<https://github.com/struphy-hub/struphy-tutorials>`_.

.. highlight:: python

Basics
------

Choose a model::

    from struphy.models import Maxwell
    model = Maxwell()

Set initial conditions::

    from struphy import perturbations

    fun1 = perturbations.ModesCos(ls=(1,), amps=(1e-1,), comp=1)
    fun2 = perturbations.ModesSin(ls=(2,), amps=(1e-2,), comp=2)

    model.em_fields.e_field.add_perturbation(perturbation=fun1)
    model.em_fields.e_field.add_perturbation(perturbation=fun2)

Create and run a simulation::

    from struphy import Simulation
    sim = Simulation(model=model, verbose=True)
    sim.run(verbose=True)

Define another simulation with different parameters::

    from struphy import Time, grids, DerhamOptions

    time_opts = Time(dt=0.1, Tend=0.3)
    grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
    derham_opts = DerhamOptions(degree=(3, 1, 1))

    sim2 = Simulation(model=model,
                      time_opts=time_opts,
                      grid=grid,
                      derham_opts=derham_opts,
                      verbose=True,
                      )
    
Check the :ref:`api_guide` for more details on the available options and how to use them.
Compare, serialize and save simulations::

    sim3 = Simulation.from_dict(sim2.to_dict())
    assert sim2 == sim3

    sim4 = sim2.spawn_sister()
    assert sim2 == sim4

    # only in devel so far - not yet in main
    sim2.export("sim2.yaml")
    sim5 = Simulation.from_file("sim2.yaml")
    assert sim2 == sim5    

Run, post process and load plotting data::

    sim2.run(verbose=True)
    sim2.pproc()
    sim2.load_plotting_data()

Plot snapshots::

    from matplotlib import pyplot as plt

    e_field = sim2.spline_values.em_fields.e_field_log

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for n, (time, data) in enumerate(e_field.data.items()):

        # Plot 1: First component in x-direction
        axes[n, 0].plot(data[0][:, 0, 0])
        axes[n, 0].set_title(f"E_1 (x-direction) at time {time}")
        axes[n, 0].set_xlabel("x")
        axes[n, 0].set_ylabel("Value")
        axes[n, 0].set_ylim(-1.5e-1, 1.5e-1)  # Set y-limits for better comparison across time steps

        # Plot 2: Add your second plot
        axes[n, 1].plot(data[1][:, 0, 0])
        axes[n, 1].set_title(f"E_2 (x-direction) at time {time} ")
        axes[n, 1].set_xlabel("x")
        axes[n, 1].set_ylabel("Value")
        axes[n, 1].set_ylim(-1.5e-1, 1.5e-1)  # Set y-limits for better comparison across time steps

        # Plot 3: Add your third plot
        axes[n, 2].plot(data[2][:, 0, 0])
        axes[n, 2].set_title(f"E_3 (x-direction) at time {time}")
        axes[n, 2].set_xlabel("x")
        axes[n, 2].set_ylabel("Value")
        axes[n, 2].set_ylim(-1.5e-1, 1.5e-1)

        plt.tight_layout()
    plt.show()


LinearMHD in DESC equilibrium
-----------------------------

This is the setup for running the linear MHD model in a DESC equilibrium::

    from struphy.models import LinearMHD
    from struphy import Simulation
    from struphy import domains, equils

    model = LinearMHD()
    domain = domains.DESCunit()
    equil = equils.DESCequilibrium(use_nfp=False)

    sim = Simulation(
        model=model,
        domain=domain,
        equil=equil,
    )

    sim.show_domain(scalars="absB0", window_size = (850, 250), zoom_factor=2.0)

Check the :ref:`DESCunit <domains_avail>` and :ref:`DESCequilibrium <equils_avail>` classes for more details on the available options and how to use them.
You can adapt the grid and time options as needed for running the simulation.


Kinetic model: two-stream instability
-------------------------------------

This is the setup for running the two-stream instability with the Vlasov-Maxwell one species model. 
Start by choosing a good model for this test, and set the species properties for your test::

    from struphy.models import VlasovAmpereOneSpecies
    model = VlasovAmpereOneSpecies(with_B0 = False)
    model.kinetic_ions.set_species_properties(alpha=1.0, epsilon=-1.0)

We then set the basic options for this test::

    from struphy import (
        DerhamOptions,
        Time,
        domains,
        grids,
        )

    time_opts = Time(dt = 0.1, Tend = 50.0, split_algo = "LieTrotter")
    domain = domains.Cuboid(r1 = 31.42)
    equil = None
    grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
    derham_opts = DerhamOptions(degree=(3, 1, 1))

Next, instantiate the simulation with the above parameters::

    from struphy import Simulation

    sim = Simulation(
        model=model,
        params_path=__file__,
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
    )

We can now set the particle parameters for the PIC simulation::

    from struphy import (
        perturbations,
        BinningPlot,
        LoadingParameters,
        WeightsParameters,
        maxwellians,
        )

    loading_params = LoadingParameters(ppc=1000, moments=(0.0, 0.0, 0.0, 3.0, 1.0, 1.0))
    weights_params = WeightsParameters(control_variate=True)
    model.kinetic_ions.set_markers(loading_params=loading_params,
                                weights_params=weights_params,
                                bufsize = 0.4,
                                )
    model.kinetic_ions.set_sorting_boxes(boxes_per_dim=(16, 1, 1), do_sort=True) 

    binplot = BinningPlot(slice='e1_v1', n_bins= (128, 128), ranges= ((0.,1.), (-10.0,10.0)))
    model.kinetic_ions.set_save_data(binning_plots=(binplot,))

Finally, we set the two-stream background::

    maxwellian_1 = maxwellians.Maxwellian3D(n=(0.5, None), u1 = (3.0, None))
    maxwellian_2 = maxwellians.Maxwellian3D(n=(0.5, None), u1 = (-3.0, None))
    background = maxwellian_1 + maxwellian_2
    model.kinetic_ions.var.add_background(background)

and the corresponding initial condition::

    perturbation = perturbations.ModesCos(amps = (0.001,), ls = (1,))
    init1 = maxwellians.Maxwellian3D(n = (0.5, perturbation), u1 = (3.0, None)) 
    init2 = maxwellians.Maxwellian3D(n = (0.5, perturbation), u1 = (-3.0, None))
    init = init1 + init2
    model.kinetic_ions.var.add_initial_condition(init)

Press run::
    
    sim.run(verbose=True)

   


