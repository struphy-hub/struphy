.. _userguide:

Userguide
=========

This section contains some basic commands to get started with Struphy. 
It is not a tutorial, but rather a collection of examples that show 
how to use the different features of Struphy. 
For more detailed tutorials, please refer to [github.com/struphy-hub/struphy-tutorials](https://github.com/struphy-hub/struphy-tutorials).

.. highlight:: python

Basics
------

Create and run a simulation::

    from struphy import Simulation
    from struphy.models import Maxwell

    model = Maxwell()
    sim = Simulation(model=model, verbose=True)
    sim.run(verbose=True)

Define another simulation with different parameters::

    from struphy import Time, grids, DerhamOptions

    time_opts = Time(dt=0.1, Tend=0.3)
    grid = grids.TensorProductGrid(Nel=(32, 1, 1))
    derham_opts = DerhamOptions(p=(3, 1, 1))

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

Run, post process raw data and load plotting data::

    sim2.run(verbose=True)
    sim2.pproc()
    sim2.load_plotting_data()

Plot snapshots::

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


