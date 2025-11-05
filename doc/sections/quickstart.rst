.. _quickstart:

Quickstart
==========

Get familiar with Struphy right away through the tutorials on `mybinder <https://mybinder.org/v2/gh/struphy-hub/struphy-tutorials/main>`_ - no installation needed.

What follows is an introduction to the CLI (command line interface) of Struphy.
For a more in-depth manual please go to :ref:`userguide`.

Get help on Struphy console commands::

    struphy -h

Check if kernels are compiled::

    struphy compile

Display available kinetic models::

    struphy --kinetic

Generate default parameters for the model :class:`~struphy.models.kinetic.VlasovMaxwellOneSpecies`::

    struphy params VlasovMaxwellOneSpecies

After hitting enter on prompt, the default launch file ``params_VlasovMaxwellOneSpecies.py`` is created
in the current working directory (cwd). Let us rename it for convenience::

    mv params_VlasovMaxwellOneSpecies.py test_struphy.py

The file ``test_struphy`` contains all information of a simulation with the above model. 
We can change the parameters therein to our liking. 
Then, we can now run a simulation simply with::

    python test_struphy.py

By default, the produced data is in ``sim_1`` in the cwd::

    ls sim_1/ 

The data can be accessed through the Struphy API. If ``ipython`` is installed::

    ipython
    
and then::

    from struphy.main import pproc, load_data
    import os
    path = os.path.join(os.getcwd(), "sim_1")
    pproc(path)
    simdata = load_data(path)

The variable ``simdata`` is of type :class:`~struphy.main.SimData` and holds grid and orbit information.
You can deduce the kind of info held from the screen output. For instance, you have access several ``grids``
as well as to, for instance::

    print(simdata.spline_values["em_fields"]["e_field_log"].keys())
    print(simdata.orbits["kinetic_ions"].shape)
    print(simdata.f["kinetic_ions"]["e1"].keys())

Under ``simdata.spline_values`` you find dictionaries holding splines values at the pre-defined ``simdata.grids_log``
(or the physical grid); the keys are the time points of evaluation.

Under ``simdata.orbits`` you find numpy arrays holding orbit data, indexed by ``[time, particle, attribute]``.

Under ``simdata.f`` you find binning data, in this case a 1d binning plot in the first logical coordinate :math:`\eta_1`-direction.
 
Parallel simulations can invoked from the same launch file for instance by::

    mpirun -n 4 struphy_test.py

If you want to learn more please check out the :ref:`userguide`.

            
