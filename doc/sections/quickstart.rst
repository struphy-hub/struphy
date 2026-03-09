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

The file ``test_struphy.py`` contains all information for a simulation with the above model. 
We can change the parameters therein to our liking. 
Then, we can run a simulation simply with::

    python test_struphy.py

By default, the produced data is in ``sim_1`` in the cwd::

    ls sim_1/ 

Check the :ref:`userguide` or the `tutorials <https://github.com/struphy-hub/struphy-tutorials>`_ for how to process and display the raw data.
 
Parallel simulations can invoked from the same launch file for instance by::

    pip install -U mpi4py
    mpirun -n 4 python struphy_test.py

            
