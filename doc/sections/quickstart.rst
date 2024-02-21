.. _quickstart:

Quickstart
==========

Get help on Struphy console commands::

    struphy -h

Check if kernels are compiled::

    struphy compile --status

Check the current I/O paths::

    struphy -p

Set the I/O paths to the current working directory::

    struphy --set-i .
    struphy --set-o .

Get a list of available Struphy models::

    struphy run -h

Let us run the model `VlasovMaxwell <https://struphy.pages.mpcdf.de/struphy/sections/models.html#struphy.models.kinetic.VlasovMaxwell>`_::

    struphy run VlasovMaxwell

After hitting ``enter`` on prompt, the parameter file ``params_VlasovMaxwell.yml`` is created
in the current input path (cwd). Let us rename it for convenience::

    mv params_VlasovMaxwell.yml test.yml

We can now run a simulation with these parameters and save the data to ``my_first_sim/``::

    struphy run VlasovMaxwell -i test.yml -o my_first_sim

The produced data is in the expected folder in the current output path (cwd)::

    ls my_first_sim/ 

Let us post-process the raw simulation data::

    struphy pproc -d my_first_sim

The results of post-processing are stored under ``my_first_sim/post_processing/``. In particular, 
the data of the FEEC-fields is stored under::

    ls my_first_sim/post_processing/fields_data/

and the data of the kinetic particles is stored under::

    ls my_first_sim/post_processing/kinetic_data/

Check out :ref:`Tutorial 2 - Data, post processing and standard plots <tutorials>`
for a deeper discussion on Struphy data.

Our first simulation ran for just three time steps. Let us change the end-time of the simulation by opening the parameter file::

    vi test.yml

and setting ``time/Tend`` to ``0.1``. Save, quit and run again, but this time on 2 MPI processes, 
and saving to a different folder::

    struphy run VlasovMaxwell -i test.yml -o another_sim --mpi 2

This time we ran for 20 time steps. The physical time unit of the run can be known via::

    struphy units VlasovMaxwell -i test.yml

Please refer to :ref:`Tutorial 1 - Run Struphy main file in a notebook <tutorials>` 
for more information on the units used in Struphy.
For completeness, let us post-process the data of the second run::

    struphy pproc -d another_sim

Let us now double the number of markers used in the simulation:: 

    vi test.yml

by changing ``kinetic/electrons/markers/ppc`` from 10 to 20, and then running::

    struphy run VlasovMaxwell -i test.yml -o sim_20 --mpi 2

Finally, each Struphy model has some specific options to it, which in the case of ``VlasovMaxwell`` can be inspected via::

    struphy params VlasovMaxwell --options

These options can be set in the parameter file. They usually refer to different types of solvers or solution methods.

If you want to learn more about using Struphy, please check out the :ref:`userguide`
as well as the :ref:`tutorials`.

            
