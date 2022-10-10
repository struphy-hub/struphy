.. _quickstart:

Quickstart
==========

Get help::

    struphy 

Check if kernels are compiled::

    struphy compile

Get the ``<install_path>`` and default struphy paths::

    struphy -p

Run Maxwell model with default input parameters and save data to ``<install_path>/io/out/sim_1/``::

    struphy run Maxwell -o sim_1

Post process data::

    struphy pproc sim_1

You can now open ``paraview`` and load the data from the folder ``<install_path>/io/out/sim_1/vtk/``.

Let us do a second run with different parameters. Open the default parameter file (for example with ``vim``)::

    vi <install_path>/io/in/parameters.yml

Change the number of elements under ``grid/Nel`` to ``[4, 4, 64]``, save and quit, and run a second simulation
which saves data to ``<install_path>/io/out/sim_2``::

    struphy run Maxwell -o sim_2

Profile the runs::

    struphy profile sim_1 sim_2

Check out available examples including post-processing and diagnostics::

    struphy example --help

Run Maxwell example (serial)::

    struphy example maxwell

Simulate particle orbits in a tokamak::

    struphy example orbits_tokamak

            
