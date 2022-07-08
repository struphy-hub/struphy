.. _quickstart:

Quickstart
==========

Get help::

    struphy 

Check if kernels are compiled::

    struphy compile

Get the ``<install_path>``::

    struphy -p

Run Maxwell model with default input parameters and save data to ``<install_path>/io/out/sim_1``::

    struphy run Maxwell -o sim_1

Open the default parameter file (for example with ``vim``)::

    vi <install_path>/io/in/parameters.yml

Change the number of elements under ``grid/Nel`` to ``[4, 4, 64]``, save and quit, and run a second simulation
which saves data to ``<install_path>/io/out/sim_2``::

    struphy run Maxwell -o sim_2

Profile the runs::

    struphy profile sim_1 sim_2

Run Maxwell tests including post-processing and simple diagnostics in serial::

    example_maxwell_serial

Same with three mpi processes::

    example_maxwell_mpi_3

The source of the routines used in these post-processing examples is in ``<install_path>/examples/example_diagnostics_1dfft.py`` 


            
