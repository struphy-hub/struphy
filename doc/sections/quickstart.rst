.. _quickstart:

Quickstart
==========

Get help::

    struphy -h

Check if kernels are compiled::

    struphy compile

Get the ``<install_path>`` and default struphy paths::

    struphy -p

Run the model ``Maxwell`` with default input parameters and save data to ``<install_path>/io/out/my_first_sim/``::

    struphy run Maxwell -o my_first_sim

Run the model ``LinearMHD`` with default input parameters, on two processes, and save data to a different folder::

    struphy run LinearMHD --mpi 2 -o sim_mhd

Post process data::

    struphy pproc -d sim_mhd

You can now open ``paraview`` and load the data from the folder ``<install_path>/io/out/sim_mhd/post_processing/fields_data/vtk/``.
Let us do a second run with different parameters. Open the default parameter file (for example with ``vim``)::

    vi <install_path>/io/in/parameters.yml

Change the number of elements under ``grid/Nel``, save and quit, and run a second simulation
which saves data to ``<install_path>/io/out/sim_mhd_2``::

    struphy run LinearMHD --mpi 2 -o sim_mhd_2 

Profile the runs::

    struphy profile sim_mhd sim_mhd_2

Check out available Struphy examples which include post-processing and diagnostics::

    struphy example --help

Run Maxwell example (serial)::

    struphy example maxwell

Simulate particle orbits in a tokamak::

    struphy example orbits_tokamak

            
