.. _quickstart:

Quickstart
==========

Get help::

    struphy -h

Check if kernels are compiled::

    struphy compile

Get the ``<install_path>`` which ends with `/struphy`::

    struphy -p

Write the install path to a variable for easier use::

    STR_PATH=<install_path>

Let us run the model ``Maxwell`` with default input parameters and save the data to ``<install_path>/io/out/my_first_sim/``::

    struphy run Maxwell -o my_first_sim

Let us change some input parameters. Open the default parameter file (for example with ``vim``)::

    vi $STR_PATH/io/inp/parameters.yml

Change the number of elements under ``grid/Nel`` from ``[16, 32, 32]`` to ``[32, 32, 32]``, save and quit.
Let us now run ``Maxwell`` again, but this time on 2 processes, and saving to a different folder::

    struphy run Maxwell -o another_sim --mpi 2

Post process the data of the two runs::

    struphy pproc -d my_first_sim
    struphy pproc -d another_sim

Profile the runs::

    struphy profile my_first_sim another_sim 

Let us now run a more complicated hybrid model, namely ``LinearMHDVlasovCC``, on 2 processes, by specifying different in- and output files::

    struphy run LinearMHDVlasovCC -i params_mhd_vlasov.yml -o sim_hybrid --mpi 2

The output is stored in ``<install_path>/io/out/sim_hybrid/``. Post process the data::

    struphy pproc -d sim_hybrid

You can inspect the generated data via::

    ls $STR_PATH/io/out/sim_hybrid/post_processing/ 

Finally, let us run in Tokamak geometry. For this, let us run the model ``LinearMHD`` on 2 processes, by specifying a given input file,
and save the data to a different folder::

    struphy run LinearMHD -i params_mhd.yml -o sim_mhd --mpi 2

You can inspect the input file via::
    
    vi $STR_PATH/io/in/params_mhd.yml

Post process data::

    struphy pproc -d sim_mhd

You can now open ``paraview`` and load the data from the folder ``<install_path>/io/out/sim_mhd/post_processing/fields_data/vtk/``.

Check out available Struphy examples which include post-processing and diagnostics::

    struphy example --help

Simulate the MHD dispersion relation in a slab::

    struphy example linearmhd

Simulate guiding-center orbits in a tokamak::

    struphy example gc_orbits_tokamak

The same for full-orbit Vlasov::

    struphy example orbits_tokamak

            
