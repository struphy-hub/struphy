.. _quickstart:

Quickstart
==========

Get help::

    struphy

Check if kernels are compiled::

    struphy compile

Set the default output path to the current working direcory::

    struphy --set-o .

Let us run the model ``Maxwell`` with default input parameters and save the data to ``my_first_sim/``::

    struphy run Maxwell -o my_first_sim

Let us change some input parameters.
First, get the current I/O paths::

    struphy -p

Open the default parameter file (for example with ``vim``) in the current ``<input_path>``::

    vi <input_path>/parameters.yml

Change the number of elements under ``grid/Nel`` by doubling the first entry in the 3-list, save and quit.
Let us now run ``Maxwell`` again, but this time on 2 processes, and saving to a different folder::

    struphy run Maxwell -o another_sim --mpi 2

Post process the data of the two runs::

    struphy pproc -d my_first_sim
    struphy pproc -d another_sim

Profile the runs::

    struphy profile my_first_sim another_sim 

Let us now run a more complicated hybrid model, namely ``LinearMHDVlasovCC``, on 2 processes, by specifying different in- and output files::

    struphy run LinearMHDVlasovCC -i <input_path>/params_mhd_vlasov.yml -o sim_hybrid --mpi 2

The output is stored in ``sim_hybrid/``. Post process the data::

    struphy pproc -d sim_hybrid

You can inspect the generated data via::

    ls sim_hybrid/post_processing/ 

More info about the post-processed data can be found in `Tutorial_02 <https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/notebooks/tutorial_02_postproc_standard_plotting.ipynb>`_.

Finally, let us run in Tokamak geometry. For this, let us run the model ``LinearMHD`` on 2 processes, by specifying a given input file,
and save the data to a different folder::

    struphy run LinearMHD -i <input_path>/params_mhd.yml -o sim_mhd --mpi 2

You can inspect the input file via::
    
    vi <input_path>/params_mhd.yml

Post process data::

    struphy pproc -d sim_mhd

You can now open ``paraview`` and load the data from the folder ``sim_mhd/post_processing/fields_data/vtk/``.

**Please check out the** :ref:`tutorials` **to learn more about using Struphy.**

            
