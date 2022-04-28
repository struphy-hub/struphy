.. _quickstart:

Quickstart
==========

Command line interface
----------------------

:abbr:`STRUPHY (STRUcture-Preserving HYbrid codes)` is command-line based, i.e. there exists a binary ``<env>/bin/struphy``. 

To get help type::

    struphy 

There are three different modes, ``compile``, ``run`` and ``profile``, for calling struphy::

    struphy compile [<compile options>]
    struphy run <model> [<run options>]
    struphy profile [<profile options>] <sim1> [<sim2> ...]

All three modes come with the flags ``--user`` and ``-e``. Without those flags, struphy packages will be assumed in a platform specfifc path 
(given in ``sysconfig.get_path("platlib"``). With ``--user``, struphy packages are assumed in ``<user>/.local/lib/`` (or similar). With ``-e``, 
struphy packages are assumed in the top level of the git repository. These paths can be accessed via::

    struphy -p

which leads to an output like::

    Platform: /usr/lib64/python3.4/site-packages/struphy
    Local:    /u/spossann/.local/lib/python3.4/site-packages/struphy
    Source:  
    default in : <path_from_above>/io/inp
    default out: <path_from_above>/io/out/sim_1/

The "Source:" path pertaining to the ``-e`` flag is not given but is displayed when running for example ``struphy compile -e``.


Simple example
--------------

Let us assume that struphy has been installed and compiled in the platform specific path, henceforth called ``<platlib>``. 
Since the struphy binary is globally available, we can go to our home directory ``$HOME``, create a struphy simulations folder,
and copy input/output folders there::

    cd ~
    mkdir my_sims
    cp -r <platlib>/io/ my_sims/io/ 

We now have input files, output directories and batch scripts available in ``my_sims/io/inp/``, ``my_sims/io/out/`` and ``my_sims/io/batch/``, respectively.
Let us run the code ``maxwell_psydac`` with two different resolutions and store the results in different folders. 
First we copy the parameter file and rename it::
 
    cd my_sims/io/inp/maxwell_psydac
    cp parameters.yml params_1.yml
    
We can now edit ``params_1.yml`` and set our desired simulation parameters, including the resolution under ``grid/Nel``.
We then clone these parameters and just change the resolution for the second run::

    cp params_1.yml params_2.yml

Now let us run struphy by specifying absolute in- and output paths (we can do this in the current folder)::

    struphy run maxwell_psydac --input-abs $HOME/my_sims/io/inp/maxwell_psydac/params_1.yml --output-abs $HOME/my_sims/io/out/sim_1
    struphy run maxwell_psydac --input-abs $HOME/my_sims/io/inp/maxwell_psydac/params_2.yml --output-abs $HOME/my_sims/io/out/sim_2

For the second resolution, we wish to use mpi to investigate the scaling::
	
    struphy run maxwell_psydac --mpi 2 --input-abs $HOME/my_sims/io/inp/maxwell_psydac/params_2.yml --output-abs $HOME/my_sims/io/out/sim_3
    struphy run maxwell_psydac --mpi 4 --input-abs $HOME/my_sims/io/inp/maxwell_psydac/params_2.yml --output-abs $HOME/my_sims/io/out/sim_4

Once all simulations are finished, we can compare the run times via::

    struphy profile $HOME/my_sims/io/out/sim_1/ $HOME/my_sims/io/out/sim_2/ $HOME/my_sims/io/out/sim_3/ $HOME/my_sims/io/out/sim_4/ 


Example on  Cobra (IPP Garching)
--------------------------------

On `Cobra <https://docs.mpcdf.mpg.de/doc/computing/cobra-user-guide.html>`__ the flag ``--user`` must be used when installing/calling struphy.
Assuming struphy is installed and compiled, with the path called ``<userlib>``, we can copy input, output and batch files to ``$HOME``::

    cd ~
    cp -r <userlib>/io/ io/ 

We now have to prepare the batch scripts according to our needs. There are several defaults available::

    cd io/batch
    ls -1

yields::

    cobra_0040proc.sh
    cobra_0080proc.sh
    cobra_0160proc.sh
    cobra_0320proc.sh
    cobra_0640proc.sh
    cobra_1280proc.sh
    __init__.py
    
These are slurm batch scripts for running on 1, 2, 4, 8, 16 and 32 nodes with 40 processes each (no openmp). 
Let's say we want to do a strong scaling test for the code ``inverse_mass_test``; there is a suitable default parameter
file ``io/inp/inverse_mass_test/parameters_128x128x64.yml`` for this. Let's do::

    cd ~
    cp io/inp/inverse_mass_test/parameters_128x128x64.yml scaling_params.yml

Hence we pass our jobs to slurm via::

    struphy run inverse_mass_test --user --input-abs $HOME/scaling_params.yml --output-abs $HOME/io/out/sim_1 --batch-abs $HOME/io/batch/cobra_0040proc.sh
    struphy run inverse_mass_test --user --input-abs $HOME/scaling_params.yml --output-abs $HOME/io/out/sim_2 --batch-abs $HOME/io/batch/cobra_0080proc.sh
    struphy run inverse_mass_test --user --input-abs $HOME/scaling_params.yml --output-abs $HOME/io/out/sim_3 --batch-abs $HOME/io/batch/cobra_0160proc.sh
    struphy run inverse_mass_test --user --input-abs $HOME/scaling_params.yml --output-abs $HOME/io/out/sim_4 --batch-abs $HOME/io/batch/cobra_0320proc.sh
    struphy run inverse_mass_test --user --input-abs $HOME/scaling_params.yml --output-abs $HOME/io/out/sim_5 --batch-abs $HOME/io/batch/cobra_0640proc.sh
    struphy run inverse_mass_test --user --input-abs $HOME/scaling_params.yml --output-abs $HOME/io/out/sim_6 --batch-abs $HOME/io/batch/cobra_1280proc.sh

To check the status of the jobs::

    squeue -u <username>

To cancel a job::

    scancel <JobID>

Once all jobs have finished, we can check the scaling results via::

    struphy profile --user $HOME/io/out/sim_1/ $HOME/io/out/sim_2/ $HOME/io/out/sim_3/ $$HOME/io/out/sim_4/ $HOME/io/out/sim_5/ HOME/io/out/sim_6/ 


            
