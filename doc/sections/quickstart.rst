.. _quickstart:

Quickstart
==========

Command line interface
----------------------

Get help::

    struphy 

View installation path::

    struphy -p

which leads to an output like::

    Struphy installation path: $HOME/git_repos/struphy/struphy
    default input:             $HOME/git_repos/struphy/struphy/io/inp
    default output:            $HOME/git_repos/struphy/struphy/io/out
    template batch scripts:    $HOME/git_repos/struphy/struphy/io/batch

Compile kernels::

    pip install pyccel==0.10.1
    struphy compile
    pip install -U pyccel

Note that the older version of ``pyccel`` is needed only for compilation and will be removed completely in the near future.


Simple example
--------------

Struphy comes with some example work flows. Two simple ones are executed via::

    example_pproc_serial
    example_pproc_mpi_3

This runs a Maxwell solver in serial mode and on 3 mpi processes with random noise as initial condition, then plots the light wave dispersion relation. 

To get a little more busy, we can create a struphy simulations folder,
and copy input/output folders there::

    mkdir my_sims
    cp -r <install_path>/io/ my_sims/io/ 

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


            
