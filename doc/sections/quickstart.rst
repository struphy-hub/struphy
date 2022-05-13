.. _quickstart:

Quickstart
==========

Command line interface
----------------------

Get help::

    struphy 

View installation path::

    struphy -p

Compile kernels::

    struphy compile


Simple example
--------------

Struphy comes with some example work flows. Two simple ones are executed via::

    example_pproc_serial
    example_pproc_mpi_3

This runs a Maxwell solver in serial mode and on 3 mpi processes with random noise as initial condition, 
then plots the light wave dispersion relation. 

For more information see :ref:`userguide`.


Example on  Cobra (IPP Garching)
--------------------------------

Let's assume Struphy is installed in the installation path ``<path>`` which can be determined with::

    pip show struphy

under ``Location:``. We can copy default folders to ``$HOME``::

    cd ~
    cp -r <path>/io/ io/ 

There are several default batch scripts available for Cobra::

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
For a scaling test of the model ``maxwell``::

    struphy run maxwell --input-abs $HOME/io/inp/parameters.yml --output-abs $HOME/io/out/sim_1 --batch-abs $HOME/io/batch/cobra_0040proc.sh
    struphy run maxwell --input-abs $HOME/io/inp/parameters.yml --output-abs $HOME/io/out/sim_2 --batch-abs $HOME/io/batch/cobra_0080proc.sh
    struphy run maxwell --input-abs $HOME/io/inp/parameters.yml --output-abs $HOME/io/out/sim_3 --batch-abs $HOME/io/batch/cobra_0160proc.sh
    struphy run maxwell --input-abs $HOME/io/inp/parameters.yml --output-abs $HOME/io/out/sim_4 --batch-abs $HOME/io/batch/cobra_0320proc.sh
    struphy run maxwell --input-abs $HOME/io/inp/parameters.yml --output-abs $HOME/io/out/sim_5 --batch-abs $HOME/io/batch/cobra_0640proc.sh
    struphy run maxwell --input-abs $HOME/io/inp/parameters.yml --output-abs $HOME/io/out/sim_6 --batch-abs $HOME/io/batch/cobra_1280proc.sh

To check the status of the jobs::

    squeue -u <username>

To cancel a job::

    scancel <JobID>

Once all jobs have finished, we can check the scaling results via::

    struphy profile $HOME/io/out/sim_1/ $HOME/io/out/sim_2/ $HOME/io/out/sim_3/ $$HOME/io/out/sim_4/ $HOME/io/out/sim_5/ HOME/io/out/sim_6/ 


            
