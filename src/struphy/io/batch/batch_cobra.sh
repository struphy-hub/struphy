#!/bin/bash -l
#
#SBATCH -o ./sim.out
#SBATCH -e ./sim.err
#SBATCH -D ./
#SBATCH -J STRUPHY
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#
# Request memory in [MB] if run is on a shared node (tiny partition on Cobra)
#SBATCH --mem=40000

module purge
module load gcc/9 openmpi anaconda/3/2021.11 mpi4py

# load Python virtual environment (absolute path)
source /u/floho/struphy/my_venv/bin/activate

# set number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# request larger stacksize in [MB] (needed for some runs)
#export OMP_STACKSIZE=512m

# suppress warnings
export OMPI_MCA_mpi_warn_on_fork=0

# Run command added by Struphy
srun python3 -m cProfile -o /cobra/u/floho/struphy/struphy/io/out/sim_tae_16/profile_tmp -s time /cobra/u/floho/struphy/struphy/models/main.py LinearMHD -i /u/floho/struphy/struphy/io/out/sim_tae_16/parameters.yml -o /cobra/u/floho/struphy/struphy/io/out/sim_tae_16 --runtime 600 > /cobra/u/floho/struphy/struphy/io/out/sim_tae_16/struphy.out
