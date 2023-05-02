#!/bin/bash -l
# Standard output and error:
#SBATCH -o /cobra/u/spossann/slurm-p_256.out
#SBATCH -e /cobra/u/spossann/slurm-p_256.err
# Initial working directory:
#SBATCH -D /cobra/u/spossann/
# Job Name:
#SBATCH -J p_256
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=32
# For OpenMP:
#SBATCH --cpus-per-task=1
#
#SBATCH --mail-type=none
#SBATCH --mail-user=spossann@ipp.mpg.de
#
# Memory limit for the job
#SBATCH --mem=64000MB
# Wall clock limit:
#SBATCH --time=00:60:00

module purge
module load gcc/9 openmpi anaconda/3/2021.11 mpi4py

# load Python virtual environment
source struphy/env/bin/activate

# Set the number of OMP threads *per process* to avoid overloading of the node!
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# For pinning threads correctly:
export OMP_PLACES=cores

# suppress warnings
OMPI_MCA_mpi_warn_on_fork=0
export OMPI_MCA_mpi_warn_on_fork

# Run command automatically added by Struphy
srun python3 -m cProfile -o /cobra/u/spossann/struphy/env/lib/python3.9/site-packages/struphy/io/out/sim_5/profile_tmp -s time /cobra/u/spossann/struphy/env/lib/python3.9/site-packages/struphy/main.py Maxwell /cobra/u/spossann/struphy/env/lib/python3.9/site-packages/struphy/io/inp/tests/params_maxwell_1.yml /cobra/u/spossann/struphy/env/lib/python3.9/site-packages/struphy/io/out/sim_5/ None /cobra/u/spossann/struphy/env/lib/python3.9/site-packages/struphy/io/out/sim_5/meta.txt > /cobra/u/spossann/struphy/env/lib/python3.9/site-packages/struphy/io/out/sim_5/struphy.out
