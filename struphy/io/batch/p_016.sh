#!/bin/bash -l
# Standard output and error:
#SBATCH -o /cobra/u/spossann/slurm-p_16.out
#SBATCH -e /cobra/u/spossann/slurm-p_16.err
# Initial working directory:
#SBATCH -D /cobra/u/spossann/
# Job Name:
#SBATCH -J p_16
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
# For OpenMP:
#SBATCH --cpus-per-task=1
#
#SBATCH --mail-type=none
#SBATCH --mail-user=spossann@ipp.mpg.de
#
# Memory limit for the job
#SBATCH --mem=32000MB
# Wall clock limit:
#SBATCH --time=00:60:00

module purge
module load gcc/9 openmpi anaconda/3/2021.11 mpi4py

# load Python virtual environment
source struphy/env/bin/activate

# Set the number of OMP threads *per process* to avoid overloading of the node!
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# For pinning threads correctly
export OMP_PLACES=cores

# suppress warnings
OMPI_MCA_mpi_warn_on_fork=0
export OMPI_MCA_mpi_warn_on_fork


# Run command added by Struphy
srun python3 -m cProfile -o /home/spossann/git_repos/struphy/struphy/io/out/sim_1/profile_tmp -s time /home/spossann/git_repos/struphy/struphy/console/run.py Maxwell /home/spossann/git_repos/struphy/struphy/io/inp/parameters.yml /home/spossann/git_repos/struphy/struphy/io/out/sim_1 False 300 > /home/spossann/git_repos/struphy/struphy/io/out/sim_1/struphy.out