#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./slurm-%j.out
#SBATCH -e ./slurm-%j.err
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J test_struphy
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
# For OpenMP:
#SBATCH --cpus-per-task=1
#
#SBATCH --mail-type=none
#SBATCH --mail-user=tkche@rzg.mpg.de
#
# Memory limit for the job
#SBATCH --mem=8192MB
# Wall clock limit:
#SBATCH --time=00:20:00

# module purge
module load gcc/9 openmpi/4 anaconda/3/2021.05 mpi4py/3.0.3 h5py-mpi/2.10

# Set the number of OMP threads *per process* to avoid overloading of the node!
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# For pinning threads correctly:
export OMP_PLACES=cores 

# source ~/struphy-env/bin/activate
# echo `which python`

export PATH="${PATH}:$HOME/.local/bin"
export PYTHONPATH="${PYTHONPATH}:$HOME/.local/bin"

# Run the program:
# srun struphy run maxwell_psydac --user --mpi $SLURM_JOB_NUM_NODES > "slurm-$SLURM_JOB_ID.log"
