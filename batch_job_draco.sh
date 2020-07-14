#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob_struphy.out.%j
#SBATCH -e ./tjob_struphy.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J struphy
# Queue (Partition):
#SBATCH --partition=general
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
# for OpenMP:
#SBATCH --cpus-per-task=32
#
# Request 80 GB of main memory per node in units of MB:
#SBATCH --mem=81920
#
#SBATCH --mail-type=all
#SBATCH --mail-user=username@rzg.mpg.de
# Wall clock limit:
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# For pinning threads correctly:
#export OMP_PLACES=threads
#export SLURM_HINT=multithread

# Run the program
module load anaconda/3/5.1

python3 STRUPHY.py
