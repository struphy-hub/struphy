#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob_STRUPHY.out.%j
#SBATCH -e ./tjob_STRUPHY.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J STRUPHY
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Enable Hyperthreading:
#SBATCH --ntasks-per-core=1
# for OpenMP:
#SBATCH --cpus-per-task=16
#
# Request 40 GB of main memory per node in units of MB:
#SBATCH --mem=40960
#
#SBATCH --mail-type=all
#SBATCH --mail-user=floho@rzg.mpg.de
# Wall clock limit:
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# For pinning threads correctly:
#export OMP_PLACES=threads
#export SLURM_HINT=multithread

# Run the program
module purge

module load anaconda/3/5.1
module load gcc/9

python3 STRUPHY.py
