#!/bin/bash -l

# Standard output and error:
#SBATCH -o ./sim.out
#SBATCH -e ./sim.err
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J test_struphy_mpi
# Queue (Partition):
#SBATCH --partition=express
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#
#SBATCH --mail-type=none
#SBATCH --mail-user=<userid>@rzg.mpg.de
#
# Wall clock limit:
#SBATCH --time=00:10:00

#Run the program:
srun python3 STRUPHY_mpi.py > prog_2_16.out
#srun python3 sum_mpi.py