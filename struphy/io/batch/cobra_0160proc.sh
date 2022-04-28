#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./slurm-%j.out
#SBATCH -e ./slurm-%j.err
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J p_160
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
# For OpenMP:
#SBATCH --cpus-per-task=1
#
#SBATCH --mail-type=none
#SBATCH --mail-user=spossann@ipp.mpg.de
#
# Wall clock limit:
#SBATCH --time=00:10:00

module purge
module load gcc openmpi anaconda/3/2020.02 mpi4py
 
# Set the number of OMP threads *per process* to avoid overloading of the node!
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# For pinning threads correctly:
export OMP_PLACES=cores 

export PATH="${PATH}:$HOME/.local/bin"
export PYTHONPATH="${PYTHONPATH}:$(python3 -m site --user-site)"

# Run command added by Struphy
srun python3 -m cProfile -o /u/spossann/io/out/sim_3/profile_tmp -s time /u/spossann/.local/lib/python3.7/site-packages/struphy/models/codes/exec.py inverse_mass_test /u/spossann/io/inp/inverse_mass_test/parameters_128x128x64.yml /u/spossann/io/out/sim_3/ None /u/spossann/io/out/sim_3/meta.txt > /u/spossann/io/out/sim_3/struphy.out
