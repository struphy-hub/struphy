#!/bin/bash -l
#SBATCH -o ./job_struphy_%j.out
#SBATCH -e ./job_struphy_%j.err
#SBATCH -D ./
#SBATCH -J job_struphy
#SBATCH --nodes=1                # number of compute nodes
#SBATCH --ntasks-per-node=72     # number of MPI processes (max 144 on raven with hyperthreading)
##SBATCH --mail-type=
##SBATCH --mail-user=
#SBATCH --time=00:10:00

if [ ! -d "env_struphy" ]; then python3 -m venv env_struphy; fi
source env_struphy/bin/activate
LIKWID_PREFIX=$(realpath $(dirname $(which likwid-topology))/..)
export LD_LIBRARY_PATH=$LIKWID_PREFIX/lib
export LD_LIBRARY_PATH=/mpcdf/soft/SLE_15/packages/skylake/likwid/gcc_12-12.1.0/5.3.0/lib:$LD_LIBRARY_PATH

# Set the number of OMP threads *per process* to avoid overloading of the node!
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# For pinning threads correctly
#export OMP_PLACES=cores
KMP_AFFINITY=scatter

struphy test performance --mpi 1  > struphy_0001.out
struphy test performance --mpi 2  > struphy_0002.out
struphy test performance --mpi 4  > struphy_0004.out
struphy test performance --mpi 8  > struphy_0008.out
struphy test performance --mpi 16 > struphy_0016.out
struphy test performance --mpi 32 > struphy_0032.out
struphy test performance --mpi 64 > struphy_0064.out