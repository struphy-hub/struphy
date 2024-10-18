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

# Load modules
module purge
module load gcc/12 openmpi/4.1 anaconda/3/2023.03 git/2.43 pandoc/3.1 likwid/5.2

# Set up environment
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

# Array of MPI process counts
mpi_procs=(1 2 4 8 16 32 64)

# Loop through the process counts and run the performance tests
for procs in "${mpi_procs[@]}"; do
    # Create a directory for the simulation
    dir="sim_$(printf "%04d" $procs)/"
    mkdir -p "$dir"
    
    # Run the performance test and output to respective files
    struphy test performance --mpi "$procs" > "${dir}struphy_$(printf "%04d" $procs).out"
done