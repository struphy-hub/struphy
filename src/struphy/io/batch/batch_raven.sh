#!/bin/bash -l
#SBATCH -o ./str_performance%j.out
#SBATCH -e ./str_performance%j.err
#SBATCH -D ./
#SBATCH -J str_performance
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

# Get the current date in YYYYMMDD format
current_date=$(date +"%Y%m%d")

# Define the base output path, incorporating the current date
output_path="output/${current_date}"

# Create the base output directory
mkdir -p "$output_path"

# Save environment variables to a file
env > "${output_path}/environment_variables.txt"

# Copy the current SLURM script (this script itself)
cp "$0" "${output_path}/slurm_script.sh"

# Save LIKWID topology information to files
likwid-topology > "${output_path}/likwid-topology.txt"
likwid-topology -g > "${output_path}/likwid-topology-g.txt"

# Save a list of loaded modules
module list > "${output_path}/module_list.txt"

# Save a list of all environment variables
printenv > "${output_path}/printenv.txt"

# Save SLURM-specific environment variables to a separate file
for var in $(env | grep ^SLURM_ | cut -d= -f1); do
    echo "$var=${!var}" >> "${output_path}/SLURM_VARIABLES.txt"
done

# Array of MPI process counts to test
mpi_procs=(1 2 4 8 16 32 64)

# Loop through the MPI process counts and run performance tests
for procs in "${mpi_procs[@]}"; do
    # Create a directory for the specific MPI process count
    dir="${output_path}/sim_$(printf "%04d" $procs)/"
    mkdir -p "$dir"
    
    # Run the Struphy performance test with the specified MPI processes
    # Save the output to a file named based on the process count
    struphy test performance --mpi "$procs" > "${dir}struphy.out"
done

# Summary message to indicate the script has finished running
echo "Performance tests completed. Output saved to ${output_path}"
