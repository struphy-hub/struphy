#!/usr/bin/env bash
# set -euo pipefail

# Load HPC modules
module purge
#module load "$1"   # Pass modules as first argument
source ./setup/modules.sh load
module list

# For gvec
export FC=`which gfortran`
export CC=`which gcc`
export CXX=`which g++`

# Python virtual environment setup
which python
python --version
python -m venv env
source env/bin/activate

# Install Struphy
pip install --upgrade pip
pip install --no-binary=mpi4py mpi4py
pip install ".[dev,phys,doc]"
pip list
struphy -h
struphy --refresh-models

# Test mpirun
# echo "OMPI oversubscribe: $OMPI_MCA_rmaps_base_oversubscribe"
which mpirun
mpirun --version
python -c "from mpi4py import MPI; print(MPI)"
mpirun --oversubscribe --report-bindings -n 4 python -c "from mpi4py import MPI; comm=MPI.COMM_WORLD; print(f'Hello from rank {comm.Get_rank()} of {comm.Get_size()}'); assert comm.Get_size()==4"

# Compile kernels
pyccel --version
struphy compile -y # --language "$2"  # Pass compile language as second argument

# # Run tests depending on type
# TEST_TYPE="$3"  # unit, model, or verification
# case "$TEST_TYPE" in
#   unit)
#     struphy test Maxwell
#     struphy test unit --mpi 2
#     ;;
#   model)
#     struphy test Maxwell
#     struphy test models --mpi 2
#     ;;
#   verification)
#     struphy test Maxwell
#     struphy test verification --mpi 2
#     ;;
#   *)
#     echo "Unknown test type: $TEST_TYPE"
#     exit 1
#     ;;
# esac
