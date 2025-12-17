#!/usr/bin/env bash
# Arguments:
# 1: language (for compilation)
# 2: test_type (unit, model, verification, or install; default=install)

# set -euo pipefail

# Parse arguments
LANGUAGE="${1:-fortran}"         # Default language if none given
TEST_TYPE="${2:-install}"        # Default test_type is 'install'

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
if [ ! -d "env" ]; then
    python -m venv env
fi
source env/bin/activate

# Install Struphy
pip install --upgrade pip
pip install --no-binary=mpi4py mpi4py
pip install ".[dev,phys,doc]"
pip list

# Verify struphy installation
struphy -h
struphy --refresh-models

# Test mpirun
which mpirun
mpirun --version
python -c "from mpi4py import MPI; print(MPI)"
mpirun --oversubscribe --report-bindings -n 4 python -c "
from mpi4py import MPI
comm = MPI.COMM_WORLD
print(f'Hello from rank {comm.Get_rank()} of {comm.Get_size()}')
assert comm.Get_size() == 4
"

# Compile kernels
pyccel --version
struphy compile -y --language "$LANGUAGE"

# Run tests based on type
case "$TEST_TYPE" in
    install)
        echo "Install completed. No tests run."
        ;;
    unit)
        struphy test unit --mpi 2
        ;;
    model)
        struphy test models --mpi 2
        ;;
    verification)
        struphy test verification --mpi 2
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        exit 1
        ;;
esac
