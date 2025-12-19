#!/usr/bin/env bash
# Arguments:
# 1: language (for compilation)
# 2: test_type (unit, model, verification, or install; default=install)

# set -euo pipefail

echo "========================================"
echo "Starting Struphy HPC setup script"
echo "========================================"
echo

# Parse arguments
LANGUAGE="${1:-fortran}"         # Default language if none given
TEST_TYPE="${2:-install}"        # Default test_type is 'install'
echo "Selected compilation language: $LANGUAGE"
echo "Selected test type: $TEST_TYPE"
echo

# Load HPC modules
echo "----------------------------------------"
echo "Purging and loading HPC modules"
echo "----------------------------------------"
module purge
source ./setup/modules.sh load
module list
echo

# Python virtual environment setup
echo "----------------------------------------"
echo "Setting up Python virtual environment"
echo "----------------------------------------"
which python
python --version
if [ ! -d "env" ]; then
    echo "Creating virtual environment 'env'"
    python -m venv env
else
    echo "Virtual environment 'env' already exists"
fi
source env/bin/activate
echo "Activated virtual environment"
echo

# Install Struphy and dependencies
echo "----------------------------------------"
echo "Installing Struphy and dependencies"
echo "----------------------------------------"
pip install --upgrade pip
pip install -e ".[dev]"
pip uninstall mpi4py -y
pip install --no-binary=mpi4py mpi4py
echo "Installed packages:"
pip list
echo

# Verify struphy installation
echo "----------------------------------------"
echo "Verifying Struphy installation"
echo "----------------------------------------"
struphy -h
struphy --refresh-models
echo

# Test mpirun
echo "----------------------------------------"
echo "Testing MPI installation"
echo "----------------------------------------"
echo "mpirun path:"
which mpirun
mpirun --version
python -c "from mpi4py import MPI; print('mpi4py import successful:', MPI)"
echo
echo "Running test with 4 MPI ranks:"
mpirun --oversubscribe --report-bindings -n 4 python -c "
from mpi4py import MPI
comm = MPI.COMM_WORLD
print(f'Hello from rank {comm.Get_rank()} of {comm.Get_size()}')
assert comm.Get_size() == 4
"
echo

# Compile kernels
echo "----------------------------------------"
echo "Compiling kernels with Pyccel"
echo "----------------------------------------"
pyccel --version
struphy compile -y --language "$LANGUAGE"
echo

# Test Maxwell model
echo "----------------------------------------"
echo "Testing Maxwell model"
echo "----------------------------------------"
struphy params Maxwell -y
echo "Running single-process test:"
python params_Maxwell.py
echo "Running 4-process MPI test:"
mpirun -n 4 python params_Maxwell.py
echo

# Model tests
echo "----------------------------------------"
echo "Running Model tests (2 MPI processes)"
echo "----------------------------------------"
struphy test models --mpi 2
echo

# Verification tests
echo "----------------------------------------"
echo "Running Verification tests (2 MPI processes)"
echo "----------------------------------------"
struphy test verification --mpi 2
echo

# Unit tests
# echo "----------------------------------------"
# echo "Running Unit tests (2 MPI processes)"
# echo "----------------------------------------"
# struphy test unit --mpi 2
# echo

# # Run tests based on type
# case "$TEST_TYPE" in
#     install)
#         echo "Install completed. No tests run."
#         ;;
#     unit)
#         struphy test unit --mpi 2
#         ;;
#     model)
#         struphy test models --mpi 2
#         ;;
#     verification)
#         struphy test verification --mpi 2
#         ;;
#     *)
#         echo "Unknown test type: $TEST_TYPE"
#         exit 1
#         ;;
# esac
