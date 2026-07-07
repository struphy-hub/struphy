set -euo pipefail

echo "========================================"
echo "Starting Struphy setup"
echo "========================================"
echo

LANGUAGE="fortran"
echo "Selected compilation language: $LANGUAGE"
echo

echo "----------------------------------------"
echo "Purging and loading modules"
echo "----------------------------------------"
module purge
source ./setup/modules.sh load
module list
echo

echo "----------------------------------------"
echo "Setting up Python virtual environment"
echo "----------------------------------------"
which python
python --version
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment '.venv'"
    python -m venv .venv
else
    echo "Virtual environment '.venv' already exists"
fi
source .venv/bin/activate
echo "Activated virtual environment"
echo

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

echo "----------------------------------------"
echo "Verifying Struphy installation"
echo "----------------------------------------"
struphy -h
echo

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

echo "----------------------------------------"
echo "Compiling kernels with Pyccel"
echo "----------------------------------------"
pyccel --version
struphy compile -y --language "$LANGUAGE"
echo
