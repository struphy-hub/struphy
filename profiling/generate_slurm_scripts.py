import os
from slurm_script_generator.slurm_script import SlurmScript

# Determine paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))

commands = """

echo "========================================"
echo "Starting Struphy setup script"
echo "========================================"
echo

# Parse arguments
LANGUAGE="fortran"         # Default language if none given
echo "Selected compilation language: $LANGUAGE"
echo

# Load modules
echo "----------------------------------------"
echo "Purging and loading modules"
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
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment '.venv'"
    python -m venv .venv
else
    echo "Virtual environment '.venv' already exists"
fi
source .venv/bin/activate
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

# Run Struphy tests
echo "----------------------------------------"
"""

ntasks_per_node_list = [1, 2, 4, 8]
model = "Maxwell"

for ntasks_per_node in ntasks_per_node_list:
    commands += f"""
echo "Running Struphy tests with {ntasks_per_node} MPI ranks"
mpirun --oversubscribe --report-bindings -n {ntasks_per_node} struphy test {model}
"""

custom_commands = commands.splitlines()


    
script = SlurmScript(
    job_name=f"struphy_test_{model}",
    nodes=1,
    ntasks_per_node=max(ntasks_per_node_list),
    time="00:15:00",
    custom_commands=custom_commands,
)
# Save the generated script in the repository root directory
output_path = os.path.join(repo_root, f"job_struphy_test_{model}.sh")
script.save(output_path)
print(f"Generated {output_path}")