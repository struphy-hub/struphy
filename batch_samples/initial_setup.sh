# Load environment.
module purge
module load gcc/9 openmpi/4 anaconda/3/2021.05 mpi4py/3.0.3 h5py-mpi/2.10

# Setup virtual environment.
# pip install --user virtualenv
# python3 -m venv ~/struphy-env
# source ~/struphy-env/bin/activate
# echo `which python`

export PATH="${PATH}:$HOME/.local/bin"
export PYTHONPATH="${PYTHONPATH}:$HOME/.local/bin"

# Clone struphy.
git clone -b maxwell_parallel https://gitlab.mpcdf.mpg.de/clapp/hylife.git struphy
cd struphy

# Install submodules of struphy.
git submodule init
git submodule update
cd gvec_to_python
git pull origin master
pip install . -r requirements.txt
pip install sympy==1.6.1
cd ..
cd psydac
git pull origin devel
export CC="mpicc"
export HDF5_MPI="ON"
export HDF5_DIR=/path/to/hdf5/openmpi
pip install -r requirements.txt
pip install -r requirements_extra.txt --no-build-isolation
pip install .
cd ..

# Install struphy.
# We don't install right after git clone because numpy version.
pip install --user .
struphy compile --user
pip install -U pyccel
