MODULES_INTEL="intel-oneapi-compilers-classic/2021.10.0 \
intel-oneapi-mkl/2024.0.0--intel-oneapi-mpi--2021.12.1 \
python/3.11.7"


# openmpi/4.1.6--gcc--12.3.0
MODULES_GCC="gcc/12.3.0 \
python/3.11.7 \
hdf5/1.14.3--gcc--12.3.0 \
cmake/3.27.9 \
netcdf-fortran/4.6.1--gcc--12.3.0 \
netlib-scalapack/2.2.0--openmpi--4.1.6--gcc--12.3.0-ucx1.20"


# For GVEC
# Should be fixed so it works with both gcc and intel
export FC=`which gfortran`
export CC=`which gcc`
export CXX=`which g++`
