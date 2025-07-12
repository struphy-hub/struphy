# Here is how to build the image and upload it to the mpcdf gitlab registry:
#
# We suppose you are in the struphy repo directory. 
# Start the docker engine and run "docker login" with the following token:
#
# TOKEN=gldt-CgMRBMtePbSwdWTxKw4Q; echo "$TOKEN" | docker login gitlab-registry.mpcdf.mpg.de -u gitlab+deploy-token-162 --password-stdin
# docker info
# docker build -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/mpcdf-gcc-openmpi-with-struphy --provenance=false -f docker/mpcdf-gcc-openmpi-with-struphy.dockerfile .
# docker push gitlab-registry.mpcdf.mpg.de/struphy/struphy/mpcdf-gcc-openmpi-with-struphy

FROM gitlab-registry.mpcdf.mpg.de/mpcdf/ci-module-image/gcc_14-openmpi_5_0:latest

RUN source ./mpcdf/soft/SLE_15/packages/x86_64/Modules/5.4.0/etc/profile.d/modules.sh \
    && module load gcc/14 openmpi/5.0 python-waterboa/2024.06 git graphviz/8 \
    && module load cmake netcdf-serial mkl hdf5-serial \
    && export FC=`which gfortran` \
    && export CC=`which gcc` \
    && export CXX=`which g++` \
    && git clone https://gitlab.mpcdf.mpg.de/struphy/struphy.git struphy_c_ \
    && cd struphy_c_ \
    && python3 -m venv env_c_ \
    && source env_c_/bin/activate \
    && pip install -U pip \
    && pip install -e .[phys] --no-cache-dir --no-binary mpi4py \
    && struphy compile \
    && deactivate
    
RUN source ./mpcdf/soft/SLE_15/packages/x86_64/Modules/5.4.0/etc/profile.d/modules.sh \
    && module load gcc/14 openmpi/5.0 python-waterboa/2024.06 git graphviz/8 \
    && module load cmake netcdf-serial mkl hdf5-serial \
    && export FC=`which gfortran` \
    && export CC=`which gcc` \
    && export CXX=`which g++` \
    && git clone https://gitlab.mpcdf.mpg.de/struphy/struphy.git struphy_fortran_\
    && cd struphy_fortran_ \
    && python3 -m venv env_fortran_ \
    && source env_fortran_/bin/activate \
    && pip install -U pip \
    && pip install -e .[phys] --no-cache-dir --no-binary mpi4py \
    && struphy compile --language fortran -y \
    && deactivate 

RUN source ./mpcdf/soft/SLE_15/packages/x86_64/Modules/5.4.0/etc/profile.d/modules.sh \
    && module load gcc/14 openmpi/5.0 python-waterboa/2024.06 git graphviz/8 \
    && module load cmake netcdf-serial mkl hdf5-serial \
    && export FC=`which gfortran` \
    && export CC=`which gcc` \
    && export CXX=`which g++` \
    && git clone https://gitlab.mpcdf.mpg.de/struphy/struphy.git struphy_fortran_--omp-pic\
    && cd struphy_fortran_--omp-pic \
    && python3 -m venv env_fortran_--omp-pic \
    && source env_fortran_--omp-pic/bin/activate \
    && pip install -U pip \
    && pip install -e .[phys] --no-cache-dir --no-binary mpi4py \
    && struphy compile --language fortran --omp-pic -y \
    && deactivate 