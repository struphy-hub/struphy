FROM fedora:36

# install linux packages 
RUN dnf install -y python3-pip \
    && dnf install -y gcc \
    && dnf install -y gfortran \ 
    && dnf install -y blas-devel lapack-devel \ 
    && dnf install -y openmpi openmpi-devel \
    && dnf install -y libgomp \
    && dnf install -y git \
    && dnf install -y environment-modules \
    && dnf install -y python3-mpi4py-openmpi \
    && dnf install -y python3-devel 

# create new working dir
WORKDIR /struphy/

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
