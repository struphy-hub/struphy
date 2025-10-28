# Here is how to build the image and upload it to the Github package registry:
#
# We suppose you are in the struphy repo directory. 
# Start the docker engine and login to the Github package registry using a github personal acces token (classic):
#
# export CR_PAT=YOUR_TOKEN
# echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
# docker info
# docker build -t ghcr.io/struphy-hub/struphy/fedora-with-reqs:latest --provenance=false -f docker/fedora-latest.dockerfile .
# docker push ghcr.io/struphy-hub/struphy/fedora-with-reqs:latest

FROM fedora:latest

RUN echo "Refreshing repositories and installing basic tools..." \
    && dnf install -y wget yum-utils make openssl-devel bzip2-devel libffi-devel zlib-devel \
    && dnf update -y 

RUN echo "Installing GCC and MPI libraries..." \
    && dnf install -y gcc \ 
    && dnf install -y gfortran \ 
    && dnf install -y blas-devel lapack-devel \ 
    && dnf install -y openmpi openmpi-devel \
    && dnf install -y git \
    && dnf install -y pandoc 
    
RUN echo "Installing Python and development tools..." \
    && dnf install -y python3-devel \
    && dnf install -y python3-mpi4py-openmpi \
    && python3 -m ensurepip \
    && python3 -V 

RUN echo "Installing additional tools..." \
    && dnf install -y g++ cmake netcdf netcdf-devel netcdf-fortran netcdf-fortran-devel pkgconf \
    && export FC=`which gfortran` \ 
    && export CC=`which gcc` \ 
    && export CXX=`which g++` 

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_pml=ob1
ENV OMPI_MCA_btl=tcp,self