# Here is how to build the image and upload it to the Github package registry:
#
# We suppose you are in the struphy repo directory. 
# Start the docker engine and login to the Github package registry using a github personal acces token (classic):
#
# export CR_PAT=YOUR_TOKEN
# echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
# docker info
# docker build -t ghcr.io/struphy-hub/struphy/opensuse-with-reqs:latest --provenance=false -f docker/opensuse-latest.dockerfile .
# docker push ghcr.io/struphy-hub/struphy/opensuse-with-reqs:latest

FROM opensuse/tumbleweed:latest

RUN echo "Refreshing repositories and installing basic tools..." \
    && zypper clean --all \
    && zypper refresh \
    && zypper install -y grep sed coreutils \
    && zypper install -y pkg-config \
    && zypper install -y meson ninja \
    && zypper clean --all

RUN echo "Installing GCC and MPI libraries..." \
    && zypper install -y gcc-fortran gcc \
    && zypper install -y blas-devel lapack-devel \
    && zypper install -y openmpi openmpi-devel openmpi4-devel \
    && zypper install -y libgomp1 \
    && zypper clean --all

RUN echo "Installing Python and development tools..." \
    && zypper refresh \
    && zypper install -y python3 python3-devel python3-pip python3-virtualenv python3-pkgconfig \
    && python3 --version || echo "Python installation failed!" \
    && zypper clean --all

RUN echo "Installing additional tools..." \
    && zypper install -y git pandoc vim make \
    # for gvec
    && zypper install -y gcc-c++ cmake netcdf \
    && zypper addrepo -G https://download.opensuse.org/repositories/science/openSUSE_Tumbleweed/science.repo \
    && zypper install -y netcdf-fortran-devel \
    && export FC=`which gfortran` \ 
    && export CC=`which gcc` \ 
    && export CXX=`which g++` \
    && zypper clean --all

# Allow mpirun to run as root (for OpenMPI)
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_pml=ob1
ENV OMPI_MCA_btl=tcp,self
ENV PATH="/usr/lib64/mpi/gcc/openmpi4/bin/:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib64/mpi/gcc/openmpi4/lib64:$LD_LIBRARY_PATH"