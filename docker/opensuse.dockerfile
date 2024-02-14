# Here is how to build the image and upload it to the mpcdf gitlab registry:
#
# We suppose you are in the struphy repo directory. Start the docker engine and then run:
#
# docker info
# docker login gitlab-registry.mpcdf.mpg.de -u docker_api -p glpat--z6kJtobeG-xM_LdL6k6
# docker build -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/opensuse -f docker/opensuse.dockerfile .
# docker push gitlab-registry.mpcdf.mpg.de/struphy/struphy/opensuse

FROM opensuse/tumbleweed:latest

# Refresh repository indexes
RUN zypper refresh

# Install Python 3.9 and development tools
RUN zypper install -y python39 python39-devel

RUN zypper install -y python39-pip python3-virtualenv

# Install other dependencies
RUN zypper install -y python3-pip \
    && zypper install -y python3-virtualenv \
    && zypper install -y gcc-fortran gcc \
    && zypper install -y lapack-devel openmpi-devel \
    && zypper install -y blas-devel openmpi \
    && zypper install -y libgomp1 \
    && zypper install -y git \
    && zypper install -y pandoc \
    && zypper install -y sqlite3 \
    && zypper install -y vim \
    && zypper install -y make

# Create a new working directory
WORKDIR /opensuse_latest/

# Create a virtual environment
RUN python3.9 -m venv /opensuse_latest/venv

# Activate the virtual environment
ENV PATH="/opensuse_latest/venv/bin:$PATH"

# Upgrade pip in the virtual environment
RUN pip install --upgrade pip

# Allow mpirun to run as root (for OpenMPI)
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_pml=ob1
ENV OMPI_MCA_btl=tcp,self

# Add the OpenMPI binaries to PATH
ENV PATH="/usr/lib64/mpi/gcc/openmpi4/bin/:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib64/mpi/gcc/openmpi4/lib64:$LD_LIBRARY_PATH"

WORKDIR /opensuse/