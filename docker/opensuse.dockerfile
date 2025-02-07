# Here is how to build the image and upload it to the mpcdf gitlab registry:
#
# We suppose you are in the struphy repo directory. 
# Start the docker engine and run "docker login" with the current token from https://struphy.pages.mpcdf.de/struphy/sections/install.html#user-install, then:
#
# docker info
# docker build -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/struphy_opensuse_latest -f docker/opensuse.dockerfile .
# docker push gitlab-registry.mpcdf.mpg.de/struphy/struphy/struphy_opensuse_latest

FROM opensuse/tumbleweed:latest

RUN echo "Refreshing repositories and installing basic tools..." \
    && zypper clean --all \
    && zypper refresh \
    && zypper install -y grep sed coreutils \
    && zypper install -y pkg-config \
    && zypper install -y meson ninja \
    && zypper clean --all

RUN echo "Installing Python and development tools..." \
    && zypper refresh \
    && zypper install -y python3 python3-devel python3-pip python3-venv python3-pkgconfig \
    && python3 --version || echo "Python installation failed!" \
    && zypper clean --all

RUN echo "Installing GCC and MPI libraries..." \
    && zypper install -y gcc-fortran gcc \
    && zypper install -y lapack-devel openmpi-devel \
    && zypper install -y blas-devel openmpi \
    && zypper install -y libgomp1 \
    && zypper clean --all

RUN echo "Installing additional tools..." \
    && zypper install -y git pandoc sqlite3 vim make \
    && zypper clean --all

RUN echo "Setting up Python virtual environment..." \
    && python3 -m venv /opensuse_latest/venv \
    && opensuse_latest/venv/bin/pip install --upgrade pip

RUN echo "Reinstalling python3-devel..." \
    && zypper install -y python3-devel

# Create a new working directory
WORKDIR /struphy_install/

# Allow mpirun to run as root (for OpenMPI)
ENV PATH="/opensuse_latest/venv/bin:$PATH"
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_pml=ob1
ENV OMPI_MCA_btl=tcp,self
ENV PATH="/usr/lib64/mpi/gcc/openmpi4/bin/:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib64/mpi/gcc/openmpi4/lib64:$LD_LIBRARY_PATH"