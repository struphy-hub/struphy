# Here is how to build the image and upload it to the mpcdf gitlab registry:
#
# We suppose you are in the struphy repo directory. 
# Start the docker engine and run "docker login" with the current token from https://struphy.pages.mpcdf.de/struphy/sections/install.html#user-install, then:
#
# docker info
# docker build -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/struphy_opensuse_python_3_11 -f docker/opensuse.dockerfile .
# docker push gitlab-registry.mpcdf.mpg.de/struphy/struphy/struphy_opensuse_python_3_11

FROM opensuse/tumbleweed:latest

# Install other dependencies
RUN zypper refresh \
    && zypper install -y python311 python311-devel \
    && zypper install -y python311-pip python3-virtualenv \
    && zypper install -y gcc-fortran gcc \
    && zypper install -y lapack-devel openmpi-devel \ 
    && zypper install -y blas-devel openmpi \
    && zypper install -y libgomp1 \
    && zypper install -y git \
    && zypper install -y pandoc \ 
    && zypper install -y sqlite3 \
    && zypper install -y vim \
    && zypper install -y make \
    && python3 -m venv /opensuse_latest/venv

# Create a new working directory
WORKDIR /struphy_install/

COPY dist/struphy*.whl .

# Allow mpirun to run as root (for OpenMPI)
ENV PATH="/opensuse_latest/venv/bin:$PATH"
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_pml=ob1
ENV OMPI_MCA_btl=tcp,self
ENV PATH="/usr/lib64/mpi/gcc/openmpi4/bin/:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib64/mpi/gcc/openmpi4/lib64:$LD_LIBRARY_PATH"