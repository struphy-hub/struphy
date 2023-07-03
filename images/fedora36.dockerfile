# Here is how to build the image and upload it to the mpcdf gitlab registry:
#
# We suppose you are in the struphy repo directory. Start the docker engine and then run:
#
# docker info
# docker login gitlab-registry.mpcdf.mpg.de -u docker_api -p glpat--z6kJtobeG-xM_LdL6k6
# docker build -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/fedora36 -f images/fedora36.dockerfile .
# docker push gitlab-registry.mpcdf.mpg.de/struphy/struphy/fedora36

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
WORKDIR /your_working_dir/

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
