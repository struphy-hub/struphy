FROM ubuntu:20.04

# install linux packages
RUN apt update -y && apt clean \
    && apt install -y gfortran gcc git curl vim \
    && DEBIAN_FRONTEND=noninteractive TZ="Europe/Berlin" apt-get install -y liblapack-dev libopenmpi-dev \
    && apt install -y libblas-dev openmpi-bin libomp-dev libomp5 libhdf5-openmpi-dev \
    && apt install -y python3-pip python3-mpi4py python3-venv

# create new working dir
WORKDIR /struphy/

# some python pacakges and alias
RUN pip install pytest coverage h5py pylint build wheel \
    && echo 'alias python="python3"' >> /etc/bash.bashrc 

# install gvec_to_python 
RUN curl -O --header "PRIVATE-TOKEN: glpat-5QH11Kx-65GGiykzR5xo" "https://gitlab.mpcdf.mpg.de/api/v4/projects/5368/jobs/2080073/artifacts/dist/gvec_to_python-1.0.2-py3-none-any.whl" \
    && pip install gvec_to_python-1.0.2-py3-none-any.whl \
    && rm gvec_to_python-1.0.2-py3-none-any.whl 

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

