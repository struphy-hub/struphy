# Here is how to build the image and upload it to the mpcdf gitlab registry:
#
# We suppose you are in the struphy repo directory. 
# Start the docker engine and run "docker login" with the following token:
#
# TOKEN=gldt-CgMRBMtePbSwdWTxKw4Q; echo "$TOKEN" | docker login gitlab-registry.mpcdf.mpg.de -u gitlab+deploy-token-162 --password-stdin
# docker info
# docker build -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu-latest --provenance=false -f docker/ubuntu-latest.dockerfile .
# docker push gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu-latest

FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

# install linux packages
RUN apt update -y && apt clean \
    && apt install -y software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt update -y \
    && apt install -y python3 \
    && apt install -y python3-dev \
    && apt install -y python3-pip \
    && apt install -y python3-venv \
    && apt install -y gfortran gcc \
    && apt install -y liblapack-dev libblas-dev \
    && apt install -y libopenmpi-dev openmpi-bin \
    && apt install -y libomp-dev libomp5 \
    && apt install -y git \
    && apt install -y pandoc graphviz \
    && bash -c "source ~/.bashrc" \
    # for gvec
    && apt install -y g++ liblapack3 cmake cmake-curses-gui zlib1g-dev libnetcdf-dev libnetcdff-dev \
    && export FC=`which gfortran` \ 
    && export CC=`which gcc` \ 
    && export CXX=`which g++` 

# Create a new working directory
WORKDIR /install_struphy_here/

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

