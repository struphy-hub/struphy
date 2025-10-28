# Here is how to build the image and upload it to the Github package registry:
#
# We suppose you are in the struphy repo directory. 
# Start the docker engine and login to the Github package registry using a github personal acces token (classic):
#
# export CR_PAT=YOUR_TOKEN
# echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
# docker info
# docker build -t ghcr.io/struphy-hub/struphy/ubuntu-with-reqs:latest --provenance=false -f docker/ubuntu-latest.dockerfile .
# docker push ghcr.io/struphy-hub/struphy/ubuntu-with-reqs:latest

FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

# install linux packages
RUN apt update -y && apt clean \
    && apt install -y software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt update -y 

RUN apt install -y python3 \
    && apt install -y python3-dev \
    && apt install -y python3-pip \
    && apt install -y python3-venv 

RUN apt install -y gfortran gcc \
    && apt install -y liblapack-dev libblas-dev 

RUN apt install -y libopenmpi-dev openmpi-bin \
    && apt install -y libomp-dev libomp5 

RUN apt install -y git \
    && apt install -y pandoc graphviz \
    && bash -c "source ~/.bashrc" 

# for gvec
RUN apt install -y g++ liblapack3 cmake cmake-curses-gui zlib1g-dev libnetcdf-dev libnetcdff-dev \
    && export FC=`which gfortran` \ 
    && export CC=`which gcc` \ 
    && export CXX=`which g++` 

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

