# Here is how to build the image and upload it to the mpcdf gitlab registry:
#
# We suppose you are in the struphy repo directory. 
# Start the docker engine and run "docker login" with the following token:
#
# TOKEN=gldt-CgMRBMtePbSwdWTxKw4Q; echo "$TOKEN" | docker login gitlab-registry.mpcdf.mpg.de -u gitlab+deploy-token-162 --password-stdin
# docker info
# docker build -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu-latest-with-struphy --provenance=false -f docker/ubuntu-latest-with-struphy.dockerfile .
# docker push gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu-latest-with-struphy

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

# install three versions of struphy
RUN git clone https://gitlab.mpcdf.mpg.de/struphy/struphy.git struphy_c_ \
    && cd struphy_c_ \
    && python3 -m venv env_c_ \
    && . env_c_/bin/activate \
    && pip install -U pip \
    && pip install -e .[phys] --no-cache-dir \
    && struphy compile \
    && deactivate
    
RUN git clone https://gitlab.mpcdf.mpg.de/struphy/struphy.git struphy_fortran_\
    && cd struphy_fortran_ \
    && python3 -m venv env_fortran_ \
    && . env_fortran_/bin/activate \
    && pip install -U pip \
    && pip install -e .[phys] --no-cache-dir \
    && struphy compile --language fortran -y \
    && deactivate 

RUN git clone https://gitlab.mpcdf.mpg.de/struphy/struphy.git struphy_fortran_--omp-pic\
    && cd struphy_fortran_--omp-pic \
    && python3 -m venv env_fortran_--omp-pic \
    && . env_fortran_--omp-pic/bin/activate \
    && pip install -U pip \
    && pip install -e .[phys] --no-cache-dir \
    && struphy compile --language fortran --omp-pic -y \
    && deactivate 

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
