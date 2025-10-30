# Here is how to build the image and upload it to the Github package registry:
#
# We suppose you are in the struphy repo directory. 
# Start the docker engine and login to the Github package registry using a github personal acces token (classic):
#
# export CR_PAT=YOUR_TOKEN
# echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdinn
# docker info
# docker build -t ghcr.io/struphy-hub/struphy/ubuntu-with-struphy:latest --provenance=false -f docker/ubuntu-latest-with-struphy.dockerfile .
# docker push ghcr.io/struphy-hub/struphy/ubuntu-with-struphy:latest

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

# install three versions of struphy
RUN git clone https://github.com/struphy-hub/struphy.git struphy_c_ \
    && cd struphy_c_ \
    && python3 -m venv env_c_ \
    && . env_c_/bin/activate \
    && pip install -U pip \
    && pip install -e .[phys,mpi,doc] --no-cache-dir \
    && struphy compile --status \
    && struphy compile \
    && deactivate
    
RUN git clone https://github.com/struphy-hub/struphy.git struphy_fortran_\
    && cd struphy_fortran_ \
    && python3 -m venv env_fortran_ \
    && . env_fortran_/bin/activate \
    && pip install -U pip \
    && pip install -e .[phys,mpi,doc] --no-cache-dir \
    && struphy compile --status \
    && struphy compile --language fortran -y \
    && deactivate 

RUN git clone https://github.com/struphy-hub/struphy.git struphy_fortran_--omp-pic\
    && cd struphy_fortran_--omp-pic \
    && python3 -m venv env_fortran_--omp-pic \
    && . env_fortran_--omp-pic/bin/activate \
    && pip install -U pip \
    && pip install -e .[phys,mpi,doc] --no-cache-dir \
    && struphy compile --status \
    && struphy compile --language fortran --omp-pic -y \
    && deactivate 

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
