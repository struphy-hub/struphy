# Here is how to build the image and upload it to the mpcdf gitlab registry:
#
# We suppose you are in the struphy repo directory. Start the docker engine and then run:
#
# docker info
# docker login gitlab-registry.mpcdf.mpg.de -u docker_api -p glpat--z6kJtobeG-xM_LdL6k6
# docker build -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/struphy_ubuntu_python_3_11 -f docker/ubuntu.dockerfile .
# docker push gitlab-registry.mpcdf.mpg.de/struphy/struphy/struphy_ubuntu_python_3_11

FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

# install linux packages
RUN apt update -y && apt clean \
    && apt install -y software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt update -y \
    && apt install -y python3.11 \
    && apt install -y python3.11-dev \
    && apt install -y python3-pip \
    && apt install -y python3.11-venv \
    && apt install -y gfortran gcc \
    && apt install -y liblapack-dev libopenmpi-dev \
    && apt install -y libblas-dev openmpi-bin \
    && apt install -y libomp-dev libomp5 \
    && apt install -y git \
    && apt install -y pandoc \
    && apt install -y sqlite3 \
    && bash -c "echo 'alias python3=python3.11' >> ~/.bashrc" \
    && bash -c "source ~/.bashrc" 

# create new working dir
WORKDIR /struphy_install/

COPY dist/struphy*.whl .

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

