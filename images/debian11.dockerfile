FROM debian:11

# install linux packages
RUN apt update -y && apt clean \
    && apt install -y python3-pip \
    && apt install -y gfortran gcc \
    && DEBIAN_FRONTEND=noninteractive TZ="Europe/Berlin" apt-get install -y liblapack-dev libopenmpi-dev \
    && apt install -y libblas-dev openmpi-bin \
    && apt install -y libomp-dev libomp5 \
    && apt install -y git

# create new working dir
WORKDIR /struphy/

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
