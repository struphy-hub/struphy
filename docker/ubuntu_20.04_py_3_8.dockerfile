# Here is how to build the image and upload it to the mpcdf gitlab registry:
#
# We suppose you are in the struphy repo directory. Start the docker engine and then run:
#
# docker info
# docker login gitlab-registry.mpcdf.mpg.de -u docker_api -p glpat--z6kJtobeG-xM_LdL6k6
# docker build -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu -f docker/ubuntu.dockerfile .
# docker push gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu

FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt install -y \
    python3.8 \
    gfortran gcc \
    liblapack-dev libopenmpi-dev \
    libblas-dev openmpi-bin \
    libomp-dev libomp5 \
    git \
    pandoc \
    make \
    python3.8-venv \
    python3.8-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
# Install your Python packages in this environment
RUN python3.8 -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install sympy==1.5 struphy gvec_to_python \
    && struphy compile

# Set environment variable to ensure commands run inside the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# create new working dir
WORKDIR /ubuntu_20_04_py_3_8/

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

