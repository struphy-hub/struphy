# Here is how to build the image and upload it to the mpcdf gitlab registry:
#
# We suppose you are in the struphy repo directory. 
# Start the docker engine and run "docker login" with the current token from https://struphy.pages.mpcdf.de/struphy/sections/install.html#user-install, then:
#
# docker info
# docker build -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu -f docker/ubuntu.dockerfile .
# docker push gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu

FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt install -y \
    python3.9 \
    gfortran gcc \
    liblapack-dev libopenmpi-dev \
    libblas-dev openmpi-bin \
    libomp-dev libomp5 \
    git \
    pandoc \
    make \
    python3.9-venv \
    python3.9-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
# Install your Python packages in this environment
RUN python3.9 -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install sympy==1.5 struphy gvec_to_python \
    && struphy compile

# Set environment variable to ensure commands run inside the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /ubuntu_20_04_py_3.9/

ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1