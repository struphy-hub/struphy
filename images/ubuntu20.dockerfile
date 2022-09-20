FROM ubuntu:20.04

# install linux pacakges
RUN apt update -y && apt clean \
    && apt install -y gfortran gcc git curl vim \
    && DEBIAN_FRONTEND=noninteractive TZ="Europe/Berlin" apt-get install -y liblapack-dev libopenmpi-dev \
    && apt install -y libblas-dev openmpi-bin libomp-dev libomp5 libhdf5-openmpi-dev \
    && apt install -y python3-pip python3-mpi4py python3-venv

# create new working dir
WORKDIR /struphy/

# install gvec_to_python 
RUN curl -O --header "PRIVATE-TOKEN: glpat-5QH11Kx-65GGiykzR5xo" "https://gitlab.mpcdf.mpg.de/api/v4/projects/5368/jobs/1679220/artifacts/dist/gvec_to_python-0.1.2-py3-none-any.whl" \
    && pip install gvec_to_python-0.1.2-py3-none-any.whl \
    && pip install sympy==1.6.1 \
    && rm gvec_to_python-0.1.2-py3-none-any.whl

# install psydac from submodule
RUN git clone https://github.com/pyccel/psydac.git \
    && cd psydac \
    && git checkout 0be048d57040dc6b62b5d901f9805bcd30e35537 \
    && python3 -m pip install -r requirements.txt \
    && python3 -m pip install -r requirements_extra.txt --no-build-isolation \
    && pip install . \
    && cd .. \
    && rm -rf psydac/ 

# compile psydac kernels 
RUN PSYDAC=$(python3 -c "import psydac as _; print(_.__path__[0])") \
    && pyccel $PSYDAC/core/kernels.py \
    && pyccel $PSYDAC/core/bsplines_pyccel.py 

# additional python pacakges and alias
RUN pip install pytest coverage h5py pylint build wheel \
    && echo 'alias python="python3"' >> /etc/bash.bashrc 

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

