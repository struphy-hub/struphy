FROM fedora:36

# install linux packages
RUN yum install -y dnf 
RUN dnf update -y
RUN dnf install -y gcc-gfortran 
RUN dnf install -y blas-devel lapack-devel 
RUN dnf install -y openmpi-devel 
RUN dnf install -y libgomp 
# RUN yum update -y && yum clean all
# RUN yum install -y gfortran gcc git curl
# RUN yum install -y lapack-devel.x86_64 openmpi.x86_64 
# RUN yum install -y blas.x86_64 epel-release
# RUN yum update -y

# Python 3.8
RUN yum install -y gcc openssl-devel bzip2-devel libffi-devel zlib-devel wget make \
    && cd /opt && wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tgz \
    && tar xzf Python-3.8.12.tgz \
    && cd Python-3.8.12 \
    && ./configure --enable-optimizations \
    && make altinstall \
    && cd .. && rm Python-3.8.12.tgz \
    && python3.8 -V
RUN yum install -y python3 python-pip 

# create new working dir
WORKDIR /struphy/

RUN dnf install -y git

RUN ls /usr/bin/python*

RUN echo "alias python3='python3.8'" >> ~/.aliases 

# install gvec_to_python from submodule
RUN curl -O --header "PRIVATE-TOKEN: glpat-5QH11Kx-65GGiykzR5xo" "https://gitlab.mpcdf.mpg.de/api/v4/projects/5368/jobs/1679220/artifacts/dist/gvec_to_python-0.1.2-py3-none-any.whl" \
    && python3.8 -m pip install --upgrade pip \
    && python3.8 --version && python3 --version && pip --version \
    && pip install gvec_to_python-0.1.2-py3-none-any.whl \
    && pip install sympy==1.6.1 

RUN gcc --version && gfortran --version 

RUN yum install -y openmpi-devel
ENV CC=/usr/lib64/openmpi/bin/mpicc
ENV CC="mpicc"
ENV HDF5_MPI="ON"

# install psydac from submodule
RUN git clone https://github.com/pyccel/psydac.git \
    && echo 'change something here if not using the cache' \
    && cd psydac \
    && git checkout 0be048d57040dc6b62b5d901f9805bcd30e35537 \
    && python3 -m pip install -r requirements.txt \
    && python3 -m pip install -r requirements_extra.txt --no-build-isolation \
    && pip install . 

# compile psydac kernels 
RUN PSYDAC=$(python3 -c "import psydac as _; print(_.__path__[0])") \
    && pyccel $PSYDAC/core/kernels.py \
    && pyccel $PSYDAC/core/bsplines_pyccel.py \ 
    && pyccel $PSYDAC/linalg/kernels.py \
    && pyccel $PSYDAC/feec/dof_kernels.py

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
