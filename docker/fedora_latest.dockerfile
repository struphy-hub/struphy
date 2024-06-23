# Here is how to build the image and upload it to the mpcdf gitlab registry:
#
# We suppose you are in the struphy repo directory. Start the docker engine and then run:
#
# docker info
# docker login gitlab-registry.mpcdf.mpg.de -u docker_api -p glpat--z6kJtobeG-xM_LdL6k6
# docker build -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/fedora -f docker/fedora.dockerfile .
# docker push gitlab-registry.mpcdf.mpg.de/struphy/struphy/fedora

FROM fedora:latest

# install linux packages 
RUN dnf install -y wget yum-utils make gcc openssl-devel bzip2-devel libffi-devel zlib-devel \
    && dnf update -y \
    && dnf install -y gfortran \ 
    && dnf install -y blas-devel lapack-devel \ 
    && dnf install -y openmpi openmpi-devel \
    && dnf install -y libgomp \
    && dnf install -y git \
    && dnf install -y environment-modules \
    && dnf install -y python3-mpi4py-openmpi \
    && dnf install -y python3-devel \
    # && wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz \
    # && tar xzf Python-3.10.14.tgz \
    # && cd Python-3.10.14 \
    # && ./configure --with-system-ffi --with-computed-gotos --enable-loadable-sqlite-extensions \
    # && make -j ${nproc} \
    # && make altinstall \
    # $$ echo "alias python3=python3.10" >> ~/.bashrc \
    # $$ source ~/.bashrc \
    && dnf install -y pandoc \
    && dnf install -y sqlite

# create new working dir
WORKDIR /work_dir/

COPY . .

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_pml=ob1
ENV OMPI_MCA_btl=tcp,self
