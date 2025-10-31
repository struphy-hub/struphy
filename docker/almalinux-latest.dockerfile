# Here is how to build the image and upload it to the Github package registry:
#
# We suppose you are in the struphy repo directory. 
# Start the docker engine and login to the Github package registry using a github personal acces token (classic):
#
# export CR_PAT=YOUR_TOKEN
# echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
# docker info
# docker build -t ghcr.io/struphy-hub/struphy/almalinux-with-reqs:latest --provenance=false -f docker/almalinux-latest.dockerfile .
# docker push ghcr.io/struphy-hub/struphy/almalinux-with-reqs:latest

FROM almalinux:latest

RUN echo "Refreshing repositories and installing basic tools..." \
    && yum install -y wget yum-utils make openssl-devel bzip2-devel libffi-devel zlib-devel \
    && yum update -y && yum clean all 

RUN echo "Installing GCC and MPI libraries..." \
    && yum install -y gcc \ 
    && yum install -y gfortran \ 
    && yum install -y openmpi openmpi-devel \
    && yum install -y libgomp \
    && yum install -y git \
    && yum install -y environment-modules 

RUN echo "Installing Python and development tools..." \
    && wget https://www.python.org/ftp/python/3.12.8/Python-3.12.8.tgz \
    && tar xzf Python-3.12.8.tgz \
    && cd Python-3.12.8 \
    && ./configure --with-system-ffi --with-computed-gotos --enable-loadable-sqlite-extensions \
    && make -j $(nproc) \
    && make altinstall \
    && alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1 \
    && alternatives --set python3 /usr/local/bin/python3.12 \
    && mv /usr/local/lib/libpython3.12.a libpython3.12.a.bak \
    && python3 -V  # Verify Python version

RUN echo "Installing additional tools..." \
    && yum install -y g++ cmake which flexiblas-devel \
    && yum install -y epel-release \
    && yum install -y netcdf-devel netcdf-fortran-devel \
    && export FC=`which gfortran` \ 
    && export CC=`which gcc` \ 
    && export CXX=`which g++`

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_pml=ob1
ENV OMPI_MCA_btl=tcp,self
ENV PATH="/usr/lib64/openmpi/bin:$PATH"
