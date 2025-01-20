# Here is how to build the image and upload it to the mpcdf gitlab registry:
#
# We suppose you are in the struphy repo directory. 
# Start the docker engine and run "docker login" with the current token from https://struphy.pages.mpcdf.de/struphy/sections/install.html#user-install, then:
#
# docker info
# docker build -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/struphy_almalinux_latest -f docker/almalinux.dockerfile .
# docker push gitlab-registry.mpcdf.mpg.de/struphy/struphy/struphy_almalinux_latest

FROM almalinux:latest

RUN yum install -y wget yum-utils make openssl-devel bzip2-devel libffi-devel zlib-devel \
    && yum update -y && yum clean all \
    && yum install -y gcc \ 
    && yum install -y gfortran \ 
    && yum install -y openmpi openmpi-devel \
    && yum install -y libgomp \
    && yum install -y git \
    && yum install -y environment-modules \
    && yum install -y sqlite-devel \
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

# create new working dir
WORKDIR /almalinux_latest/

COPY dist/struphy*.whl .

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_pml=ob1
ENV OMPI_MCA_btl=tcp,self
ENV PATH="/usr/lib64/openmpi/bin:$PATH"
