# Here is how to build the image and upload it to the mpcdf gitlab registry:
#
# We suppose you are in the struphy repo directory. Start the docker engine and then run:
#
# docker info
# docker login gitlab-registry.mpcdf.mpg.de -u docker_api -p glpat--z6kJtobeG-xM_LdL6k6
# docker build -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/struphy_almalinux_python_3_10 -f docker/almalinux.dockerfile .
# docker push gitlab-registry.mpcdf.mpg.de/struphy/struphy/struphy_almalinux_python_3_10

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
    && wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz \
    && tar xzf Python-3.10.14.tgz \
    && cd Python-3.10.14 \
    && ./configure --with-system-ffi --with-computed-gotos --enable-loadable-sqlite-extensions \
    && make -j ${nproc} \
    && make altinstall \
    && echo "alias python3=python3.10" >> ~/.bashrc \
    && source ~/.bashrc \
    && mv /usr/local/lib/libpython3.10.a libpython3.10.a.bak

# create new working dir
WORKDIR /almalinux_latest/

COPY dist/struphy*.whl .

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_pml=ob1
ENV OMPI_MCA_btl=tcp,self
ENV PATH="/usr/lib64/openmpi/bin:$PATH"